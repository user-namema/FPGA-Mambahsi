"""
测量当前严格空间划分 BothMamba FP32/QAT 模型在 GPU 上的tile延迟和整图吞吐。

整幅场景仍按训练/部署协议分成 16x16 tile。默认 --batch-size=1，
与FPGA逐tile执行协议对齐；也可调大batch测量GPU吞吐。UP 610x340场景
对应 ceil(610/16) * ceil(340/16) = 858 个tile。

脚本报告三组主要时间：
  1. single_tile：一个已在GPU的16x16 tile的单次网络前向延迟；
  2. model_only：整图全部tile的网络前向，不含上采样/argmax/拼接；
  3. full_gpu_pipeline：网络前向 + 上采样 + argmax + 整图拼接。

整图测试默认把tile预先放在GPU，用于测量稳态GPU推理。加上
--measure-h2d 后，还会报告每个batch现场从CPU传到GPU的墙钟时间。

默认 --model-kind=fp32，用于测量不包含fake-quant开销的模型本体。
--model-kind=qat 测量 QAT fake-quant；--model-kind=fixed 从同一 QAT checkpoint
编译硬件定点语义路径（FP64精确整数MAC、INT64状态），计时前与FPGA参考逐码核对。

运行示例：
    python benchmark_gpu_batch1_dense16.py \
        --qat-run-dir results/SPATIAL_SPLIT_3WAY_DENSE_QAT/\
UP_int8_lsq_freeze20_detach_bias_lr1/run_seed0 \
        --dataset UP \
        --data-path ./data \
        --output-dir results/GPU_BENCHMARK_BATCH1/UP_seed0 \
        --batch-size 1 \
        --warmup-tiles 200 \
        --repeats 20 \
        --single-tile-trials 200 \
        --measure-h2d
"""

import argparse
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analyze_alog_dynamics import restore_model
from both_FPGA_single_qat_source import (
    TILE_SIZE, extract_state_dict, infer_fp32_dir, json_default,
    load_json_object, load_spatial_masks, make_dense16_tile,
    resolve_existing_path, transform_with_saved_preprocess,
    validate_dense16_blocks, validate_qat_run_metadata, validate_raw_data,
)
from train_mambahsi_spatial_split_dense_qat import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU可配batch的16x16 tile延迟与整图推理计时"
    )
    parser.add_argument(
        "--model-kind",
        choices=("fp32", "qat", "fixed"),
        default="fp32",
        help="fp32=量化前；qat=fake-quant；fixed=硬件定点语义GPU模拟",
    )
    parser.add_argument(
        "--qat-run-dir",
        default=None,
        help="QAT模式必需；FP32模式仅在未给--fp32-dir时用于推断路径",
    )
    parser.add_argument("--dataset", default="UP")
    parser.add_argument("--fixed-verify-tiles", type=int, default=3,
                        help="定点模式计时前与原FPGA模拟器逐码验证的tile数，至少2")
    parser.add_argument("--data-path", default="./data")
    parser.add_argument("--output-dir", default="./results/GPU_BENCHMARK_BATCH1")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--fp32-dir", default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--token-num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="整图测试每次合并的tile数；1与FPGA协议对齐",
    )
    parser.add_argument(
        "--warmup-steps",
        "--warmup-tiles",
        dest="warmup_steps",
        type=int,
        default=100,
        help="整图正式计时前执行的batch前向次数",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="整幅场景重复计时次数",
    )
    parser.add_argument(
        "--measure-h2d",
        action="store_true",
        help="额外测量每个batch从CPU传到GPU的整图管线",
    )
    parser.add_argument(
        "--single-tile-warmup",
        type=int,
        default=200,
        help="单tile延迟正式计时前的预热次数",
    )
    parser.add_argument(
        "--single-tile-trials",
        type=int,
        default=200,
        help="单tile同步延迟重复次数；0表示不测",
    )
    parser.add_argument(
        "--single-tile-index",
        type=int,
        default=0,
        help="选择第几个tile进行单tile延迟测试",
    )
    parser.add_argument(
        "--single-tile-only",
        action="store_true",
        help="只测一个tile，跳过整图测试",
    )
    parser.add_argument(
        "--save-prediction",
        action="store_true",
        help="保存最后一次整图预测（0-based类别索引）",
    )
    parser.add_argument('--allow-reference-scan', action='store_true',
                        help='Allow slow torch_reference (reported separately); default requires CUDA scan')
    parser.add_argument('--allow-tf32', action='store_true', help='Enable TF32 explicitly; default disabled')
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_seconds(values, tile_count, pixel_count, forward_calls):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(values <= 0):
        raise ValueError("计时结果必须是非空正数序列。")
    median_seconds = float(np.median(values))
    mean_seconds = float(np.mean(values))
    return {
        "repeats": int(values.size),
        "samples_ms": (values * 1000.0).tolist(),
        "mean_ms_per_scene": mean_seconds * 1000.0,
        "std_ms_per_scene": float(np.std(values, ddof=0)) * 1000.0,
        "median_ms_per_scene": median_seconds * 1000.0,
        "min_ms_per_scene": float(np.min(values)) * 1000.0,
        "p95_ms_per_scene": percentile(values, 95) * 1000.0,
        "median_ms_per_tile": median_seconds * 1000.0 / tile_count,
        "median_ms_per_forward_call": (
            median_seconds * 1000.0 / forward_calls
        ),
        "median_scenes_per_second": 1.0 / median_seconds,
        "median_tiles_per_second": tile_count / median_seconds,
        "median_megapixels_per_second": pixel_count / median_seconds / 1e6,
    }


def print_summary(
    title,
    summary,
    unit_label="full scene",
    rate_label="scene",
):
    print(f"\n[{title}]")
    print(
        f"  {unit_label}: "
        f"median={summary['median_ms_per_scene']:.3f} ms, "
        f"mean={summary['mean_ms_per_scene']:.3f} +/- "
        f"{summary['std_ms_per_scene']:.3f} ms, "
        f"min={summary['min_ms_per_scene']:.3f} ms, "
        f"p95={summary['p95_ms_per_scene']:.3f} ms"
    )
    print(
        f"  per tile  : {summary['median_ms_per_tile']:.6f} ms"
    )
    print(
        "  per call  : "
        f"{summary['median_ms_per_forward_call']:.6f} ms"
    )
    print(
        f"  throughput: {summary['median_scenes_per_second']:.4f} "
        f"{rate_label}/s, "
        f"{summary['median_megapixels_per_second']:.4f} Mpixel/s"
    )


def resolve_artifacts(args):
    qat_run_dir = None
    qat_result = {}
    if args.qat_run_dir is not None:
        qat_run_dir = resolve_existing_path(args.qat_run_dir, "QAT run目录")
        if not qat_run_dir.is_dir():
            raise NotADirectoryError(f"QAT run路径不是目录：{qat_run_dir}")
        qat_result = load_json_object(qat_run_dir / "result.json")
        validate_qat_run_metadata(qat_result, args.dataset, args.seed)

    if args.fp32_dir is not None:
        fp32_dir = resolve_existing_path(args.fp32_dir, "FP32数据集结果目录")
    elif qat_run_dir is not None:
        fp32_dir = infer_fp32_dir(qat_result, qat_run_dir)
    else:
        raise ValueError(
            "FP32模式请提供--fp32-dir；或提供--qat-run-dir供脚本推断"
            "与QAT对齐的FP32结果目录。"
        )
    if not fp32_dir.is_dir():
        raise NotADirectoryError(f"FP32结果路径不是目录：{fp32_dir}")

    if args.model_kind in ("qat", "fixed"):
        if qat_run_dir is None:
            raise ValueError("--model-kind=qat时必须提供--qat-run-dir。")
        default_model_path = qat_run_dir / "best_qat_foldaware.pth"
        model_label = "QAT best_qat_foldaware.pth"
        run_dir = qat_run_dir
        metadata = qat_result
    else:
        run_dir = fp32_dir / f"run_seed{args.seed}"
        default_model_path = run_dir / "best_model.pth"
        model_label = "FP32 best_model.pth"
        result_path = run_dir / "result.json"
        metadata = load_json_object(result_path) if result_path.exists() else {}
        if "dataset" in metadata and str(metadata["dataset"]) != args.dataset:
            raise RuntimeError(
                f"FP32 result.json数据集不匹配：{metadata['dataset']} != {args.dataset}"
            )
        if "seed" in metadata and int(metadata["seed"]) != int(args.seed):
            raise RuntimeError(
                f"FP32 result.json seed不匹配：{metadata['seed']} != {args.seed}"
            )

    model_path = resolve_existing_path(
        default_model_path if args.model_path is None else args.model_path,
        model_label,
        roots=(run_dir, fp32_dir),
    )
    return run_dir, metadata, model_path, fp32_dir


def load_scene_and_blocks(args, fp32_dir):
    preprocess_start = time.perf_counter()
    raw_data, raw_gt, _ = load_dataset(args.dataset, args.data_path)
    raw_data, gt, class_count = validate_raw_data(
        raw_data, raw_gt, args.dataset
    )
    image = transform_with_saved_preprocess(
        raw_data,
        fp32_dir / "train_only_preprocess.npz",
    )
    masks = load_spatial_masks(fp32_dir, gt.shape)
    split_info = load_json_object(fp32_dir / "spatial_split.json")
    blocks = validate_dense16_blocks(split_info, masks, gt.shape)
    preprocess_seconds = time.perf_counter() - preprocess_start

    expected_tiles = math.ceil(gt.shape[0] / TILE_SIZE) * math.ceil(
        gt.shape[1] / TILE_SIZE
    )
    if len(blocks) != expected_tiles:
        raise RuntimeError(
            f"tile数量异常：保存协议为{len(blocks)}，"
            f"按场景尺寸计算应为{expected_tiles}。"
        )
    return image, gt, class_count, blocks, preprocess_seconds


def load_deploy_model(
    args,
    model_metadata,
    model_path,
    image,
    blocks,
    class_count,
    device,
):
    run = Path(model_path).parent
    # Load every architecture field from the saved configuration, including D/z/A.
    net, config = restore_model(run, device, checkpoint=model_path,
                                artifact_dir=args.fp32_dir)
    if int(config['in_channels']) != image.shape[2] or int(config['num_classes']) != class_count:
        raise ValueError('Checkpoint input channels/classes do not match this dataset')
    for name in ('hidden_dim', 'token_num'):
        override = getattr(args, name, None)
        if override is not None and int(override) != int(config[name]):
            raise ValueError('Requested {} differs from checkpoint'.format(name))
    expected = 'QAT' if args.model_kind in ('qat', 'fixed') else 'FP32'
    if net._analysis_checkpoint_kind != expected:
        raise ValueError('Checkpoint kind does not match --model-kind')
    import importlib
    module = importlib.import_module(type(net).__module__)
    backend = module.selective_scan_backend()
    if args.model_kind != 'fixed' and backend != 'mamba_ssm_optimized' and not getattr(args, 'allow_reference_scan', False):
        raise RuntimeError('CUDA selective scan unavailable; install matching mamba_ssm or explicitly use --allow-reference-scan')
    net._benchmark_config = config
    net._benchmark_scan_backend = backend
    if args.model_kind == "fixed":
        from fixed_gpu_backend import FixedGPUModel
        net = FixedGPUModel(net, config)
    return net, int(config['hidden_dim']), int(config['token_num'])


def build_cpu_tiles(image, blocks):
    records = []
    for block in blocks:
        tile, valid_height, valid_width = make_dense16_tile(image, block)
        if tile.shape[0] != 1 or tuple(tile.shape[-2:]) != (TILE_SIZE, TILE_SIZE):
            raise RuntimeError(f"tile形状不符合batch=1协议：{tuple(tile.shape)}")
        records.append(
            {
                "tile": tile.to(dtype=torch.float32).contiguous(),
                "top": int(block["top"]),
                "left": int(block["left"]),
                "valid_height": int(valid_height),
                "valid_width": int(valid_width),
            }
        )
    return records


def preload_gpu_tiles(cpu_records, device):
    return [
        {
            **record,
            "tile": record["tile"].to(device=device, non_blocking=False),
        }
        for record in cpu_records
    ]


def make_tile_batches(records, batch_size):
    """在计时外预先组batch，避免把torch.cat时间误算为模型推理。"""
    if batch_size <= 0:
        raise ValueError("batch_size必须为正整数。")
    batches = []
    for start in range(0, len(records), batch_size):
        group = records[start : start + batch_size]
        batches.append(
            {
                "tiles": torch.cat([record["tile"] for record in group], dim=0),
                "items": [
                    {key: value for key, value in record.items() if key != "tile"}
                    for record in group
                ],
            }
        )
    return batches


def preload_gpu_batches(cpu_batches, device):
    return [
        {
            "tiles": batch["tiles"].to(device=device, non_blocking=False),
            "items": batch["items"],
        }
        for batch in cpu_batches
    ]


@torch.inference_mode()
def warm_up(net, gpu_batches, warmup_steps):
    if warmup_steps <= 0:
        return
    for batch in (gpu_batches[0], gpu_batches[-1]):
        net(batch["tiles"])  # warm full and tail shapes
    for index in range(warmup_steps):
        logits = net(gpu_batches[index % len(gpu_batches)]["tiles"])
        dense = F.interpolate(
            logits,
            size=(TILE_SIZE, TILE_SIZE),
            mode="bilinear",
            align_corners=True,
        )
        torch.argmax(dense, dim=1)
    torch.cuda.synchronize()


def timed_cuda_call(function):
    """同时返回同步墙钟时间和CUDA event时间。"""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start_event.record()
    result = function()
    end_event.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    device_seconds = start_event.elapsed_time(end_event) / 1000.0
    return wall_seconds, device_seconds, result


@torch.inference_mode()
def run_model_only(net, gpu_batches):
    last_logits = None
    for batch in gpu_batches:
        last_logits = net(batch["tiles"])
    return last_logits


@torch.inference_mode()
def run_full_pipeline(net, batches, scene_shape, device, transfer_each_batch):
    canvas = torch.empty(scene_shape, dtype=torch.uint8, device=device)
    for batch in batches:
        if transfer_each_batch:
            tiles = batch["tiles"].to(device=device, non_blocking=False)
        else:
            tiles = batch["tiles"]
        logits = net(tiles)
        dense_logits = F.interpolate(
            logits,
            size=(TILE_SIZE, TILE_SIZE),
            mode="bilinear",
            align_corners=True,
        )
        prediction = torch.argmax(dense_logits, dim=1).to(torch.uint8)
        for item_index, item in enumerate(batch["items"]):
            top = item["top"]
            left = item["left"]
            valid_height = item["valid_height"]
            valid_width = item["valid_width"]
            canvas[
                top : top + valid_height,
                left : left + valid_width,
            ] = prediction[
                item_index,
                :valid_height,
                :valid_width,
            ]
    return canvas


def benchmark_repeated(title, repeats, function):
    wall_samples = []
    device_samples = []
    last_result = None
    for repeat_index in range(repeats):
        wall_seconds, device_seconds, last_result = timed_cuda_call(function)
        wall_samples.append(wall_seconds)
        device_samples.append(device_seconds)
        print(
            f"  {title} {repeat_index + 1:02d}/{repeats}: "
            f"wall={wall_seconds * 1000.0:.3f} ms, "
            f"CUDA={device_seconds * 1000.0:.3f} ms"
        )
    return wall_samples, device_samples, last_result


@torch.inference_mode()
def benchmark_single_tile(net, tile, warmup_steps, trials):
    """
    每次只调用一次net(tile)，并在每次前后同步。

    CUDA event结果表示该次前向在CUDA时间线上的延迟；墙钟结果
    还包含Python调度和同步的主机开销。
    """
    for _ in range(warmup_steps):
        net(tile)
    torch.cuda.synchronize()

    wall_samples = []
    device_samples = []
    progress_every = max(1, trials // 10)
    for trial in range(trials):
        wall_seconds, device_seconds, _ = timed_cuda_call(lambda: net(tile))
        wall_samples.append(wall_seconds)
        device_samples.append(device_seconds)
        if (
            trial == 0
            or trial + 1 == trials
            or (trial + 1) % progress_every == 0
        ):
            print(
                f"  single_tile {trial + 1:04d}/{trials}: "
                f"wall={wall_seconds * 1000.0:.3f} ms, "
                f"CUDA={device_seconds * 1000.0:.3f} ms"
            )
    return wall_samples, device_samples


@torch.inference_mode()
def main():
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats必须为正整数。")
    if args.batch_size <= 0:
        raise ValueError("--batch-size必须为正整数。")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps不能为负数。")
    if args.single_tile_warmup < 0 or args.single_tile_trials < 0:
        raise ValueError("单tile预热/测试次数不能为负数。")
    if args.single_tile_only and args.single_tile_trials == 0:
        raise ValueError("--single-tile-only时--single-tile-trials必须大于0。")
    if args.single_tile_only and args.save_prediction:
        raise ValueError("只测单tile时不能保存整图预测。")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前PyTorch检测不到CUDA。请在含4090的服务器环境运行。"
        )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("本脚本只用于GPU基准，--device必须是cuda设备。")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)

    model_run_dir, model_metadata, model_path, fp32_dir = resolve_artifacts(args)
    args.fp32_dir = str(fp32_dir)
    image, gt, class_count, blocks, preprocess_seconds = load_scene_and_blocks(
        args, fp32_dir
    )
    net, hidden_dim, token_num = load_deploy_model(
        args,
        model_metadata,
        model_path,
        image,
        blocks,
        class_count,
        device,
    )
    cpu_records = build_cpu_tiles(image, blocks)
    tile_count = len(blocks)
    pixel_count = int(gt.size)
    if not 0 <= args.single_tile_index < tile_count:
        raise IndexError(
            f"--single-tile-index={args.single_tile_index}越界，"
            f"有效范围为0..{tile_count - 1}。"
        )

    selected_cpu_record = cpu_records[args.single_tile_index]
    selected_gpu_tile = selected_cpu_record["tile"].to(
        device=device,
        non_blocking=False,
    )
    if args.model_kind == "fixed":
        if not 2 <= args.fixed_verify_tiles <= tile_count:
            raise ValueError("--fixed-verify-tiles must be between 2 and tile count")
        indices = np.linspace(0, tile_count-1, args.fixed_verify_tiles, dtype=int)
        verification = torch.cat([cpu_records[i]["tile"] for i in indices]).to(device)
        print("[Fixed parity] comparing integer codes against FPGA reference before timing", flush=True)
        print(net.verify(verification), flush=True)
        del verification
    output_shape = tuple(net(selected_gpu_tile).shape)
    if output_shape[0] != 1 or output_shape[1] != class_count:
        raise RuntimeError(f"网络输出形状异常：{output_shape}")

    cpu_batches = make_tile_batches(cpu_records, args.batch_size)
    forward_calls = len(cpu_batches)
    gpu_batches = None
    if not args.single_tile_only:
        gpu_batches = preload_gpu_batches(cpu_batches, device)
        if args.model_kind == "fixed":
            net.verify_batch_lanes(gpu_batches[0]["tiles"])
            if len(gpu_batches[-1]["tiles"]) != len(gpu_batches[0]["tiles"]):
                net.verify_batch_lanes(gpu_batches[-1]["tiles"])

    print("\n" + "=" * 78)
    print("GPU tile latency / full-scene throughput benchmark")
    print(f"  GPU             : {torch.cuda.get_device_name(device)}")
    print(f"  PyTorch/CUDA    : {torch.__version__} / {torch.version.cuda}")
    print(f"  dataset/scene   : {args.dataset} / {tuple(gt.shape)}")
    print(f"  one tile        : 1 x {image.shape[2]} x 16 x 16")
    print(f"  network output  : {output_shape}")
    print(f"  scene batch     : {args.batch_size} tile(s) per forward")
    print(
        f"  scene workload  : {tile_count} tiles, "
        f"{forward_calls} forward calls"
    )
    print(
        f"  single tile     : warmup={args.single_tile_warmup}, "
        f"trials={args.single_tile_trials}, index={args.single_tile_index}"
    )
    print(
        f"  full scene      : warmup={args.warmup_steps}, "
        f"repeats={args.repeats}, skipped={args.single_tile_only}"
    )
    print(f"  model kind      : {args.model_kind.upper()}")
    print(f"  checkpoint      : {model_path}")
    if args.model_kind == "fp32":
        print("  execution type  : PyTorch FP32 CUDA, no fake-quant")
    elif args.model_kind == "fixed":
        print("  execution type  : Hardware-contract GPU simulation; FP64 integer MAC + INT64 state")
    else:
        print("  execution type  : PyTorch QAT fake-quant CUDA, not native INT8")
    if args.batch_size > 1:
        print(
            "  [Notice] batch>1用于GPU吞吐测试；"
            "与FPGA逐tile延迟对比时仍应使用batch=1。"
        )
    print("=" * 78)

    if gpu_batches is not None:
        warm_up(net, gpu_batches, args.warmup_steps)
    torch.cuda.reset_peak_memory_stats(device)
    memory_before = int(torch.cuda.memory_allocated(device))

    single_wall_summary = None
    single_device_summary = None
    if args.single_tile_trials > 0:
        single_wall, single_device = benchmark_single_tile(
            net,
            selected_gpu_tile,
            args.single_tile_warmup,
            args.single_tile_trials,
        )
        single_wall_summary = summarize_seconds(
            single_wall, 1, TILE_SIZE * TILE_SIZE, 1
        )
        single_device_summary = summarize_seconds(
            single_device, 1, TILE_SIZE * TILE_SIZE, 1
        )
        print_summary(
            "Single tile model forward - synchronized wall clock",
            single_wall_summary,
            unit_label="one tile",
            rate_label="tile",
        )
        print_summary(
            "Single tile model forward - CUDA events",
            single_device_summary,
            unit_label="one tile",
            rate_label="tile",
        )

    model_wall_summary = None
    model_device_summary = None
    pipeline_wall_summary = None
    pipeline_device_summary = None
    h2d_wall_summary = None
    h2d_device_summary = None
    final_canvas = None

    if not args.single_tile_only:
        model_wall, model_device, _ = benchmark_repeated(
            "model_only",
            args.repeats,
            lambda: run_model_only(net, gpu_batches),
        )
        pipeline_wall, pipeline_device, final_canvas = benchmark_repeated(
            "full_gpu_pipeline",
            args.repeats,
            lambda: run_full_pipeline(
                net,
                gpu_batches,
                gt.shape,
                device,
                transfer_each_batch=False,
            ),
        )
        model_wall_summary = summarize_seconds(
            model_wall, tile_count, pixel_count, forward_calls
        )
        model_device_summary = summarize_seconds(
            model_device, tile_count, pixel_count, forward_calls
        )
        pipeline_wall_summary = summarize_seconds(
            pipeline_wall, tile_count, pixel_count, forward_calls
        )
        pipeline_device_summary = summarize_seconds(
            pipeline_device, tile_count, pixel_count, forward_calls
        )
        print_summary(
            "Model only - synchronized wall clock", model_wall_summary
        )
        print_summary("Model only - CUDA events", model_device_summary)
        print_summary(
            "Full GPU pipeline - synchronized wall clock",
            pipeline_wall_summary,
        )
        print_summary(
            "Full GPU pipeline - CUDA events", pipeline_device_summary
        )

        if args.measure_h2d:
            h2d_wall, h2d_device, final_canvas = benchmark_repeated(
                "pipeline_with_h2d",
                args.repeats,
                lambda: run_full_pipeline(
                    net,
                    cpu_batches,
                    gt.shape,
                    device,
                    transfer_each_batch=True,
                ),
            )
            h2d_wall_summary = summarize_seconds(
                h2d_wall, tile_count, pixel_count, forward_calls
            )
            h2d_device_summary = summarize_seconds(
                h2d_device, tile_count, pixel_count, forward_calls
            )
            print_summary(
                "Pipeline with per-batch CPU-to-GPU copy - wall clock",
                h2d_wall_summary,
            )
            print_summary(
                "Pipeline with per-batch CPU-to-GPU copy - CUDA events",
                h2d_device_summary,
            )

    torch.cuda.synchronize()
    memory_peak = int(torch.cuda.max_memory_allocated(device))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / (
        f"gpu_batch{args.batch_size}_benchmark_{timestamp}.json"
    )

    report = {
        "build": "current-architecture-fixed-gpu-20260918",
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixed_backend_sha256": hashlib.sha256(Path(__file__).with_name("fixed_gpu_backend.py").read_bytes()).hexdigest() if args.model_kind == "fixed" else None,
        "model_config": net._benchmark_config,
        "selective_scan_backend": net._benchmark_scan_backend,
        "checkpoint_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "dtype": "FP64 integer MAC / INT64 fixed state" if args.model_kind == "fixed" else "float32",
        "fixed_contract": net.manifest() if args.model_kind == "fixed" else None,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "model_kind": args.model_kind,
        "scene_shape": list(gt.shape),
        "input_channels": int(image.shape[2]),
        "class_count": int(class_count),
        "tile_size": TILE_SIZE,
        "tile_count": tile_count,
        "batch_size": int(args.batch_size),
        "network_forward_calls_per_scene": forward_calls,
        "network_output_shape_for_one_tile": list(output_shape),
        "hidden_dim": hidden_dim,
        "token_num": token_num,
        "warmup_steps": int(args.warmup_steps),
        "single_tile": {
            "tile_index": int(args.single_tile_index),
            "top": int(selected_cpu_record["top"]),
            "left": int(selected_cpu_record["left"]),
            "warmup": int(args.single_tile_warmup),
            "trials": int(args.single_tile_trials),
            "wall_clock": single_wall_summary,
            "cuda_events": single_device_summary,
        },
        "preprocess_once_seconds_not_in_inference": preprocess_seconds,
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "memory_allocated_before_timing_bytes": memory_before,
            "peak_memory_allocated_bytes": memory_peak,
        },
        "artifacts": {
            "model_run_dir": str(model_run_dir),
            "fp32_dir": str(fp32_dir),
            "checkpoint": str(model_path),
        },
        "timing_scope": {
            "single_tile": (
                "已在GPU的1xCx16x16输入，每个trial只执行一次net(tile)"
            ),
            "model_only": (
                f"整图{tile_count}个tile，batch={args.batch_size}，"
                "输入预先放入GPU，不含上采样/argmax/拼接"
            ),
            "full_gpu_pipeline": (
                "输入预先放入GPU，网络前向+"
                "双线性上采样+argmax+整图拼接"
            ),
            "pipeline_with_h2d": (
                f"每个batch({args.batch_size}个tile)CPU->GPU后执行完整管线"
                if args.measure_h2d and not args.single_tile_only
                else "not measured"
            ),
            "excluded": "磁盘读取、checkpoint加载、PCA预处理和画图",
            "execution_type": (
                "Hardware-contract GPU simulation: FP64 integer MAC, INT64 state; not native INT8"
                if args.model_kind == "fixed" else
                "PyTorch FP32 CUDA operators; no fake-quant"
                if args.model_kind == "fp32"
                else (
                    "PyTorch QAT fake-quant CUDA operators; "
                    "not TensorRT INT8 and not FPGA"
                )
            ),
        },
        "model_only": (
            {
                "wall_clock": model_wall_summary,
                "cuda_events": model_device_summary,
            }
            if model_wall_summary is not None
            else None
        ),
        "full_gpu_pipeline": (
            {
                "wall_clock": pipeline_wall_summary,
                "cuda_events": pipeline_device_summary,
            }
            if pipeline_wall_summary is not None
            else None
        ),
        "pipeline_with_h2d": (
            {
                "wall_clock": h2d_wall_summary,
                "cuda_events": h2d_device_summary,
            }
            if h2d_wall_summary is not None
            else None
        ),
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=json_default)

    if args.save_prediction:
        prediction_path = output_dir / (
            f"gpu_batch{args.batch_size}_prediction_{timestamp}.npy"
        )
        prediction = final_canvas.detach().cpu().numpy()
        np.save(prediction_path, prediction, allow_pickle=False)
        print(f"\nPrediction saved: {prediction_path}")

    print(f"\nBenchmark JSON saved: {result_path}")
    if args.single_tile_only:
        print(
            "Primary end-to-end number: "
            "single_tile.wall_clock.median_ms_per_tile "
            "(输入已在GPU，单次net(tile)+同步，包含PyTorch/CPU发射开销)。"
        )
        print(
            "Secondary device-only number: "
            "single_tile.cuda_events.median_ms_per_tile "
            "(主要反映GPU命令/kernel执行，不代表batch=1可达端到端吞吐)。"
        )
    else:
        print(
            "Primary full-scene number: "
            "full_gpu_pipeline.wall_clock.median_ms_per_scene "
            f"(稳态GPU，batch={args.batch_size}，不含磁盘/PCA)。"
        )


if __name__ == "__main__":
    main()
