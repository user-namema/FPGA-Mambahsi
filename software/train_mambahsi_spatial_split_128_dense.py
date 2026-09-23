"""
MambaHSI 严格空间隔离的可配置tile输入/稠密输出训练脚本。

与中心像素分类不同，本脚本把每个空间块作为语义分割样本：
    input : [B, bands, tile_size, tile_size]
    model : [B, classes, tile_size/4, tile_size/4]（两次下采样）
    output: 双线性上采样回 [B, classes, tile_size, tile_size]

训练损失在训练块内全部有效训练标签上计算；验证和测试分别使用完全独立
的空间块。每个验证/测试块输出完整类别图并拼回原图。训练、验证、测试
原始像素两两不重叠，PCA和归一化也只在训练空间拟合。
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

import utils.data_load_operate as data_load_operate
from mambahsi_ablation_model import (
    MODEL_CONFIG_FIELDS,
    MODEL_FIXED_CONFIG,
    MODEL_VARIANT_NAMES,
    MambaHSI,
    resolve_model_config,
    selective_scan_backend,
)

try:
    from utils.visual_predict import visualize_predict
except ImportError:
    visualize_predict = None

try:
    from calflops import calculate_flops
except ImportError:
    calculate_flops = None


DATASET_CONFIGS = {
    "UP": {
        "class_count": 9,
        "data_file": "UP/PaviaU.mat",
        "label_file": "UP/PaviaU_gt.mat",
    },
    "HanChuan": {
        "class_count": 16,
        "data_file": "HanChuan/WHU_Hi_HanChuan.mat",
        "label_file": "HanChuan/WHU_Hi_HanChuan_gt.mat",
    },
    "HongHu": {
        "class_count": 22,
        "data_file": "HongHu/WHU_Hi_HongHu.mat",
        "label_file": "HongHu/WHU_Hi_HongHu_gt.mat",
        "class_names": [
            "Red roof",
            "Road",
            "Bare soil",
            "Cotton",
            "Cotton firewood",
            "Rape",
            "Chinese cabbage",
            "Pakchoi",
            "Cabbage",
            "Tuber mustard",
            "Brassica parachinensis",
            "Brassica chinensis",
            "Small Brassica chinensis",
            "Lactuca sativa",
            "Celtuce",
            "Film covered lettuce",
            "Romaine lettuce",
            "Carrot",
            "White radish",
            "Garlic sprout",
            "Broad bean",
            "Tree",
        ],
    },
    "Houston": {
        "class_count": 15,
        "data_file": "Houston/Houston.mat",
        "label_file": "Houston/Houston_GT.mat",
    },
}
DATASET_ORDER = tuple(DATASET_CONFIGS)
SPLIT_TARGET_RATIO = (0.5, 0.2, 0.3)


def parse_seed_list(text):
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("seeds 不能为空，例如 --seeds 0,1,2")
    return values


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("布尔参数请使用 true 或 false。")


def build_parser():
    parser = argparse.ArgumentParser(
        description="MambaHSI strict spatial split with configurable dense tiles"
    )
    parser.add_argument(
        "--dataset", choices=[*DATASET_ORDER, "all"], default="UP"
    )
    parser.add_argument(
        "--dataset_index", type=int, choices=range(len(DATASET_ORDER)), default=None
    )
    parser.add_argument("--data_set_path", type=str, default="./data")
    parser.add_argument("--work_dir", type=str, default="./results")
    parser.add_argument("--exp_name", type=str, default="SPATIAL_SPLIT_3WAY_DENSE")
    parser.add_argument("--run_tag", type=str, default="strict_3way")

    parser.add_argument("--tile_size", type=int, default=128)
    parser.add_argument(
        "--fallback_tile_size",
        type=int,
        default=64,
        help=(
            "当前tile尺寸无法满足逐类空间划分时自动重试的尺寸；"
            "默认64，设为0关闭自动回退。回退时split_block_size也设为该值。"
        ),
    )
    parser.add_argument(
        "--split_strategy",
        choices=["auto", "straight", "blocks"],
        default="auto",
        help="auto 优先连续三段；失败后采用训练/验证/测试空间块。",
    )
    parser.add_argument(
        "--split_axis",
        choices=["auto", "vertical", "horizontal"],
        default="auto",
    )
    parser.add_argument(
        "--split_block_size",
        type=int,
        default=128,
        help="train/validation/test 分配的空间块尺寸；必须不小于tile_size。",
    )
    parser.add_argument("--split_trials", type=int, default=30000)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument(
        "--min_train_samples",
        type=int,
        default=100,
        help="空间划分后训练侧每类至少包含的标签数。",
    )
    parser.add_argument(
        "--min_val_samples",
        type=int,
        default=50,
        help="独立验证空间内每类至少包含的标签数。",
    )
    parser.add_argument("--min_test_samples", type=int, default=100)
    parser.add_argument(
        "--train_samples",
        type=int,
        default=100,
        help=(
            "0=使用训练侧全部标签；正数=每类无放回使用至多该数量标签。"
            "空间划分的逐类下限由--min_train_samples单独控制。"
        ),
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=10,
        help=(
            "兼容旧命令；现在验证使用独立验证块内全部标签。"
            "逐类最低要求取max(val_samples, min_val_samples)。"
        ),
    )

    parser.add_argument("--pca_components", type=int, default=16)
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help=(
            "MambaHSI主干通道数；未指定时current/custom=32，"
            "original/original_safe=64，original_safe_matched=32。"
        ),
    )
    parser.add_argument(
        "--model_variant",
        "--model-variant",
        choices=MODEL_VARIANT_NAMES,
        default="current",
        help=(
            "结构预设：current=当前FPGA部署网络；"
            "original/original_safe=原容量(hidden64/head128)；"
            "original_safe_matched=当前容量(hidden32/head64)；"
            "两者都修正Spa跨batch串接；"
            "custom=从current开始并由下列参数覆盖。"
        ),
    )
    parser.add_argument(
        "--branch_mode",
        "--branch-mode",
        choices=["spa", "spe", "both"],
        default=None,
        help="空间分支、光谱分支或双分支；未指定时继承model_variant。",
    )
    parser.add_argument(
        "--fusion_mode",
        "--fusion-mode",
        choices=["sum", "mean", "softmax"],
        default=None,
        help="双分支融合；spa/spe单分支消融必须使用sum占位。",
    )
    parser.add_argument(
        "--skip_scale",
        "--skip-scale",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="每个Mamba块最终残差fusion + skip_scale*x。",
    )
    parser.add_argument(
        "--use_z",
        "--use-z",
        dest="use_z",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
        help="恢复Mamba z门控；可写--use_z或--use_z true/false。",
    )
    parser.add_argument(
        "--use_D",
        "--use-D",
        dest="use_D",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
        help="启用SSM直接通路D*x；可写--use_D或--use_D true/false。",
    )
    parser.add_argument(
        "--A_mode",
        "--a-mode",
        dest="A_mode",
        choices=["shared", "per_channel"],
        default=None,
        help="全部d_inner通道共享一组A，或每通道独立A。",
    )
    parser.add_argument(
        "--norm_path",
        "--norm-path",
        choices=["bn", "gn"],
        default=None,
        help="bn复现当前部署路径；gn复现原网络的分支后GroupNorm路径。",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "silu"],
        default=None,
        help=(
            "统一控制卷积后、分支后及patch/head激活；"
            "z门始终使用原生Mamba的SiLU，以保持消融轴独立。"
        ),
    )
    parser.add_argument(
        "--head_dim",
        "--head-dim",
        type=int,
        choices=[32, 64, 128],
        default=None,
        help="分类头中间通道数。",
    )
    parser.add_argument(
        "--token_num",
        "--token-num",
        type=int,
        default=None,
        help="SpeMamba光谱token数；hidden_dim不可整除时自动零填充并裁回。",
    )
    parser.add_argument(
        "--d_state",
        "--d-state",
        type=int,
        default=None,
        help="每个Mamba核的SSM状态维度。",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--class_weight",
        choices=["none", "inverse", "sqrt_inverse"],
        default="none",
        help="每类固定相同训练样本数时建议使用none。",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="全局L2梯度范数裁剪阈值；设为0关闭裁剪。",
    )
    parser.add_argument(
        "--gradient_diagnostics",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="记录每个batch的有效标签、类别构成和裁剪前后梯度范数。",
    )
    parser.add_argument(
        "--diagnostic_batch_interval",
        type=int,
        default=1,
        help="每隔多少个batch写一次详细梯度诊断。",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "step", "exponential", "cosine", "plateau"],
        default="step",
        help="学习率调度器；plateau依据验证集选择指标调整。",
    )
    parser.add_argument("--scheduler_step_size", type=int, default=20)
    parser.add_argument("--scheduler_gamma", type=float, default=0.9)
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=3,
        help="ReduceLROnPlateau连续多少次验证不改善后降低学习率。",
    )
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-6)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=12,
        help="连续多少次验证无显著改善后早停；设为0关闭。",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-4,
        help="验证选择指标至少提高多少才重置早停计数。",
    )
    parser.add_argument(
        "--selection_metric",
        choices=["OA", "mAcc"],
        default="mAcc",
        help="独立验证块上选择最佳权重的指标，默认mAcc以减弱类别不平衡。",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("0"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--record_computecost",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "是否在首次seed执行dummy前向形状探测和calflops统计；"
            "默认关闭，参数量和模型结构仍会记录。"
        ),
    )
    return parser


def resolve_model_arguments(args):
    """Apply a model preset, then let explicit command-line fields override it."""
    overrides = {
        name: getattr(args, name)
        for name in MODEL_CONFIG_FIELDS
    }
    config = resolve_model_config(args.model_variant, **overrides)
    for name in MODEL_CONFIG_FIELDS:
        setattr(args, name, config[name])
    args.model_config = {
        **config,
        **MODEL_FIXED_CONFIG,
    }
    return args


def model_config_slug(args):
    """Readable, collision-resistant directory name for architecture runs."""
    config = args.model_config
    return (
        f"{config['model_variant']}"
        f"_h{config['hidden_dim']}"
        f"_br-{config['branch_mode']}"
        f"_fu-{config['fusion_mode']}"
        f"_sk{config['skip_scale']}"
        f"_z{int(config['use_z'])}"
        f"_D{int(config['use_D'])}"
        f"_A-{config['A_mode']}"
        f"_n-{config['norm_path']}"
        f"_a-{config['activation']}"
        f"_head{config['head_dim']}"
        f"_tok{config['token_num']}"
        f"_state{config['d_state']}"
    )


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_logger(path, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def save_json(path, value):
    def convert(item):
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"cannot serialize {type(item)}")

    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, default=convert)


def append_jsonl(path, records):
    if not records:
        return

    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"cannot serialize {type(item)}")

    with open(path, "a", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False, default=convert) + "\n"
            )


def validate_data(raw_data, raw_gt, dataset_name, pca_components):
    data = np.asarray(raw_data)
    gt = np.squeeze(np.asarray(raw_gt)).astype(np.int64, copy=False)
    if data.ndim != 3 or gt.ndim != 2 or data.shape[:2] != gt.shape:
        raise ValueError(
            f"{dataset_name}: data={data.shape}, gt={gt.shape}，空间尺寸不匹配。"
        )
    foreground = np.unique(gt)
    foreground = foreground[foreground > 0]
    expected = np.arange(1, foreground.size + 1)
    if not np.array_equal(foreground, expected):
        raise ValueError(f"标签必须连续为1..C，实际为 {foreground.tolist()}")
    class_count = DATASET_CONFIGS[dataset_name]["class_count"]
    if foreground.size != class_count:
        raise ValueError(
            f"{dataset_name}: GT为{foreground.size}类，配置为{class_count}类。"
        )
    if not 1 <= pca_components <= data.shape[2]:
        raise ValueError(
            f"pca_components={pca_components}，原始波段数={data.shape[2]}。"
        )
    if not np.isfinite(data).all():
        raise ValueError("原始影像包含NaN或Inf。")
    return data.astype(np.float32, copy=False), gt, class_count


def counts_in_mask(gt, mask, class_count):
    return np.asarray(
        [np.count_nonzero(mask & (gt == label)) for label in range(1, class_count + 1)],
        dtype=np.int64,
    )


def split_score(
    train_counts,
    val_counts,
    test_counts,
    all_counts,
    train_area_fraction,
    val_area_fraction,
    test_area_fraction,
):
    denominator = np.maximum(all_counts, 1)
    worst_fraction = min(
        float(np.min(train_counts / denominator)),
        float(np.min(val_counts / denominator)),
        float(np.min(test_counts / denominator)),
    )
    area_penalty = (
        abs(float(train_area_fraction) - SPLIT_TARGET_RATIO[0])
        + abs(float(val_area_fraction) - SPLIT_TARGET_RATIO[1])
        + abs(float(test_area_fraction) - SPLIT_TARGET_RATIO[2])
    )
    return worst_fraction - 0.03 * area_penalty


def straight_split(
    gt,
    class_count,
    required_train,
    required_val,
    required_test,
    tile_size,
    split_axis,
):
    height, width = gt.shape
    axes = []
    if split_axis in {"auto", "vertical"}:
        axes.append(("vertical", width))
    if split_axis in {"auto", "horizontal"}:
        axes.append(("horizontal", height))
    labels = np.arange(1, class_count + 1)[:, None, None]
    one_hot = gt[None, :, :] == labels
    all_counts = counts_in_mask(gt, gt > 0, class_count)
    best = None
    closest = None
    for axis_name, length in axes:
        if length < 3 * tile_size:
            continue
        reduced = one_hot.sum(axis=1 if axis_name == "vertical" else 2)
        prefix = np.concatenate(
            [np.zeros((class_count, 1), dtype=np.int64), np.cumsum(reduced, axis=1)],
            axis=1,
        )
        step = max(tile_size // 2, 1)
        cut_positions = list(range(tile_size, length - tile_size + 1, step))
        assignments = (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )
        for cut1 in cut_positions:
            for cut2 in cut_positions:
                if cut2 - cut1 < tile_size or length - cut2 < tile_size:
                    continue
                segment_counts = (
                    prefix[:, cut1],
                    prefix[:, cut2] - prefix[:, cut1],
                    prefix[:, -1] - prefix[:, cut2],
                )
                segment_lengths = (cut1, cut2 - cut1, length - cut2)
                for train_index, val_index, test_index in assignments:
                    train_counts = segment_counts[train_index]
                    val_counts = segment_counts[val_index]
                    test_counts = segment_counts[test_index]
                    deficit = int(
                        np.maximum(required_train - train_counts, 0).sum()
                        + np.maximum(required_val - val_counts, 0).sum()
                        + np.maximum(required_test - test_counts, 0).sum()
                    )
                    candidate = {
                        "deficit": deficit,
                        "axis": axis_name,
                        "cut1": cut1,
                        "cut2": cut2,
                        "train_segment": train_index,
                        "val_segment": val_index,
                        "test_segment": test_index,
                        "train_counts": train_counts.copy(),
                        "val_counts": val_counts.copy(),
                        "test_counts": test_counts.copy(),
                    }
                    if closest is None or deficit < closest["deficit"]:
                        closest = candidate
                    if deficit:
                        continue
                    score = split_score(
                        train_counts,
                        val_counts,
                        test_counts,
                        all_counts,
                        segment_lengths[train_index] / length,
                        segment_lengths[val_index] / length,
                        segment_lengths[test_index] / length,
                    )
                    if best is None or score > best["score"]:
                        best = {**candidate, "score": score}
    if best is None:
        return None, closest

    cut1 = best["cut1"]
    cut2 = best["cut2"]
    if best["axis"] == "vertical":
        regions = [
            (0, height, 0, cut1),
            (0, height, cut1, cut2),
            (0, height, cut2, width),
        ]
    else:
        regions = [
            (0, cut1, 0, width),
            (cut1, cut2, 0, width),
            (cut2, height, 0, width),
        ]
    return {
        **best,
        "strategy": "straight",
        "train_regions": [regions[best["train_segment"]]],
        "val_regions": [regions[best["val_segment"]]],
        "test_regions": [regions[best["test_segment"]]],
    }, closest


def enumerate_blocks(height, width, block_size):
    return [
        (top, min(top + block_size, height), left, min(left + block_size, width))
        for top in range(0, height, block_size)
        for left in range(0, width, block_size)
    ]


def block_split(
    gt,
    class_count,
    required_train,
    required_val,
    required_test,
    block_size,
    trials,
    seed,
):
    height, width = gt.shape
    blocks = enumerate_blocks(height, width, block_size)
    block_counts = np.asarray(
        [
            counts_in_mask(
                gt[top:bottom, left:right],
                gt[top:bottom, left:right] > 0,
                class_count,
            )
            for top, bottom, left, right in blocks
        ],
        dtype=np.int64,
    )
    available = block_counts.sum(axis=0)
    if np.any(available < required_train + required_val + required_test):
        return None, {
            "deficit": int(
                np.maximum(
                    required_train + required_val + required_test - available,
                    0,
                ).sum()
            ),
            "available_counts": available,
        }
    block_areas = np.asarray(
        [(bottom - top) * (right - left) for top, bottom, left, right in blocks],
        dtype=np.int64,
    )
    total_area = int(block_areas.sum())
    rng = np.random.default_rng(seed + block_size * 997)
    best = None
    closest = None
    completed = 0
    while completed < trials:
        count = min(256, trials - completed)
        assignment = rng.choice(
            3,
            size=(count, len(blocks)),
            p=np.asarray(SPLIT_TARGET_RATIO),
        ).astype(np.int8)
        for row in range(count):
            for group in range(3):
                if not np.any(assignment[row] == group):
                    assignment[row, group] = group
        train_mask = assignment == 0
        val_mask = assignment == 1
        test_mask = assignment == 2
        train_batch = train_mask.astype(np.int64) @ block_counts
        val_batch = val_mask.astype(np.int64) @ block_counts
        test_batch = test_mask.astype(np.int64) @ block_counts
        deficits = (
            np.maximum(required_train - train_batch, 0).sum(axis=1)
            + np.maximum(required_val - val_batch, 0).sum(axis=1)
            + np.maximum(required_test - test_batch, 0).sum(axis=1)
        )
        failure_index = int(np.argmin(deficits))
        failure = {
            "deficit": int(deficits[failure_index]),
            "train_counts": train_batch[failure_index].copy(),
            "val_counts": val_batch[failure_index].copy(),
            "test_counts": test_batch[failure_index].copy(),
        }
        if closest is None or failure["deficit"] < closest["deficit"]:
            closest = failure
        for index in np.flatnonzero(deficits == 0):
            train_area = int(train_mask[index].astype(np.int64) @ block_areas)
            val_area = int(val_mask[index].astype(np.int64) @ block_areas)
            test_area = int(test_mask[index].astype(np.int64) @ block_areas)
            score = split_score(
                train_batch[index],
                val_batch[index],
                test_batch[index],
                available,
                train_area / total_area,
                val_area / total_area,
                test_area / total_area,
            )
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "assignment": assignment[index].copy(),
                    "train_counts": train_batch[index].copy(),
                    "val_counts": val_batch[index].copy(),
                    "test_counts": test_batch[index].copy(),
                }
        completed += count
    if best is None:
        return None, closest

    train_regions = []
    val_regions = []
    test_regions = []
    block_records = []
    group_names = ("train", "validation", "test")
    region_groups = (train_regions, val_regions, test_regions)
    for group, region in zip(best["assignment"], blocks):
        target = region_groups[int(group)]
        target.append(region)
        top, bottom, left, right = region
        block_records.append(
            {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "split": group_names[int(group)],
            }
        )
    return {
        **best,
        "strategy": "blocks",
        "block_size": block_size,
        "blocks": block_records,
        "train_regions": train_regions,
        "val_regions": val_regions,
        "test_regions": test_regions,
    }, closest


def format_failure(failures, required_train, required_val, required_test):
    lines = [
        "找不到满足逐类数量要求的空间划分。",
        f"要求 train每类>={required_train}, validation每类>={required_val}, "
        f"test每类>={required_test}。",
    ]
    for name, failure in failures:
        if failure is None:
            lines.append(f"{name}: 无有效候选")
        else:
            lines.append(f"{name}: 最小总缺口={failure.get('deficit')}")
            if "train_counts" in failure:
                lines.append(f"  train={failure['train_counts'].tolist()}")
            if "val_counts" in failure:
                lines.append(f"  val  ={failure['val_counts'].tolist()}")
            if "test_counts" in failure:
                lines.append(f"  test ={failure['test_counts'].tolist()}")
    return "\n".join(lines)


def choose_split(gt, class_count, args):
    required_train = (
        min(args.train_samples, args.min_train_samples)
        if args.train_samples > 0
        else args.min_train_samples
    )
    required_val = max(args.min_val_samples, args.val_samples)
    required_test = args.min_test_samples
    failures = []
    if args.split_strategy in {"auto", "straight"}:
        result, failure = straight_split(
            gt,
            class_count,
            required_train,
            required_val,
            required_test,
            args.tile_size,
            args.split_axis,
        )
        if result is not None:
            return result
        failures.append(("straight", failure))
        if args.split_strategy == "straight":
            raise ValueError(
                format_failure(
                    failures, required_train, required_val, required_test
                )
            )
    result, failure = block_split(
        gt,
        class_count,
        required_train,
        required_val,
        required_test,
        args.split_block_size,
        args.split_trials,
        args.split_seed,
    )
    if result is not None:
        if failures:
            result["straight_failure"] = failures[0][1]
        return result
    failures.append(("blocks", failure))
    raise ValueError(
        format_failure(failures, required_train, required_val, required_test)
    )


def regions_to_mask(shape, regions):
    mask = np.zeros(shape, dtype=bool)
    for top, bottom, left, right in regions:
        if mask[top:bottom, left:right].any():
            raise AssertionError("空间区域发生重叠。")
        mask[top:bottom, left:right] = True
    return mask


def make_tiles(regions, tile_size):
    tiles = []
    for region_id, (region_top, region_bottom, region_left, region_right) in enumerate(regions):
        for top in range(region_top, region_bottom, tile_size):
            bottom = min(top + tile_size, region_bottom)
            for left in range(region_left, region_right, tile_size):
                right = min(left + tile_size, region_right)
                tiles.append((top, bottom, left, right, region_id))
    return tiles


def fit_train_only_preprocess(
    raw_data, train_region, components, save_path, random_state, logger
):
    height, width, bands = raw_data.shape
    flat = raw_data.reshape(-1, bands).astype(np.float32, copy=False)
    fit_data = flat[train_region.reshape(-1)]
    logger.info(
        "PCA start | train_pixels=%d bands=%d components=%d",
        fit_data.shape[0],
        bands,
        components,
    )
    start = time.perf_counter()
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=random_state,
    )
    pca.fit(fit_data)
    transformed = pca.transform(flat).reshape(height, width, components)
    train_features = transformed[train_region]
    channel_min = train_features.min(axis=0)
    channel_max = train_features.max(axis=0)
    channel_range = np.maximum(channel_max - channel_min, 1e-12)
    normalized = np.clip(
        (transformed - channel_min) / channel_range, 0.0, 1.0
    ).astype(np.float32)
    np.savez_compressed(
        save_path,
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        channel_min=channel_min.astype(np.float32),
        channel_max=channel_max.astype(np.float32),
        fitted_pixel_count=np.asarray([fit_data.shape[0]], dtype=np.int64),
    )
    logger.info(
        "PCA + transform finished | seconds=%.2f output=%s",
        time.perf_counter() - start,
        normalized.shape,
    )
    return normalized


def create_label_maps(
    gt,
    train_region,
    val_region,
    test_region,
    class_count,
    train_samples,
    seed,
):
    rng = np.random.default_rng(seed)
    flat_gt = gt.reshape(-1)
    flat_train_region = train_region.reshape(-1)
    flat_val_region = val_region.reshape(-1)
    flat_test_region = test_region.reshape(-1)
    train_indices = []
    val_indices = []
    test_indices = []
    for label in range(1, class_count + 1):
        candidates = np.flatnonzero(flat_train_region & (flat_gt == label))
        candidates = rng.permutation(candidates)
        if train_samples > 0:
            candidates = candidates[:train_samples]
        train_indices.extend(candidates)
        val_indices.extend(
            np.flatnonzero(flat_val_region & (flat_gt == label))
        )
        test_indices.extend(
            np.flatnonzero(flat_test_region & (flat_gt == label))
        )
    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    train_labels = np.full(gt.size, -1, dtype=np.int64)
    val_labels = np.full(gt.size, -1, dtype=np.int64)
    test_labels = np.full(gt.size, -1, dtype=np.int64)
    train_labels[train_indices] = flat_gt[train_indices] - 1
    val_labels[val_indices] = flat_gt[val_indices] - 1
    test_labels[test_indices] = flat_gt[test_indices] - 1
    return (
        train_labels.reshape(gt.shape),
        val_labels.reshape(gt.shape),
        test_labels.reshape(gt.shape),
        train_indices,
        val_indices,
        test_indices,
    )


class DenseTileDataset(Dataset):
    def __init__(self, image, label_map, tiles, tile_size, require_label):
        self.image = np.asarray(image, dtype=np.float32)
        self.label_map = np.asarray(label_map, dtype=np.int64)
        self.tile_size = int(tile_size)
        if require_label:
            self.tiles = [
                tile
                for tile in tiles
                if np.any(
                    self.label_map[tile[0] : tile[1], tile[2] : tile[3]] >= 0
                )
            ]
        else:
            self.tiles = list(tiles)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        top, bottom, left, right, region_id = self.tiles[index]
        image = self.image[top:bottom, left:right]
        labels = self.label_map[top:bottom, left:right]
        valid_height, valid_width = image.shape[:2]
        pad_height = self.tile_size - valid_height
        pad_width = self.tile_size - valid_width
        if pad_height or pad_width:
            pad_mode = "reflect" if min(valid_height, valid_width) > 1 else "edge"
            image = np.pad(
                image,
                ((0, pad_height), (0, pad_width), (0, 0)),
                mode=pad_mode,
            )
            labels = np.pad(
                labels,
                ((0, pad_height), (0, pad_width)),
                mode="constant",
                constant_values=-1,
            )
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        labels = np.ascontiguousarray(labels)
        return (
            torch.from_numpy(image),
            torch.from_numpy(labels),
            top,
            left,
            valid_height,
            valid_width,
            region_id,
        )


def make_loader(dataset, batch_size, shuffle, workers, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def dense_logits(model, image, tile_size):
    low_resolution = model(image)
    if low_resolution.ndim != 4:
        raise RuntimeError(
            f"模型输出应为BxCxHxW，实际为 {tuple(low_resolution.shape)}"
        )
    return F.interpolate(
        low_resolution,
        size=(tile_size, tile_size),
        mode="bilinear",
        align_corners=True,
    )


def log_model_structure_and_cost(
    model,
    optimizer,
    args,
    input_channels,
    class_count,
    device,
    run_dir,
    logger,
):
    """恢复原训练脚本的网络结构、优化器和计算量输出。"""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    compatibility_parameters = sum(
        parameter.numel()
        for parameter in model.shared_params.parameters()
    )
    effective_parameters = total_parameters - compatibility_parameters
    logger.info(
        "Model configuration | in_channels=%d hidden_dim=%d classes=%d "
        "dense_tile=%dx%d scan_backend=%s config=%s",
        input_channels,
        args.hidden_dim,
        class_count,
        args.tile_size,
        args.tile_size,
        selective_scan_backend(),
        json.dumps(args.model_config, ensure_ascii=False, sort_keys=True),
    )
    logger.info("Model structure:\n%s", model)
    logger.info("Optimizer:\n%s", optimizer)
    logger.info(
        "Parameters | total=%s effective_forward=%s "
        "checkpoint_compatibility_only=%s trainable=%s non_trainable=%s",
        f"{total_parameters:,}",
        f"{effective_parameters:,}",
        f"{compatibility_parameters:,}",
        f"{trainable_parameters:,}",
        f"{total_parameters - trainable_parameters:,}",
    )

    structure_path = run_dir / "model_structure.txt"
    structure_path.write_text(
        "Model configuration\n"
        f"in_channels={input_channels}\n"
        f"hidden_dim={args.hidden_dim}\n"
        f"classes={class_count}\n"
        f"tile_size={args.tile_size}\n"
        f"selective_scan_backend={selective_scan_backend()}\n"
        "model_ablation_config="
        f"{json.dumps(args.model_config, ensure_ascii=False, sort_keys=True)}\n"
        f"total_parameters={total_parameters}\n"
        f"effective_forward_parameters={effective_parameters}\n"
        f"checkpoint_compatibility_only_parameters={compatibility_parameters}\n"
        f"trainable_parameters={trainable_parameters}\n\n"
        f"{model}\n\nOptimizer\n{optimizer}\n",
        encoding="utf-8",
    )

    # Dummy forward and third-party FLOPs profilers are optional.  Keep them
    # away from the model used for training unless explicitly requested: some
    # custom Mamba variants do not support these probing/profile paths.
    if not args.record_computecost:
        return

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            dummy_input = torch.zeros(
                1,
                input_channels,
                args.tile_size,
                args.tile_size,
                device=device,
            )
            raw_output = model(dummy_input)
            resized_output = F.interpolate(
                raw_output,
                size=(args.tile_size, args.tile_size),
                mode="bilinear",
                align_corners=True,
            )
        logger.info(
            "Tensor shapes | input=%s raw_model_output=%s dense_output=%s",
            tuple(dummy_input.shape),
            tuple(raw_output.shape),
            tuple(resized_output.shape),
        )
    except Exception as error:
        logger.warning("网络形状探测失败，训练继续：%s", error)
    finally:
        model.train(was_training)

    if calculate_flops is None:
        logger.warning("未安装calflops，跳过FLOPs/MACs统计。")
        return
    try:
        model.eval()
        flops, macs, parameters = calculate_flops(
            model=model,
            input_shape=(
                1,
                input_channels,
                args.tile_size,
                args.tile_size,
            ),
        )
        logger.info(
            "Compute cost | parameters=%s FLOPs=%s MACs=%s",
            parameters,
            flops,
            macs,
        )
    except Exception as error:
        logger.warning("FLOPs/MACs统计失败，训练继续：%s", error)
    finally:
        model.train(was_training)


def compute_class_weights(label_map, class_count, mode, device):
    valid = label_map[label_map >= 0]
    counts = np.bincount(valid, minlength=class_count).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"训练标签存在空类别：counts={counts.tolist()}")
    if mode == "none":
        weights = np.ones(class_count, dtype=np.float64)
    elif mode == "inverse":
        weights = 1.0 / counts
    else:
        weights = 1.0 / np.sqrt(counts)
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device), counts


def build_scheduler(optimizer, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma,
        )
    if args.scheduler == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.scheduler_gamma
        )
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.max_epoch, 1),
            eta_min=args.scheduler_min_lr,
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_gamma,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )


def gradient_l2_norm(parameters):
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().norm(2)
        squared_norm += float(grad_norm) ** 2
    return squared_norm**0.5


def cross_entropy_denominator(labels, class_weights):
    valid_labels = labels[labels >= 0]
    if valid_labels.numel() == 0:
        return 0.0
    if class_weights is None:
        return float(valid_labels.numel())
    return float(class_weights[valid_labels].sum().detach())


def train_one_epoch(
    model,
    loader,
    optimizer,
    class_weights,
    tile_size,
    class_count,
    device,
    grad_clip_norm=0.0,
    collect_diagnostics=False,
    diagnostic_batch_interval=1,
):
    model.train()
    loss_numerator = 0.0
    loss_denominator = 0.0
    valid_pixels = 0
    batch_diagnostics = []
    grad_norms_before = []
    grad_norms_after = []
    batch_valid_counts = []
    epoch_class_counts = np.zeros(class_count, dtype=np.int64)
    for batch_index, (image, labels, *_) in enumerate(loader, start=1):
        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        valid = labels >= 0
        count = int(valid.sum())
        if count == 0:
            continue
        batch_class_counts = torch.bincount(
            labels[valid], minlength=class_count
        ).detach().cpu().numpy()
        epoch_class_counts += batch_class_counts
        optimizer.zero_grad(set_to_none=True)
        logits = dense_logits(model, image, tile_size)
        loss = F.cross_entropy(
            logits,
            labels,
            weight=class_weights,
            ignore_index=-1,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"batch={batch_index}出现非有限loss：{float(loss.detach())}"
            )
        loss.backward()
        grad_norm_before = gradient_l2_norm(model.parameters())
        if not np.isfinite(grad_norm_before):
            raise FloatingPointError(
                f"batch={batch_index}出现非有限梯度范数：{grad_norm_before}"
            )
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        grad_norm_after = gradient_l2_norm(model.parameters())
        optimizer.step()
        denominator = cross_entropy_denominator(labels, class_weights)
        loss_numerator += float(loss.detach()) * denominator
        loss_denominator += denominator
        valid_pixels += count
        batch_valid_counts.append(count)
        grad_norms_before.append(grad_norm_before)
        grad_norms_after.append(grad_norm_after)
        if collect_diagnostics and batch_index % diagnostic_batch_interval == 0:
            batch_diagnostics.append(
                {
                    "batch": batch_index,
                    "tile_count": int(image.shape[0]),
                    "valid_pixels": count,
                    "class_counts": batch_class_counts,
                    "active_class_count": int(np.count_nonzero(batch_class_counts)),
                    "loss": float(loss.detach()),
                    "loss_denominator": denominator,
                    "grad_norm_before_clip": grad_norm_before,
                    "grad_norm_after_clip": grad_norm_after,
                    "was_clipped": bool(
                        grad_clip_norm > 0 and grad_norm_before > grad_clip_norm
                    ),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )
    if not batch_valid_counts:
        raise RuntimeError("本epoch没有任何有效训练标签。")
    batch_valid_array = np.asarray(batch_valid_counts, dtype=np.float64)
    grad_before_array = np.asarray(grad_norms_before, dtype=np.float64)
    grad_after_array = np.asarray(grad_norms_after, dtype=np.float64)
    summary = {
        "loss": loss_numerator / max(loss_denominator, 1e-12),
        "loss_numerator": loss_numerator,
        "loss_denominator": loss_denominator,
        "valid_pixels": valid_pixels,
        "batch_count": len(batch_valid_counts),
        "batch_valid_pixels_min": int(batch_valid_array.min()),
        "batch_valid_pixels_max": int(batch_valid_array.max()),
        "batch_valid_pixels_mean": float(batch_valid_array.mean()),
        "batch_valid_pixels_std": float(batch_valid_array.std()),
        "epoch_class_counts": epoch_class_counts,
        "grad_norm_before_clip_mean": float(grad_before_array.mean()),
        "grad_norm_before_clip_max": float(grad_before_array.max()),
        "grad_norm_after_clip_mean": float(grad_after_array.mean()),
        "grad_norm_after_clip_max": float(grad_after_array.max()),
        "clipped_batch_count": int(
            np.count_nonzero(
                grad_before_array > grad_clip_norm
                if grad_clip_norm > 0
                else np.zeros_like(grad_before_array, dtype=bool)
            )
        ),
    }
    return summary, batch_diagnostics


@torch.no_grad()
def stitch_prediction(model, loader, scene_shape, tile_size, device):
    model.eval()
    prediction = np.full(scene_shape, -1, dtype=np.int64)
    visit_count = np.zeros(scene_shape, dtype=np.uint8)
    for image, _, tops, lefts, heights, widths, _ in loader:
        image = image.to(device, non_blocking=True)
        batch_prediction = torch.argmax(
            dense_logits(model, image, tile_size), dim=1
        ).cpu().numpy()
        for item in range(batch_prediction.shape[0]):
            top = int(tops[item])
            left = int(lefts[item])
            height = int(heights[item])
            width = int(widths[item])
            if visit_count[top : top + height, left : left + width].any():
                raise AssertionError("输出tile发生重叠，无法进行唯一拼接。")
            prediction[top : top + height, left : left + width] = batch_prediction[
                item, :height, :width
            ]
            visit_count[top : top + height, left : left + width] = 1
    return prediction, visit_count


def classification_metrics(target, prediction, class_count):
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (target, prediction), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = support + predicted - true_positive
    total = confusion.sum()
    per_class_acc = np.divide(
        true_positive, support, out=np.zeros_like(true_positive), where=support > 0
    )
    iou = np.divide(
        true_positive, union, out=np.zeros_like(true_positive), where=union > 0
    )
    oa = float(true_positive.sum() / total) if total else 0.0
    expected = float((support * predicted).sum() / (total * total)) if total else 0.0
    kappa = (oa - expected) / (1.0 - expected) if expected < 1.0 else 0.0
    return {
        "OA": oa,
        "mAcc": float(per_class_acc.mean()),
        "Kappa": float(kappa),
        "mIoU": float(iou.mean()),
        "per_class_acc": per_class_acc,
        "IoU": iou,
        "support": support.astype(np.int64),
        "confusion_matrix": confusion,
    }


def evaluate_prediction(prediction, label_map, class_count):
    valid = label_map >= 0
    if np.any(prediction[valid] < 0):
        missing = int(np.count_nonzero(prediction[valid] < 0))
        raise AssertionError(f"有{missing}个评估像素没有被任何tile覆盖。")
    return classification_metrics(
        label_map[valid], prediction[valid], class_count
    )


def summarize_class_confusions(confusion, top_k=3, class_names=None):
    """Return the strongest off-diagonal destinations for every true class."""
    confusion = np.asarray(confusion, dtype=np.int64)
    summaries = []
    for true_index, row in enumerate(confusion):
        support = int(row.sum())
        correct = int(row[true_index])
        mistakes = row.copy()
        mistakes[true_index] = 0
        order = np.argsort(mistakes)[::-1]
        top_confusions = []
        for predicted_index in order:
            count = int(mistakes[predicted_index])
            if count <= 0 or len(top_confusions) >= top_k:
                break
            top_confusions.append(
                {
                    "predicted_class": int(predicted_index + 1),
                    "predicted_class_name": (
                        class_names[predicted_index]
                        if class_names is not None
                        else None
                    ),
                    "count": count,
                    "rate_over_true_support": (
                        float(count / support) if support else 0.0
                    ),
                }
            )
        summaries.append(
            {
                "true_class": int(true_index + 1),
                "true_class_name": (
                    class_names[true_index] if class_names is not None else None
                ),
                "support": support,
                "correct": correct,
                "accuracy": float(correct / support) if support else 0.0,
                "top_confusions": top_confusions,
            }
        )
    return summaries


def plot_confusion_matrix(confusion, path, normalized):
    """Plot a count or row-normalized test confusion matrix using 1-based IDs."""
    confusion = np.asarray(confusion, dtype=np.int64)
    class_count = confusion.shape[0]
    ticks = np.arange(class_count)
    labels = [str(index + 1) for index in range(class_count)]

    if normalized:
        support = confusion.sum(axis=1, keepdims=True)
        values = np.divide(
            confusion.astype(np.float64),
            support,
            out=np.zeros_like(confusion, dtype=np.float64),
            where=support > 0,
        ) * 100.0
        color_values = values
        title = "Test confusion matrix (row-normalized, %)"
        colorbar_label = "Percentage of true class (%)"
        cmap = "Blues"
        vmax = 100.0
    else:
        values = confusion
        # A logarithmic color scale keeps minority-class errors visible when a
        # majority class has tens of thousands of test pixels.
        color_values = np.log1p(confusion.astype(np.float64))
        title = "Test confusion matrix (counts; logarithmic color scale)"
        colorbar_label = "log(1 + count)"
        cmap = "magma"
        vmax = None

    side = max(9.0, class_count * 0.52)
    figure, axis = plt.subplots(figsize=(side, side * 0.92))
    image = axis.imshow(
        color_values,
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        aspect="equal",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    axis.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )
    axis.tick_params(axis="x", labelrotation=90)

    font_size = 5.0 if class_count >= 20 else 6.5
    for row in range(class_count):
        for column in range(class_count):
            if normalized:
                value = float(values[row, column])
                if value < 1.0:
                    continue
                label = f"{value:.0f}"
                text_color = "white" if value >= 50.0 else "black"
            else:
                value = int(values[row, column])
                if value <= 0:
                    continue
                label = str(value)
                text_color = "white" if color_values[row, column] >= 0.55 * np.max(color_values) else "black"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=font_size,
            )

    # HongHu class 8 is a known difficult class. Highlighting row 8 makes its
    # error destinations immediately visible, while remaining harmless for
    # datasets with fewer than eight classes.
    if class_count >= 8:
        axis.add_patch(
            plt.Rectangle(
                (-0.5, 6.5),
                class_count,
                1.0,
                fill=False,
                edgecolor="red",
                linewidth=2.0,
            )
        )
        axis.get_yticklabels()[7].set_color("red")
        axis.get_yticklabels()[7].set_fontweight("bold")

    figure.tight_layout()
    figure.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(figure)


def save_prediction_png(prediction, display_mask, class_count, path):
    display = np.where(display_mask, prediction + 1, 0)
    masked = np.ma.masked_where(display == 0, display)
    plt.figure(figsize=(8, 7))
    plt.imshow(masked, cmap=plt.get_cmap("gist_ncar", class_count + 1))
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close()


def save_project_palette_prediction(
    gt,
    prediction,
    evaluation_label_map,
    prediction_path,
    gt_path,
    logger,
):
    """使用原MambaHSI工程的固定颜色表，保证同一类别与GT颜色一致。"""
    if visualize_predict is None:
        logger.warning(
            "utils.visual_predict 不可用，无法生成原项目同色板预测图。"
        )
        return
    evaluation_mask = evaluation_label_map >= 0
    masked_gt = np.where(evaluation_mask, gt, 0).astype(np.int64)
    # visualize_predict 接受0..C-1预测；掩码外的值不会在
    # only_vis_label=True时显示，仍设为0以避免负索引。
    masked_prediction = np.where(evaluation_mask, prediction, 0).astype(
        np.int64
    )
    visualize_predict(
        masked_gt,
        masked_prediction,
        str(prediction_path),
        str(gt_path),
        only_vis_label=True,
    )


def save_split_png(gt, train_region, val_region, test_region, path):
    canvas = np.zeros((*gt.shape, 3), dtype=np.float32)
    canvas[train_region] = (0.12, 0.55, 0.95)
    canvas[val_region] = (0.98, 0.75, 0.10)
    canvas[test_region] = (0.95, 0.32, 0.18)
    canvas[gt == 0] *= 0.4
    plt.figure(figsize=(8, 7))
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_loss(losses, path, tile_size):
    plt.figure(figsize=(7, 4.5))
    plt.plot(np.arange(1, len(losses) + 1), losses, color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Masked dense cross-entropy")
    plt.title(f"{tile_size}x{tile_size} dense-output training loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_training_diagnostics(history, path, tile_size):
    epochs = np.asarray([item["epoch"] for item in history])
    losses = np.asarray([item["loss"] for item in history])
    grad_max = np.asarray(
        [item["grad_norm_before_clip_max"] for item in history]
    )
    grad_mean = np.asarray(
        [item["grad_norm_before_clip_mean"] for item in history]
    )
    learning_rates = np.asarray([item["lr"] for item in history])
    valid_min = np.asarray([item["batch_valid_pixels_min"] for item in history])
    valid_max = np.asarray([item["batch_valid_pixels_max"] for item in history])
    valid_mean = np.asarray([item["batch_valid_pixels_mean"] for item in history])

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes[0, 0].plot(epochs, losses, color="tab:blue")
    axes[0, 0].set_ylabel("Exact masked CE")
    axes[0, 0].set_title("Training loss")

    axes[0, 1].plot(epochs, grad_mean, label="mean", color="tab:green")
    axes[0, 1].plot(epochs, grad_max, label="max", color="tab:red", alpha=0.8)
    axes[0, 1].set_ylabel("Global L2 gradient norm")
    axes[0, 1].set_title("Gradient before clipping")
    axes[0, 1].legend()

    axes[1, 0].fill_between(
        epochs, valid_min, valid_max, color="tab:orange", alpha=0.25, label="min-max"
    )
    axes[1, 0].plot(epochs, valid_mean, color="tab:orange", label="mean")
    axes[1, 0].set_ylabel("Supervised pixels / batch")
    axes[1, 0].set_title("Batch supervision density")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, learning_rates, color="tab:purple")
    axes[1, 1].set_ylabel("Learning rate")
    axes[1, 1].set_title("Learning-rate schedule")
    axes[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    figure.suptitle(f"{tile_size}x{tile_size} training diagnostics")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def split_public_info(split):
    return {
        key: value
        for key, value in split.items()
        if key
        not in {"assignment", "train_regions", "val_regions", "test_regions"}
    }


def run_seed(
    args,
    dataset_name,
    image,
    gt,
    class_count,
    split,
    train_region,
    val_region,
    test_region,
    dataset_dir,
    seed,
    device,
    logger,
):
    set_seed(seed)
    run_dir = dataset_dir / f"run_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (
        train_label,
        val_label,
        test_label,
        train_indices,
        val_indices,
        test_indices,
    ) = create_label_maps(
        gt,
        train_region,
        val_region,
        test_region,
        class_count,
        args.train_samples,
        seed,
    )
    np.savez_compressed(
        run_dir / "sample_indices.npz",
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )

    train_tiles = make_tiles(split["train_regions"], args.tile_size)
    val_tiles = make_tiles(split["val_regions"], args.tile_size)
    test_tiles = make_tiles(split["test_regions"], args.tile_size)
    train_dataset = DenseTileDataset(
        image, train_label, train_tiles, args.tile_size, require_label=True
    )
    val_dataset = DenseTileDataset(
        image, val_label, val_tiles, args.tile_size, require_label=False
    )
    test_dataset = DenseTileDataset(
        image, test_label, test_tiles, args.tile_size, require_label=False
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            f"没有包含训练标签的{args.tile_size}x{args.tile_size}空间块。"
        )
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, seed
    )
    val_loader = make_loader(
        val_dataset, args.eval_batch_size, False, args.num_workers, seed
    )
    test_loader = make_loader(
        test_dataset, args.eval_batch_size, False, args.num_workers, seed
    )

    model = MambaHSI(
        in_channels=image.shape[2],
        num_classes=class_count,
        hidden_dim=args.hidden_dim,
        branch_mode=args.branch_mode,
        fusion_mode=args.fusion_mode,
        skip_scale=args.skip_scale,
        use_z=args.use_z,
        use_D=args.use_D,
        A_mode=args.A_mode,
        norm_path=args.norm_path,
        activation=args.activation,
        head_dim=args.head_dim,
        token_num=args.token_num,
        d_state=args.d_state,
    ).to(device)
    parameter_count_total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    compatibility_parameter_count = sum(
        parameter.numel()
        for parameter in model.shared_params.parameters()
    )
    parameter_count_effective = (
        parameter_count_total - compatibility_parameter_count
    )
    run_model_config = {
        **args.model_config,
        "in_channels": image.shape[2],
        "num_classes": class_count,
        "selective_scan_backend": selective_scan_backend(),
    }
    save_json(
        run_dir / "model_config.json",
        run_model_config,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, args)
    if seed == args.seeds[0]:
        log_model_structure_and_cost(
            model=model,
            optimizer=optimizer,
            args=args,
            input_channels=image.shape[2],
            class_count=class_count,
            device=device,
            run_dir=run_dir,
            logger=logger,
        )
    class_weights, train_class_counts = compute_class_weights(
        train_label, class_count, args.class_weight, device
    )
    logger.info(
        "seed=%d train_tiles=%d val_tiles=%d test_tiles=%d "
        "train_labels=%d val_labels=%d test_labels=%d class_weight=%s",
        seed,
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        train_indices.size,
        val_indices.size,
        test_indices.size,
        args.class_weight,
    )
    logger.info("train class counts=%s", train_class_counts.astype(int).tolist())
    logger.info(
        "training controls | lr=%.8g batch_size=%d grad_clip_norm=%.4g "
        "scheduler=%s early_stopping_patience=%d min_delta=%.6g",
        args.lr,
        args.batch_size,
        args.grad_clip_norm,
        args.scheduler,
        args.early_stopping_patience,
        args.early_stopping_min_delta,
    )

    best_val_score = -1.0
    early_stopping_reference = -1.0
    best_val_metrics = None
    best_epoch = 0
    best_path = run_dir / "best_model.pth"
    losses = []
    history = []
    diagnostics_path = run_dir / "batch_gradient_diagnostics.jsonl"
    if diagnostics_path.exists():
        diagnostics_path.unlink()
    validations_without_improvement = 0
    stopped_early = False
    stop_epoch = args.max_epoch
    start = time.perf_counter()
    for epoch in range(1, args.max_epoch + 1):
        epoch_summary, batch_diagnostics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            class_weights,
            args.tile_size,
            class_count,
            device,
            grad_clip_norm=args.grad_clip_norm,
            collect_diagnostics=args.gradient_diagnostics,
            diagnostic_batch_interval=args.diagnostic_batch_interval,
        )
        loss = epoch_summary["loss"]
        losses.append(loss)
        epoch_summary["epoch"] = epoch
        epoch_summary["lr"] = float(optimizer.param_groups[0]["lr"])
        epoch_summary["val_OA"] = None
        epoch_summary["val_mAcc"] = None
        epoch_summary["val_Kappa"] = None
        epoch_summary["val_mIoU"] = None
        for record in batch_diagnostics:
            record["epoch"] = epoch
        append_jsonl(diagnostics_path, batch_diagnostics)
        should_stop = False
        if epoch == 1 or epoch % args.eval_interval == 0 or epoch == args.max_epoch:
            val_prediction, val_visits = stitch_prediction(
                model,
                val_loader,
                gt.shape,
                args.tile_size,
                device,
            )
            if not np.all(val_visits[val_region] == 1):
                raise AssertionError("独立验证空间没有被验证tile完整覆盖。")
            if np.any(val_visits[train_region]) or np.any(val_visits[test_region]):
                raise AssertionError("验证输入越过空间划分边界。")
            val_metrics = evaluate_prediction(
                val_prediction, val_label, class_count
            )
            epoch_summary["val_OA"] = val_metrics["OA"]
            epoch_summary["val_mAcc"] = val_metrics["mAcc"]
            epoch_summary["val_Kappa"] = val_metrics["Kappa"]
            epoch_summary["val_mIoU"] = val_metrics["mIoU"]
            logger.info(
                "seed=%d epoch=%d/%d loss=%.6f lr=%.8g "
                "grad_mean/max=%.4f/%.4f clipped=%d/%d "
                "valid_px_batch[min/mean/max]=%d/%.1f/%d "
                "val_OA=%.6f val_mAcc=%.6f",
                seed,
                epoch,
                args.max_epoch,
                loss,
                epoch_summary["lr"],
                epoch_summary["grad_norm_before_clip_mean"],
                epoch_summary["grad_norm_before_clip_max"],
                epoch_summary["clipped_batch_count"],
                epoch_summary["batch_count"],
                epoch_summary["batch_valid_pixels_min"],
                epoch_summary["batch_valid_pixels_mean"],
                epoch_summary["batch_valid_pixels_max"],
                val_metrics["OA"],
                val_metrics["mAcc"],
            )
            current_score = val_metrics[args.selection_metric]
            checkpoint_improved = current_score > best_val_score
            significant_improvement = current_score > (
                early_stopping_reference + args.early_stopping_min_delta
            )
            if checkpoint_improved or best_val_metrics is None:
                best_val_score = current_score
                best_val_metrics = dict(val_metrics)
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
            if significant_improvement:
                early_stopping_reference = current_score
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1
            epoch_summary["checkpoint_improved"] = checkpoint_improved
            epoch_summary["significant_validation_improvement"] = (
                significant_improvement
            )
            epoch_summary["validations_without_improvement"] = (
                validations_without_improvement
            )
            if args.scheduler == "plateau":
                scheduler.step(current_score)
            if (
                args.early_stopping_patience > 0
                and validations_without_improvement
                >= args.early_stopping_patience
            ):
                stopped_early = True
                stop_epoch = epoch
                should_stop = True
                logger.info(
                    "EARLY STOP seed=%d epoch=%d best_epoch=%d best_%s=%.6f",
                    seed,
                    epoch,
                    best_epoch,
                    args.selection_metric,
                    best_val_score,
                )
        if scheduler is not None and args.scheduler != "plateau":
            scheduler.step()
        epoch_summary["lr_after_scheduler"] = float(
            optimizer.param_groups[0]["lr"]
        )
        history.append(epoch_summary)
        if should_stop:
            break
    train_seconds = time.perf_counter() - start
    plot_loss(losses, run_dir / "train_loss_curve.png", args.tile_size)
    plot_training_diagnostics(
        history, run_dir / "training_diagnostics.png", args.tile_size
    )
    save_json(run_dir / "training_history.json", history)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_start = time.perf_counter()
    test_prediction, test_visits = stitch_prediction(
        model,
        test_loader,
        gt.shape,
        args.tile_size,
        device,
    )
    test_seconds = time.perf_counter() - test_start
    if not np.all(test_visits[test_region] == 1):
        raise AssertionError(
            f"测试空间没有被{args.tile_size}x{args.tile_size}输出完整覆盖。"
        )
    if np.any(test_visits[train_region]) or np.any(test_visits[val_region]):
        raise AssertionError("测试输入/输出越过空间划分边界。")
    test_metrics = evaluate_prediction(test_prediction, test_label, class_count)
    class_confusions = summarize_class_confusions(
        test_metrics["confusion_matrix"],
        top_k=3,
        class_names=DATASET_CONFIGS[dataset_name].get("class_names"),
    )
    confusion_counts_path = run_dir / "test_confusion_matrix_counts.png"
    confusion_normalized_path = (
        run_dir / "test_confusion_matrix_row_normalized.png"
    )
    class_confusions_path = run_dir / "test_class_confusions.json"
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        confusion_counts_path,
        normalized=False,
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        confusion_normalized_path,
        normalized=True,
    )
    save_json(class_confusions_path, class_confusions)
    if class_count >= 8:
        logger.info(
            "CLASS CONFUSION true_class=8 details=%s",
            json.dumps(class_confusions[7], ensure_ascii=False),
        )
    np.save(run_dir / "test_prediction_full_region.npy", test_prediction)
    save_prediction_png(
        test_prediction,
        test_region,
        class_count,
        run_dir / "test_prediction_dense_region.png",
    )
    save_prediction_png(
        test_prediction,
        test_label >= 0,
        class_count,
        run_dir / "test_prediction_labeled_pixels.png",
    )
    save_project_palette_prediction(
        gt=gt,
        prediction=test_prediction,
        evaluation_label_map=test_label,
        prediction_path=run_dir / "test_prediction_same_palette.png",
        gt_path=run_dir / "test_gt_same_palette.png",
        logger=logger,
    )
    result = {
        "dataset": dataset_name,
        "seed": seed,
        "protocol": (
            "strict train/validation/test raw-pixel-disjoint spatial split; "
            f"dense {args.tile_size} input/output; train-only PCA/minmax"
        ),
        "split_strategy": split["strategy"],
        "tile_size": args.tile_size,
        "split_block_size": split.get("block_size"),
        "train_samples_per_class": args.train_samples,
        "train_label_count": train_indices.size,
        "val_label_count": val_indices.size,
        "test_label_count": test_indices.size,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "stop_epoch": stop_epoch,
        "selection_metric": args.selection_metric,
        "best_val_score": best_val_score,
        "best_val_OA": best_val_metrics["OA"],
        "best_val_mAcc": best_val_metrics["mAcc"],
        "best_val_Kappa": best_val_metrics["Kappa"],
        "best_val_mIoU": best_val_metrics["mIoU"],
        "test_OA": test_metrics["OA"],
        "test_mAcc": test_metrics["mAcc"],
        "test_Kappa": test_metrics["Kappa"],
        "test_mIoU": test_metrics["mIoU"],
        "test_per_class_acc": test_metrics["per_class_acc"],
        "test_IoU": test_metrics["IoU"],
        "test_support": test_metrics["support"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "class_confusions": class_confusions,
        "confusion_matrix_counts_path": confusion_counts_path,
        "confusion_matrix_row_normalized_path": confusion_normalized_path,
        "class_confusions_path": class_confusions_path,
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "weight_path": best_path,
        "optimizer": "Adam",
        "initial_lr": args.lr,
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": args.weight_decay,
        "class_weight_mode": args.class_weight,
        "batch_size": args.batch_size,
        "model_config": run_model_config,
        "selective_scan_backend": selective_scan_backend(),
        "parameter_count_total": parameter_count_total,
        "parameter_count_effective": parameter_count_effective,
        "checkpoint_compatibility_parameter_count": (
            compatibility_parameter_count
        ),
        "grad_clip_norm": args.grad_clip_norm,
        "scheduler": args.scheduler,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "training_history_path": run_dir / "training_history.json",
        "gradient_diagnostics_path": diagnostics_path,
    }
    save_json(run_dir / "result.json", result)
    logger.info(
        "FINAL seed=%d OA=%.6f mAcc=%.6f Kappa=%.6f mIoU=%.6f best_epoch=%d",
        seed,
        test_metrics["OA"],
        test_metrics["mAcc"],
        test_metrics["Kappa"],
        test_metrics["mIoU"],
        best_epoch,
    )
    return result


def run_dataset(args, dataset_name, device):
    config = DATASET_CONFIGS[dataset_name]
    data_root = Path(args.data_set_path)
    data_file = data_root / config["data_file"]
    gt_file = data_root / config["label_file"]
    if not data_file.exists() or not gt_file.exists():
        raise FileNotFoundError(f"缺少数据文件：\n{data_file}\n{gt_file}")
    raw_data, raw_gt = data_load_operate.load_data(dataset_name, str(data_root))
    raw_data, gt, class_count = validate_data(
        raw_data, raw_gt, dataset_name, args.pca_components
    )
    requested_tile_size = args.tile_size
    requested_block_size = args.split_block_size
    effective_args = argparse.Namespace(**vars(args))
    first_split_error = None
    try:
        split = choose_split(gt, class_count, effective_args)
    except ValueError as error:
        first_split_error = str(error)
        fallback = int(args.fallback_tile_size)
        if fallback <= 0 or fallback >= args.tile_size:
            raise
        if fallback < 16 or fallback % 4 != 0:
            raise ValueError(
                "--fallback_tile_size 必须是大于等于16且能被4整除的尺寸。"
            ) from error
        effective_args.tile_size = fallback
        effective_args.split_block_size = fallback
        print(
            f"{dataset_name}: {requested_tile_size}x{requested_tile_size} "
            f"空间划分失败，自动重试 {fallback}x{fallback}。"
        )
        try:
            split = choose_split(gt, class_count, effective_args)
        except ValueError as fallback_error:
            raise ValueError(
                f"{requested_tile_size}x{requested_tile_size} 与 "
                f"{fallback}x{fallback} 均无法满足空间划分。\n\n"
                f"[第一次失败]\n{first_split_error}\n\n"
                f"[回退失败]\n{fallback_error}"
            ) from fallback_error
    args = effective_args
    split["requested_tile_size"] = requested_tile_size
    split["requested_split_block_size"] = requested_block_size
    split["effective_tile_size"] = args.tile_size
    split["effective_split_block_size"] = args.split_block_size
    split["used_tile_fallback"] = args.tile_size != requested_tile_size
    if first_split_error is not None:
        split["initial_split_failure"] = first_split_error
    train_region = regions_to_mask(gt.shape, split["train_regions"])
    val_region = regions_to_mask(gt.shape, split["val_regions"])
    test_region = regions_to_mask(gt.shape, split["test_regions"])
    if (
        np.any(train_region & val_region)
        or np.any(train_region & test_region)
        or np.any(val_region & test_region)
    ):
        raise AssertionError("train/validation/test空间发生重叠。")
    if not np.all(train_region | val_region | test_region):
        raise AssertionError("存在未分配给train/validation/test的原始像素。")

    dataset_dir = (
        Path(args.work_dir)
        / args.exp_name
        / f"{dataset_name}_{args.run_tag}"
        / model_config_slug(args)
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(
        dataset_dir / "train.log",
        f"dense{args.tile_size}_{dataset_name}_{time.time_ns()}",
    )
    train_counts = counts_in_mask(gt, train_region, class_count)
    val_counts = counts_in_mask(gt, val_region, class_count)
    test_counts = counts_in_mask(gt, test_region, class_count)
    split_info = split_public_info(split)
    split_info.update(
        {
            "dataset": dataset_name,
            "image_shape": list(gt.shape),
            "tile_size": args.tile_size,
            "requested_tile_size": requested_tile_size,
            "requested_split_block_size": requested_block_size,
            "effective_tile_size": args.tile_size,
            "effective_split_block_size": args.split_block_size,
            "used_tile_fallback": args.tile_size != requested_tile_size,
            "actual_train_counts": train_counts,
            "actual_val_counts": val_counts,
            "actual_test_counts": test_counts,
            "all_counts": counts_in_mask(gt, gt > 0, class_count),
            "train_raw_pixel_count": int(train_region.sum()),
            "val_raw_pixel_count": int(val_region.sum()),
            "test_raw_pixel_count": int(test_region.sum()),
            "train_val_raw_pixel_overlap": int(
                np.count_nonzero(train_region & val_region)
            ),
            "train_test_raw_pixel_overlap": int(
                np.count_nonzero(train_region & test_region)
            ),
            "val_test_raw_pixel_overlap": int(
                np.count_nonzero(val_region & test_region)
            ),
            "pca_fit_scope": "train spatial region only",
        }
    )
    save_json(dataset_dir / "spatial_split.json", split_info)
    np.savez_compressed(
        dataset_dir / "spatial_split_masks.npz",
        train_region=train_region,
        val_region=val_region,
        test_region=test_region,
    )
    save_split_png(
        gt,
        train_region,
        val_region,
        test_region,
        dataset_dir / "spatial_split.png",
    )
    compact = {key: value for key, value in split_info.items() if key != "blocks"}
    if "straight_failure" in compact:
        failure = compact["straight_failure"]
        compact["straight_failure"] = {
            "deficit": failure.get("deficit"),
            "axis": failure.get("axis"),
            "cut1": failure.get("cut1"),
            "cut2": failure.get("cut2"),
        }
    # 完整首次失败原因已写入 spatial_split.json，终端只打印简短摘要。
    if "initial_split_failure" in compact:
        compact["initial_split_failure"] = compact[
            "initial_split_failure"
        ].splitlines()[0]
    logger.info("dataset=%s data=%s gt=%s", dataset_name, raw_data.shape, gt.shape)
    logger.info(
        "dense tile request=%dx%d effective=%dx%d | split block request=%d effective=%d",
        requested_tile_size,
        requested_tile_size,
        args.tile_size,
        args.tile_size,
        requested_block_size,
        args.split_block_size,
    )
    logger.info(
        "split=%s",
        json.dumps(
            compact,
            ensure_ascii=False,
            default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item,
        ),
    )
    image = fit_train_only_preprocess(
        raw_data,
        train_region,
        args.pca_components,
        dataset_dir / "train_only_preprocess.npz",
        args.split_seed,
        logger,
    )
    results = [
        run_seed(
            args,
            dataset_name,
            image,
            gt,
            class_count,
            split,
            train_region,
            val_region,
            test_region,
            dataset_dir,
            seed,
            device,
            logger,
        )
        for seed in args.seeds
    ]
    summary = {
        "dataset": dataset_name,
        "runs": len(results),
        "seeds": args.seeds,
        "model_config": {
            **args.model_config,
            "in_channels": image.shape[2],
            "num_classes": class_count,
            "selective_scan_backend": selective_scan_backend(),
        },
        "hidden_dim": args.hidden_dim,
        "selective_scan_backend": selective_scan_backend(),
        "split_strategy": split["strategy"],
        "tile_size": args.tile_size,
        "split_block_size": split.get("block_size"),
    }
    for key, name in (
        ("test_OA", "OA"),
        ("test_mAcc", "mAcc"),
        ("test_Kappa", "Kappa"),
        ("test_mIoU", "mIoU"),
    ):
        values = np.asarray([result[key] for result in results]) * 100.0
        summary[f"{name}_percent_mean"] = float(values.mean())
        summary[f"{name}_percent_std"] = float(
            values.std(ddof=1) if values.size > 1 else 0.0
        )
        summary[f"{name}_percent_values"] = values
    save_json(dataset_dir / "summary.json", summary)
    logger.info("SUMMARY %s", summary)
    return summary


def resolve_device(text):
    if text.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA不可用，自动切换到CPU。")
        return torch.device("cpu")
    return torch.device(text)


def main():
    args = resolve_model_arguments(build_parser().parse_args())
    if args.dataset_index is not None:
        indexed = DATASET_ORDER[args.dataset_index]
        if args.dataset not in {"UP", indexed}:
            raise ValueError("--dataset 与 --dataset_index 指向不同数据集。")
        args.dataset = indexed
    if args.tile_size < 16 or args.tile_size % 4 != 0:
        raise ValueError("--tile_size 必须大于等于16且能被4整除，例如64或128。")
    if args.split_block_size < args.tile_size:
        raise ValueError("--split_block_size 不能小于 --tile_size。")
    if args.split_block_size % args.tile_size != 0:
        raise ValueError("--split_block_size 必须是 --tile_size 的整数倍。")
    if (
        args.train_samples < 0
        or args.val_samples <= 0
        or args.min_train_samples <= 0
        or args.min_val_samples <= 0
        or args.min_test_samples <= 0
    ):
        raise ValueError(
            "train_samples必须>=0，train/validation/test最低数量必须>0。"
        )
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch_size和eval_batch_size必须大于0。")
    if args.hidden_dim <= 0:
        raise ValueError("hidden_dim必须大于0。")
    if args.max_epoch <= 0 or args.eval_interval <= 0:
        raise ValueError("max_epoch和eval_interval必须大于0。")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("lr必须大于0，weight_decay不能为负。")
    if args.grad_clip_norm < 0:
        raise ValueError("grad_clip_norm不能为负；设为0表示关闭。")
    if args.diagnostic_batch_interval <= 0:
        raise ValueError("diagnostic_batch_interval必须大于0。")
    if args.scheduler_step_size <= 0:
        raise ValueError("scheduler_step_size必须大于0。")
    if not 0 < args.scheduler_gamma < 1:
        raise ValueError("scheduler_gamma必须位于(0, 1)。")
    if args.scheduler_patience < 0 or args.scheduler_min_lr < 0:
        raise ValueError("scheduler_patience和scheduler_min_lr不能为负。")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience不能为负；设为0表示关闭。")
    if args.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta不能为负。")
    device = resolve_device(args.device)
    selected = list(DATASET_ORDER) if args.dataset == "all" else [args.dataset]
    summaries = [run_dataset(args, name, device) for name in selected]
    output = (
        Path(args.work_dir)
        / args.exp_name
        / (
            f"all_datasets_summary_{args.run_tag}_"
            f"{model_config_slug(args)}.json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, summaries)
    print(f"全部完成，汇总已保存到：{output}")


if __name__ == "__main__":
    main()
