"""
与当前 16x16 严格空间划分 QAT 完全对齐的 BothMamba FPGA INT8 模拟。

放置位置：
    与 train_mambahsi_spatial_split_dense_qat.py、utils/ 同一目录。

依赖：
    本文件内置 Conv/Linear 的整数 MAC、INT32 bias 和重量化模拟；
    单文件QAT脚本提供与训练完全相同的MambaHSI和量化/BN融合实现。

运行示例：
    python both_FPGA_21patch_dual.py \
        --qat-run-dir results/SPATIAL_SPLIT_3WAY_DENSE_QAT/\
UP_int8_lsq_freeze20_detach_bias_lr1/run_seed0 \
        --dataset UP \
        --data-path ./data \
        --output-dir ./sim_results_both_dense16

重要：
    1. 只接受 tile_size=16、split_block_size=16 的 blocks 协议；
    2. 必须复用 FP32 保存的 train_only_preprocess.npz、空间 mask 和样本索引；
    3. 不再重新拟合 PCA、不裁剪场景、不使用旧随机划分，也不做重叠滑窗；
    4. checkpoint 必须是当前 BothMamba QAT 的 best_qat_foldaware.pth。
"""

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.evaluation import Evaluator
import train_mambahsi_spatial_split_dense_qat as _qat_source

# 与产生checkpoint/保存预测的单文件QAT共用唯一模型和量化代码源。
from train_mambahsi_spatial_split_dense_qat import (
    MambaHSI,
    build_configured_model,
    DEFAULT_MODEL_CONFIG,
    d_path_simulator_compatible,
    resolve_ssm_contract,
    load_dataset,
    run_selective_scan as _qat_run_selective_scan,
    _reference_selective_scan as _qat_reference_selective_scan,
    freeze_lsq_initialization,
    fuse_qat_model_bns_for_deploy,
    prepare_qat_model,
    validate_fpga_qat_contract,
)

from ssm_error_ablation import (
    SSMNumericConfig, add_numeric_arguments, config_from_args,
    compile_coefficients, scan_codes, ErrorRecorder,
)
from ssm_d_path import compile_d_path, certify_d_requant


def run_selective_scan(u, delta, A, B, C, D=None):
    """Device-safe adapter, including for older server-side QAT modules."""
    if not u.is_cuda:
        return _qat_reference_selective_scan(u, delta, A, B, C, D=D)
    return _qat_run_selective_scan(u, delta, A, B, C, D=D)


@contextlib.contextmanager
def _device_safe_qat_scan():
    """Use the adapter inside QAT forwards as well as staged diagnostics.

    Restore the imported module on success or failure. No checkpoint tensors,
    quantization configuration, or CUDA-kernel error handling are changed.
    """
    original = _qat_source.run_selective_scan
    _qat_source.run_selective_scan = run_selective_scan
    try:
        yield
    finally:
        _qat_source.run_selective_scan = original


def ssm_readout_to_int8(value, physical_scale, output_scale, numeric):
    """Only change the final SSM readout boundary; recurrence is untouched.

    Integer SSM values are codes; counterfactual SSM values are physical units.
    The ideal branch introduces no intermediate rounding to the state grid.
    """
    numeric = SSMNumericConfig.from_dict(numeric)
    if numeric.resolved_readout_requantization == 'hardware':
        return requantize_int8(value, physical_scale, output_scale)
    physical = value.double()
    if numeric.error_source == 'all':
        physical = physical * physical_scale.double()
    return round_half_away_from_zero(physical / output_scale.double()).clamp(
        INT8_MIN, INT8_MAX).to(torch.int8)


def selective_scan_fn(u, delta, A, B, C, D=None, z=None,
                      delta_bias=None, delta_softplus=True):
    if z is not None or delta_bias is not None or not delta_softplus:
        raise ValueError("Diagnostic scan expects no z/bias and softplus=True")
    return run_selective_scan(u, delta, A, B, C, D=D)


INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
TILE_SIZE = 16
SPLIT_BLOCK_SIZE = 16
EXPECTED_SKIP_SCALE = 2.0
SSM_LUT_SCHEMA_VERSION = 4
SSM_DT_OUTPUT_BITS = 8
SSM_DT_ADDRESS_COUNT = 1 << SSM_DT_OUTPUT_BITS
SSM_A_FRACTION_BITS = 24
SSM_A_LOGICAL_BITS = 1 + SSM_A_FRACTION_BITS
# K(dt) folds both the dynamic-B scale and the constant SSM-u scale:
#     K = softplus(dt_code*s_dt) * s_B * s_u.
# A signed INT8 B code multiplied by the largest 19-bit unsigned K code fits
# in a signed 27-bit DSP48E2 input even for B=-128.  The fractional position is
# selected per core, up to Q24, so Block2/Spa automatically uses Q23 while the
# other currently exported cores retain Q24.
SSM_K_LOGICAL_BITS = 19
SSM_K_PREFERRED_FRACTION_BITS = 24
SSM_KB_SIGNED_BITS = 27
SSM_COE_WORD_BITS = 32
SSM_LUT_ROUNDING = "half_away_from_zero"
SIMULATOR_BUILD = "single-qat-source-d-path-batch-requant-k-20260914"


def round_half_away_from_zero(value):
    """与QAT Round.forward及常见RTL舍入规则一致。"""
    value = torch.as_tensor(value)
    return torch.sign(value) * torch.floor(torch.abs(value) + 0.5)


def choose_unsigned_fraction_bits(value, total_bits=32, preferred_bits=32):
    """为一个非负、共享格式的ROM选择尽可能高的小数位数。"""
    value = torch.as_tensor(value, dtype=torch.float64)
    if value.numel() == 0:
        raise ValueError("定点格式选择要求非空系数张量。")
    if not torch.isfinite(value).all():
        raise FloatingPointError("定点格式选择遇到NaN或Inf。")
    if torch.any(value < 0):
        raise ValueError("unsigned定点格式不能编码负系数。")

    max_value = float(value.max().item())
    max_code = (1 << int(total_bits)) - 1
    if max_value == 0.0:
        return int(preferred_bits)

    # 先按解析式估算，再用与RTL相同的舍入规则复核边界。
    fraction_bits = min(
        int(preferred_bits),
        int(math.floor(math.log2(max_code / max_value))),
    )
    while fraction_bits >= 0:
        rounded_max = int(
            round_half_away_from_zero(max_value * (2.0**fraction_bits)).item()
        )
        if rounded_max <= max_code:
            return fraction_bits
        fraction_bits -= 1
    raise OverflowError(
        f"最大系数{max_value:.9g}无法放入unsigned {total_bits}-bit ROM。"
    )


def get_m_int_shift(multiplier, q_bits=16):
    """把浮点重量化比例编译为signed q_bits整数乘数和公共移位。"""
    multiplier = torch.as_tensor(multiplier, dtype=torch.float64)
    max_multiplier = float(multiplier.abs().max().item())
    if max_multiplier == 0.0:
        return torch.zeros_like(multiplier, dtype=torch.int32), 0
    if not math.isfinite(max_multiplier):
        raise FloatingPointError("重量化multiplier必须有限。")
    shift = int(
        math.floor(math.log2(((1 << (q_bits - 1)) - 1) / max_multiplier))
    )
    if shift < -62 or shift > 62:
        raise OverflowError(
            f"重量化shift={shift}超出本地signed INT64模拟支持范围[-62, 62]。"
        )
    scaled = multiplier * float(2.0**shift)
    multiplier_int = round_half_away_from_zero(scaled).to(torch.int32)
    return multiplier_int, shift


def apply_multiplier_shift(accumulator, multiplier_int, shift):
    """signed INT64乘法后执行对称half-away-from-zero移位舍入。"""
    accumulator = accumulator.to(torch.int64)
    multiplier_int = multiplier_int.to(device=accumulator.device, dtype=torch.int64)
    if accumulator.numel() and multiplier_int.numel():
        max_product = int(accumulator.abs().max().item()) * int(
            multiplier_int.abs().max().item()
        )
        if max_product > torch.iinfo(torch.int64).max:
            raise OverflowError("重量化乘法超出signed INT64模拟容器。")
    product = accumulator * multiplier_int
    if shift > 0:
        rounding_bias = 1 << (shift - 1)
        if max_product < rounding_bias:
            return torch.zeros_like(product)
        if max_product > torch.iinfo(torch.int64).max - rounding_bias:
            raise OverflowError("重量化舍入偏置会使signed INT64容器溢出。")
        magnitude = (product.abs() + rounding_bias) >> shift
        return torch.where(product < 0, -magnitude, magnitude)
    if shift < 0:
        left_shift = -shift
        if max_product > (torch.iinfo(torch.int64).max >> left_shift):
            raise OverflowError("重量化左移会使signed INT64容器溢出。")
        return product << (-shift)
    return product


def report_fixed_float(name, reference, codes=None, scale=None, fixed=None):
    """统一打印一个硬件边界的定点反量化值与浮点参考值。"""
    reference = torch.as_tensor(reference).detach().double()
    if fixed is None:
        if codes is None or scale is None:
            raise ValueError(f"{name}: codes/scale或fixed至少提供一组。")
        fixed = (
            torch.as_tensor(codes).detach().double()
            * torch.as_tensor(scale).detach().double().reshape(())
        )
    else:
        fixed = torch.as_tensor(fixed).detach().double()

    if fixed.shape != reference.shape:
        raise ValueError(
            f"{name}: 定点/浮点形状不一致："
            f"{tuple(fixed.shape)} vs {tuple(reference.shape)}。"
        )
    if fixed.numel() == 0:
        raise ValueError(f"{name}: 不比较空张量。")
    if not torch.isfinite(fixed).all() or not torch.isfinite(reference).all():
        raise FloatingPointError(f"{name}: 定点反量化值或浮点参考含NaN/Inf。")

    fixed_flat = fixed.reshape(-1)
    reference_flat = reference.reshape(-1)
    difference = (fixed_flat - reference_flat).abs()
    cosine = F.cosine_similarity(
        fixed_flat,
        reference_flat,
        dim=0,
        eps=1e-12,
    ).item()
    print(
        f"  fixed_vs_float [{name}] | "
        f"fixed=[{fixed_flat.min().item():.8g}, {fixed_flat.max().item():.8g}] | "
        f"float=[{reference_flat.min().item():.8g}, "
        f"{reference_flat.max().item():.8g}] | "
        f"cosine={cosine:.8f} | MAE={difference.mean().item():.8g} | "
        f"MaxAE={difference.max().item():.8g}"
    )
    return {
        "cosine": cosine,
        "mae": difference.mean().item(),
        "maxae": difference.max().item(),
    }


def sim_layer_int8(
    x_int,
    s_in,
    module,
    name,
    output_dir,
    ref_val=None,
    s_out_target=None,
):
    """用FP64承载精确整数MAC，再按RTL规则完成INT32/INT8重量化。"""
    device = x_int.device
    weight_bits = int(getattr(module, "nbit_w", 8))
    output_bits = int(getattr(module, "nbit_a", 8))
    weight_min = -(1 << (weight_bits - 1))
    weight_max = (1 << (weight_bits - 1)) - 1
    output_min = -(1 << (output_bits - 1))
    output_max = (1 << (output_bits - 1)) - 1
    s_in = scalar_scale(s_in, device).abs().clamp_min(1e-12)
    s_weight = scalar_scale(module.lsq_w.s.detach(), device).abs().clamp_min(1e-12)
    weight_int = round_half_away_from_zero(
        module.weight.detach() / s_weight
    ).clamp(weight_min, weight_max)

    if isinstance(module, nn.Conv2d):
        kernel_h, kernel_w = (
            (module.kernel_size, module.kernel_size)
            if isinstance(module.kernel_size, int)
            else tuple(module.kernel_size)
        )
        reduction_terms = (
            (module.in_channels // module.groups) * kernel_h * kernel_w
        )
    elif isinstance(module, nn.Conv1d):
        kernel = (
            module.kernel_size
            if isinstance(module.kernel_size, int)
            else module.kernel_size[0]
        )
        reduction_terms = (module.in_channels // module.groups) * kernel
    elif isinstance(module, nn.Linear):
        reduction_terms = module.in_features
    else:
        raise TypeError(f"{name}: 不支持的定点层类型{type(module).__name__}。")
    worst_case_mac = (
        int(x_int.to(torch.int64).abs().max().item())
        * int(weight_int.abs().max().item())
        * int(reduction_terms)
    )
    if worst_case_mac > INT32_MAX:
        raise OverflowError(
            f"{name}: 保守部分和上界{worst_case_mac}超出signed INT32；"
            "即使最终卷积值发生抵消也不满足硬件累加器合同。"
        )

    # FP64 can represent all integer MAC values exactly while their magnitude is
    # below 2^53. This avoids the old FP32 accumulator losing integer LSBs.
    x_exact = x_int.to(torch.float64)
    weight_exact = weight_int.to(device=device, dtype=torch.float64)
    if isinstance(module, nn.Conv2d):
        accumulator_float = F.conv2d(
            x_exact,
            weight_exact,
            None,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
        )
    elif isinstance(module, nn.Conv1d):
        accumulator_float = F.conv1d(
            x_exact,
            weight_exact,
            None,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
        )
    elif isinstance(module, nn.Linear):
        accumulator_float = F.linear(x_exact, weight_exact, None)
    if accumulator_float.abs().max().item() >= 2**53:
        raise OverflowError(f"{name}: 整数MAC超过FP64精确整数范围2^53。")
    accumulator_rounded = torch.round(accumulator_float)
    integer_error = (accumulator_float - accumulator_rounded).abs().max().item()
    if integer_error > 1e-6:
        raise ArithmeticError(
            f"{name}: FP64卷积结果偏离整数网格{integer_error:.3e}，"
            "不能安全解释为整数MAC；请改用--device cpu复核。"
        )
    accumulator = accumulator_rounded.to(torch.int64)
    if torch.any(accumulator < INT32_MIN) or torch.any(accumulator > INT32_MAX):
        raise OverflowError(f"{name}: 原始MAC超出signed INT32累加器范围。")

    accumulator_scale = s_in * s_weight
    bias_int = None
    if module.bias is not None:
        # The current QAT SymmetricQuantFunction and static bias exporter use
        # torch.round (ties-to-even), so the hardware simulator mirrors it here.
        bias_int = torch.round(module.bias.detach() / accumulator_scale)
        if not torch.isfinite(bias_int).all():
            raise FloatingPointError(f"{name}: bias整数码包含NaN或Inf。")
        if torch.any(bias_int < INT32_MIN) or torch.any(bias_int > INT32_MAX):
            raise OverflowError(f"{name}: bias整数码超出signed INT32。")
        bias_int = bias_int.to(device=device, dtype=torch.int64)
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            view = [1, -1] + [1] * (accumulator.ndim - 2)
            accumulator = accumulator + bias_int.view(view)
        else:
            accumulator = accumulator + bias_int

    if torch.any(accumulator < INT32_MIN) or torch.any(accumulator > INT32_MAX):
        raise OverflowError(f"{name}: MAC+bias超出signed INT32累加器范围。")
    if s_out_target is None:
        physical = accumulator.to(torch.float64) * accumulator_scale.double()
        s_out = (physical.abs().max() / output_max).clamp_min(1e-12).float()
    else:
        s_out = scalar_scale(s_out_target, device).abs().clamp_min(1e-12)
    multiplier = accumulator_scale / s_out
    multiplier_int, shift = get_m_int_shift(multiplier, q_bits=16)
    requantized = apply_multiplier_shift(
        accumulator,
        multiplier_int.to(device),
        shift,
    )
    output_int = requantized.clamp(output_min, output_max)
    output_dtype = torch.int16 if output_bits > 8 else torch.int8
    output_int = output_int.to(output_dtype)

    print("-" * 100)
    print(
        f"Layer: {name:25s} (INT{output_bits}) | "
        f"acc=[{int(accumulator.min())}, {int(accumulator.max())}] | "
        f"out=[{int(output_int.min())}, {int(output_int.max())}] | "
        f"s_in={s_in.item():.8g} s_w={s_weight.item():.8g} "
        f"s_out={s_out.item():.8g} M_int={int(multiplier_int.item())} "
        f"shift={shift}"
    )
    if ref_val is not None:
        report_fixed_float(
            name,
            ref_val,
            codes=output_int,
            scale=s_out,
        )

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            output_path / f"{name}_out_int.txt",
            output_int.detach().cpu().numpy().reshape(-1),
            fmt="%d",
        )
        np.savetxt(
            output_path / f"{name}_M_int.txt",
            multiplier_int.detach().cpu().numpy().reshape(-1),
            fmt="%d",
        )
        np.savetxt(
            output_path / f"{name}_M_shift.txt",
            np.asarray([shift], dtype=np.int32),
            fmt="%d",
        )
        np.savetxt(
            output_path / f"{name}_weight_int8.txt",
            weight_int.detach().cpu().numpy().astype(np.int8).reshape(-1),
            fmt="%d",
        )
        if bias_int is not None:
            np.savetxt(
                output_path / f"{name}_bias_int32.txt",
                bias_int.detach().cpu().numpy().astype(np.int32).reshape(-1),
                fmt="%d",
            )
    return output_int, s_out, multiplier


def scalar_scale(scale, device):
    """把 LSQ scale 统一成位于 device 上的标量 Tensor。"""
    scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
    if scale.numel() != 1:
        raise RuntimeError(
            "FPGA仅支持per-tensor scale，但收到"
            f"{scale.numel()}个scale。"
        )
    return scale.reshape(())


def get_activation_scale(module, fallback, device):
    """读取 Quan 层的输入激活 scale；普通层则沿用上游 scale。"""
    if hasattr(module, "lsq_a") and hasattr(module.lsq_a, "s"):
        return scalar_scale(module.lsq_a.s.detach(), device)
    return scalar_scale(fallback, device)


def get_boundary_scale(boundary, fallback, device):
    """读取显式 ActivationFakeQuant 硬件边界的输出 scale。"""
    if hasattr(boundary, "lsq_a") and hasattr(boundary.lsq_a, "s"):
        return scalar_scale(boundary.lsq_a.s.detach(), device)
    return scalar_scale(fallback, device)


def sim_layer_requant(
    x_int,
    s_in,
    module,
    name,
    output_dir,
    s_out_target,
    ref_val=None,
    output_bits=8,
):
    """
    层输出必须由“下一层输入/显式边界”的 scale 重量化。

    QAT 中 nbit_a 描述当前层输入，而这里还需显式指定硬件输出位宽。
    因此使用只读浅副本明确
    指定输出位宽，并禁止动态生成输出 scale。
    """
    if s_out_target is None:
        raise ValueError(f"{name}: 必须提供下一层输入 scale")

    if not hasattr(module, "_fpga_weight_stats_reported"):
        weight_scale = module.lsq_w.s.detach()
        if weight_scale.numel() != 1:
            raise RuntimeError(
                f"{name}: FPGA仅支持per-tensor权重scale，"
                f"当前元素数={weight_scale.numel()}"
            )
        weight_scale = weight_scale.reshape(()).abs().clamp_min(1e-12)
        weight_bits = getattr(module, "nbit_w", 8)
        qmin = -(1 << (weight_bits - 1))
        qmax = (1 << (weight_bits - 1)) - 1
        raw_weight_int = round_half_away_from_zero(
            module.weight.detach() / weight_scale
        )
        weight_int = raw_weight_int.clamp(qmin, qmax)
        clipped_ratio = (
            (raw_weight_int < qmin) | (raw_weight_int > qmax)
        ).float().mean().item() * 100.0
        endpoint_ratio = (
            (weight_int == qmin) | (weight_int == qmax)
        ).float().mean().item() * 100.0
        print(
            f"[Weight Per-Tensor] {name}: "
            f"scale={weight_scale.item():.8g}, "
            f"INT range=[{int(weight_int.min().item())}, "
            f"{int(weight_int.max().item())}], "
            f"endpoint={endpoint_ratio:.4f}%, "
            f"clipped={clipped_ratio:.4f}%"
        )
        if clipped_ratio > 1.0:
            print(
                f"  [Warning] {name} 权重实际裁剪率超过1%，"
                "需要延长QAT或检查该层scale。"
            )
        module._fpga_weight_stats_reported = True

    if module.bias is not None and not hasattr(module, "_fpga_bias_stats_reported"):
        weight_scale = scalar_scale(module.lsq_w.s.detach(), x_int.device).abs()
        accumulator_scale = (
            scalar_scale(s_in, x_int.device).abs() * weight_scale
        ).clamp_min(1e-12)
        # Bias fake quant/export in the current QAT uses torch.round (ties-to-even).
        raw_bias_int = torch.round(module.bias.detach() / accumulator_scale)
        if not torch.isfinite(raw_bias_int).all():
            raise FloatingPointError(f"{name}: INT32 bias code包含NaN或Inf。")
        overflow = (raw_bias_int < INT32_MIN) | (raw_bias_int > INT32_MAX)
        if overflow.any():
            raise OverflowError(
                f"{name}: {int(overflow.sum().item())}个bias超出signed INT32累加器范围。"
            )
        print(
            f"[Bias INT32] {name}: scale={accumulator_scale.item():.8g}, "
            f"code_range=[{int(raw_bias_int.min().item())}, "
            f"{int(raw_bias_int.max().item())}]"
        )
        module._fpga_bias_stats_reported = True

    module_view = copy.copy(module)
    module_view.nbit_a = output_bits
    result = sim_layer_int8(
        x_int,
        s_in,
        module_view,
        name,
        output_dir,
        ref_val=ref_val,
        s_out_target=scalar_scale(
            s_out_target,
            x_int.device,
        ),
    )
    if output_bits > 8 and output_dir:
        # INT9 使用有符号 int16 容器导出，避免被误转成 int8。
        out_int = result[0]
        np.savetxt(
            os.path.join(output_dir, f"{name}_out_int.txt"),
            out_int.detach().cpu().numpy().astype(np.int16).flatten(),
            fmt="%d",
        )
    return result


def requantize_int8(x_int8, s_in, s_out):
    """把一个 INT8 张量从 s_in 重量化到 s_out。"""
    s_in = scalar_scale(s_in, x_int8.device)
    s_out = scalar_scale(s_out, x_int8.device).clamp_min(1e-12)
    multiplier_int, shift = get_m_int_shift(s_in / s_out, q_bits=16)
    y = apply_multiplier_shift(
        x_int8.to(torch.int64),
        multiplier_int.to(x_int8.device),
        shift,
    )
    return y.clamp(INT8_MIN, INT8_MAX).to(torch.int8)


def export_input_requant_contract(
    name,
    x_in,
    s_in,
    s_out,
    y_out,
    output_dir,
):
    """Export the exact branch-input requantization used before in_proj."""
    if not output_dir:
        return
    device = x_in.device
    s_in = scalar_scale(s_in, device)
    s_out = scalar_scale(s_out, device).clamp_min(1e-12)
    multiplier_int, shift = get_m_int_shift(s_in / s_out, q_bits=16)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    arrays = {
        f"{name}_in_int.txt": x_in.detach().cpu().numpy().reshape(-1),
        f"{name}_out_int.txt": y_out.detach().cpu().numpy().reshape(-1),
        f"{name}_M_int.txt": np.asarray([int(multiplier_int.item())]),
        f"{name}_M_shift.txt": np.asarray([int(shift)]),
    }
    for filename, values in arrays.items():
        np.savetxt(output_path / filename, values, fmt="%d")

    with (output_path / f"{name}_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "name": name,
                "input_scale": float(s_in.item()),
                "output_scale": float(s_out.item()),
                "multiplier": int(multiplier_int.item()),
                "shift": int(shift),
                "operation": "clip_int8(round_half_away(x*M/2^shift))",
                "rounding": "round-half-away-from-zero",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def align_and_add_int8(x_a, s_a, x_b, s_b, s_out=None):
    """FPGA 残差/融合加法：先用定点乘数对齐 scale，再做饱和加法。"""
    device = x_a.device
    s_a = scalar_scale(s_a, device)
    s_b = scalar_scale(s_b, device)
    if s_out is None:
        s_out = torch.maximum(s_a, s_b)
    else:
        s_out = scalar_scale(s_out, device)

    m_a, shift_a = get_m_int_shift(s_a / s_out, q_bits=16)
    m_b, shift_b = get_m_int_shift(s_b / s_out, q_bits=16)

    y_a = apply_multiplier_shift(x_a.to(torch.int32), m_a.to(device), shift_a)
    y_b = apply_multiplier_shift(x_b.to(torch.int32), m_b.to(device), shift_b)
    y = (y_a + y_b).clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    return y, s_out


def export_align_add_contract(
    name,
    x_a,
    s_a,
    x_b,
    s_b,
    s_out,
    y_out,
    output_dir,
):
    """导出残差/融合加法的完整整数合同。

    RTL必须分别对a/b执行half-away重量化，然后相加并饱和；
    不可以先相加再只做一次移位。
    """
    if not output_dir:
        return
    device = x_a.device
    s_a = scalar_scale(s_a, device)
    s_b = scalar_scale(s_b, device)
    s_out = scalar_scale(s_out, device)
    m_a, shift_a = get_m_int_shift(s_a / s_out, q_bits=16)
    m_b, shift_b = get_m_int_shift(s_b / s_out, q_bits=16)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    arrays = {
        f"{name}_a_in_int.txt": x_a.detach().cpu().numpy().reshape(-1),
        f"{name}_b_in_int.txt": x_b.detach().cpu().numpy().reshape(-1),
        f"{name}_a_M_int.txt": np.asarray([int(m_a.item())]),
        f"{name}_a_M_shift.txt": np.asarray([int(shift_a)]),
        f"{name}_b_M_int.txt": np.asarray([int(m_b.item())]),
        f"{name}_b_M_shift.txt": np.asarray([int(shift_b)]),
        f"{name}_out_int.txt": y_out.detach().cpu().numpy().reshape(-1),
    }
    for filename, values in arrays.items():
        np.savetxt(output_path / filename, values, fmt="%d")

    with (output_path / f"{name}_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "name": name,
                "a_scale": float(s_a.item()),
                "b_scale": float(s_b.item()),
                "output_scale": float(s_out.item()),
                "a_multiplier": int(m_a.item()),
                "a_shift": int(shift_a),
                "b_multiplier": int(m_b.item()),
                "b_shift": int(shift_b),
                "operation": (
                    "clip_int8(round_away(a*M_a/2^shift_a) + "
                    "round_away(b*M_b/2^shift_b))"
                ),
                "rounding": "round-half-away-from-zero",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def export_first_tile_coe(x_uint8, output_dir):
    """以UINT8原码00..FF导出第一个16x16 tile的各PCA通道。"""
    coe_dir = os.path.join(output_dir, "fpga_input_coe")
    os.makedirs(coe_dir, exist_ok=True)

    channels = (
        x_uint8.squeeze(0)
        .detach()
        .cpu()
        .to(torch.int16)
        .clamp(0, 255)
    )
    for channel_idx, channel in enumerate(channels):
        values = channel.flatten().tolist()
        path = os.path.join(coe_dir, f"img_channel_{channel_idx:02d}.coe")
        with open(path, "w", encoding="utf-8") as file:
            file.write("memory_initialization_radix=16;\n")
            file.write("memory_initialization_vector=\n")
            for idx, value in enumerate(values):
                ending = ";\n" if idx == len(values) - 1 else ",\n"
                file.write(f"{value:02X}{ending}")
    print(f"First patch COE exported to: {coe_dir}")


def export_pcie_input_tiles(image, blocks, input_scale, output_dir, device,
                            print_all=False):
    """Export the exact UINT8 Patch input; no re-fit, re-scale or new padding."""
    export_dir = Path(output_dir) / "pcie_input"
    export_dir.mkdir(parents=True, exist_ok=True)
    channels = int(image.shape[2])
    if channels != 16:
        raise ValueError(f"PCIe Patch contract requires 16 channels, got {channels}")
    scale = float(input_scale.item())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid input scale: {scale}")
    ordered = sorted(blocks, key=lambda b: (b["top"], b["left"]))
    tile_bytes = TILE_SIZE * TILE_SIZE * channels
    manifest_path = export_dir / "manifest.json"
    manifest = {
        "complete": False, "dtype": "uint8", "input_scale": scale,
        "scene_shape_hwc": list(image.shape), "tile_shape_hwc": [16, 16, 16],
        "tile_count": len(ordered), "bytes_per_tile": tile_bytes,
        "layout": "tile raster order -> pixel y -> pixel x -> channel c",
        "word_layout": "128-bit word: channel c at [8*c +: 8]; channel 0 is first binary byte",
        "quantization": "round_half_away_from_zero(tile / input_scale), clamp 0..255",
        "padding": "make_dense16_tile: reflect if min(valid_h,valid_w)>1 else edge; pad before quantization",
        "files": {"binary": "scene_tiles_uint8.bin", "hex_words": "scene_tiles_128b.mem",
                  "all_values": "scene_tiles_uint8.txt"},
        "tiles": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    digest = hashlib.sha256()
    print(f"[PCIe input export] {len(ordered)} tiles -> {export_dir}", flush=True)
    with (export_dir / "scene_tiles_uint8.bin").open("wb") as binary, \
         (export_dir / "scene_tiles_128b.mem").open("w", encoding="ascii") as hex_file, \
         (export_dir / "scene_tiles_uint8.txt").open("w", encoding="ascii") as text_file:
        text_file.write("# UINT8 decimal; tile y x c00..c15; includes ALL padded pixels\n")
        with torch.no_grad():
            for index, block in enumerate(ordered):
                tile_cpu, valid_h, valid_w = make_dense16_tile(image, block)
                tile = tile_cpu.to(device=device, dtype=torch.float32)
                # Identical expression and device/float32 carrier to simulate_tile_dual_int8.
                quantized = round_half_away_from_zero(tile / input_scale).clamp(0, 255)
                hwc = quantized.to(torch.uint8).squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
                payload = hwc.tobytes(order="C")
                if len(payload) != tile_bytes:
                    raise AssertionError("Unexpected PCIe tile byte count")
                binary.write(payload)
                digest.update(payload)
                lines = []
                for pixel, values in enumerate(hwc.reshape(-1, channels)):
                    hex_file.write(values.tobytes()[::-1].hex().upper() + "\n")
                    y, x = divmod(pixel, TILE_SIZE)
                    lines.append(f"{index} {y} {x} " + " ".join(str(int(v)) for v in values) + "\n")
                text_file.writelines(lines)
                if print_all:
                    sys.stdout.write(f"[PCIe TILE {index}] top={block['top']} left={block['left']}\n")
                    sys.stdout.writelines(lines)
                manifest["tiles"].append({
                    "tile_index": index, "top": int(block["top"]), "left": int(block["left"]),
                    "valid_height": valid_h, "valid_width": valid_w,
                    "byte_offset": index * tile_bytes, "bytes": tile_bytes,
                    "crc32": f"{zlib.crc32(payload) & 0xffffffff:08X}",
                })
    manifest.update(complete=True, total_bytes=len(ordered) * tile_bytes,
                    sha256=digest.hexdigest())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[PCIe input export PASS] UINT8 bytes={manifest['total_bytes']}, "
          f"SHA256={manifest['sha256']}\nBinary: {export_dir / 'scene_tiles_uint8.bin'}", flush=True)
    return manifest


def _write_32bit_coe(path, values):
    """按C-order写出32-bit原始位模式；同时支持signed/unsigned payload。"""
    flattened = values.detach().cpu().to(torch.int64).reshape(-1).tolist()
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("memory_initialization_radix=16;\n")
        handle.write("memory_initialization_vector=\n")
        for index, value in enumerate(flattened):
            ending = ";\n" if index == len(flattened) - 1 else ",\n"
            handle.write(f"{int(value) & 0xFFFFFFFF:08X}{ending}")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_ssm_luts(output_dir, name_prefix, cache):
    """Export the Abar and scale-folded K(dt) ROMs for one SSM core."""
    lut_dir = Path(output_dir) / "ssm_luts"
    lut_dir.mkdir(parents=True, exist_ok=True)
    a_lut = cache["a_bar_q24"]
    k_lut = cache["k_multiplier_u19"]
    k_fraction_bits = int(cache["k_fraction_bits"])
    a_base = lut_dir / f"{name_prefix}_Abar_uq1_24"
    k_base = lut_dir / f"{name_prefix}_K_q{k_fraction_bits}_u19"

    a_numpy = np.ascontiguousarray(
        a_lut.detach().cpu().numpy().astype("<i4", copy=False)
    )
    k_numpy = np.ascontiguousarray(
        k_lut.detach().cpu().numpy().astype("<u4", copy=False)
    )
    a_npy_path = Path(f"{a_base}.npy")
    k_npy_path = Path(f"{k_base}.npy")
    a_txt_path = Path(f"{a_base}.txt")
    k_txt_path = Path(f"{k_base}.txt")
    a_coe_path = Path(f"{a_base}.coe")
    k_coe_path = Path(f"{k_base}.coe")

    np.save(a_npy_path, a_numpy)
    np.save(k_npy_path, k_numpy)
    np.savetxt(
        a_txt_path,
        a_numpy.reshape(-1),
        fmt="%d",
    )
    np.savetxt(
        k_txt_path,
        k_numpy.reshape(-1),
        fmt="%u",
    )
    _write_32bit_coe(a_coe_path, a_lut)
    _write_32bit_coe(k_coe_path, k_lut)

    manifest = {
        "numeric_config": cache["numeric_config"],
        "range_certificate": cache["range_certificate"],
        "rtl_baseline_compatible": True,
        "schema_version": SSM_LUT_SCHEMA_VERSION,
        "name": name_prefix,
        "dt_proj_input_bits": 9,
        "dt_output_bits": SSM_DT_OUTPUT_BITS,
        "dt_code_min": INT8_MIN,
        "dt_code_max": INT8_MAX,
        "dt_address_formula": "address = dt_output_code + 128",
        "dt_address_count": SSM_DT_ADDRESS_COUNT,
        "rounding": SSM_LUT_ROUNDING,
        "dt_bias_contract": (
            "dt_proj INT8 output already contains its one and only bias; "
            "selective_scan delta_bias=None and the LUT adds no bias"
        ),
        "Abar_shape": list(a_lut.shape),
        "Abar_axis_order": ["dt_address", "state"],
        "Abar_flat_address_formula": (
            "dt_address * d_state + state"
        ),
        "Abar_entry_count": int(a_lut.numel()),
        "Abar_logical_bits": SSM_A_LOGICAL_BITS,
        "Abar_min_code": int(a_lut.min().item()),
        "Abar_max_code": int(a_lut.max().item()),
        "Abar_format": (
            "unsigned UQ1.24 stored in int32; exact 1.0 is code 16777216"
        ),
        "Abar_formula": (
            "round_half_away(exp(softplus(dt_code*s_dt)"
            "*(-exp(A_log_shared[state])))*2^24)"
        ),
        "Abar_coe_contract": (
            "one 32-bit hexadecimal word per scalar entry; bits[24:0] are the "
            "unsigned UQ1.24 payload and bits[31:25] are zero"
        ),
        "K_shape": list(k_lut.shape),
        "K_axis_order": ["dt_address"],
        "K_flat_address_formula": "dt_address",
        "K_entry_count": int(k_lut.numel()),
        "K_logical_bits": SSM_K_LOGICAL_BITS,
        "K_signed": False,
        "K_fraction_bits": k_fraction_bits,
        "K_min_code": int(k_lut.min().item()),
        "K_max_code": int(k_lut.max().item()),
        "K_formula": (
            "round_half_away(softplus(dt_code*s_dt)*s_B*s_u"
            f"*2^{k_fraction_bits})"
        ),
        "K_coe_contract": (
            "one unsigned 32-bit hexadecimal word per scalar entry; only "
            f"bits[{SSM_K_LOGICAL_BITS - 1}:0] are used. Multiply K by signed "
            "INT8 B to form signed KB, then multiply KB by signed INT8 U. "
            f"The KBU result is Q{k_fraction_bits}; left shift it by "
            f"{2 * SSM_A_FRACTION_BITS - k_fraction_bits} before adding it to "
            "Abar_Q24*state_Q24, then round the common accumulator by 24 bits."
        ),
        "coe_word_bits": SSM_COE_WORD_BITS,
        "rom_generation_dtype": "float64 from frozen float32 parameters/scales",
        "content_signature_sha256": cache["content_signature_sha256"],
        "Abar_raw_little_endian_int32_sha256": hashlib.sha256(
            a_numpy.tobytes(order="C")
        ).hexdigest(),
        "K_raw_little_endian_uint32_sha256": hashlib.sha256(
            k_numpy.tobytes(order="C")
        ).hexdigest(),
        "Abar_coe_sha256": _sha256_file(a_coe_path),
        "K_coe_sha256": _sha256_file(k_coe_path),
        "Abar_npy_sha256": _sha256_file(a_npy_path),
        "K_npy_sha256": _sha256_file(k_npy_path),
        "Abar_txt_sha256": _sha256_file(a_txt_path),
        "K_txt_sha256": _sha256_file(k_txt_path),
        "s_dt": cache["s_dt"],
        "s_B": cache["s_b"],
        "s_u": cache["s_u"],
        "flat_file_order": "C-order; last axis varies fastest",
    }
    with (lut_dir / f"{name_prefix}_lut_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    for legacy_path in lut_dir.glob(f"{name_prefix}_Bmult_uq*"):
        if legacy_path.is_file():
            legacy_path.unlink()


def get_or_build_ssm_luts(m, s_dt, s_b, s_u, name_prefix, output_dir, device):
    numeric = getattr(m, "_ssm_numeric_config", SSMNumericConfig())
    theta = m.A_log_shared.detach().float()
    if theta.numel() != m.d_state:
        raise ValueError("channel-shared pole dimension mismatch")
    values = [float(scalar_scale(x, device)) for x in (s_dt, s_b, s_u)]
    signature = hashlib.sha256(
        json.dumps([numeric.to_dict(), values], sort_keys=True).encode()
        + theta.cpu().numpy().tobytes()).hexdigest()
    cache = m.__dict__.get("_fpga_ssm_lut_cache")
    if cache is None or cache['content_signature_sha256'] != signature:
        tables = compile_coefficients(theta, *values, numeric)
        cache = dict(
            content_signature_sha256=signature,
            a_bar_q24=tables['a'].to(device),
            k_multiplier_u19=tables['k'].to(device),
            k_fraction_bits=tables['k_fraction_bits'],
            a_float=(-torch.exp(theta)).unsqueeze(0).expand(m.d_inner, -1).contiguous(),
            s_dt=values[0], s_b=values[1], s_u=values[2],
            tables=tables, numeric_config=numeric.to_dict(),
            range_certificate=tables['range_certificate'], exported_dirs=set())
        m.__dict__['_fpga_ssm_lut_cache'] = cache
        print(f"[SSM ROM] {name_prefix}: {numeric.dt_count} addresses, "
              f"A UQ1.{numeric.a_fraction_bits}, K U{numeric.k_bits}/Q{tables['k_fraction_bits']}, "
              f"state {numeric.state_bits}/{numeric.state_fraction_bits}, "
              f"range proof {tables['range_certificate']['required_signed_bits']} signed bits")
    if output_dir:
        key = str(Path(output_dir).resolve())
        if key not in cache['exported_dirs']:
            if numeric.rtl_baseline_compatible and not getattr(m, 'use_D', False):
                _export_ssm_luts(output_dir, name_prefix, cache)
            else:
                lut_dir = Path(output_dir) / 'ssm_luts'
                lut_dir.mkdir(parents=True, exist_ok=True)
                a_name = f'{name_prefix}_Abar_uq1_{numeric.a_fraction_bits}.npy'
                k_name = f'{name_prefix}_K_q{cache["k_fraction_bits"]}_u{numeric.k_bits}.npy'
                np.save(lut_dir / a_name, cache['tables']['a'].numpy())
                np.save(lut_dir / k_name, cache['tables']['k'].numpy())
                manifest = dict(schema_version=5, name=name_prefix,
                    numeric_config=numeric.to_dict(), rtl_baseline_compatible=False,
                    representation=('integer-format-variant' if numeric.error_source == 'all' else 'floating-counterfactual'),
                    Abar_file=a_name, K_file=k_name, dt_address_count=numeric.dt_count,
                    dt_address_formula=f'dt_code + {-numeric.dt_min}',
                    range_certificate=cache['range_certificate'],
                    content_signature_sha256=signature,
                    coefficient_error=cache['tables']['coefficient_error'],
                    approximation_scope='PWL options are numerical SFU baselines; no physical PPA measured',
                    logical_rom_bits=cache['tables']['logical_rom_bits'])
                (lut_dir / f'{name_prefix}_lut_manifest.json').write_text(json.dumps(manifest, indent=2))
            cache['exported_dirs'].add(key)
    return cache['a_float'], cache['a_bar_q24'], cache['k_multiplier_u19'], cache['k_fraction_bits']


def lookup_ssm_luts(a_bar_q24_lut, k_multiplier_u19_lut, dt_codes):
    """Lookup with a domain inferred from the actual power-of-two table size."""
    if a_bar_q24_lut.ndim != 2:
        raise ValueError('A LUT must be [addresses,state]')
    count = a_bar_q24_lut.shape[0]
    if count < 4 or count & (count - 1) or k_multiplier_u19_lut.shape != (count,):
        raise ValueError('A/K LUTs must share a power-of-two address count')
    dt_codes = dt_codes.to(torch.int64)
    if dt_codes.ndim != 2 or torch.any(dt_codes < -count//2) or torch.any(dt_codes >= count//2):
        raise ValueError('dt must be [batch,channel] within the LUT signed code domain')
    addresses = dt_codes + count//2
    return a_bar_q24_lut[addresses], k_multiplier_u19_lut[addresses]


def get_or_build_d_path(m, s_u, s_c, numeric, name_prefix, output_dir):
    if not m.use_D:
        return None
    d_quant = m.d_weight_quant(m.D.float()).detach()
    signature = (tuple(d_quant.cpu().tolist()), float(s_u), float(s_c), str(numeric.to_dict()))
    cache = getattr(m, '_fpga_d_cache', None)
    if cache is None or cache['signature'] != signature:
        table = compile_d_path(d_quant, float(s_u), float(s_c), m.d_state, numeric)
        cache = dict(signature=signature, table=table)
        m._fpga_d_cache = cache
    table = cache['table']
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        q = m.d_weight_quant.lsq_w
        codes = round_half_away_from_zero(m.D.detach() / q.s.detach()).clamp(q.Qn, q.Qp).to(torch.int64)
        np.savetxt(path / f'{name_prefix}_d_q.txt', codes.cpu().numpy(), fmt='%d')
        np.savetxt(path / f'{name_prefix}_d_coefficient.txt', table['coefficient'].numpy(), fmt='%d')
        mask = (1 << numeric.d_coefficient_bits) - 1
        digits = (numeric.d_coefficient_bits + 3) // 4
        (path / f'{name_prefix}_d_coefficient.mem').write_text(
            ''.join(f'{int(v) & mask:0{digits}X}\n' for v in table['coefficient']))
        manifest = dict(table['range_certificate'], name=name_prefix,
            d_weight_bits=q.bit, d_weight_scale=float(q.s.detach().reshape(())),
            s_u=float(s_u), s_c=float(s_c), state_fraction_bits=numeric.state_fraction_bits,
            d_raw=m.D.detach().cpu().tolist(), d_quant=d_quant.cpu().tolist(),
            lsq_max_abs_error=float((m.D.detach()-d_quant).abs().max()),
            rtl_baseline_compatible=False, coefficient_channel_order='SSM channel, independent of state/token')
        (path / f'{name_prefix}_d_manifest.json').write_text(json.dumps(manifest, indent=2))
    return table


def simulate_mamba_core_int8(
    x_int8_in,
    s_in,
    m_module,
    name_prefix,
    output_dir,
    device,
    x_ref_in,
    out_scale_target,
):
    """
    对一个 Mamba core 执行 INT8/Q8.24 FPGA 模拟。

    SpaMamba 输入：
        x_int8_in: [1, H*W, C]

    SpeMamba 输入：
        x_int8_in: [B*H*W, token_num, group_channel_num]

    本函数对 SpeMamba 的每一个像素组分别寻址 dt LUT，不能像旧脚本那样
    固定使用 dt_out_int8[0]，否则光谱分支只有第一个像素的 dt 是正确的。
    """
    m = m_module
    numeric = getattr(m, "_ssm_numeric_config", SSMNumericConfig())
    batch, sequence_length, _ = x_int8_in.shape

    with torch.no_grad():
        ref_in_proj = m.in_proj(x_ref_in)

    in_proj_target_scale = get_activation_scale(
        m.conv1d,
        s_in,
        device,
    )
    xz_int8, s_xz, _ = sim_layer_requant(
        x_int8_in,
        s_in,
        m.in_proj,
        f"{name_prefix}_in_proj",
        output_dir,
        s_out_target=in_proj_target_scale,
        ref_val=ref_in_proj,
    )

    # 当前工程已移除 z 门控，因此整个 in_proj 输出都进入卷积分支。
    x_conv_in = xz_int8.transpose(1, 2).contiguous()
    ref_conv_in = ref_in_proj.transpose(1, 2).contiguous()
    # Mamba 的 depthwise Conv1d 通常是 kernel_size=4、padding=3，
    # 因而卷积原始输出长度为 L+3。相似度检查必须先比较两个同为
    # L+3 的完整输出；随后再将 INT8/FP32 两条路径同时裁回 L。
    ref_conv_full = m.conv1d(ref_conv_in)

    conv_target_scale = get_activation_scale(m.x_proj, s_xz, device)
    x_conv_int8, s_conv, _ = sim_layer_requant(
        x_conv_in,
        s_xz,
        m.conv1d,
        f"{name_prefix}_conv1d",
        output_dir,
        s_out_target=conv_target_scale,
        ref_val=ref_conv_full,
    )
    x_conv_int8 = x_conv_int8[:, :, :sequence_length]
    ref_conv = ref_conv_full[:, :, :sequence_length]
    x_act_int8 = F.relu(x_conv_int8)
    x_act_ref = F.relu(ref_conv)

    # 参考路径直接调用原Mamba的x_proj；其forward hook只量化B/C，
    # 不改变dt片段和SSM-u路径。
    if getattr(m, '_qat_ssm_contract', {}).get('u_quantization') == 'shared':
        flat = x_act_ref.transpose(1, 2).reshape(-1, m.d_inner)
        flat, ref_s_u = m.x_proj.lsq_a(flat)
        x_act_ref = flat.reshape(batch, sequence_length, m.d_inner).transpose(1, 2).contiguous()
        ref_x_proj = m.x_proj(flat, scale_x=ref_s_u, input_is_quantized=True).reshape(batch, sequence_length, -1)
    else:
        ref_x_proj = m.x_proj(x_act_ref.transpose(1, 2).contiguous())
    dt_rank = m.dt_rank
    d_state = m.d_state
    d_inner = m.d_inner
    s_dt_input = get_activation_scale(
        m.dt_proj,
        s_conv,
        device,
    )
    s_b = get_boundary_scale(
        m.b_output_quant,
        s_conv,
        device,
    )
    s_c = get_boundary_scale(
        m.c_output_quant,
        s_conv,
        device,
    )

    # x_proj 的同一个 INT32 累加结果分成两条 requant 路径：
    # dt -> 下一层 dt_proj 输入 INT9 scale；B、C 各自进入显式 INT8 边界。
    x_proj_for_dt, _, _ = sim_layer_requant(
        x_act_int8.transpose(1, 2).contiguous(),
        s_conv,
        m.x_proj,
        f"{name_prefix}_x_proj_dt",
        output_dir,
        s_out_target=s_dt_input,
        ref_val=ref_x_proj,
        output_bits=numeric.dt_input_bits,
    )
    x_proj_for_b, _, _ = sim_layer_requant(
        x_act_int8.transpose(1, 2).contiguous(),
        s_conv,
        m.x_proj,
        f"{name_prefix}_x_proj_B",
        output_dir,
        s_out_target=s_b,
        ref_val=ref_x_proj,
        output_bits=8,
    )
    x_proj_for_c, _, _ = sim_layer_requant(
        x_act_int8.transpose(1, 2).contiguous(),
        s_conv,
        m.x_proj,
        f"{name_prefix}_x_proj_C",
        output_dir,
        s_out_target=s_c,
        ref_val=ref_x_proj,
        output_bits=8,
    )

    # 严格保持当前Mamba源码：dt_proj带bias计算一次，
    # selective_scan(delta_bias=None)不再加第二次bias。
    dt_input_int8 = x_proj_for_dt[:, :, :dt_rank]
    dt_input_ref = ref_x_proj[:, :, :dt_rank]
    b_ref = ref_x_proj[:, :, dt_rank : dt_rank + d_state]
    c_ref = ref_x_proj[:, :, dt_rank + d_state :]

    if m.dt_proj.bias is None:
        raise RuntimeError(
            f"{name_prefix}.dt_proj.bias is None before simulation. "
            "This usually means an older version of this script mutated the "
            "module through copy.copy(). Restart Python and reload the model."
        )

    ref_dt_out = m.dt_proj(dt_input_ref)
    s_dt_target = get_boundary_scale(
        m.dt_output_quant,
        s_dt_input,
        device,
    )
    dt_int8, s_dt, _ = sim_layer_requant(
        dt_input_int8,
        s_dt_input,
        m.dt_proj,
        f"{name_prefix}_dt_proj",
        output_dir,
        s_out_target=s_dt_target,
        ref_val=ref_dt_out,
        output_bits=numeric.dt_output_bits,
    )

    # SSM 的外部接口仍是定点张量，内部状态寄存器采用 Q8.24。
    # U stays signed INT8 in hardware.  Its constant activation scale is folded
    # into K(dt); u_float remains only for the floating diagnostic path.
    u_int8_all = x_act_int8.to(torch.int64)
    u_float = x_act_int8.float() * scalar_scale(s_conv, device)

    b_int8_all = x_proj_for_b[
        :, :, dt_rank : dt_rank + d_state
    ].transpose(1, 2).to(torch.int64)
    c_int8_all = x_proj_for_c[
        :, :, dt_rank + d_state :
    ].transpose(1, 2).to(torch.int64)

    # A_log_shared、dt bias和最终QAT scale均已冻结；在首个tile前离线展开
    # 完整A_bar/B系数ROM。时间循环内只按dt输出INT8码查表。
    (
        a_float,
        a_bar_q24_lut,
        k_multiplier_u19_lut,
        k_fraction_bits,
    ) = get_or_build_ssm_luts(
        m,
        s_dt,
        s_b,
        s_conv,
        name_prefix,
        output_dir,
        device,
    )

    cache = m.__dict__['_fpga_ssm_lut_cache']
    recorder = getattr(m, '_ssm_error_recorder', None)
    d_tables = get_or_build_d_path(m, s_conv, s_c, numeric, name_prefix, output_dir)
    d_components = {} if output_dir and d_tables is not None else None
    y_ssm_q24 = scan_codes(
        u_int8_all, dt_int8, b_int8_all, c_int8_all,
        cache['tables'], numeric, scales=(float(s_b), float(s_c)),
        recorder=recorder, core=name_prefix, d_tables=d_tables, components=d_components)
    y_q24_scale = scalar_scale(s_c, device) * (2. ** -numeric.state_fraction_bits)
    if numeric.error_source == 'all':
        y_ssm_sim = (y_ssm_q24.double() * y_q24_scale.double()).to(u_float.dtype)
    else:
        y_ssm_sim = y_ssm_q24.to(u_float.dtype)
    y_ssm_sim = y_ssm_sim.transpose(1, 2).contiguous()
    capture_dir = getattr(m, '_ssm_capture_dir', None) or output_dir
    if capture_dir and getattr(m, '_capture_ssm_inputs', False):
        replay_dir = Path(capture_dir) / 'ssm_replay_inputs'
        replay_dir.mkdir(parents=True, exist_ok=True)
        d_capture = {} if d_tables is None else dict(d_quant=d_tables['d_quant'].numpy())
        np.savez_compressed(replay_dir / f'{name_prefix}.npz',
            **d_capture,
            u=u_int8_all.cpu().numpy(), dt=dt_int8.cpu().numpy(),
            b=b_int8_all.cpu().numpy(), c=c_int8_all.cpu().numpy(),
            theta=m.A_log_shared.detach().cpu().numpy(),
            s_dt=float(s_dt), s_b=float(s_b), s_u=float(s_conv), s_c=float(s_c),
            numeric_config=json.dumps(numeric.to_dict()), core=name_prefix)

    # 纯浮点分支仅用于误差监控和后续参考值。
    u_ref = x_act_ref.contiguous()
    b_ref = b_ref.transpose(1, 2).contiguous()
    c_ref = c_ref.transpose(1, 2).contiguous()
    y_ssm_ref = selective_scan_fn(
        u_ref,
        ref_dt_out.transpose(1, 2).contiguous(),
        a_float,
        b_ref,
        c_ref,
        D=m.d_weight_quant(m.D.float()) if m.use_D else None,
        z=None,
        delta_bias=None,
        delta_softplus=True,
    )
    y_ssm_ref = y_ssm_ref.transpose(1, 2).contiguous()

    report_fixed_float(
        f"{name_prefix}_ssm",
        y_ssm_ref,
        fixed=y_ssm_sim,
    )

    s_out_proj_input = get_activation_scale(
        m.out_proj_linear,
        (y_ssm_sim.abs().max() / INT8_MAX).clamp_min(1e-6),
        device,
    )
    if d_tables is not None:
        d_m, d_shift = get_m_int_shift(y_q24_scale / s_out_proj_input, q_bits=16)
        d_tables['requant_certificate'] = certify_d_requant(d_tables, d_m.item(), d_shift)
    y_ssm_int8 = ssm_readout_to_int8(
        y_ssm_q24.transpose(1, 2).contiguous(), y_q24_scale, s_out_proj_input, numeric)

    # SSM RTL的第一块精确验证需要两个以往未导出的边界：
    #   1) Conv1d ReLU INT8码转换成Q8.24后的SSM-u；
    #   2) C读出后、out_proj前的Q24/INT8结果。
    # 这些文件只是硬件验证向量，不改变模型计算。
    if output_dir and numeric.error_source == 'all':
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if d_components is not None:
            for key, value in d_components.items():
                np.savetxt(output_path / f'{name_prefix}_ssm_y_{key}_codes.txt',
                           value.transpose(1, 2).cpu().numpy().reshape(-1), fmt='%d')
        y_requant_multiplier, y_requant_shift = get_m_int_shift(
            y_q24_scale / s_out_proj_input,
            q_bits=16,
        )
        np.savetxt(
            output_path / f"{name_prefix}_ssm_u_int8.txt",
            u_int8_all.detach().cpu().numpy().reshape(-1),
            fmt="%d",
        )
        np.savetxt(
            output_path / f"{name_prefix}_ssm_y_q{numeric.state_fraction_bits}.txt",
            y_ssm_q24.transpose(1, 2).contiguous()
            .detach().cpu().numpy().reshape(-1),
            fmt="%d",
        )
        np.savetxt(
            output_path / f"{name_prefix}_ssm_out_int.txt",
            y_ssm_int8.detach().cpu().numpy().reshape(-1),
            fmt="%d",
        )
        np.savetxt(
            output_path / f"{name_prefix}_ssm_out_M_int.txt",
            y_requant_multiplier.detach().cpu().numpy().reshape(-1),
            fmt="%d",
        )
        np.savetxt(
            output_path / f"{name_prefix}_ssm_out_M_shift.txt",
            np.asarray([y_requant_shift], dtype=np.int32),
            fmt="%d",
        )
        with (output_path / f"{name_prefix}_ssm_runtime_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "name": name_prefix,
                    "numeric_config": numeric.to_dict(),
                    "rtl_baseline_compatible": numeric.rtl_baseline_compatible and not m.use_D,
                    "d_path": None if d_tables is None else dict(d_tables['range_certificate'],
                        requant_certificate=d_tables['requant_certificate']),
                    "u_tensor_axis_order": ["batch", "channel", "time"],
                    "y_tensor_axis_order": ["batch", "time", "channel"],
                    "u_scale": float(scalar_scale(s_conv, device).item()),
                    "u_q_format": "signed INT8; scale folded into K(dt)",
                    "k_logical_bits": numeric.k_bits,
                    "k_fraction_bits": int(k_fraction_bits),
                    "kb_signed_bits": numeric.k_bits + 8,
                    "kbu_to_common_left_shift": int(
                        numeric.a_fraction_bits + numeric.state_fraction_bits - k_fraction_bits
                    ),
                    "y_fraction_bits": numeric.state_fraction_bits,
                    "y_physical_scale": float(y_q24_scale.item()),
                    **({
                        "kbu_to_q48_left_shift": int(48 - k_fraction_bits),
                        "y_q24_physical_scale": float(y_q24_scale.item()),
                    } if numeric.a_fraction_bits == 24 and numeric.state_fraction_bits == 24 else {}),
                    "out_proj_input_scale": float(s_out_proj_input.item()),
                    "output_requant_multiplier": int(
                        y_requant_multiplier.item()
                    ),
                    "output_requant_shift": int(y_requant_shift),
                    "rounding": "round-half-away-from-zero",
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        # Do not leave legacy U-Q24 artifacts beside the new INT8-U contract.
        for legacy_name in (
            f"{name_prefix}_ssm_u_q24.txt",
            f"{name_prefix}_ssm_u_q24_lut.txt",
        ):
            legacy_path = output_path / legacy_name
            if legacy_path.exists():
                legacy_path.unlink()

    ref_out_proj = m.out_proj_linear(y_ssm_ref)
    x_out_int8, s_out, _ = sim_layer_requant(
        y_ssm_int8,
        s_out_proj_input,
        m.out_proj_linear,
        f"{name_prefix}_out_proj",
        output_dir,
        s_out_target=out_scale_target,
        ref_val=ref_out_proj,
        output_bits=8,
    )
    return x_out_int8.to(torch.int8), scalar_scale(s_out, device), ref_out_proj


def simulate_both_mamba_block_int8(
    x_int8,
    s_in,
    x_ref,
    block,
    block_idx,
    output_dir,
    device,
):
    """模拟一个 BothMamba：Spa/Spe融合 + skip_scale块级残差。"""
    batch, channels, height, width = x_int8.shape
    prefix = f"blk{block_idx}"
    print("\n" + "=" * 60)
    print(f"Processing {prefix}: SpaMamba + SpeMamba")

    # 两个分支各自使用其 QAT 输入 scale，符合独立硬件支路的接口。
    spa_core = block.spa_mamba.mamba
    spa_input_scale = get_activation_scale(spa_core.in_proj, s_in, device)
    spa_input_4d = requantize_int8(x_int8, s_in, spa_input_scale)
    export_input_requant_contract(
        f"{prefix}_spa_input_requant",
        x_int8,
        s_in,
        spa_input_scale,
        spa_input_4d,
        output_dir,
    )
    spa_int8_in = (
        spa_input_4d.permute(0, 2, 3, 1)
        .contiguous()
        .view(1, -1, channels)
    )
    spa_ref_in = (
        x_ref.permute(0, 2, 3, 1)
        .contiguous()
        .view(1, -1, channels)
    )
    spa_residual_scale = get_boundary_scale(
        block.spa_residual_quant,
        s_in,
        device,
    )
    spa_out, s_spa_out, _ = simulate_mamba_core_int8(
        spa_int8_in,
        spa_input_scale,
        spa_core,
        f"{prefix}_spa",
        output_dir,
        device,
        spa_ref_in,
        out_scale_target=spa_residual_scale,
    )
    spa_out = (
        spa_out.view(batch, height, width, channels)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    if getattr(block.spa_mamba, "use_proj", True):
        spa_out = F.relu(spa_out)

    if block.spa_mamba.use_residual:
        spa_result, s_spa_result = align_and_add_int8(
            x_int8,
            s_in,
            spa_out,
            s_spa_out,
            s_out=spa_residual_scale,
        )
        export_align_add_contract(
            f"{prefix}_spa_residual",
            x_int8,
            s_in,
            spa_out,
            s_spa_out,
            s_spa_result,
            spa_result,
            output_dir,
        )
    else:
        spa_result, s_spa_result = spa_out, s_spa_out

    # QAT forward在分支输出后有独立fake-quant边界；这里使用同一条
    # 浮点/QAT参考路径与硬件反量化值逐层对比。
    spa_ref_result = block.spa_residual_quant(block.spa_mamba(x_ref))
    report_fixed_float(
        f"{prefix}_spa_branch",
        spa_ref_result,
        codes=spa_result,
        scale=s_spa_result,
    )

    spe = block.spe_mamba
    spe_core = spe.mamba
    spe_input_scale = get_activation_scale(spe_core.in_proj, s_in, device)
    spe_input_4d = requantize_int8(x_int8, s_in, spe_input_scale)
    export_input_requant_contract(
        f"{prefix}_spe_input_requant",
        x_int8,
        s_in,
        spe_input_scale,
        spe_input_4d,
        output_dir,
    )

    pad_channels = spe.channel_num - channels
    if pad_channels > 0:
        int8_padding = torch.zeros(
            (batch, pad_channels, height, width),
            dtype=torch.int8,
            device=device,
        )
        ref_padding = torch.zeros(
            (batch, pad_channels, height, width),
            dtype=x_ref.dtype,
            device=device,
        )
        spe_input_4d = torch.cat([spe_input_4d, int8_padding], dim=1)
        spe_ref_4d = torch.cat([x_ref, ref_padding], dim=1)
    else:
        spe_ref_4d = x_ref

    padded_channels = spe_input_4d.shape[1]
    spe_int8_in = (
        spe_input_4d.permute(0, 2, 3, 1)
        .contiguous()
        .view(
            batch * height * width,
            spe.token_num,
            spe.group_channel_num,
        )
    )
    spe_ref_in = (
        spe_ref_4d.permute(0, 2, 3, 1)
        .contiguous()
        .view(
            batch * height * width,
            spe.token_num,
            spe.group_channel_num,
        )
    )
    spe_residual_scale = get_boundary_scale(
        block.spe_residual_quant,
        s_in,
        device,
    )
    spe_out, s_spe_out, _ = simulate_mamba_core_int8(
        spe_int8_in,
        spe_input_scale,
        spe_core,
        f"{prefix}_spe",
        output_dir,
        device,
        spe_ref_in,
        out_scale_target=spe_residual_scale,
    )
    spe_out = (
        spe_out.reshape(batch, height, width, padded_channels)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    spe_out = F.relu(spe_out[:, :channels])

    if spe.use_residual:
        spe_result, s_spe_result = align_and_add_int8(
            x_int8,
            s_in,
            spe_out,
            s_spe_out,
            s_out=spe_residual_scale,
        )
        export_align_add_contract(
            f"{prefix}_spe_residual",
            x_int8,
            s_in,
            spe_out,
            s_spe_out,
            s_spe_result,
            spe_result,
            output_dir,
        )
    else:
        spe_result, s_spe_result = spe_out, s_spe_out

    spe_ref_result = block.spe_residual_quant(block.spe_mamba(x_ref))
    report_fixed_float(
        f"{prefix}_spe_branch",
        spe_ref_result,
        codes=spe_result,
        scale=s_spe_result,
    )

    if block.use_att:
        attention = block.softmax(block.weights).detach()
        spa_weight = scalar_scale(attention[0], device)
        spe_weight = scalar_scale(attention[1], device)
    else:
        spa_weight = torch.tensor(1.0, device=device)
        spe_weight = torch.tensor(1.0, device=device)

    # 权重并入各自物理 scale，INT8 数据本身不需要浮点乘法。
    fusion_scale = get_boundary_scale(
        block.fusion_quant,
        torch.maximum(s_spa_result, s_spe_result),
        device,
    )
    fusion, s_fusion = align_and_add_int8(
        spa_result,
        s_spa_result * spa_weight,
        spe_result,
        s_spe_result * spe_weight,
        s_out=fusion_scale,
    )
    export_align_add_contract(
        f"{prefix}_fusion",
        spa_result,
        s_spa_result * spa_weight,
        spe_result,
        s_spe_result * spe_weight,
        s_fusion,
        fusion,
        output_dir,
    )
    if block.use_att:
        fusion_ref = (
            spa_ref_result * attention[0]
            + spe_ref_result * attention[1]
        )
    else:
        fusion_ref = spa_ref_result + spe_ref_result
    fusion_ref = block.fusion_quant(fusion_ref)
    report_fixed_float(
        f"{prefix}_fusion",
        fusion_ref,
        codes=fusion,
        scale=s_fusion,
    )

    block_output_scale = get_boundary_scale(
        block.block_output_quant,
        s_fusion,
        device,
    )
    if block.use_residual:
        if not hasattr(block, "_fpga_skip_scale"):
            raise RuntimeError(
                f"{prefix}缺少QAT checkpoint中的_fpga_skip_scale；"
                "请重新执行与当前MambaHSI对齐的QAT。"
            )
        skip_scale = scalar_scale(block._fpga_skip_scale, device)
        if not torch.isfinite(skip_scale) or skip_scale.item() <= 0.0:
            raise RuntimeError(
                f"{prefix}.skip_scale必须是有限正数，当前为"
                f"{skip_scale.item()}"
            )
        block_out, s_out = align_and_add_int8(
            x_int8,
            s_in * skip_scale,
            fusion,
            s_fusion,
            s_out=block_output_scale,
        )
        export_align_add_contract(
            f"{prefix}_block_residual",
            x_int8,
            s_in * skip_scale,
            fusion,
            s_fusion,
            s_out,
            block_out,
            output_dir,
        )
    else:
        block_out = requantize_int8(
            fusion,
            s_fusion,
            block_output_scale,
        )
        s_out = block_output_scale

    if block.use_residual:
        block_ref = block.block_output_quant(
            fusion_ref + skip_scale * x_ref
        )
    else:
        block_ref = block.block_output_quant(fusion_ref)
    report_fixed_float(
        f"{prefix}_block_output",
        block_ref,
        codes=block_out,
        scale=s_out,
    )
    if output_dir:
        np.savetxt(
            Path(output_dir) / f"{prefix}_block_output_int.txt",
            block_out.detach().cpu().numpy().reshape(-1),
            fmt="%d",
        )
    return block_out, s_out, block_ref


def simulate_avg_pool_codes(x_codes, s_in, pool_layer, device):
    """保留2x2整数和，以scale/4精确表示QAT AvgPool输出。"""
    kernel = pool_layer.kernel_size
    if isinstance(kernel, int):
        kernel_pair = (kernel, kernel)
    else:
        kernel_pair = tuple(kernel)
    stride = pool_layer.stride
    stride_pair = (stride, stride) if isinstance(stride, int) else tuple(stride)
    padding = pool_layer.padding
    padding_pair = (padding, padding) if isinstance(padding, int) else tuple(padding)
    if (
        kernel_pair != (2, 2)
        or stride_pair != (2, 2)
        or padding_pair != (0, 0)
        or bool(getattr(pool_layer, "ceil_mode", False))
        or not bool(getattr(pool_layer, "count_include_pad", True))
        or getattr(pool_layer, "divisor_override", None) not in (None, 4)
    ):
        raise RuntimeError(
            "当前FPGA合同只支持AvgPool2d(kernel=2,stride=2,padding=0,"
            "ceil_mode=False,divisor=4)。"
        )

    elements = 4
    # x_codes在进入池化前为signed INT8。直接在整数域做2x2求和，避免
    # float AvgPool和再次舍入；signed INT10范围可精确承载[-512, 508]。
    if x_codes.ndim != 4 or x_codes.shape[-2] % 2 or x_codes.shape[-1] % 2:
        raise ValueError(
            f"AvgPool输入必须是偶数高宽的NCHW张量，当前为{tuple(x_codes.shape)}。"
        )
    summed = (
        x_codes.to(torch.int64)
        .unfold(2, 2, 2)
        .unfold(3, 2, 2)
        .sum(dim=(-1, -2))
        .to(torch.int16)
    )
    if int(summed.min().item()) < -512 or int(summed.max().item()) > 511:
        raise OverflowError("AvgPool整数和超出signed INT10范围。")
    return summed, scalar_scale(s_in, device) / elements


def simulate_tile_dual_int8(net, patch_float, input_scale, output_dir, device):
    """完成一个 16x16 tile 的双分支 FPGA 模拟。"""
    if (
        patch_float.ndim != 4
        or patch_float.shape[0] != 1
        or tuple(patch_float.shape[-2:]) != (TILE_SIZE, TILE_SIZE)
    ):
        raise ValueError(
            "FPGA模拟只接受batch=1的NCHW 16x16 tile，"
            f"当前为{tuple(patch_float.shape)}。"
        )
    qat_reference_logits = net(patch_float)

    # ADC/Patch 输入为无符号 8 bit。用 int16 容器避免 128..255 回绕。
    x_int8 = round_half_away_from_zero(
        patch_float / input_scale
    ).clamp(0, 255).to(torch.int16)

    ref_patch_conv = net.patch_embedding[0](patch_float)
    ref_patch = net.patch_embedding[1](ref_patch_conv)
    patch_output_scale = get_boundary_scale(
        net.patch_output_quant,
        input_scale,
        device,
    )
    x_int8, s_current, _ = sim_layer_requant(
        x_int8,
        input_scale,
        net.patch_embedding[0],
        "patch_embed",
        output_dir,
        s_out_target=patch_output_scale,
        ref_val=ref_patch,
        output_bits=8,
    )
    x_int8 = F.relu(x_int8).to(torch.int8)
    x_ref = net.patch_output_quant(F.relu(ref_patch))

    for block_idx in range(3):
        block = net.mamba[block_idx * 2]
        x_int8, s_current, x_ref = simulate_both_mamba_block_int8(
            x_int8,
            s_current,
            x_ref,
            block,
            block_idx,
            output_dir,
            device,
        )

        if block_idx < 2:
            pool_layer = net.mamba[block_idx * 2 + 1]
            x_int8, s_current = simulate_avg_pool_codes(
                x_int8,
                s_current,
                pool_layer,
                device,
            )
            x_ref = pool_layer(x_ref)
            report_fixed_float(
                f"pool{block_idx}",
                x_ref,
                codes=x_int8,
                scale=s_current,
            )

    head_input_scale = get_activation_scale(
        net.cls_head[0],
        s_current,
        device,
    )
    x_int8 = requantize_int8(x_int8, s_current, head_input_scale)

    ref_head1 = net.cls_head[1](net.cls_head[0](x_ref))
    x_head_int8, s_head, _ = sim_layer_requant(
        x_int8,
        head_input_scale,
        net.cls_head[0],
        "head_conv1",
        output_dir,
        ref_val=ref_head1,
        s_out_target=get_activation_scale(
            net.cls_head[3],
            head_input_scale,
            device,
        ),
    )
    x_head_int8 = F.relu(x_head_int8).to(torch.int8)
    ref_head1 = F.relu(ref_head1)

    ref_logits = net.logits_quant(net.cls_head[3](ref_head1))
    logits_scale = get_boundary_scale(
        net.logits_quant,
        s_head,
        device,
    )
    logits_int8, s_logits, _ = sim_layer_requant(
        x_head_int8,
        s_head,
        net.cls_head[3],
        "head_conv2",
        output_dir,
        s_out_target=logits_scale,
        ref_val=ref_logits,
        output_bits=8,
    )
    for path_name, logits in (
        ("QAT direct", qat_reference_logits),
        ("staged FP", ref_logits),
        ("INT8", logits_int8),
    ):
        if tuple(logits.shape[-2:]) != (4, 4):
            raise RuntimeError(
                f"{path_name} logits空间尺寸应为4x4（16x16经两次池化），"
                f"实际为{tuple(logits.shape)}。"
            )
    return (
        qat_reference_logits,
        ref_logits,
        logits_int8.to(torch.int8),
        s_logits,
    )


def load_json_object(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到协议文件：{path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path}必须包含JSON对象。")
    return value


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("QAT checkpoint必须是state_dict或包含state_dict的字典。")
    if checkpoint and all(str(key).startswith("module.") for key in checkpoint):
        checkpoint = {
            str(key)[len("module.") :]: value for key, value in checkpoint.items()
        }
    return checkpoint


def resolve_existing_path(value, description, roots=()):
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path]
    if not path.is_absolute():
        candidates.extend(Path(root) / path for root in roots)
    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.exists():
            return candidate
    joined = "\n".join(str(candidate) for candidate in checked)
    raise FileNotFoundError(f"找不到{description}，已检查：\n{joined}")


def validate_raw_data(raw_data, raw_gt, dataset_name):
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
        raise ValueError(f"标签必须连续为1..C，实际为{foreground.tolist()}。")
    if not np.isfinite(data).all():
        raise ValueError("原始影像包含NaN或Inf。")
    return data.astype(np.float32, copy=False), gt, int(foreground.size)


def transform_with_saved_preprocess(raw_data, preprocess_path):
    """严格复用FP32训练区拟合的PCA/whitening/min-max参数。"""
    preprocess_path = Path(preprocess_path)
    if not preprocess_path.exists():
        raise FileNotFoundError(
            f"找不到FP32预处理参数：{preprocess_path}\n"
            "FPGA模拟禁止重新拟合PCA或对整图重新计算min-max。"
        )
    with np.load(preprocess_path, allow_pickle=False) as saved:
        required = {
            "pca_mean",
            "pca_components",
            "explained_variance",
            "channel_min",
            "channel_max",
        }
        missing = sorted(required.difference(saved.files))
        if missing:
            raise KeyError(f"{preprocess_path}缺少字段：{missing}")
        mean = saved["pca_mean"].astype(np.float64)
        components = saved["pca_components"].astype(np.float64)
        explained_variance = saved["explained_variance"].astype(np.float64)
        channel_min = saved["channel_min"].astype(np.float64)
        channel_max = saved["channel_max"].astype(np.float64)

    height, width, bands = raw_data.shape
    if mean.shape != (bands,) or components.ndim != 2 or components.shape[1] != bands:
        raise ValueError(
            "FP32预处理参数与当前数据波段数不匹配："
            f"data_bands={bands}, mean={mean.shape}, components={components.shape}"
        )
    if (
        explained_variance.shape != (components.shape[0],)
        or channel_min.shape != (components.shape[0],)
        or channel_max.shape != (components.shape[0],)
    ):
        raise ValueError("FP32 PCA/min-max参数的输出通道维度不一致。")

    flat = raw_data.reshape(-1, bands).astype(np.float64, copy=False)
    transformed = (flat - mean) @ components.T
    transformed /= np.sqrt(np.maximum(explained_variance, 1e-12))
    channel_range = np.maximum(channel_max - channel_min, 1e-12)
    normalized = np.clip(
        (transformed - channel_min) / channel_range,
        0.0,
        1.0,
    ).reshape(height, width, components.shape[0]).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("复用FP32预处理后出现NaN或Inf。")
    print(
        "[Exact FP32 preprocess] "
        f"path={preprocess_path}, output={normalized.shape}, "
        f"range=[{normalized.min():.6f}, {normalized.max():.6f}]"
    )
    return normalized


def load_index_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到样本索引：{path}")
    with np.load(path, allow_pickle=False) as saved:
        required = ("train_indices", "val_indices", "test_indices")
        missing = [key for key in required if key not in saved.files]
        if missing:
            raise KeyError(f"{path}缺少字段：{missing}")
        return {
            key: np.asarray(saved[key], dtype=np.int64).reshape(-1)
            for key in required
        }


def load_spatial_masks(fp32_dir, scene_shape):
    mask_path = Path(fp32_dir) / "spatial_split_masks.npz"
    if not mask_path.exists():
        raise FileNotFoundError(f"找不到FP32空间划分：{mask_path}")
    with np.load(mask_path, allow_pickle=False) as saved:
        keys = ("train_region", "val_region", "test_region")
        missing = [key for key in keys if key not in saved.files]
        if missing:
            raise KeyError(f"{mask_path}缺少字段：{missing}")
        masks = {key: np.asarray(saved[key], dtype=bool) for key in keys}
    for key, mask in masks.items():
        if mask.shape != tuple(scene_shape):
            raise ValueError(
                f"{key}形状{mask.shape}与当前GT{tuple(scene_shape)}不一致。"
            )
    total = sum(mask.astype(np.uint8) for mask in masks.values())
    if not np.all(total == 1):
        raise ValueError("FP32 train/validation/test空间mask必须互斥且完整覆盖场景。")
    return masks


def validate_sample_indices(indices, masks, gt, class_count):
    flat_gt = gt.reshape(-1)
    flat_masks = {key: value.reshape(-1) for key, value in masks.items()}
    mask_key = {
        "train_indices": "train_region",
        "val_indices": "val_region",
        "test_indices": "test_region",
    }
    all_parts = []
    for key, values in indices.items():
        if values.size == 0:
            raise ValueError(f"{key}为空。")
        if np.any(values < 0) or np.any(values >= flat_gt.size):
            raise IndexError(f"{key}包含越界像素。")
        if np.unique(values).size != values.size:
            raise ValueError(f"{key}内部存在重复像素。")
        if np.any(flat_gt[values] <= 0) or np.any(flat_gt[values] > class_count):
            raise ValueError(f"{key}必须只包含有效的1..C标签像素。")
        if not np.all(flat_masks[mask_key[key]][values]):
            raise ValueError(f"{key}与保存的{mask_key[key]}不一致。")
        all_parts.append(values)
    concatenated = np.concatenate(all_parts)
    if np.unique(concatenated).size != concatenated.size:
        raise ValueError("train/validation/test样本索引发生重叠。")

    for key, region_key in (("val_indices", "val_region"), ("test_indices", "test_region")):
        expected = np.flatnonzero(flat_masks[region_key] & (flat_gt > 0))
        if not np.array_equal(np.sort(indices[key]), expected):
            raise ValueError(f"{key}没有完整覆盖对应空间区域内的全部标签。")


def validate_dense16_blocks(split_info, masks, scene_shape):
    if split_info.get("strategy") != "blocks":
        raise ValueError(
            f"FPGA 16x16模拟只接受blocks划分，实际为{split_info.get('strategy')!r}。"
        )
    block_size = int(split_info.get("block_size", split_info.get("split_block_size", -1)))
    effective_tile = int(split_info.get("effective_tile_size", split_info.get("tile_size", -1)))
    if block_size != SPLIT_BLOCK_SIZE or effective_tile != TILE_SIZE:
        raise ValueError(
            "FPGA模拟协议必须为tile_size=16且split_block_size=16，"
            f"实际tile={effective_tile}, block={block_size}。"
        )
    records = split_info.get("blocks")
    if not isinstance(records, list) or not records:
        raise ValueError("spatial_split.json缺少可复现的blocks列表。")

    height, width = scene_shape
    coverage = np.zeros(scene_shape, dtype=np.uint8)
    split_to_mask = {
        "train": masks["train_region"],
        "validation": masks["val_region"],
        "test": masks["test_region"],
    }
    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"blocks[{index}]必须是字典。")
        try:
            top = int(record["top"])
            bottom = int(record["bottom"])
            left = int(record["left"])
            right = int(record["right"])
            split_name = str(record["split"])
        except KeyError as error:
            raise KeyError(f"blocks[{index}]缺少字段：{error.args[0]}") from error
        if split_name not in split_to_mask:
            raise ValueError(f"blocks[{index}].split={split_name!r}无效。")
        if not (0 <= top < bottom <= height and 0 <= left < right <= width):
            raise ValueError(f"blocks[{index}]坐标越界。")
        if top % TILE_SIZE or left % TILE_SIZE:
            raise ValueError(f"blocks[{index}]未与16像素网格对齐。")
        if bottom - top > TILE_SIZE or right - left > TILE_SIZE:
            raise ValueError(f"blocks[{index}]大于16x16。")
        region = (slice(top, bottom), slice(left, right))
        if coverage[region].any():
            raise ValueError(f"blocks[{index}]与此前block重叠。")
        if not np.all(split_to_mask[split_name][region]):
            raise ValueError(f"blocks[{index}]与保存的{split_name}空间mask不一致。")
        coverage[region] = 1
        normalized.append(
            {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "split": split_name,
            }
        )
    if not np.all(coverage == 1):
        raise ValueError("保存的blocks没有完整且唯一地覆盖整幅场景。")
    normalized.sort(key=lambda item: (item["top"], item["left"]))
    return normalized


def make_dense16_tile(image, block):
    top, bottom = block["top"], block["bottom"]
    left, right = block["left"], block["right"]
    tile = image[top:bottom, left:right]
    valid_height, valid_width = tile.shape[:2]
    pad_height = TILE_SIZE - valid_height
    pad_width = TILE_SIZE - valid_width
    if pad_height or pad_width:
        pad_mode = "reflect" if min(valid_height, valid_width) > 1 else "edge"
        tile = np.pad(
            tile,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode=pad_mode,
        )
    tile = np.ascontiguousarray(tile.transpose(2, 0, 1))
    return torch.from_numpy(tile).unsqueeze(0), valid_height, valid_width


@torch.no_grad()
def reproduce_saved_qat_test_prediction(
    net,
    image,
    blocks,
    scene_shape,
    batch_size,
    device,
):
    """按QAT测试时的batch方式复算融合BN后的部署预测。"""
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("QAT参考batch_size必须为正整数。")
    test_blocks = [block for block in blocks if block["split"] == "test"]
    if not test_blocks:
        raise ValueError("空间划分中没有test blocks。")

    prediction = np.full(scene_shape, -1, dtype=np.int64)
    visits = np.zeros(scene_shape, dtype=np.uint8)
    net.eval()
    for start in range(0, len(test_blocks), batch_size):
        batch_blocks = test_blocks[start : start + batch_size]
        tiles = []
        shapes = []
        for block in batch_blocks:
            tile, valid_height, valid_width = make_dense16_tile(image, block)
            tiles.append(tile)
            shapes.append((valid_height, valid_width))
        inputs = torch.cat(tiles, dim=0).to(
            device=device,
            dtype=torch.float32,
        )
        logits = net(inputs)
        logits = F.interpolate(
            logits,
            size=(TILE_SIZE, TILE_SIZE),
            mode="bilinear",
            align_corners=True,
        )
        batch_prediction = torch.argmax(logits, dim=1).cpu().numpy()

        for item, (block, valid_shape) in enumerate(
            zip(batch_blocks, shapes)
        ):
            valid_height, valid_width = valid_shape
            top, left = block["top"], block["left"]
            rows = slice(top, top + valid_height)
            cols = slice(left, left + valid_width)
            if visits[rows, cols].any():
                raise AssertionError("QAT批量参考的test tile发生重叠。")
            prediction[rows, cols] = batch_prediction[
                item,
                :valid_height,
                :valid_width,
            ]
            visits[rows, cols] = 1
    return prediction, visits


def verify_saved_qat_test_prediction(
    reproduced,
    expected,
    test_indices,
    batch_size,
):
    """验证checkpoint、预处理与QAT保存预测可以按原batch严格复现。"""
    if reproduced.shape != expected.shape:
        raise ValueError(
            f"复算/保存QAT预测形状不一致：{reproduced.shape} vs {expected.shape}。"
        )
    reproduced_test = reproduced.reshape(-1)[test_indices]
    expected_test = expected.reshape(-1)[test_indices]
    if np.any(reproduced_test < 0) or np.any(expected_test < 0):
        raise ValueError("复算或保存的QAT预测没有覆盖全部测试索引。")
    matches = int(np.count_nonzero(reproduced_test == expected_test))
    total = int(len(test_indices))
    rate = matches / total
    print(
        "[Saved QAT Reproduction] "
        f"batch={batch_size}, match={rate * 100:.6f}% ({matches}/{total})"
    )
    if matches != total:
        raise RuntimeError(
            "已按result.json记录的真实eval_batch_size复算融合BN QAT，"
            "但仍无法复现保存预测。这不是FPGA定点误差；请检查checkpoint、"
            "单文件QAT脚本版本和CUDA确定性。"
        )
    return rate


def resolve_qat_reference_batch_size(qat_result, requested_batch_size):
    """Return the batch that actually generated the saved QAT prediction.

    A command-line batch is allowed only for legacy result files that do not
    record ``eval_batch_size``.  For current runs, overriding that metadata
    would compare two different execution modes and falsely attribute the
    mismatch to checkpoint or INT8 arithmetic.
    """
    requested_batch_size = int(requested_batch_size)
    if "eval_batch_size" in qat_result:
        saved_batch_size = int(qat_result["eval_batch_size"])
        if saved_batch_size <= 0:
            raise ValueError(
                "QAT result.json中的eval_batch_size必须为正整数，"
                f"当前为{saved_batch_size}。"
            )
        if requested_batch_size > 0 and requested_batch_size != saved_batch_size:
            print(
                "[QAT Reference Batch] "
                f"命令请求batch={requested_batch_size}，但保存预测由"
                f"eval_batch_size={saved_batch_size}生成。"
                "强一致性校验必须使用保存值；逐tile batch=1仍由后续FPGA路径"
                "单独执行和比较。"
            )
        return saved_batch_size

    legacy_batch_size = requested_batch_size if requested_batch_size > 0 else 8
    print(
        "[Compatibility] QAT result.json缺少eval_batch_size；"
        f"使用batch={legacy_batch_size}复算保存预测。"
    )
    return legacy_batch_size


@torch.no_grad()
def diagnose_qat_batch_invariance(net, image, blocks, batch_size, device):
    """检查相同tile按批前向与逐tile前向是否具有相同语义。"""
    probe_count = min(max(int(batch_size), 1), 8)
    probe_blocks = [
        block for block in blocks if block["split"] == "test"
    ][:probe_count]
    if not probe_blocks:
        raise ValueError("没有test block可用于QAT batch不变性检查。")
    inputs = torch.cat(
        [make_dense16_tile(image, block)[0] for block in probe_blocks],
        dim=0,
    ).to(device=device, dtype=torch.float32)

    batched_logits = F.interpolate(
        net(inputs),
        size=(TILE_SIZE, TILE_SIZE),
        mode="bilinear",
        align_corners=True,
    )
    single_logits = torch.cat(
        [
            F.interpolate(
                net(inputs[index : index + 1]),
                size=(TILE_SIZE, TILE_SIZE),
                mode="bilinear",
                align_corners=True,
            )
            for index in range(inputs.shape[0])
        ],
        dim=0,
    )
    difference = (batched_logits.double() - single_logits.double()).abs()
    cosine = F.cosine_similarity(
        batched_logits.double().reshape(-1),
        single_logits.double().reshape(-1),
        dim=0,
        eps=1e-12,
    ).item()
    batched_prediction = torch.argmax(batched_logits, dim=1)
    single_prediction = torch.argmax(single_logits, dim=1)
    prediction_match = (
        (batched_prediction == single_prediction).double().mean().item()
    )
    metrics = {
        "probe_tiles": int(inputs.shape[0]),
        "cosine": float(cosine),
        "logit_MAE": float(difference.mean().item()),
        "logit_MaxAE": float(difference.max().item()),
        "prediction_match": float(prediction_match),
    }
    print(
        "[QAT Batch Invariance Probe] "
        f"tiles={metrics['probe_tiles']}, batch/all-vs-single "
        f"cosine={metrics['cosine']:.8f}, "
        f"MAE={metrics['logit_MAE']:.8g}, "
        f"MaxAE={metrics['logit_MaxAE']:.8g}, "
        f"prediction_match={metrics['prediction_match'] * 100:.6f}%"
    )
    if metrics["prediction_match"] < 1.0 or metrics["logit_MaxAE"] > 1e-5:
        print(
            "  [Warning] QAT模型对推理batch分组敏感。若误差明显大于浮点舍入，"
            "需检查SpaMamba是否把batch维拼进序列；FPGA结论应以batch=1参考为准。"
        )
    return metrics


def classification_metrics(labels, predictions, class_count):
    evaluator = Evaluator(num_class=class_count)
    evaluator.add_batch(labels, predictions)
    oa = evaluator.Pixel_Accuracy()
    macc, _ = evaluator.Pixel_Accuracy_Class()
    kappa = evaluator.Kappa()
    return oa, macc, kappa


def detailed_classification_metrics(labels, predictions, class_count):
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = support + predicted - true_positive
    total = int(confusion.sum())
    per_class_acc = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    iou = np.divide(
        true_positive,
        union,
        out=np.zeros_like(true_positive),
        where=union > 0,
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


def agreement_result(prediction_a, prediction_b, indices):
    if len(indices) == 0:
        return 0, 0, float("nan")
    values_a = prediction_a.reshape(-1)[indices]
    values_b = prediction_b.reshape(-1)[indices]
    matches = int((values_a == values_b).sum())
    return matches, len(indices), matches / len(indices)


def print_scope_agreements(
    predict_true,
    predict_ref,
    predict_int8,
    scopes,
):
    pairs = (
        ("QAT-direct / staged-FP", predict_true, predict_ref),
        ("staged-FP / INT8", predict_ref, predict_int8),
        ("QAT-direct / INT8", predict_true, predict_int8),
    )
    print("\n[Prediction Agreement by Pixel Scope]")
    for pair_name, prediction_a, prediction_b in pairs:
        print(f"  {pair_name}")
        for scope_name, indices in scopes.items():
            matches, total, rate = agreement_result(
                prediction_a,
                prediction_b,
                indices,
            )
            print(
                f"    {scope_name:12s}: {rate * 100:7.3f}% "
                f"({matches}/{total})"
            )


def print_mismatch_attribution(
    pair_name,
    prediction_a,
    prediction_b,
    partition_scopes,
):
    flat_a = prediction_a.reshape(-1)
    flat_b = prediction_b.reshape(-1)
    total_mismatches = int((flat_a != flat_b).sum())
    print(f"\n[Mismatch Attribution] {pair_name}")
    print(f"  total mismatches: {total_mismatches}")
    for scope_name, indices in partition_scopes.items():
        mismatch_count = int((flat_a[indices] != flat_b[indices]).sum())
        share = (
            mismatch_count / total_mismatches * 100.0
            if total_mismatches
            else 0.0
        )
        print(
            f"  {scope_name:12s}: {mismatch_count:6d} "
            f"({share:6.2f}% of all mismatches)"
        )


def logits_to_pixels(logits):
    """[1,C,H,W] -> [H*W,C]，保留float硬件反量化值。"""
    return (
        logits.detach()
        .float()
        .squeeze(0)
        .permute(1, 2, 0)
        .reshape(-1, logits.shape[1])
    )


def print_logit_diagnostics(
    name,
    reference_logits,
    candidate_logits,
    reference_prediction,
    candidate_prediction,
    indices,
):
    device = reference_logits.device
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
    ref_pixels = logits_to_pixels(reference_logits).index_select(
        0,
        index_tensor,
    )
    candidate_pixels = logits_to_pixels(candidate_logits).index_select(
        0,
        index_tensor,
    )
    difference = (ref_pixels - candidate_pixels).abs()
    cosine = F.cosine_similarity(
        ref_pixels.reshape(-1),
        candidate_pixels.reshape(-1),
        dim=0,
    ).item()

    top2 = torch.topk(ref_pixels, k=2, dim=1).values
    reference_margin = top2[:, 0] - top2[:, 1]
    agree_mask = torch.as_tensor(
        reference_prediction.reshape(-1)[indices]
        == candidate_prediction.reshape(-1)[indices],
        device=device,
    )

    def margin_summary(mask):
        values = reference_margin[mask]
        if values.numel() == 0:
            return "N/A"
        return (
            f"mean={values.mean().item():.6f}, "
            f"median={values.median().item():.6f}, "
            f"p10={torch.quantile(values, 0.10).item():.6f}"
        )

    print(f"\n[Logit Diagnostics on Test Pixels] {name}")
    print(
        f"  cosine={cosine:.8f}, "
        f"MAE={difference.mean().item():.8f}, "
        f"MaxAE={difference.max().item():.8f}"
    )
    print(
        "  reference top1 margin when predictions agree:    "
        + margin_summary(agree_mask)
    )
    print(
        "  reference top1 margin when predictions disagree: "
        + margin_summary(~agree_mask)
    )


def print_error_exchange(
    labels,
    predict_true_test,
    predict_int8_test,
):
    true_correct = predict_true_test == labels
    int8_correct = predict_int8_test == labels
    same_prediction = predict_true_test == predict_int8_test

    categories = (
        ("both correct", true_correct & int8_correct),
        ("QAT correct / INT8 wrong", true_correct & ~int8_correct),
        ("QAT wrong / INT8 correct", ~true_correct & int8_correct),
        (
            "both wrong / same class",
            ~true_correct & ~int8_correct & same_prediction,
        ),
        (
            "both wrong / different class",
            ~true_correct & ~int8_correct & ~same_prediction,
        ),
    )
    print("\n[Test-set Error Exchange: QAT-direct vs INT8]")
    for name, mask in categories:
        count = int(mask.sum())
        print(
            f"  {name:31s}: {count:6d} "
            f"({count / len(labels) * 100:7.3f}%)"
        )


def print_per_class_diagnostics(
    labels,
    predict_true_test,
    predict_ref_test,
    predict_int8_test,
    class_count,
):
    print("\n[Per-class Test Diagnostics]")
    print(
        "  class support | QAT-acc staged-acc INT8-acc | "
        "QAT/INT8-match staged/INT8-match"
    )
    for class_index in range(class_count):
        mask = labels == class_index
        support = int(mask.sum())
        if support == 0:
            continue
        qat_acc = (predict_true_test[mask] == labels[mask]).mean()
        ref_acc = (predict_ref_test[mask] == labels[mask]).mean()
        int8_acc = (predict_int8_test[mask] == labels[mask]).mean()
        qat_match = (
            predict_true_test[mask] == predict_int8_test[mask]
        ).mean()
        ref_match = (
            predict_ref_test[mask] == predict_int8_test[mask]
        ).mean()
        print(
            f"  {class_index + 1:5d} {support:7d} | "
            f"{qat_acc * 100:7.3f}% {ref_acc * 100:10.3f}% "
            f"{int8_acc * 100:8.3f}% | "
            f"{qat_match * 100:13.3f}% {ref_match * 100:16.3f}%"
        )


def print_tile_coverage_diagnostics(
    coverage_count,
    predict_true,
    predict_ref,
    predict_int8,
    test_idx,
):
    coverage_flat = coverage_count.reshape(-1)
    print("\n[Dense16 Tile Coverage Diagnostics]")
    print(
        "  cover pixels | QAT/INT8 all  staged/INT8 all | "
        "QAT/INT8 test staged/INT8 test"
    )
    for coverage in np.unique(coverage_flat):
        all_indices = np.flatnonzero(coverage_flat == coverage)
        test_indices = test_idx[coverage_flat[test_idx] == coverage]
        _, _, qat_all = agreement_result(
            predict_true,
            predict_int8,
            all_indices,
        )
        _, _, ref_all = agreement_result(
            predict_ref,
            predict_int8,
            all_indices,
        )
        _, _, qat_test = agreement_result(
            predict_true,
            predict_int8,
            test_indices,
        )
        _, _, ref_test = agreement_result(
            predict_ref,
            predict_int8,
            test_indices,
        )
        print(
            f"  {int(coverage):5d} {len(all_indices):6d} | "
            f"{qat_all * 100:12.3f}% {ref_all * 100:15.3f}% | "
            f"{qat_test * 100:13.3f}% {ref_test * 100:16.3f}%"
        )


def evaluate_predictions(
    gt,
    predict_true,
    predict_ref,
    predict_int8,
    final_true,
    final_ref,
    final_int8,
    coverage_count,
    class_count,
    sample_indices,
    expected_qat_prediction,
    reproduced_qat_prediction,
    qat_reference_batch_size,
    qat_result,
):
    """在FP32/QAT保存的同一组像素上输出完整诊断。"""
    gt_flat = gt.reshape(-1)
    train_idx = sample_indices["train_indices"]
    val_idx = sample_indices["val_indices"]
    test_idx = sample_indices["test_indices"]
    labeled_idx = np.flatnonzero(gt_flat > 0)
    background_idx = np.flatnonzero(gt_flat == 0)
    all_idx = np.arange(gt_flat.size, dtype=np.int64)
    labels = gt_flat[test_idx] - 1

    print(
        f"\n[Exact Saved Evaluation Split] seed={qat_result['seed']}, "
        f"train={len(train_idx)}, val={len(val_idx)}, "
        f"test={len(test_idx)}, labeled={len(labeled_idx)}, "
        f"background={len(background_idx)}, all={len(all_idx)}"
    )

    if expected_qat_prediction.shape != gt.shape:
        raise ValueError(
            "QAT保存的部署预测形状与GT不一致："
            f"prediction={expected_qat_prediction.shape}, gt={gt.shape}。"
        )
    expected_test = expected_qat_prediction.reshape(-1)[test_idx]
    if np.any(expected_test < 0):
        raise ValueError("QAT保存的部署预测没有覆盖全部测试索引。")
    reproduced_test = reproduced_qat_prediction.reshape(-1)[test_idx]
    saved_matches = int(np.count_nonzero(reproduced_test == expected_test))
    saved_rate = saved_matches / len(test_idx)
    print(
        "[Protocol Parity] reproduced saved-batch QAT / saved deploy prediction: "
        f"{saved_rate * 100:.6f}% ({saved_matches}/{len(test_idx)})"
    )
    if saved_matches != len(test_idx):
        raise RuntimeError(
            "原QAT batch参考无法复现训练脚本保存的部署预测。"
        )
    direct_test = predict_true.reshape(-1)[test_idx]
    batch_matches = int(np.count_nonzero(direct_test == reproduced_test))
    batch_rate = batch_matches / len(test_idx)
    print(
        "[Batch-mode Sensitivity] FPGA tile batch=1 / saved QAT "
        f"batch={qat_reference_batch_size}: {batch_rate * 100:.6f}% "
        f"({batch_matches}/{len(test_idx)})"
    )
    scopes = {
        "all pixels": all_idx,
        "background": background_idx,
        "all labeled": labeled_idx,
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx,
    }
    print_scope_agreements(
        predict_true,
        predict_ref,
        predict_int8,
        scopes,
    )
    for pair_name, first, second in (
        (
            f"saved-QAT(batch={qat_reference_batch_size}) / QAT-direct(batch=1)",
            reproduced_qat_prediction,
            predict_true,
        ),
        (
            f"saved-QAT(batch={qat_reference_batch_size}) / INT8",
            reproduced_qat_prediction,
            predict_int8,
        ),
    ):
        matches, total, rate = agreement_result(first, second, test_idx)
        print(
            f"  {pair_name}: test={rate * 100:.3f}% ({matches}/{total})"
        )

    partition_scopes = {
        "background": background_idx,
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx,
    }
    print_mismatch_attribution(
        "QAT-direct / INT8",
        predict_true,
        predict_int8,
        partition_scopes,
    )
    print_mismatch_attribution(
        "staged-FP / INT8",
        predict_ref,
        predict_int8,
        partition_scopes,
    )

    predict_saved_test = reproduced_test
    predict_true_test = direct_test
    predict_ref_test = predict_ref.reshape(-1)[test_idx]
    predict_int8_test = predict_int8.reshape(-1)[test_idx]
    saved_qat_metrics = classification_metrics(
        labels,
        predict_saved_test,
        class_count,
    )
    qat_metrics = classification_metrics(
        labels,
        predict_true_test,
        class_count,
    )
    ref_metrics = classification_metrics(
        labels,
        predict_ref_test,
        class_count,
    )
    int8_metrics = classification_metrics(
        labels,
        predict_int8_test,
        class_count,
    )
    detailed = {
        "qat_saved_batch": detailed_classification_metrics(
            labels, predict_saved_test, class_count
        ),
        "qat_direct": detailed_classification_metrics(
            labels, predict_true_test, class_count
        ),
        "staged_fp": detailed_classification_metrics(
            labels, predict_ref_test, class_count
        ),
        "int8_hw": detailed_classification_metrics(
            labels, predict_int8_test, class_count
        ),
    }

    saved_oa = float(qat_result["qat_deploy_test_OA"])
    saved_macc = float(qat_result["qat_deploy_test_mAcc"])
    if (
        abs(detailed["qat_saved_batch"]["OA"] - saved_oa) > 1e-12
        or abs(detailed["qat_saved_batch"]["mAcc"] - saved_macc) > 1e-12
    ):
        raise RuntimeError(
            "复算的saved-batch QAT虽通过逐像素核验，但指标与result.json不一致："
            f"current_OA={detailed['qat_saved_batch']['OA']:.12f}, "
            f"saved_OA={saved_oa:.12f}, "
            f"current_mAcc={detailed['qat_saved_batch']['mAcc']:.12f}, "
            f"saved_mAcc={saved_macc:.12f}。"
        )

    print("\n[Final Quantitative Results on the Same Fixed Test Set]")
    print(
        f"  Saved QAT batch={qat_reference_batch_size}: "
        f"OA={saved_qat_metrics[0]:.4f}, "
        f"mAcc={saved_qat_metrics[1]:.4f}, "
        f"Kappa={saved_qat_metrics[2]:.4f}"
    )
    print(
        f"  QAT direct batch=1: OA={qat_metrics[0]:.4f}, "
        f"mAcc={qat_metrics[1]:.4f}, Kappa={qat_metrics[2]:.4f}"
    )
    print(
        f"  Staged FP:  OA={ref_metrics[0]:.4f}, "
        f"mAcc={ref_metrics[1]:.4f}, Kappa={ref_metrics[2]:.4f}"
    )
    print(
        f"  INT8 HW:    OA={int8_metrics[0]:.4f}, "
        f"mAcc={int8_metrics[1]:.4f}, Kappa={int8_metrics[2]:.4f}"
    )
    print(
        "  Saved-batch reported loss (saved QAT - INT8): "
        f"{(saved_qat_metrics[0] - int8_metrics[0]) * 100:.3f} percentage points"
    )
    print(
        "  Deployment-mode loss (QAT batch=1 - INT8): "
        f"{(qat_metrics[0] - int8_metrics[0]) * 100:.3f} percentage points"
    )
    print(
        "  Reference-path loss (QAT direct - staged FP): "
        f"{(qat_metrics[0] - ref_metrics[0]) * 100:.3f} percentage points"
    )

    print_error_exchange(
        labels,
        predict_true_test,
        predict_int8_test,
    )
    print_per_class_diagnostics(
        labels,
        predict_true_test,
        predict_ref_test,
        predict_int8_test,
        class_count,
    )
    print_logit_diagnostics(
        "QAT-direct / INT8",
        final_true,
        final_int8,
        predict_true,
        predict_int8,
        test_idx,
    )
    print_logit_diagnostics(
        "QAT-direct / staged-FP",
        final_true,
        final_ref,
        predict_true,
        predict_ref,
        test_idx,
    )
    print_logit_diagnostics(
        "staged-FP / INT8",
        final_ref,
        final_int8,
        predict_ref,
        predict_int8,
        test_idx,
    )
    print_tile_coverage_diagnostics(
        coverage_count,
        predict_true,
        predict_ref,
        predict_int8,
        test_idx,
    )
    agreements = {}
    for name, first, second in (
        (
            "qat_saved_batch_vs_qat_direct_batch1",
            reproduced_qat_prediction,
            predict_true,
        ),
        (
            "qat_saved_batch_vs_int8",
            reproduced_qat_prediction,
            predict_int8,
        ),
        ("qat_direct_vs_staged_fp", predict_true, predict_ref),
        ("staged_fp_vs_int8", predict_ref, predict_int8),
        ("qat_direct_vs_int8", predict_true, predict_int8),
    ):
        matches, total, rate = agreement_result(first, second, test_idx)
        agreements[name] = {
            "matches": matches,
            "total": total,
            "rate": rate,
        }
    return {
        "saved_qat_prediction_match": saved_rate,
        "qat_reference_batch_size": int(qat_reference_batch_size),
        "qat_batch1_vs_saved_batch_match": batch_rate,
        "metrics": detailed,
        "test_agreements": agreements,
    }


def save_prediction_images(gt, predict_true, predict_ref, predict_int8, output_dir):
    """保存整图预测和 mask。"""
    try:
        from utils.visual_predict import visualize_predict

        def save_one(prediction, name):
            prediction_path = os.path.join(output_dir, name)
            gt_path = os.path.join(output_dir, "00_Ground_Truth.png")
            visualize_predict(
                gt,
                prediction,
                prediction_path,
                gt_path,
                only_vis_label=False,
            )
            visualize_predict(
                gt,
                prediction,
                prediction_path.replace(".png", "_mask.png"),
                gt_path,
                only_vis_label=True,
            )

        save_one(predict_true, "11_QAT_EndToEnd_Stitched.png")
        save_one(predict_ref, "22_QAT_ManualRef_Stitched.png")
        save_one(predict_int8, "33_INT8_BOTH_Stitched.png")
        print(f"Prediction images saved to: {output_dir}")
    except Exception as error:
        print(f"Image generation failed: {error}")


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"无法JSON序列化{type(value)}")


def validate_qat_run_metadata(qat_result, dataset_name, seed):
    if qat_result.get("dataset") != dataset_name:
        raise ValueError(
            f"QAT result数据集为{qat_result.get('dataset')!r}，"
            f"但命令指定{dataset_name!r}。"
        )
    if int(qat_result.get("seed", -1)) != int(seed):
        raise ValueError(
            f"QAT result seed={qat_result.get('seed')}，但命令指定seed={seed}。"
        )
    if int(qat_result.get("tile_size", -1)) != TILE_SIZE:
        raise ValueError("当前FPGA模拟只接受由16x16 QAT产生的checkpoint。")
    if qat_result.get("split_strategy") != "blocks":
        raise ValueError("当前FPGA模拟只接受blocks空间划分。")
    if int(qat_result.get("split_block_size", -1)) != SPLIT_BLOCK_SIZE:
        raise ValueError("当前FPGA模拟只接受split_block_size=16。")
    if float(qat_result.get("bn_fold_test_prediction_match", 0.0)) != 1.0:
        raise ValueError("QAT运行没有通过BN融合前后100%预测一致性检查。")

    residual_contract = qat_result.get("residual_contract")
    if not isinstance(residual_contract, list) or len(residual_contract) != 3:
        raise ValueError("QAT result缺少三层BothMamba残差合同。")
    invalid = [
        item
        for item in residual_contract
        if not bool(item.get("block_use_residual"))
        or not math.isclose(
            float(item.get("block_skip_scale", float("nan"))),
            EXPECTED_SKIP_SCALE,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or bool(item.get("spa_use_residual"))
        or bool(item.get("spe_use_residual"))
    ]
    if invalid:
        raise ValueError(
            "FPGA模拟要求三层均为fusion+2*x且分支内部无残差，"
            f"不匹配项为{invalid}。"
        )


def infer_fp32_dir(qat_result, qat_run_dir):
    pretrained = qat_result.get("pretrained_fp32_path")
    if not pretrained:
        raise KeyError("QAT result.json缺少pretrained_fp32_path，需显式传--fp32-dir。")
    pretrained_path = resolve_existing_path(
        pretrained,
        "QAT记录的FP32 checkpoint",
        roots=tuple(qat_run_dir.parents),
    )
    expected_run_name = f"run_seed{qat_result['seed']}"
    if (
        pretrained_path.name != "best_model.pth"
        or pretrained_path.parent.name != expected_run_name
    ):
        raise ValueError(
            "pretrained_fp32_path不符合<FP32数据集目录>/run_seedN/best_model.pth结构："
            f"{pretrained_path}"
        )
    return pretrained_path.parent.parent


def summarize_ssm_roms(net):
    """Summarize the six channel-shared Abar/K ROM pairs."""
    ssm_roms = []
    total_packed_bits = 0
    total_aligned_bits = 0
    for block_index in range(3):
        block = net.mamba[block_index * 2]
        for branch_name, branch in (
            ("spa", block.spa_mamba),
            ("spe", block.spe_mamba),
        ):
            cache = branch.mamba.__dict__.get("_fpga_ssm_lut_cache")
            if cache is None:
                raise RuntimeError(
                    f"blk{block_index}_{branch_name}没有生成SSM ROM缓存。"
                )
            a_shape = list(cache["a_bar_q24"].shape)
            k_shape = list(cache["k_multiplier_u19"].shape)
            a_entries = int(cache["a_bar_q24"].numel())
            k_entries = int(cache["k_multiplier_u19"].numel())
            packed_bits = (
                a_entries * (cache["numeric_config"]["a_fraction_bits"] + 1)
                + k_entries * cache["numeric_config"]["k_bits"]
            )
            aligned_bits = (a_entries + k_entries) * 32
            total_packed_bits += packed_bits
            total_aligned_bits += aligned_bits
            ssm_roms.append(
                {
                    "core": f"blk{block_index}_{branch_name}",
                    "Abar_shape": a_shape,
                    "K_shape": k_shape,
                    "coefficient_error": cache["tables"]["coefficient_error"],
                    "K_signed": False,
                    "K_logical_bits": cache["numeric_config"]["k_bits"],
                    "K_fraction_bits": int(cache["k_fraction_bits"]),
                    "U_storage_bits": 8,
                    "packed_bits": packed_bits,
                    "packed_MiB": packed_bits / 8.0 / (1 << 20),
                    "int32_aligned_bits": aligned_bits,
                    "int32_aligned_MiB": aligned_bits / 8.0 / (1 << 20),
                }
            )
    return ssm_roms, total_packed_bits, total_aligned_bits


@_device_safe_qat_scan()
def simulate_mamba_hsi_dense16_both(
    qat_run_dir,
    dataset_name="UP",
    data_path="./data",
    output_dir="./sim_results_both_dense16",
    model_path=None,
    fp32_dir=None,
    hidden_dim=32,
    token_num=4,
    seed=0,
    device_name="auto",
    progress_every=25,
    verbose_every_tile=False,
    qat_reference_batch_size=0,
    numeric_config=None,
    report_ssm_errors=False,
    capture_ssm_inputs=False,
    ssm_analysis_split="test",
    ssm_capture_tiles=1,
    print_input_tiles=False,
):
    """按保存的blocks逐块执行无重叠16x16 QAT/INT8 FPGA模拟。"""
    print(
        f"[Simulator Build] {SIMULATOR_BUILD} | "
        f"model={MambaHSI.__module__} | QAT={prepare_qat_model.__module__}"
    )
    expected_qat_module = "train_mambahsi_spatial_split_dense_qat"
    if (
        MambaHSI.__module__ != expected_qat_module
        or prepare_qat_model.__module__ != expected_qat_module
        or fuse_qat_model_bns_for_deploy.__module__ != expected_qat_module
    ):
        raise RuntimeError(
            "模拟器没有使用生成当前checkpoint的单文件QAT代码。"
            f"期望module={expected_qat_module!r}，实际model="
            f"{MambaHSI.__module__!r}, prepare="
            f"{prepare_qat_model.__module__!r}, fuse="
            f"{fuse_qat_model_bns_for_deploy.__module__!r}。"
        )
    qat_run_dir = resolve_existing_path(qat_run_dir, "QAT run目录")
    if not qat_run_dir.is_dir():
        raise NotADirectoryError(f"QAT run路径不是目录：{qat_run_dir}")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"输出目录非空，拒绝混入旧架构/旧run文件：{output_dir}\n"
            "请为本次模拟指定一个新的--output-dir。"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    qat_result = load_json_object(qat_run_dir / "result.json")
    validate_qat_run_metadata(qat_result, dataset_name, seed)
    model_config = dict(DEFAULT_MODEL_CONFIG)
    model_config.update(qat_result.get('model_config', {}))
    if not d_path_simulator_compatible(model_config):
        raise ValueError('Simulator supports the current S0 topology with optional D only')
    ssm_contract = resolve_ssm_contract(qat_result.get('qat_ssm_contract'))
    trained_numeric = SSMNumericConfig.from_dict(qat_result.get('numeric_config'))
    numeric = SSMNumericConfig.from_dict(numeric_config or trained_numeric)
    if numeric.error_source == 'd-only' and not model_config['use_D']:
        raise ValueError('d-only requires a D1 checkpoint')
    print(f"[QAT contract] use_D={model_config['use_D']}, U={ssm_contract['u_quantization']}, "
          f"dt_input_bits={numeric.dt_input_bits}, dt_output_bits={numeric.dt_output_bits}, "
          f"D_weight_bits={ssm_contract['d_weight_bits']}")
    for field in ('dt_input_bits', 'dt_output_bits'):
        if getattr(numeric, field) != getattr(trained_numeric, field):
            raise ValueError(f'{field} differs from checkpoint; train a paired QAT variant first')
    (output_dir / 'numeric_config.json').write_text(json.dumps(numeric.to_dict(), indent=2))
    if ssm_analysis_split not in ('train', 'validation', 'test', 'all') or ssm_capture_tiles < 1:
        raise ValueError('Invalid SSM analysis split or capture tile limit')
    recorder = ErrorRecorder() if report_ssm_errors else None
    captured_tiles = []
    if model_path is None:
        model_path = qat_run_dir / "best_qat_foldaware.pth"
    model_path = resolve_existing_path(
        model_path,
        "QAT best_qat_foldaware.pth",
        roots=(qat_run_dir,),
    )
    if fp32_dir is None:
        fp32_dir = infer_fp32_dir(qat_result, qat_run_dir)
    else:
        fp32_dir = resolve_existing_path(fp32_dir, "FP32数据集结果目录")
    if not fp32_dir.is_dir():
        raise NotADirectoryError(f"FP32结果路径不是目录：{fp32_dir}")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA，但当前PyTorch检测不到可用CUDA设备。")
    scan_backend = (
        "mamba_ssm_optimized"
        if device.type == "cuda" and _qat_source._optimized_selective_scan_fn is not None
        else "torch_reference"
    )
    print(f"[Selective scan] device={device}, backend={scan_backend}", flush=True)

    print(f"Loading {dataset_name} from {data_path}")
    raw_data, raw_gt, _ = load_dataset(dataset_name, data_path)
    raw_data, gt, class_count = validate_raw_data(raw_data, raw_gt, dataset_name)
    image = transform_with_saved_preprocess(
        raw_data,
        fp32_dir / "train_only_preprocess.npz",
    )
    pca_channels = int(image.shape[2])

    masks = load_spatial_masks(fp32_dir, gt.shape)
    split_info = load_json_object(fp32_dir / "spatial_split.json")
    if split_info.get("dataset") not in (None, dataset_name):
        raise ValueError(
            f"spatial_split.json数据集为{split_info.get('dataset')!r}，"
            f"当前为{dataset_name!r}。"
        )
    blocks = validate_dense16_blocks(split_info, masks, gt.shape)

    fp32_indices = load_index_file(
        fp32_dir / f"run_seed{seed}" / "sample_indices.npz"
    )
    qat_indices = load_index_file(qat_run_dir / "sample_indices.npz")
    for key in fp32_indices:
        if not np.array_equal(fp32_indices[key], qat_indices[key]):
            raise ValueError(f"QAT与FP32的{key}不一致。")
    validate_sample_indices(qat_indices, masks, gt, class_count)
    for count_key, index_key in (
        ("train_label_count", "train_indices"),
        ("val_label_count", "val_indices"),
        ("test_label_count", "test_indices"),
    ):
        if int(qat_result.get(count_key, -1)) != len(qat_indices[index_key]):
            raise ValueError(f"QAT result中的{count_key}与保存索引不一致。")

    expected_prediction_path = qat_run_dir / "qat_deploy_test_prediction.npy"
    if not expected_prediction_path.exists():
        raise FileNotFoundError(
            f"找不到QAT部署预测基准：{expected_prediction_path}"
        )
    expected_qat_prediction = np.load(
        expected_prediction_path,
        allow_pickle=False,
    )

    print(
        "[Protocol] "
        f"scene={gt.shape}, PCA={pca_channels}, tile=16, block=16, "
        f"blocks={len(blocks)}, seed={seed}, device={device}"
    )
    print(f"[Artifacts] QAT={qat_run_dir}")
    print(f"[Artifacts] FP32={fp32_dir}")
    print(f"[Artifacts] checkpoint={model_path}")

    if hidden_dim != 32 or token_num != 4:
        raise ValueError(
            "当前FPGA/QAT合同固定为hidden_dim=32、token_num=4；"
            f"当前收到hidden_dim={hidden_dim}、token_num={token_num}。"
        )
    net = build_configured_model(pca_channels, class_count, model_config)
    prepare_qat_model(net, nbit=8, numeric_config=numeric, model_config=model_config,
                      ssm_contract=ssm_contract)
    for module in net.modules():
        if hasattr(module, 'A_log_shared'):
            module._ssm_error_recorder = recorder
            module._capture_ssm_inputs = capture_ssm_inputs
    state_dict = extract_state_dict(torch.load(model_path, map_location="cpu"))
    try:
        net.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Checkpoint与PCA={pca_channels}/hidden_dim={hidden_dim}/"
            f"token_num={token_num}/current BothMamba不匹配。"
        ) from error

    first_tile_cpu, _, _ = make_dense16_tile(image, blocks[0])
    fusion_probe = first_tile_cpu.to(device=device, dtype=torch.float32)
    net.to(device).eval()
    validate_fpga_qat_contract(net, expect_bn=True)
    freeze_lsq_initialization(net)
    fuse_qat_model_bns_for_deploy(net, validation_input=fusion_probe)
    validate_fpga_qat_contract(net, expect_bn=False)

    qat_reference_batch_size = resolve_qat_reference_batch_size(
        qat_result,
        qat_reference_batch_size,
    )

    reproduced_qat_prediction, reproduced_qat_visits = (
        reproduce_saved_qat_test_prediction(
            net,
            image,
            blocks,
            gt.shape,
            qat_reference_batch_size,
            device,
        )
    )
    if not np.all(reproduced_qat_visits[masks["test_region"]] == 1):
        raise AssertionError("批量QAT参考没有完整覆盖test空间。")
    if np.any(reproduced_qat_visits[masks["train_region"]]) or np.any(
        reproduced_qat_visits[masks["val_region"]]
    ):
        raise AssertionError("批量QAT参考越过test空间边界。")
    verify_saved_qat_test_prediction(
        reproduced_qat_prediction,
        expected_qat_prediction,
        qat_indices["test_indices"],
        qat_reference_batch_size,
    )
    qat_batch_invariance_probe = diagnose_qat_batch_invariance(
        net,
        image,
        blocks,
        qat_reference_batch_size,
        device,
    )

    input_scale = get_activation_scale(
        net.patch_embedding[0],
        torch.tensor(1.0 / 255.0),
        device,
    )
    print(
        f"ADC input: UINT8 [0, 255], scale={input_scale.item():.8f}, "
        f"scene_float_range=[{image.min():.6f}, {image.max():.6f}]"
    )

    export_pcie_input_tiles(image, blocks, input_scale, output_dir, device,
                            print_all=print_input_tiles)

    full_h, full_w = gt.shape
    true_canvas = torch.zeros((1, class_count, full_h, full_w), device=device)
    ref_canvas = torch.zeros_like(true_canvas)
    int8_canvas = torch.zeros_like(true_canvas)
    coverage_count = torch.zeros(
        (full_h, full_w), dtype=torch.uint8, device=device
    )
    first_tile_dir = output_dir / "first_tile_layers"
    first_tile_dir.mkdir(parents=True, exist_ok=True)

    total_tiles = len(blocks)
    with torch.no_grad():
        for tile_index, block in enumerate(blocks, start=1):
            selected = ssm_analysis_split == 'all' or block['split'] == ssm_analysis_split
            capture = capture_ssm_inputs and selected and len(captured_tiles) < ssm_capture_tiles
            capture_dir = output_dir / 'replay_tiles' / f'tile_{tile_index:06d}'
            for module in net.modules():
                if hasattr(module, 'A_log_shared'):
                    module._ssm_error_recorder = recorder if selected else None
                    module._capture_ssm_inputs = capture
                    module._ssm_capture_dir = str(capture_dir) if capture else None
            if capture:
                captured_tiles.append(dict(tile_index=tile_index, block=block, directory=str(capture_dir)))
            tile_cpu, valid_height, valid_width = make_dense16_tile(image, block)
            tile = tile_cpu.to(device=device, dtype=torch.float32)
            if tile_index == 1:
                first_tile_uint8 = round_half_away_from_zero(
                    tile / input_scale
                ).clamp(0, 255).to(torch.int16)
                export_first_tile_coe(first_tile_uint8, str(first_tile_dir))
                with (output_dir / "first_tile.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(block, handle, ensure_ascii=False, indent=2)

            layer_output_dir = str(first_tile_dir) if tile_index == 1 else None
            if tile_index == 1 or verbose_every_tile:
                simulated = simulate_tile_dual_int8(
                    net, tile, input_scale, layer_output_dir, device
                )
            else:
                # 本地整数核会逐层打印；正式整图有数百个tile，只保留首tile明细。
                with contextlib.redirect_stdout(io.StringIO()):
                    simulated = simulate_tile_dual_int8(
                        net, tile, input_scale, layer_output_dir, device
                    )
            true_logits, ref_logits, logits_int8, logits_scale = simulated

            true_up = F.interpolate(
                true_logits,
                size=(TILE_SIZE, TILE_SIZE),
                mode="bilinear",
                align_corners=True,
            )
            ref_up = F.interpolate(
                ref_logits,
                size=(TILE_SIZE, TILE_SIZE),
                mode="bilinear",
                align_corners=True,
            )
            int8_up = F.interpolate(
                logits_int8.float() * scalar_scale(logits_scale, device),
                size=(TILE_SIZE, TILE_SIZE),
                mode="bilinear",
                align_corners=True,
            )

            top, left = block["top"], block["left"]
            rows = slice(top, top + valid_height)
            cols = slice(left, left + valid_width)
            if coverage_count[rows, cols].any():
                raise AssertionError(f"tile{tile_index}输出与此前tile重叠。")
            true_canvas[:, :, rows, cols] = true_up[:, :, :valid_height, :valid_width]
            ref_canvas[:, :, rows, cols] = ref_up[:, :, :valid_height, :valid_width]
            int8_canvas[:, :, rows, cols] = int8_up[:, :, :valid_height, :valid_width]
            coverage_count[rows, cols] = 1

            if (
                tile_index == 1
                or tile_index == total_tiles
                or tile_index % max(int(progress_every), 1) == 0
            ):
                print(
                    f"Tile {tile_index}/{total_tiles} done "
                    f"split={block['split']} top={top} left={left} "
                    f"valid={valid_height}x{valid_width}"
                )

    if not torch.all(coverage_count == 1).item():
        missing = int(torch.count_nonzero(coverage_count != 1).item())
        raise AssertionError(f"16x16 tile拼接后仍有{missing}个像素未被唯一覆盖。")

    predict_true = torch.argmax(true_canvas, dim=1).squeeze(0).cpu().numpy()
    predict_ref = torch.argmax(ref_canvas, dim=1).squeeze(0).cpu().numpy()
    predict_int8 = torch.argmax(int8_canvas, dim=1).squeeze(0).cpu().numpy()
    # 先保存原始结果，再执行严格协议核验；若核验失败，诊断产物仍可保留。
    np.save(output_dir / "qat_direct_prediction.npy", predict_true)
    np.save(
        output_dir / "qat_saved_batch_reproduced_prediction.npy",
        reproduced_qat_prediction,
    )
    np.save(output_dir / "staged_fp_prediction.npy", predict_ref)
    np.save(output_dir / "int8_hw_prediction.npy", predict_int8)
    np.save(output_dir / "tile_coverage_count.npy", coverage_count.cpu().numpy())
    evaluation = evaluate_predictions(
        gt,
        predict_true,
        predict_ref,
        predict_int8,
        true_canvas,
        ref_canvas,
        int8_canvas,
        coverage_count.cpu().numpy(),
        class_count,
        qat_indices,
        expected_qat_prediction,
        reproduced_qat_prediction,
        qat_reference_batch_size,
        qat_result,
    )

    save_prediction_images(
        gt,
        predict_true,
        predict_ref,
        predict_int8,
        str(output_dir),
    )

    ssm_roms, total_rom_bits, total_aligned_rom_bits = summarize_ssm_roms(net)
    print(
        f"[SSM ROM Total] packed25/19={total_rom_bits / 8.0 / (1 << 20):.3f}MiB, "
        f"int32-aligned={total_aligned_rom_bits / 8.0 / (1 << 20):.3f}MiB, "
        f"cores={len(ssm_roms)}"
    )

    if capture_ssm_inputs:
        (output_dir / 'replay_inputs_manifest.json').write_text(json.dumps(dict(
            dataset=dataset_name, seed=seed, analysis_split=ssm_analysis_split,
            checkpoint=str(model_path), numeric_config=numeric.to_dict(), tiles=captured_tiles,
            model_config=model_config, qat_ssm_contract=ssm_contract,
            scope='selected actual integer-path inputs; local paired replay, not full-test OA'), indent=2))
    if recorder is not None:
        recorder.write(output_dir / 'ssm_local_error_by_position.csv')
    result = {
        "numeric_config": numeric.to_dict(),
        "selective_scan_backend": scan_backend,
        "rtl_baseline_compatible": numeric.rtl_baseline_compatible and not model_config['use_D'],
        "model_config": model_config,
        "qat_ssm_contract": ssm_contract,
        "d_path_rom_bits": sum(m._fpga_d_cache['table']['range_certificate']['coefficient_rom_bits']
                               for m in net.modules() if hasattr(m, '_fpga_d_cache')),
        "d_path_manifests": {name: dict(m._fpga_d_cache['table']['range_certificate'],
            requant_certificate=m._fpga_d_cache['table'].get('requant_certificate'))
            for name, m in net.named_modules() if hasattr(m, '_fpga_d_cache')},
        "ssm_analysis_split": ssm_analysis_split,
        "ssm_execution_mode": ("exact-integer" if numeric.resolved_readout_requantization == 'hardware'
                               else "integer-ssm-ideal-readout" if numeric.error_source == 'all'
                               else "floating-counterfactual"),
        "ssm_readout_requantization": numeric.resolved_readout_requantization,
        "ssm_local_error_reference": "same quantized U/dt/B/C; full precision A/K; D reference is frozen LSQ D, d-only isolates coefficient folding",
        "dataset": dataset_name,
        "seed": int(seed),
        "protocol": (
            "exact saved 5:2:3-target raw-pixel-disjoint blocks; "
            "exact FP32 train-only PCA/minmax; non-overlapping dense 16x16 tiles; "
            "current fold-aware QAT; per-tensor fixed-point FPGA simulation"
        ),
        "simulation_scope": (
            "integer Conv/Linear/bias/requant/fusion/residual/pooling with Q8.24 "
            "SSM state; Abar and K=delta*s_B*s_u use offline-generated "
            "dt-code ROMs while dynamic signed INT8 B and U are multiplied at runtime; "
            "dt_proj applies its bias once and selective_scan uses delta_bias=None; "
            "floating selective_scan is retained only as a diagnostic reference"
        ),
        "ssm_lut_contract": (
            "dt_proj input INT9; its INT8 output already contains the single "
            "dt_proj bias; selective_scan delta_bias=None; 256 addresses; "
            "per-core channel-shared Abar[address,state] UQ1.24 and unsigned "
            "19-bit K[address], with per-core K fractional bits up to Q24; "
            "U remains signed INT8 and has no lookup table"
        ),
        "integer_mac_backend": (
            "FP64 exact-integer carrier checked on the integer grid, signed INT32 "
            "partial-sum/final-accumulator contract"
        ),
        "ssm_readout_accumulator": "signed INT64 before requantization to INT8",
        "pool_contract": (
            "signed INT10 2x2 sum with physical scale=pool_input_tensor_scale/4"
        ),
        "ssm_roms": ssm_roms,
        "ssm_rom_total_packed_bits": total_rom_bits,
        "ssm_rom_total_packed_MiB": total_rom_bits / 8.0 / (1 << 20),
        "ssm_rom_total_int32_aligned_bits": total_aligned_rom_bits,
        "ssm_rom_total_int32_aligned_MiB": (
            total_aligned_rom_bits / 8.0 / (1 << 20)
        ),
        "tile_size": TILE_SIZE,
        "split_block_size": SPLIT_BLOCK_SIZE,
        "scene_shape": list(gt.shape),
        "pca_channels": pca_channels,
        "class_count": class_count,
        "tile_count": total_tiles,
        "qat_run_dir": str(qat_run_dir),
        "fp32_dir": str(fp32_dir),
        "qat_checkpoint": str(model_path),
        "qat_checkpoint_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "qat_batch_invariance_probe": qat_batch_invariance_probe,
        "qat_best_epoch": int(qat_result["best_epoch"]),
        "input_scale": float(input_scale.item()),
        **evaluation,
    }
    if model_config['use_D']:
        result['simulation_scope'] = 'Integer SSM with C+D readout before one output requantization; D RTL not implemented.'
        result['ssm_lut_contract'] = 'A/K variant schema 5 plus independent per-channel D manifests; see numeric_config.'
    if not numeric.rtl_baseline_compatible:
        result['simulation_scope'] = 'Integer outer graph with configurable SSM; see numeric_config and ssm_execution_mode. Not validated against baseline RTL.'
        result['ssm_lut_contract'] = 'Variant schema 5; address and coefficient formats are in per-core manifests.'
    with (output_dir / "fpga_simulation_result.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=json_default)
    print(f"\n16x16 BothMamba FPGA simulation finished: {output_dir}")
    return result




def parse_args():
    parser = argparse.ArgumentParser(
        description="严格对齐当前QAT的16x16 BothMamba FPGA INT8模拟"
    )
    parser.add_argument(
        "--qat-run-dir",
        required=True,
        help="QAT的run_seedN目录，内含result.json、sample_indices和部署预测",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="可选；默认使用<qat-run-dir>/best_qat_foldaware.pth",
    )
    parser.add_argument(
        "--fp32-dir",
        default=None,
        help="可选；默认由QAT result.json中的pretrained_fp32_path推断",
    )
    parser.add_argument("--dataset", default="UP")
    parser.add_argument("--data-path", default="./data")
    parser.add_argument(
        "--output-dir",
        default="./sim_results_both_dense16",
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--token-num", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--print-input-tiles", action="store_true",
        help="终端打印所有量化UINT8 tile（含padding）；默认仍完整导出pcie_input下的bin/mem/txt",
    )
    parser.add_argument(
        "--verbose-every-tile",
        action="store_true",
        help="打印每个tile的逐层定点诊断；默认只打印并导出第一个tile",
    )
    parser.add_argument(
        "--qat-reference-batch-size",
        type=int,
        default=0,
        help=(
            "仅用于缺少eval_batch_size的旧QAT run；新run始终读取"
            "result.json中的真实保存batch，避免用错误batch核对保存预测。"
            "旧run传0时回退为历史默认值8"
        ),
    )
    add_numeric_arguments(parser, boundary_defaults=False)
    parser.add_argument('--report-ssm-errors', action='store_true',
                        help='Aggregate same-input local state/readout error by core and sequence position')
    parser.add_argument('--capture-ssm-inputs', action='store_true',
                        help='Capture selected split tiles per core for paired offline error replay')
    parser.add_argument('--ssm-analysis-split', choices=['train','validation','test','all'], default='test')
    parser.add_argument('--ssm-capture-tiles', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    saved = load_json_object(Path(args.qat_run_dir) / 'result.json').get('numeric_config')
    numeric = config_from_args(args, saved)
    simulate_mamba_hsi_dense16_both(
        qat_run_dir=args.qat_run_dir,
        dataset_name=args.dataset,
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_path=args.model,
        fp32_dir=args.fp32_dir,
        hidden_dim=args.hidden_dim,
        token_num=args.token_num,
        seed=args.seed,
        device_name=args.device,
        progress_every=args.progress_every,
        verbose_every_tile=args.verbose_every_tile,
        qat_reference_batch_size=args.qat_reference_batch_size,
        numeric_config=numeric,
        report_ssm_errors=args.report_ssm_errors,
        capture_ssm_inputs=args.capture_ssm_inputs,
        ssm_analysis_split=args.ssm_analysis_split,
        ssm_capture_tiles=args.ssm_capture_tiles,
        print_input_tiles=args.print_input_tiles,
    )


if __name__ == '__main__':
    main()
