#!/usr/bin/env python3
# Generated v5 nonlinear accuracy propagation experiment. Original v5 source is unchanged.
"""
与当前 16x16 严格空间划分 QAT 完全对齐的 BothMamba FPGA INT8 模拟。

放置位置：
    与 both_QAT_21patch_aligned.py、model/、utils/ 同一目录。

依赖：
    本文件内置 Conv/Linear 的整数 MAC、INT32 bias 和重量化模拟；
    model/MambaHSI.py 提供 MambaHSI、BothMamba、SpaMamba、SpeMamba。

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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model/QAT/CUDA dependencies are imported only when starting inference.
# --help and --self-test work without the server's model/ tree.
NL = None
NL_BASE_SHA256 = 'ba139c67a74061e6439df4b6d495281aa7ae553bd13a315921e62da19637d74d'


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
    if NL is not None:
        NL.boundary(name, codes, scale, fixed)
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


def freeze_lsq_initialization(net):
    """防止仿真时 LSQ scale 被输入 patch 再次初始化。"""
    for module in net.modules():
        if hasattr(module, "init_state"):
            module.init_state = 999999
            module.g = 1.0


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


def _n3_get_or_build_ssm_luts(m, s_dt, s_b, s_u, name_prefix, output_dir, device):
    """
    离线生成并缓存当前core共享的B系数LUT和A_bar UQ1.24 ROM。

    dt INT8码已是带一次dt_proj.bias的dt_proj输出；当前Mamba源码在
    selective_scan中传delta_bias=None，因此这里只计算
    softplus(dt_code * s_dt)，绝不再加bias。per-tensor s_dt/s_B与共享
    A_log_shared使全部d_inner通道共用同一份256地址ROM。
    """
    s_dt = scalar_scale(s_dt, device).abs().clamp_min(1e-12)
    s_b = scalar_scale(s_b, device).abs().clamp_min(1e-12)
    s_u = scalar_scale(s_u, device).abs().clamp_min(1e-12)
    a_log_shared = m.A_log_shared.detach().to(device=device, dtype=torch.float32)
    if a_log_shared.numel() != int(m.d_state):
        raise RuntimeError(
            f"{name_prefix}: A_log_shared={a_log_shared.numel()}与"
            f"d_state={m.d_state}不一致。"
        )

    s_dt_value = float(s_dt.item())
    s_b_value = float(s_b.item())
    s_u_value = float(s_u.item())
    signature_digest = hashlib.sha256()
    signature_digest.update(
        json.dumps(
            {
                "schema_version": SSM_LUT_SCHEMA_VERSION,
                "dt_code_min": INT8_MIN,
                "dt_code_max": INT8_MAX,
                "rounding": SSM_LUT_ROUNDING,
                "delta_bias_mode": "none",
                "dt_code_already_includes_dt_proj_bias": True,
                "lut_shared_across_d_inner": True,
                "input_term": "K(dt)*B_int8*U_int8",
                "k_logical_bits": SSM_K_LOGICAL_BITS,
                "k_preferred_fraction_bits": SSM_K_PREFERRED_FRACTION_BITS,
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    signature_digest.update(
        np.asarray([s_dt_value, s_b_value, s_u_value], dtype="<f8").tobytes()
    )
    signature_digest.update(
        a_log_shared.detach().cpu().contiguous().numpy().astype("<f4").tobytes()
    )
    content_signature = signature_digest.hexdigest()
    signature = (str(device), content_signature)
    cache = m.__dict__.get("_fpga_ssm_lut_cache")
    if cache is None or cache.get("signature") != signature:
        with torch.no_grad():
            # ROM在CPU float64中确定性生成；完成后仅把整数表搬到模拟设备。
            dt_codes = torch.arange(
                INT8_MIN,
                INT8_MAX + 1,
                dtype=torch.float64,
            )
            a_log_rom = a_log_shared.detach().cpu().double()
            # 只在ROM生成阶段计算非线性。部署运行时仅使用dt整数码寻址。
            # dt_codes已经由带bias的dt_proj输出量化而来，不再加bias。
            possible_delta = F.softplus(dt_codes * s_dt_value)
            a_shared_rom = -torch.exp(a_log_rom)
            a_bar_float = torch.exp(
                possible_delta.unsqueeze(-1) * a_shared_rom.view(1, -1)
            )
            a_bar_q24 = round_half_away_from_zero(
                a_bar_float * float(1 << 24)
            )
            k_multiplier_float = possible_delta * s_b_value * s_u_value
            k_fraction_bits = choose_unsigned_fraction_bits(
                k_multiplier_float,
                total_bits=SSM_K_LOGICAL_BITS,
                preferred_bits=SSM_K_PREFERRED_FRACTION_BITS,
            )
            k_multiplier_u19 = round_half_away_from_zero(
                k_multiplier_float * float(2.0**k_fraction_bits)
            )

        if torch.any(a_bar_q24 < 0) or torch.any(a_bar_q24 > (1 << 24)):
            raise OverflowError(f"{name_prefix}: A_bar ROM超出unsigned UQ1.24。")
        if torch.any(k_multiplier_u19 < 0) or torch.any(
            k_multiplier_u19 > ((1 << SSM_K_LOGICAL_BITS) - 1)
        ):
            raise OverflowError(
                f"{name_prefix}: folded K ROM exceeds unsigned "
                f"{SSM_K_LOGICAL_BITS}-bit."
            )
        cache = {
            "signature": signature,
            "content_signature_sha256": content_signature,
            "a_bar_q24": a_bar_q24.to(device=device, dtype=torch.int64),
            "k_multiplier_u19": k_multiplier_u19.to(
                device=device, dtype=torch.int64
            ),
            "k_fraction_bits": int(k_fraction_bits),
            "a_float": (-torch.exp(a_log_shared)).unsqueeze(0)
            .expand(int(m.d_inner), -1)
            .contiguous(),
            "s_dt": s_dt_value,
            "s_b": s_b_value,
            "s_u": s_u_value,
            "exported_dirs": set(),
        }
        m.__dict__["_fpga_ssm_lut_cache"] = cache
        a_entries = int(cache["a_bar_q24"].numel())
        k_entries = int(cache["k_multiplier_u19"].numel())
        packed_mib = (
            a_entries * SSM_A_LOGICAL_BITS
            + k_entries * SSM_K_LOGICAL_BITS
        ) / 8.0 / (1 << 20)
        aligned_mib = (a_entries + k_entries) * 4.0 / (1 << 20)
        print(
            f"[SSM ROM] {name_prefix}: Abar={tuple(a_bar_q24.shape)} UQ1.24, "
            f"K={tuple(k_multiplier_u19.shape)} U{SSM_K_LOGICAL_BITS}/Q{k_fraction_bits}, "
            "shared_across_channels=True, "
            f"packed={packed_mib:.3f}MiB, int32-aligned={aligned_mib:.3f}MiB"
        )

    if output_dir:
        export_key = (
            str((Path(output_dir) / "ssm_luts").resolve()),
            str(name_prefix),
        )
        if export_key not in cache["exported_dirs"]:
            _export_ssm_luts(output_dir, name_prefix, cache)
            cache["exported_dirs"].add(export_key)
    return (
        cache["a_float"],
        cache["a_bar_q24"],
        cache["k_multiplier_u19"],
        int(cache["k_fraction_bits"]),
    )


def lookup_ssm_luts(a_bar_q24_lut, k_multiplier_u19_lut, dt_codes):
    """Look up channel-shared Abar and folded K using INT8 dt codes."""
    dt_codes = dt_codes.to(torch.int64)
    if torch.any(dt_codes < INT8_MIN) or torch.any(dt_codes > INT8_MAX):
        raise OverflowError("SSM LUT地址输入超出signed INT8范围。")
    if dt_codes.ndim != 2:
        raise ValueError(f"dt_codes应为[B,d_inner]，实际为{tuple(dt_codes.shape)}。")
    if a_bar_q24_lut.ndim != 2 or a_bar_q24_lut.shape[0] != SSM_DT_ADDRESS_COUNT:
        raise ValueError(
            f"A_bar LUT应为[256,d_state]，实际为"
            f"{tuple(a_bar_q24_lut.shape)}。"
        )
    if k_multiplier_u19_lut.shape != (SSM_DT_ADDRESS_COUNT,):
        raise ValueError(
            f"K LUT must be [256], got {tuple(k_multiplier_u19_lut.shape)}."
        )

    addresses = dt_codes - INT8_MIN
    # addresses为[B,d_inner]；高级索引后直接得到
    # Abar[B,d_inner,d_state] and K[B,d_inner], with no channel replicas.
    a_bar_q24 = a_bar_q24_lut[addresses]
    k_multiplier_u19 = k_multiplier_u19_lut[addresses]
    return a_bar_q24, k_multiplier_u19


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
    ref_x_proj = m.x_proj(
        x_act_ref.transpose(1, 2).contiguous()
    )
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
        output_bits=9,
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
        output_bits=8,
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

    if NL is not None:
        NL.inputs(name_prefix, u_int8_all, dt_int8, b_int8_all, c_int8_all,
                  a_bar_q24_lut, k_multiplier_u19_lut, k_fraction_bits)

    h_q24 = torch.zeros(
        (batch, d_inner, d_state),
        dtype=torch.int64,
        device=device,
    )
    y_ssm_q24 = torch.zeros(
        (batch, d_inner, sequence_length),
        dtype=torch.int64,
        device=device,
    )

    for time_idx in range(sequence_length):
        u_t_int8 = u_int8_all[:, :, time_idx]

        # [batch, d_inner]，SpeMamba 的每个像素独立寻址。
        a_bar_q24, k_scale_u19 = lookup_ssm_luts(
            a_bar_q24_lut,
            k_multiplier_u19_lut,
            dt_int8[:, time_idx, :],
        )

        b_t_int8 = b_int8_all[:, :, time_idx]
        kb_qf = k_scale_u19.unsqueeze(-1) * b_t_int8.unsqueeze(1)
        kb_min = -(1 << (SSM_KB_SIGNED_BITS - 1))
        kb_max = (1 << (SSM_KB_SIGNED_BITS - 1)) - 1
        if torch.any(kb_qf < kb_min) or torch.any(kb_qf > kb_max):
            raise OverflowError(
                f"{name_prefix}: K*B at time={time_idx} exceeds signed "
                f"{SSM_KB_SIGNED_BITS}-bit DSP operand."
            )
        kbu_qf = kb_qf * u_t_int8.unsqueeze(-1)
        if torch.any(a_bar_q24 < 0) or torch.any(a_bar_q24 > (1 << 24)):
            raise OverflowError(
                f"{name_prefix}: A_bar UQ1.24在time={time_idx}超出[0, 1]。"
            )
        ah_bound = int(a_bar_q24.abs().max().item()) * int(
            h_q24.abs().max().item()
        )
        input_align_shift = 2 * SSM_A_FRACTION_BITS - k_fraction_bits
        bx_bound = int(kbu_qf.abs().max().item()) << input_align_shift
        if ah_bound + bx_bound > torch.iinfo(torch.int64).max:
            raise OverflowError(
                f"{name_prefix}: SSM Q48状态累加在time={time_idx}可能超出signed INT64。"
            )
        # Ah is Q48. KBU is Q{k_fraction_bits}; this wiring-only left shift
        # aligns both terms before the single common round-to-Q24 operation.
        accumulator = (
            a_bar_q24 * h_q24
            + (kbu_qf << input_align_shift)
        )
        h_q24 = apply_multiplier_shift(
            accumulator,
            torch.tensor(1, dtype=torch.int32, device=device),
            24,
        ).clamp(
            INT32_MIN,
            INT32_MAX,
        )

        y_bound = (
            int(h_q24.abs().max().item())
            * int(c_int8_all[:, :, time_idx].abs().max().item())
            * int(d_state)
        )
        if y_bound > torch.iinfo(torch.int64).max:
            raise OverflowError(
                f"{name_prefix}: C读出累加在time={time_idx}可能超出signed INT64。"
            )
        y_q24 = torch.sum(
            h_q24 * c_int8_all[:, :, time_idx].unsqueeze(1),
            dim=-1,
        )
        y_ssm_q24[:, :, time_idx] = y_q24
        if NL is not None:
            NL.state(name_prefix, time_idx, h_q24, y_q24, accumulator)

    y_q24_scale = scalar_scale(s_c, device) / float(1 << 24)
    y_ssm_sim = (
        y_ssm_q24.double() * y_q24_scale.double()
    ).to(u_float.dtype)
    y_ssm_sim = y_ssm_sim.transpose(1, 2).contiguous()

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
        D=None,
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
    y_ssm_int8 = requantize_int8(
        y_ssm_q24.transpose(1, 2).contiguous(),
        y_q24_scale,
        s_out_proj_input,
    )

    # SSM RTL的第一块精确验证需要两个以往未导出的边界：
    #   1) Conv1d ReLU INT8码转换成Q8.24后的SSM-u；
    #   2) C读出后、out_proj前的Q24/INT8结果。
    # 这些文件只是硬件验证向量，不改变模型计算。
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
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
            output_path / f"{name_prefix}_ssm_y_q24.txt",
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
                    "u_tensor_axis_order": ["batch", "channel", "time"],
                    "y_tensor_axis_order": ["batch", "time", "channel"],
                    "u_scale": float(scalar_scale(s_conv, device).item()),
                    "u_q_format": "signed INT8; scale folded into K(dt)",
                    "k_logical_bits": SSM_K_LOGICAL_BITS,
                    "k_fraction_bits": int(k_fraction_bits),
                    "kb_signed_bits": SSM_KB_SIGNED_BITS,
                    "kbu_to_q48_left_shift": int(
                        2 * SSM_A_FRACTION_BITS - k_fraction_bits
                    ),
                    "y_q24_physical_scale": float(y_q24_scale.item()),
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
    if block.spa_mamba.use_proj:
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


def _single_backend_tile(net, patch_float, input_scale, output_dir, device):
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
            "按训练时batch方式复算的融合BN QAT仍无法复现保存预测。"
            "这不是FPGA定点误差；请检查--qat-reference-batch-size、"
            "checkpoint、服务器model/MambaHSI.py版本和CUDA确定性。"
        )
    return rate


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
                a_entries * SSM_A_LOGICAL_BITS
                + k_entries * SSM_K_LOGICAL_BITS
            )
            aligned_bits = (a_entries + k_entries) * 32
            total_packed_bits += packed_bits
            total_aligned_bits += aligned_bits
            ssm_roms.append(
                {
                    "core": f"blk{block_index}_{branch_name}",
                    "Abar_shape": a_shape,
                    "K_shape": k_shape,
                    "K_signed": False,
                    "K_logical_bits": SSM_K_LOGICAL_BITS,
                    "K_fraction_bits": int(cache["k_fraction_bits"]),
                    "U_storage_bits": 8,
                    "packed_bits": packed_bits,
                    "packed_MiB": packed_bits / 8.0 / (1 << 20),
                    "int32_aligned_bits": aligned_bits,
                    "int32_aligned_MiB": aligned_bits / 8.0 / (1 << 20),
                }
            )
    return ssm_roms, total_packed_bits, total_aligned_bits


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
):
    """按保存的blocks逐块执行无重叠16x16 QAT/INT8 FPGA模拟。"""
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
    if NL is not None:
        NL.setup(output_dir)

    qat_result = load_json_object(qat_run_dir / "result.json")
    validate_qat_run_metadata(qat_result, dataset_name, seed)
    if NL is not None:
        nl_validate_run_width(qat_result)
    if model_path is None:
        model_path = qat_run_dir / "best_qat_foldaware.pth"
    model_path = resolve_existing_path(
        model_path,
        "QAT best_qat_foldaware.pth",
        roots=(qat_run_dir,),
    )
    if NL is not None and 'n2' in NL.methods:
        if _sha256_file(model_path) != NL_N2_CONFIG['checkpoint_sha256']:
            raise ValueError('N2 checkpoint differs from nonlinear_compare_n2_v1; '
                             'use the matching v5 run or regenerate a matched comparison package.')
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

    print(f"Loading {dataset_name} from {data_path}")
    raw_data, raw_gt = data_load_operate.load_data(dataset_name, data_path)
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

    net = MambaHSI(
        in_channels=pca_channels,
        hidden_dim=hidden_dim,
        num_classes=class_count,
        mamba_type="both",
        token_num=token_num,
        use_att=False,
    )
    prepare_qat_model(net, nbit=8)
    state_dict = extract_state_dict(torch.load(model_path, map_location="cpu"))
    try:
        net.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Checkpoint与PCA={pca_channels}/hidden_dim={hidden_dim}/"
            f"token_num={token_num}/mamba_type='both'/use_att=False不匹配。"
        ) from error

    first_tile_cpu, _, _ = make_dense16_tile(image, blocks[0])
    fusion_probe = first_tile_cpu.to(device=device, dtype=torch.float32)
    net.to(device).eval()
    validate_fpga_qat_contract(net, expect_bn=True)
    freeze_lsq_initialization(net)
    fuse_qat_model_bns_for_deploy(net, validation_input=fusion_probe)
    validate_fpga_qat_contract(net, expect_bn=False)

    if int(qat_reference_batch_size) <= 0:
        if "eval_batch_size" in qat_result:
            qat_reference_batch_size = int(qat_result["eval_batch_size"])
        else:
            # 现有成功QAT run由默认eval_batch_size=8产生；新run会把该字段
            # 直接写入result.json，不再依赖兼容回退。
            qat_reference_batch_size = 8
            print(
                "[Compatibility] QAT result.json缺少eval_batch_size；"
                "按该训练脚本历史默认值8复算保存预测。"
            )
    if int(qat_reference_batch_size) <= 0:
        raise ValueError("--qat-reference-batch-size必须为正数或0(自动)。")

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

    result = {
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
        "qat_batch_invariance_probe": qat_batch_invariance_probe,
        "qat_best_epoch": int(qat_result["best_epoch"]),
        "input_scale": float(input_scale.item()),
        **evaluation,
    }
    with (output_dir / "fpga_simulation_result.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=json_default)
    print(f"\n16x16 BothMamba FPGA simulation finished: {output_dir}")
    if NL is not None:
        NL.finish(gt, qat_indices, blocks, predict_int8, result)
    return result


"""Embedded by make_propagation_script.py; not a standalone entry point."""

# N0/N1 arithmetic is intentionally identical to nonlinear_compare/tools/prepare.py.
# Precomputing 256 values accelerates *numerical* simulation, not online hardware.
NL_Q = 1 << 30
NL_EXP_C = [round((-math.log(2)) ** k / math.factorial(k) * NL_Q) for k in range(11)]
NL_REC_C = [round(NL_Q / 3)] * 19
NL_LOG_C = [round(NL_Q / (2 * k + 1)) for k in range(9)]
NL_ENDPOINTS = [round(2 ** (-k / 8) * NL_Q) for k in range(9)]
NL_QAT_SOURCE = {}
NL_SCRIPT_VERSION = 'v5-nonlinear-20260914-integrated-n2'

# Frozen configuration shared with nonlinear_compare_n2_v1; no runtime fitting.
# Embedded so the server needs only this script, not the former wrapper files.
NL_N2_CONFIG = json.loads(r'''{
  "schema": 1,
  "checkpoint_sha256": "1ee46f229dda359faf0ee4a2764adf1a284a15e8926023fb1653fb6aa6e15fe3",
  "fit": {
    "schema": 1,
    "method": "S2Mamba-inspired Eq9/10/20 and two-stage Alg2 adaptation",
    "seed": 20260913,
    "steps": 5000,
    "samples": 4096,
    "fit_range": [
      -16,
      16
    ],
    "loss": "mean squared Softplus error",
    "notes": "Local coefficients, not author coefficients; no WIS/SiLU/full 16-bit network reproduction",
    "EXP_SOFT_Q30": [
      1032860726,
      1032125116
    ],
    "LOG_SOFT_Q30": [
      797976356,
      11625313
    ],
    "EXP_STATE_Q30": [
      1065150840,
      1016505539
    ],
    "LOG2E_Q30": 1549082005,
    "LN2_Q30": 744261118,
    "stage1": {
      "initial_mse": 0.00017534068155423082,
      "final_mse": 0.00015394355605531553
    },
    "stage2": {
      "initial_mse": 0.00019163078981418203,
      "final_mse": 0.00019163078981418203
    },
    "RTL_latency_registers": 13,
    "target_II": 1
  },
  "cores": {
    "blk0_spa": {
      "SDT_Q30": 62709196,
      "DECAY_Q24": [
        16951819,
        32006960,
        49326542,
        65590106,
        81603386,
        100395012,
        117768942,
        132816297,
        153381404,
        168517576,
        179741103,
        203579946,
        214101299,
        233382854,
        254762259,
        263252547
      ],
      "SBSU_Q40": 972505283,
      "K_fraction_bits": 24
    },
    "blk0_spe": {
      "SDT_Q30": 50892048,
      "DECAY_Q24": [
        16578952,
        33413728,
        49583125,
        68902355,
        78118483,
        97841484,
        113978816,
        130765526,
        157620441,
        166701476,
        174888640,
        190915644,
        223977834,
        228989827,
        248652704,
        263976586
      ],
      "SBSU_Q40": 513768001,
      "K_fraction_bits": 24
    },
    "blk1_spa": {
      "SDT_Q30": 61496036,
      "DECAY_Q24": [
        16994651,
        32818184,
        51482982,
        67523588,
        81794188,
        98384809,
        120841095,
        134517295,
        152916879,
        171352187,
        185907759,
        196644070,
        216489216,
        221547246,
        254979619,
        272173177
      ],
      "SBSU_Q40": 1988154102,
      "K_fraction_bits": 24
    },
    "blk1_spe": {
      "SDT_Q30": 57160996,
      "DECAY_Q24": [
        16526902,
        33136909,
        52387725,
        64965131,
        81422832,
        100580748,
        113063167,
        132660149,
        159232528,
        173265958,
        180686876,
        203569899,
        222293222,
        229590344,
        258411413,
        263653099
      ],
      "SBSU_Q40": 1062337329,
      "K_fraction_bits": 24
    },
    "blk2_spa": {
      "SDT_Q30": 69475760,
      "DECAY_Q24": [
        16597171,
        34167194,
        50794910,
        68140270,
        86919159,
        97274764,
        112929462,
        139154149,
        153212951,
        157689490,
        198363924,
        209031380,
        208797825,
        231939400,
        254125529,
        276731042
      ],
      "SBSU_Q40": 5103422946,
      "K_fraction_bits": 23
    },
    "blk2_spe": {
      "SDT_Q30": 60588984,
      "DECAY_Q24": [
        16008190,
        35282699,
        48846064,
        66645726,
        85555986,
        101809462,
        117990196,
        135292892,
        150868067,
        166411711,
        195878969,
        196826203,
        201806929,
        230321116,
        250368270,
        258814596
      ],
      "SBSU_Q40": 2514041171,
      "K_fraction_bits": 24
    }
  }
}''')



def nl_validate_run_width(metadata):
    """Never reinterpret a run which explicitly says its dt input was not INT9."""
    declared = {key: metadata[key] for key in ('dt_proj_input_bits', 'dt_input_bits')
                if metadata.get(key) is not None}
    numeric = metadata.get('numeric_config')
    if isinstance(numeric, dict) and numeric.get('dt_input_bits') is not None:
        declared['numeric_config.dt_input_bits'] = numeric['dt_input_bits']
    for key, value in declared.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        if not values or any(str(v) != '9' for v in values):
            raise RuntimeError(f'QAT result declares {key}={value!r}, not v5 INT9. '
                               'Select the correct v5 run/module; do NOT change this experiment to hide the mismatch.')
    return declared


def nl_dt9_contract(net, qat, repair=False):
    """Restore only constructor-time INT8/INT9 attributes, before checkpoint load.

    Do not replace quantizer modules/Parameters, alter weights/scales, or relax the
    original QAT validator. An unknown quantizer or already calibrated model fails.
    """
    blocks = [layer for layer in net.mamba if hasattr(layer, 'spa_mamba')]
    if len(blocks) != 3:
        raise RuntimeError('v5 DT9 adapter expects exactly three BothMamba blocks')
    rows = []
    for block_index, block in enumerate(blocks):
        for branch in ('spa', 'spe'):
            name = f'block{block_index}.{branch}.dt_proj'
            layer = getattr(block, f'{branch}_mamba').mamba.dt_proj
            q = getattr(layer, 'lsq_a', None)
            if not isinstance(layer, qat.QuanLinear) or not isinstance(q, qat.LsqQuantizer4input):
                raise RuntimeError(f'{name}: unsupported quantizer implementation; send the actual QAT source')
            if getattr(layer, 'quan_a', None) is not q:
                raise RuntimeError(f'{name}: quan_a and lsq_a differ; refusing to guess the forward quantizer')
            if q.all_positive or q.per_channel or q.s.numel() != 1 or not layer.quan_input:
                raise RuntimeError(f'{name}: expected enabled signed per-tensor input quantization')
            before = dict(nbit_a=int(layer.nbit_a), a_bits=int(q.a_bits), Qn=int(q.Qn), Qp=int(q.Qp))
            expected = dict(nbit_a=9, a_bits=9, Qn=-256, Qp=255)
            changed = before != expected
            if changed:
                if not repair:
                    raise RuntimeError(f'{name}: INT9 contract changed after preparation: {before}')
                if before['nbit_a'] not in (8, 9) or before['a_bits'] not in (8, 9):
                    raise RuntimeError(f'{name}: unsupported bit configuration {before}; refusing automatic conversion')
                if (before['Qn'], before['Qp']) not in ((-128, 127), (-256, 255)):
                    raise RuntimeError(f'{name}: unsupported clamp range {before}; refusing automatic conversion')
                if int(q.init_state) != 0:
                    raise RuntimeError(f'{name}: quantizer already calibrated; refusing to change its contract')
                layer.nbit_a = 9
                q.a_bits = 9
                q.Qn = -256
                q.Qp = 255
            row = dict(layer=name, before=before, after=expected, changed=changed)
            rows.append(row)
            if repair:
                action = 'RESTORED' if changed else 'OK'
                print(f'[V5 DT INPUT {action}] {name}: {before} -> {expected}; '
                      'same LSQ parameter/alias; dt output remains INT8')
    return rows


def nl_make_v5_preparer(qat):
    """Run the original converter and ALL its validations, with a local precheck.

    Temporary in-process validator binding is restored even on failure. No QAT
    source file is edited, and recursive conversion still uses the original code.
    """
    original_prepare = qat.prepare_qat_model
    original_validate = qat.validate_fpga_qat_contract
    namespace = getattr(original_prepare, '__globals__', {})
    if namespace.get('validate_fpga_qat_contract') is not original_validate:
        raise RuntimeError('Unrecognized QAT converter/validator binding; cannot apply the v5 DT9 adapter safely')

    def prepare(model, nbit=8):
        if nbit != 8:
            raise ValueError('This experiment requires v5 INT8 weights and INT9 dt input')
        audits = []

        def validate_after_constructor(net, *args, **kwargs):
            if net is not model:
                return original_validate(net, *args, **kwargs)
            audit = nl_dt9_contract(net, qat, repair=True)
            audits.extend(audit)
            if NL is not None and NL.out is not None:
                nl_json(NL.out / 'qat_dt_input_contract.json', dict(
                    version=NL_SCRIPT_VERSION, qat_source=NL_QAT_SOURCE, layers=audit,
                    note='Constructor attributes only; before strict checkpoint load. Original validator runs next.'))
            # Crucial: the original validator is called, not bypassed/caught.
            return original_validate(net, *args, **kwargs)

        namespace['validate_fpga_qat_contract'] = validate_after_constructor
        try:
            prepared = original_prepare(model, nbit=nbit)
        finally:
            namespace['validate_fpga_qat_contract'] = original_validate
        if len(audits) != 6:
            raise RuntimeError('Expected exactly one root validation of all six dt_proj layers')
        nl_dt9_contract(model, qat, repair=False)
        return prepared

    return prepare


def nl_poly(x, coefficients):
    y = coefficients[-1]
    for c in coefficients[-2::-1]:
        y = ((y * x) >> 30) + c
        if not -(1 << 31) <= y < (1 << 31):
            raise OverflowError('Online polynomial exceeds the RTL signed 32-bit contract')
    return y


def nl_exp_neg(x24, method):
    if x24 < 0:
        raise ValueError('exp_neg requires a nonnegative magnitude')
    if x24 >= 32 * (1 << 24):
        return 0
    log2e = round(math.log2(math.e) * NL_Q) if method == 'n0' else 23 * NL_Q // 16
    w = (x24 * log2e) >> 24
    n, f = w >> 30, w & (NL_Q - 1)
    if method == 'n0':
        y = max(0, nl_poly(f, NL_EXP_C))
    else:
        seg, local = f >> 27, f & ((1 << 27) - 1)
        y = NL_ENDPOINTS[seg] - (
            (NL_ENDPOINTS[seg] - NL_ENDPOINTS[seg + 1]) * local >> 27
        )
    return y >> n if n < 32 else 0


def nl_n2_exp_q30(x24, coefficients):
    """S2Mamba-inspired base conversion, affine mantissa and integer shift."""
    p = NL_N2_CONFIG['fit']
    y30 = (x24 * p['LOG2E_Q30']) >> 24
    integer, fraction = y30 >> 30, y30 & (NL_Q - 1)
    mantissa = (fraction * coefficients[0] + (coefficients[1] << 30)) >> 30
    if integer < -63:
        return 0
    if integer > 32:
        raise OverflowError('N2 exp exceeds the RTL 64-bit Q30 carrier')
    return mantissa << integer if integer >= 0 else mantissa >> -integer


def nl_n2_online(qdt, sd, decay, scale, kf):
    """Frozen local Eq.9/10/20 + Algorithm 2 adaptation, not author RTL.

    Identical integer ordering to the delivered N2 reference model. No scene,
    labels or weights are used to refit the approximation during evaluation.
    """
    signature = (sd, list(decay), scale, kf)
    if not any(signature == (c['SDT_Q30'], c['DECAY_Q24'], c['SBSU_Q40'], c['K_fraction_bits'])
               for c in NL_N2_CONFIG['cores'].values()):
        raise ValueError('N2 constants differ from the frozen hardware package. '
                         'Regenerate a matched configuration; do not bypass this check.')
    if not -128 <= qdt <= 127:
        raise ValueError('N2 requires an INT8 dt code')
    p = NL_N2_CONFIG['fit']
    magnitude = (abs(qdt) * sd + 32) >> 6
    x24 = -magnitude if qdt < 0 else magnitude
    if abs(x24) > 16 * (1 << 24):
        raise ValueError('N2 Softplus fit domain [-16,16] exceeded')
    alpha = NL_Q + nl_n2_exp_q30(x24, p['EXP_SOFT_Q30'])
    w = alpha.bit_length() - 1 - 30
    if w < 0:
        raise ValueError('N2 alpha is below 1')
    k30 = alpha >> w
    log_slope, log_bias = p['LOG_SOFT_Q30']
    delta30 = w * p['LN2_Q30'] + (((k30 - NL_Q) * log_slope) >> 30) + log_bias
    delta24 = max(0, delta30 + 32) >> 6
    a = []
    for d in decay:
        mag = (delta24 * d + (1 << 23)) >> 24
        a.append(min(1 << 24, (nl_n2_exp_q30(-mag, p['EXP_STATE_Q30']) + 32) >> 6))
    raw_k = (delta24 * scale + (1 << (63 - kf))) >> (64 - kf)
    return a, min((1 << 19) - 1, raw_k), raw_k > (1 << 19) - 1


def nl_online(qdt, sd, decay, scale, kf, method):
    if method == 'n2':
        return nl_n2_online(qdt, sd, decay, scale, kf)
    if method not in ('n0', 'n1') or not -128 <= qdt <= 127:
        raise ValueError('Invalid online backend or dt code')
    x24 = (abs(qdt) * sd + 32) >> 6
    y = nl_exp_neg(x24, method)
    if method == 'n0':
        r = ((NL_Q - y) * round(NL_Q / 3)) >> 30
        recip = nl_poly(r, NL_REC_C)
        z = y * recip >> 30
        log1p = (2 * z * nl_poly(z * z >> 30, NL_LOG_C)) >> 30
    else:
        # The chosen Eq.6 approximation is deliberately NOT corrected at zero.
        log1p = y
    delta = ((log1p + 32) >> 6) + (x24 if qdt > 0 else 0)
    a = []
    for d in decay:
        mag = (delta * d + (1 << 23)) >> 24
        a.append(min(1 << 24, (nl_exp_neg(mag, method) + 32) >> 6))
    raw_k = (delta * scale + (1 << (63 - kf))) >> (64 - kf)
    return a, min((1 << 19) - 1, raw_k), raw_k > (1 << 19) - 1


def nl_stats(reference, actual):
    r = torch.as_tensor(reference).detach().double().reshape(-1)
    a = torch.as_tensor(actual).detach().to(r.device).double().reshape(-1)
    if r.numel() != a.numel() or not torch.isfinite(r).all() or not torch.isfinite(a).all():
        raise ValueError('Non-finite or inconsistent comparison tensor')
    e = a - r
    return dict(count=r.numel(), mismatches=int(torch.count_nonzero(e).item()),
                sum_abs=float(e.abs().sum().item()), sum_sq=float((e * e).sum().item()),
                max_abs=float(e.abs().max().item()) if e.numel() else 0.,
                ref_sq=float((r * r).sum().item()), actual_sq=float((a * a).sum().item()),
                dot=float((a * r).sum().item()))


def nl_merge(target, item):
    for key, value in item.items():
        target[key] = max(target.get(key, 0), value) if key == 'max_abs' else target.get(key, 0) + value


def nl_finish_stats(value):
    v = dict(value)
    n = v['count']
    v['MAE'] = v['sum_abs'] / n if n else None
    v['RMSE'] = math.sqrt(v['sum_sq'] / n) if n else None
    v['match_fraction'] = 1 - v['mismatches'] / n if n else None
    v['relative_L2'] = math.sqrt(v['sum_sq'] / v['ref_sq']) if v['ref_sq'] else None
    denom = math.sqrt(v['ref_sq'] * v['actual_sq'])
    v['cosine'] = v['dot'] / denom if denom else None
    return v


def nl_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)


def nl_recurrence_step(a, k, u, b, c, h, kf):
    shift = 48 - kf
    bx = k.unsqueeze(-1) * b.unsqueeze(1) * u.unsqueeze(-1)
    bound = int(a.abs().max()) * int(h.abs().max()) + (int(bx.abs().max()) << shift)
    if bound > (1 << 63) - 1 - (1 << 23):
        raise OverflowError('Replay Q48 accumulator exceeds the checked INT64 carrier')
    accumulator = a * h + (bx << shift)
    raw = apply_multiplier_shift(accumulator, torch.tensor(1, device=h.device), 24)
    clipped = int(torch.count_nonzero((raw < INT32_MIN) | (raw > INT32_MAX)).item())
    state = raw.clamp(INT32_MIN, INT32_MAX)
    return state, (state * c.unsqueeze(1)).sum(-1), clipped


class NonlinearExperiment:
    def __init__(self, args):
        self.args = args
        self.methods = ['n0', 'n1', 'n2'] if args.nonlinear_backend == 'all' else (
            [] if args.nonlinear_backend == 'n3' else [args.nonlinear_backend]
        )
        self.method = 'n3'
        self.tile = -1
        self.out = None
        self.tables = {}
        self.coefficient_report = {}
        self.layers = {}
        self.isolated = {}
        self.state_e2e = {}
        self.n3_boundaries = {}
        self.n3_states = {}
        self.n3_logits = []
        self.variant_logits = {m: [] for m in self.methods}
        self.variant_scales = {m: [] for m in self.methods}
        self.n3_scales = []
        self.input_digest = hashlib.sha256()
        self.input_hashes = []
        self.dt_hist = {}
        self.replay_expected = {}

    def setup(self, output_dir):
        self.out = Path(output_dir) / 'nonlinear_experiment'
        self.out.mkdir(parents=True, exist_ok=True)
        self.run_metadata = dict(
            status='RUNNING_NOT_COMPLETE', backends=['n3', *self.methods],
            script_version=NL_SCRIPT_VERSION, qat_source=NL_QAT_SOURCE,
            arguments=vars(self.args), baseline_source_sha256=NL_BASE_SHA256,
            script_sha256=_sha256_file(Path(__file__)),
            reference='Frozen v5 integer N3, not floating QAT',
            n0='Q30 polynomial online arithmetic, NOT vendor FP32 IP',
            n1='FastMamba-inspired exp/PWL and Eq.6 softplus adaptation, NOT author RTL',
            warning='Cached software runtime is NOT online hardware latency or resource usage',
        )
        if 'n2' in self.methods:
            nl_json(self.out / 'n2_config.json', NL_N2_CONFIG)
            self.run_metadata['n2_config'] = NL_N2_CONFIG
            self.run_metadata['n2_config_sha256'] = _sha256_file(self.out / 'n2_config.json')
        nl_json(self.out / 'run_manifest.json', self.run_metadata)

    def coefficients(self, m, s_dt, s_b, s_u, name, output_dir, device):
        base = _n3_get_or_build_ssm_luts(m, s_dt, s_b, s_u, name,
                                       output_dir if self.method == 'n3' else None, device)
        key = id(m), name
        cache = m.__dict__['_fpga_ssm_lut_cache']
        signature = cache['signature']
        if key in self.tables and self.tables[key]['signature'] != signature:
            raise RuntimeError(f'{name}: frozen coefficient/scaling contract changed during experiment')
        if key not in self.tables:
            self._compile(m, name, base, cache, key, device)
        if self.method == 'n3':
            return base
        a, k = self.tables[key][self.method]
        return base[0], a, k, base[3]

    def _compile(self, m, name, base, cache, key, device):
        # Match the existing v5 generator's float32 A_log -> CPU float64 exp.
        decay = torch.exp(m.A_log_shared.detach().float().cpu().double()).tolist()
        sd = round(cache['s_dt'] * NL_Q)
        dq = [round(v * (1 << 24)) for v in decay]
        scale = round(cache['s_b'] * cache['s_u'] * (1 << 40))
        kf = int(base[3])  # ALWAYS N3's fractional position; never retune per method.
        if len(dq) != 16 or not 0 <= sd < (1 << 32) or not 0 <= scale < (1 << 48):
            raise ValueError(f'{name}: constants outside the standalone RTL comparison contract')
        if any(not 0 <= x < (1 << 48) for x in dq) or not 0 <= kf <= 24:
            raise ValueError(f'{name}: unsupported decay or K fractional width')
        methods = dict(signature=cache['signature'], n3=(base[1], base[2]))
        report = dict(s_dt=cache['s_dt'], s_B=cache['s_b'], s_u=cache['s_u'],
                      SDT_Q30=sd, SBSU_Q40=scale, DECAY_Q24=dq, K_fraction_bits=kf,
                      A_log_shared=m.A_log_shared.detach().float().cpu().tolist(),
                      decay_from_checkpoint=decay, provenance='Actual loaded checkpoint, not interval reconstruction',
                      reference_lut_check='NOT_REQUESTED', methods={})
        directory = self.out / 'coefficients'
        directory.mkdir(parents=True, exist_ok=True)
        if self.args.reference_lut_dir:
            directory_ref = Path(self.args.reference_lut_dir)
            old_a = np.load(directory_ref / f'{name}_Abar_uq1_24.npy', allow_pickle=False)
            old_k = np.load(directory_ref / f'{name}_K_q{kf}_u19.npy', allow_pickle=False)
            if not np.array_equal(old_a, base[1].cpu().numpy()) or not np.array_equal(old_k, base[2].cpu().numpy()):
                raise ValueError(f'{name}: N3 differs from saved v5 LUT. Check checkpoint/PCA/scales')
            report['reference_lut_check'] = 'ALL_256_CODES_EXACT'
        for method in ['n3', *self.methods]:
            clipped = 0
            if method != 'n3':
                values = [nl_online(q, sd, dq, scale, kf, method) for q in range(-128, 128)]
                a = torch.tensor([v[0] for v in values], dtype=torch.int64, device=device)
                k = torch.tensor([v[1] for v in values], dtype=torch.int64, device=device)
                clipped = sum(v[2] for v in values)
                methods[method] = (a, k)
            a, k = methods[method]
            np.savez(directory / f'{name}_{method}.npz', Abar=a.cpu().numpy(), K=k.cpu().numpy(), K_FRAC=kf)
            words = [sum(int(v) << (25 * j) for j, v in enumerate(row)) | (int(kv) << 400)
                     for row, kv in zip(a.cpu().tolist(), k.cpu().tolist())]
            (directory / f'{name}_{method}.mem').write_text(
                ''.join(f'{w:0105x}\n' for w in words), encoding='ascii')
            report['methods'][method] = dict(
                Abar_LSB=nl_finish_stats(nl_stats(base[1], a)),
                K_LSB=nl_finish_stats(nl_stats(base[2], k)), K_saturated_addresses=clipped,
                rtl_vector_check='NOT_REQUESTED')
            if self.args.rtl_data_dir:
                path = Path(self.args.rtl_data_dir) / f'{name}_{method}.mem'
                old = [int(line, 16) for line in path.read_text().splitlines() if line.strip()]
                if old != words:
                    raise ValueError(f'{path}: differs from actual-checkpoint coefficients. '
                                     'Regenerate RTL comparison constants/vectors from the exported decay JSON; '
                                     'do not claim RTL/SW bit-exact equivalence.')
                report['methods'][method]['rtl_vector_check'] = 'ALL_256_PACKED_WORDS_EXACT'
        report['DECAY_Q24_SV'] = "768'h" + ''.join(f'{v:012x}' for v in reversed(dq))
        self.tables[key] = methods
        self.coefficient_report[name] = report
        nl_json(directory / f'{name}_parameters.json', report)
        nl_json(directory / 'decay_from_checkpoint.json', {
            n: r['decay_from_checkpoint'] for n, r in self.coefficient_report.items()
        })

    def boundary(self, name, codes, scale, fixed):
        if self.tile < 0:
            return
        kind = 'codes' if codes is not None else 'dequantized'
        value = codes if codes is not None else fixed
        if value is None:
            return
        value = torch.as_tensor(value).detach()
        key = name + '/' + kind
        scale_value = float(torch.as_tensor(scale)) if codes is not None else None
        if self.method == 'n3':
            self.n3_boundaries[key] = value.clone(), scale_value
        else:
            reference, old_scale = self.n3_boundaries[key]
            if old_scale != scale_value:
                raise ValueError(f'{name}: backend changed the frozen boundary scale')
            if reference.shape != value.shape:
                raise ValueError(f'{name}: backend changed tensor shape')
            item = self.layers.setdefault(self.method, {}).setdefault(key, {})
            nl_merge(item, nl_stats(reference, value))

    def trace_enabled(self):
        return self.args.trace_tiles < 0 or self.tile < self.args.trace_tiles

    def inputs(self, name, u, dt, b, c, a_lut, k_lut, kf):
        # Called at the integer recurrence input, before any state is advanced.
        if self.method != 'n3':
            return
        histogram = torch.bincount((dt.long() + 128).reshape(-1), minlength=256).cpu().numpy()
        self.dt_hist[name] = self.dt_hist.get(name, np.zeros(256, dtype=np.int64)) + histogram
        if not self.trace_enabled():
            return
        tables = next(v for (_, core), v in self.tables.items() if core == name)
        states = {method: torch.zeros((*u.shape[:2], a_lut.shape[1]), dtype=torch.int64, device=u.device)
                  for method in ['n3', *self.methods]}
        expected = []
        for t in range(u.shape[2]):
            current = {}
            addr = dt[:, t, :].long() + 128
            for method in ['n3', *self.methods]:
                aa, kk = tables[method]
                h, y, clipped = nl_recurrence_step(
                    aa[addr], kk[addr], u[:, :, t], b[:, :, t], c[:, :, t], states[method], kf)
                states[method] = h
                current[method] = h, y, clipped
            expected.append((current['n3'][0].clone(), current['n3'][1].clone()))
            for method in self.methods:
                row = self.isolated.setdefault(method, {}).setdefault(name, {}).setdefault(str(t),
                    dict(state={}, readout={}, n3_saturations=0, method_saturations=0, samples=0))
                nl_merge(row['state'], nl_stats(current['n3'][0], current[method][0]))
                nl_merge(row['readout'], nl_stats(current['n3'][1], current[method][1]))
                row['n3_saturations'] += current['n3'][2]
                row['method_saturations'] += current[method][2]
                row['samples'] += 1
        self.replay_expected[name] = expected

    def state(self, name, t, h, y, raw_accumulator):
        if not self.trace_enabled():
            return
        raw = apply_multiplier_shift(raw_accumulator, torch.tensor(1, device=h.device), 24)
        clipped = int(torch.count_nonzero((raw < INT32_MIN) | (raw > INT32_MAX)))
        key = name, t
        if self.method == 'n3':
            ref_h, ref_y = self.replay_expected[name][t]
            if not torch.equal(ref_h, h) or not torch.equal(ref_y, y):
                raise AssertionError(f'{name} step {t}: isolated N3 replay is NOT bit-exact')
            self.n3_states[key] = h.clone(), y.clone(), clipped
        else:
            ref_h, ref_y, ref_clip = self.n3_states[key]
            row = self.state_e2e.setdefault(self.method, {}).setdefault(name, {}).setdefault(str(t),
                dict(state={}, readout={}, n3_saturations=0, method_saturations=0, samples=0))
            nl_merge(row['state'], nl_stats(ref_h, h))
            nl_merge(row['readout'], nl_stats(ref_y, y))
            row['n3_saturations'] += ref_clip
            row['method_saturations'] += clipped
            row['samples'] += 1

    def run_tile(self, net, tile, scale, output_dir, device):
        self.tile += 1
        self.device = device
        self.n3_boundaries.clear()
        self.n3_states.clear()
        self.replay_expected.clear()
        self.method = 'n3'
        q = round_half_away_from_zero(tile / scale).clamp(0, 255).to(torch.uint8)
        packed = q[0].permute(1, 2, 0).contiguous().cpu().numpy().tobytes()
        self.input_digest.update(packed)
        self.input_hashes.append(hashlib.sha256(packed).hexdigest())
        base = _single_backend_tile(net, tile, scale, output_dir, device)
        if self.args.expected_real5_logits and self.tile < 5:
            golden = np.load(self.args.expected_real5_logits, allow_pickle=False)
            current = base[2][0].detach().cpu().numpy()
            if golden.shape != (5, *current.shape) or not np.array_equal(golden[self.tile], current):
                raise AssertionError(f'Tile {self.tile}: N3 logits differ from --expected-real5-logits')
        self.n3_logits.append(base[2][0].detach().cpu().numpy().copy())
        self.n3_scales.append(float(base[3]))
        for method in self.methods:
            self.method = method
            # Keep existing verbose layer dumps exclusively as N3 golden artifacts.
            with contextlib.redirect_stdout(io.StringIO()):
                result = _single_backend_tile(net, tile, scale, None, device)
            if not torch.equal(result[0], base[0]) or not torch.equal(result[1], base[1]):
                raise AssertionError('Backend replacement unexpectedly changed the QAT/staged-FP reference')
            if float(result[3]) != float(base[3]):
                raise AssertionError('Backend replacement changed the frozen logit scale')
            self.variant_logits[method].append(result[2][0].detach().cpu().numpy().copy())
            self.variant_scales[method].append(float(result[3]))
        self.method = 'n3'
        return base

    def finish(self, gt, indices, blocks, baseline_prediction, result):
        if 'n2' in self.methods and _sha256_file(Path(result['qat_checkpoint'])) != NL_N2_CONFIG['checkpoint_sha256']:
            raise ValueError('N2 checkpoint differs from the frozen hardware package')
        if len(self.n3_logits) != len(blocks):
            raise AssertionError('Full-scene experiment did not finish all saved tiles')
        if len(self.coefficient_report) != 6:
            raise AssertionError('Expected all six SSM cores to be exercised')
        out = self.out
        n3 = np.stack(self.n3_logits)
        np.save(out / 'n3_tile_logits_int8.npy', n3)
        np.save(out / 'n3_prediction.npy', baseline_prediction)
        np.save(out / 'logit_scales.npy', np.asarray(self.n3_scales))
        np.savez(out / 'fixed_indices.npz', **indices)
        nl_json(out / 'tile_manifest.json', dict(blocks=blocks, input_order='tile,y,x,channel',
                    input_tiles_sha256=self.input_digest.hexdigest(), tile_sha256=self.input_hashes,
                    logit_order='tile,class,y,x', trace_tiles=min(len(blocks), self.args.trace_tiles)
                    if self.args.trace_tiles >= 0 else len(blocks)))
        if self.args.expected_labels:
            old = np.load(self.args.expected_labels, allow_pickle=False)
            if old.shape != baseline_prediction.shape or not np.array_equal(old, baseline_prediction):
                raise AssertionError('N3 full-scene labels differ from --expected-labels; do not use as v5 comparison')
        if self.args.expected_input_sha256 and self.args.expected_input_sha256.lower() != self.input_digest.hexdigest():
            raise AssertionError('Quantized full-scene input hash differs from expected v5 input')
        if self.args.expected_real5_logits:
            old = np.load(self.args.expected_real5_logits, allow_pickle=False)
            if old.shape != n3[:5].shape or not np.array_equal(old, n3[:5]):
                raise AssertionError('N3 first-five logits differ from the supplied independent golden')
        summary = dict(status='COMPLETE', backends=['n3', *self.methods],
                       baseline='N3 frozen integer network',
                       checkpoint=result['qat_checkpoint'], checkpoint_sha256=_sha256_file(Path(result['qat_checkpoint'])),
                       script_sha256=_sha256_file(Path(__file__)), baseline_source_sha256=NL_BASE_SHA256,
                       input_tiles_sha256=self.input_digest.hexdigest(),
                       tiles=len(blocks), scene_shape=list(gt.shape),
                       full_scene_v5_label_check=bool(self.args.expected_labels),
                       real5_logit_check=bool(self.args.expected_real5_logits),
                       n3_replay_bit_exact_checked_tiles=min(len(blocks), self.args.trace_tiles)
                       if self.args.trace_tiles >= 0 else len(blocks),
                       rtl_simulation_performed=False, hardware_latency_measured=False,
                       coefficients=self.coefficient_report, methods={})
        if 'n2' in self.methods:
            summary['n2_config'] = NL_N2_CONFIG
            summary['n2_config_sha256'] = self.run_metadata['n2_config_sha256']
        ref = baseline_prediction.reshape(-1)
        scopes = dict(all=np.arange(gt.size), background=np.flatnonzero(gt.reshape(-1) == 0),
                      labeled=np.flatnonzero(gt.reshape(-1) > 0),
                      train=indices['train_indices'], validation=indices['val_indices'], test=indices['test_indices'])
        baseline_metrics = {}
        for split in ('train', 'validation', 'test'):
            ix = scopes[split]
            baseline_metrics[split] = detailed_classification_metrics(gt.reshape(-1)[ix] - 1, ref[ix], n3.shape[1])
        summary['n3_metrics'] = baseline_metrics
        popcount = np.array([int(i).bit_count() if hasattr(int, 'bit_count') else bin(i).count('1')
                             for i in range(256)], dtype=np.uint8)
        for method in self.methods:
            values = np.stack(self.variant_logits[method])
            np.save(out / f'{method}_tile_logits_int8.npy', values)
            pred = np.full(gt.shape, -1, dtype=np.int64)
            for i, block in enumerate(blocks):
                logits = torch.from_numpy(values[i:i+1]).to(self.device).float() * self.n3_scales[i]
                labels = F.interpolate(logits, size=(16, 16), mode='bilinear', align_corners=True).argmax(1)[0].cpu().numpy()
                top, left = int(block['top']), int(block['left'])
                h, w = min(16, gt.shape[0] - top), min(16, gt.shape[1] - left)
                pred[top:top+h, left:left+w] = labels[:h, :w]
            if np.any(pred < 0):
                raise AssertionError('Incomplete nonlinear-backend prediction canvas')
            np.save(out / f'{method}_prediction.npy', pred)
            np.save(out / f'{method}_mismatch_mask.npy', pred != baseline_prediction)
            flat = pred.reshape(-1)
            entry = dict(logit_codes=nl_finish_stats(nl_stats(n3, values)),
                         logit_mismatched_bits=int(popcount[np.bitwise_xor(n3.view(np.uint8), values.view(np.uint8))].sum()),
                         total_logit_bits=int(n3.size * 8), agreement={}, metrics={})
            for scope, ix in scopes.items():
                matches = int(np.count_nonzero(ref[ix] == flat[ix]))
                entry['agreement'][scope] = dict(matches=matches, count=len(ix), fraction=matches / len(ix) if len(ix) else None)
            for split in ('train', 'validation', 'test'):
                ix = scopes[split]
                entry['metrics'][split] = detailed_classification_metrics(gt.reshape(-1)[ix] - 1, flat[ix], n3.shape[1])
            ix = scopes['test']
            truth = gt.reshape(-1)[ix] - 1
            rb, ab = ref[ix] == truth, flat[ix] == truth
            entry['test_error_exchange'] = dict(
                n3_correct_method_wrong=int(np.count_nonzero(rb & ~ab)),
                n3_wrong_method_correct=int(np.count_nonzero(~rb & ab)),
                both_wrong_different=int(np.count_nonzero(~rb & ~ab & (ref[ix] != flat[ix]))))
            entry['test_OA_drop_percentage_points'] = 100 * (baseline_metrics['test']['OA'] - entry['metrics']['test']['OA'])
            summary['methods'][method] = entry
            print(f"[Nonlinear {method.upper()} vs N3] test agreement={entry['agreement']['test']['fraction']:.6%}; "
                  f"OA={entry['metrics']['test']['OA']:.8f}; drop={entry['test_OA_drop_percentage_points']:.6f} pp")
        nl_json(out / 'layer_propagation.json', {
            method: {name: nl_finish_stats(v) for name, v in layers.items()}
            for method, layers in self.layers.items()})
        for filename, table in [('isolated_ssm_replay.json', self.isolated), ('end_to_end_ssm_states.json', self.state_e2e)]:
            nl_json(out / filename, {method: {core: {t: dict(row,
                state=nl_finish_stats(row['state']), readout=nl_finish_stats(row['readout']))
                for t, row in steps.items()} for core, steps in cores.items()} for method, cores in table.items()})
        nl_json(out / 'dt_code_histogram_n3.json', {k: v.tolist() for k, v in self.dt_hist.items()})
        self.run_metadata['status'] = 'COMPLETE'
        nl_json(out / 'run_manifest.json', self.run_metadata)
        # Written LAST. A partial result directory is never a completed experiment.
        nl_json(out / 'accuracy_propagation_summary.json', summary)
        print(f'[Nonlinear COMPLETE] {out / "accuracy_propagation_summary.json"}')


def get_or_build_ssm_luts(m, s_dt, s_b, s_u, name_prefix, output_dir, device):
    if NL is None:
        return _n3_get_or_build_ssm_luts(m, s_dt, s_b, s_u, name_prefix, output_dir, device)
    return NL.coefficients(m, s_dt, s_b, s_u, name_prefix, output_dir, device)


def simulate_tile_dual_int8(net, patch_float, input_scale, output_dir, device):
    if NL is None:
        return _single_backend_tile(net, patch_float, input_scale, output_dir, device)
    return NL.run_tile(net, patch_float, input_scale, output_dir, device)


def nl_load_dependencies(module_name):
    import importlib
    import inspect
    global MambaHSI, data_load_operate, Evaluator
    global fuse_qat_model_bns_for_deploy, prepare_qat_model, validate_fpga_qat_contract, selective_scan_fn, NL_QAT_SOURCE
    MambaHSI = importlib.import_module('model.MambaHSI').MambaHSI
    data_load_operate = importlib.import_module('utils.data_load_operate')
    Evaluator = importlib.import_module('utils.evaluation').Evaluator
    qat = importlib.import_module(module_name)
    inspect.signature(qat.prepare_qat_model).bind(object(), nbit=8)
    prepare_qat_model = nl_make_v5_preparer(qat)
    fuse_qat_model_bns_for_deploy = qat.fuse_qat_model_bns_for_deploy
    validate_fpga_qat_contract = qat.validate_fpga_qat_contract
    NL_QAT_SOURCE = dict(module=module_name, path=str(qat.__file__), sha256=_sha256_file(Path(qat.__file__)))
    try:
        selective_scan_fn = importlib.import_module('selective_scan_interface').selective_scan_fn
    except ImportError:
        selective_scan_fn = importlib.import_module('mamba_ssm.ops.selective_scan_interface').selective_scan_fn
    print(f'[Script] {NL_SCRIPT_VERSION}')
    print(f'[QAT module] {qat.__file__}; SHA256={NL_QAT_SOURCE["sha256"]}; '
          'v5 fixed 25/19/Q24 contract; constructor DT9 adapter; original validation retained')


def nl_self_test(rtl_data_dir=None):
    # No model, checkpoint or CUDA selective_scan is imported for this test.
    for method in ('n0', 'n1'):
        assert nl_exp_neg(0, method) == 1 << 30
        assert nl_exp_neg(32 << 24, method) == 0
        for q in range(-128, 128):
            a, k, _ = nl_online(q, round(.25 * NL_Q), [1 << 24] * 16,
                                 round(.001 * (1 << 40)), 24, method)
            assert len(a) == 16 and all(0 <= v <= 1 << 24 for v in a)
            assert 0 <= k < 1 << 19
    for name, c in NL_N2_CONFIG['cores'].items():
        expected = None
        if rtl_data_dir:
            expected = [int(s, 16) for s in (Path(rtl_data_dir) / f'{name}_n2.mem').read_text().split()]
            assert len(expected) == 256
        for index, q in enumerate(range(-128, 128)):
            a, k, _ = nl_online(q, c['SDT_Q30'], c['DECAY_Q24'], c['SBSU_Q40'], c['K_fraction_bits'], 'n2')
            assert len(a) == 16 and all(0 <= v <= 1 << 24 for v in a)
            assert 0 <= k < 1 << 19
            if expected is not None:
                actual = sum(v << (25 * state) for state, v in enumerate(a)) | (k << 400)
                assert actual == expected[index], (name, q, 'N2 reference mismatch')
        print(f'SELF TEST N2 {name}: 256 codes; RTL vectors '
              + ('EXACT' if expected is not None else 'NOT_CHECKED (provide --rtl-data-dir)'))
    a = torch.tensor([[[1 << 24]]], dtype=torch.int64)
    h = torch.tensor([[[0]]], dtype=torch.int64)
    for uval, expected in [(1, 1), (-1, -1)]:
        state, _, _ = nl_recurrence_step(a, torch.tensor([[1]]), torch.tensor([[uval]]),
                                         torch.tensor([[1]]), torch.tensor([[1]]), h, 1)
        assert state.item() == expected * (1 << 23)
    pos = torch.tensor([[(1 << 19)-1]])
    _, _, saturated = nl_recurrence_step(a, pos, torch.tensor([[127]]), torch.tensor([[127]]),
                                        torch.tensor([[1]]), h, 24)
    assert saturated == 1
    stats = nl_finish_stats(nl_stats(torch.tensor([1, 2]), torch.tensor([1, 3])))
    assert stats['mismatches'] == 1 and stats['MAE'] == .5
    print('SELF TEST PASS: arithmetic ranges, recurrence signed rounding/saturation, statistics. '
          'Not a checkpoint inference or RTL simulation PASS.')



def parse_args():
    parser = argparse.ArgumentParser(
        description="严格对齐当前QAT的16x16 BothMamba FPGA INT8模拟"
    )
    parser.add_argument(
        "--qat-run-dir",
        required=False,
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
        "--verbose-every-tile",
        action="store_true",
        help="打印每个tile的逐层定点诊断；默认只打印并导出第一个tile",
    )
    parser.add_argument(
        "--qat-reference-batch-size",
        type=int,
        default=0,
        help=(
            "复算QAT保存预测时使用的eval batch；0表示读取result.json，"
            "旧run缺字段时回退为历史默认值8"
        ),
    )
    parser.add_argument('--nonlinear-backend', choices=['all', 'n0', 'n1', 'n2', 'n3'], default='all',
                        help='Always run N3 baseline; all adds N0/N1/N2 on identical tiles; n2 adds only N2')
    parser.add_argument('--trace-tiles', type=int, default=5,
                        help='First N real tiles for detailed SSM state/replay traces; -1=all; 0=off')
    parser.add_argument('--qat-module', default='both_QAT_21patch_aligned',
                        help='Original v5 QAT module, without .py; must match the checkpoint')
    parser.add_argument('--reference-lut-dir', help='Optional v5 first_tile_layers/ssm_luts; enforce exact N3 LUTs')
    parser.add_argument('--rtl-data-dir', help='Optional nonlinear_compare/data; enforce exact per-method RTL vectors')
    parser.add_argument('--expected-labels', help='Optional saved v5 int8_hw_prediction.npy; enforce full-scene N3 parity')
    parser.add_argument('--expected-real5-logits', help='Optional real5_expected_logits.npy; enforce N3 first-five parity')
    parser.add_argument('--expected-input-sha256', help='Optional SHA256 of quantized tile,y,x,channel bytes')
    parser.add_argument('--self-test', action='store_true', help='Small local arithmetic tests, no model import')
    args = parser.parse_args()
    if not args.self_test and not args.qat_run_dir:
        parser.error('--qat-run-dir is required for inference')
    if args.trace_tiles < -1:
        parser.error('--trace-tiles must be -1 or nonnegative')
    return args


def main():
    global NL
    args = parse_args()
    if args.self_test:
        nl_self_test(args.rtl_data_dir)
        return
    nl_load_dependencies(args.qat_module)
    NL = NonlinearExperiment(args)
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
    )


if __name__ == "__main__":
    main()
