"""Configuration-driven INT8 LSQ QAT for MambaHSI architecture variants.

The configurable model implementation is embedded in this file so server-side
changes to another Python module cannot silently alter the QAT graph. Presets
and explicit CLI overrides cover branch/fusion/skip choices, z and D paths,
A sharing, normalization, activation, head width, spectral token count and
SSM state size. The saved FP32 ``model_config.json`` is checked before any
checkpoint is loaded.

Supported data contract:
    exact saved 16x16 raw-pixel-disjoint train/validation/test blocks,
    exact saved train-only PCA/min-max, and exact saved per-seed label indices.

The generated checkpoint uses scalar per-tensor weight/activation scales,
UINT8 patch input, INT9 dt input, INT8 B/C/dt output, explicit branch/fusion/
block boundaries, optional z/D quantizers, and signed-INT32 accumulator bias.
Only the ``current`` preset is accepted by the existing fixed RTL simulator;
other variants are valid QAT ablations but require matching simulator/RTL.
"""

import argparse
from ssm_error_ablation import SSMNumericConfig
import itertools
import json
import logging
import math
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import scipy.io as sio
except ImportError:  # Report only when a MATLAB dataset is actually opened.
    sio = None

try:
    import h5py
except ImportError:  # Only needed for MATLAB v7.3 files.
    h5py = None

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn as _optimized_selective_scan_fn,
    )
except ImportError:
    _optimized_selective_scan_fn = None


DATASET_CONFIGS = {
    "UP": {
        "class_count": 9,
        "data_file": "UP/PaviaU.mat",
        "label_file": "UP/PaviaU_gt.mat",
        "data_keys": ("paviaU", "PaviaU"),
        "label_keys": ("paviaU_gt", "PaviaU_gt"),
    },
    "HanChuan": {
        "class_count": 16,
        "data_file": "HanChuan/WHU_Hi_HanChuan.mat",
        "label_file": "HanChuan/WHU_Hi_HanChuan_gt.mat",
        "data_keys": ("WHU_Hi_HanChuan",),
        "label_keys": ("WHU_Hi_HanChuan_gt",),
    },
    "HongHu": {
        "class_count": 22,
        "data_file": "HongHu/WHU_Hi_HongHu.mat",
        "label_file": "HongHu/WHU_Hi_HongHu_gt.mat",
        "data_keys": ("WHU_Hi_HongHu",),
        "label_keys": ("WHU_Hi_HongHu_gt",),
    },
    "Houston": {
        "class_count": 15,
        "data_file": "Houston/Houston.mat",
        "label_file": "Houston/Houston_GT.mat",
        "data_keys": ("Houston", "Houston_data"),
        "label_keys": ("Houston_GT", "Houston_gt"),
    },
}
DATASET_ORDER = tuple(DATASET_CONFIGS)

MODEL_VARIANT_PRESETS = {
    "current": {
        "hidden_dim": 32,
        "branch_mode": "both",
        "fusion_mode": "sum",
        "skip_scale": 2,
        "use_z": False,
        "use_D": False,
        "A_mode": "shared",
        "norm_path": "bn",
        "activation": "relu",
        "head_dim": 64,
        "token_num": 4,
        "d_state": 16,
    },
    "original_safe": {
        "hidden_dim": 64,
        "branch_mode": "both",
        "fusion_mode": "softmax",
        "skip_scale": 2,
        "use_z": True,
        "use_D": True,
        "A_mode": "per_channel",
        "norm_path": "gn",
        "activation": "silu",
        "head_dim": 128,
        "token_num": 4,
        "d_state": 16,
    },
    "original": {
        "hidden_dim": 64,
        "branch_mode": "both",
        "fusion_mode": "softmax",
        "skip_scale": 2,
        "use_z": True,
        "use_D": True,
        "A_mode": "per_channel",
        "norm_path": "gn",
        "activation": "silu",
        "head_dim": 128,
        "token_num": 4,
        "d_state": 16,
    },
    "original_safe_matched": {
        "hidden_dim": 32,
        "branch_mode": "both",
        "fusion_mode": "softmax",
        "skip_scale": 2,
        "use_z": True,
        "use_D": True,
        "A_mode": "per_channel",
        "norm_path": "gn",
        "activation": "silu",
        "head_dim": 64,
        "token_num": 4,
        "d_state": 16,
    },
    "custom": {
        "hidden_dim": 32,
        "branch_mode": "both",
        "fusion_mode": "sum",
        "skip_scale": 2,
        "use_z": False,
        "use_D": False,
        "A_mode": "shared",
        "norm_path": "bn",
        "activation": "relu",
        "head_dim": 64,
        "token_num": 4,
        "d_state": 16,
    },
}
MODEL_VARIANT_NAMES = tuple(MODEL_VARIANT_PRESETS)
MODEL_CONFIG_FIELDS = tuple(MODEL_VARIANT_PRESETS["current"])
MODEL_FIXED_CONFIG = {
    "block_count": 3,
    "downsample_count": 2,
    "d_conv": 4,
    "expand": 2,
    "group_num": 4,
    "linear_bias": False,
    "conv_bias": True,
    "branch_residual": False,
    "spa_batch_safe": True,
    "dt_bias_placement": "dt_proj_output_once",
}


def resolve_model_config(model_variant="current", **overrides):
    if model_variant not in MODEL_VARIANT_PRESETS:
        raise ValueError(
            f"unknown model_variant={model_variant!r}; "
            f"choose one of {MODEL_VARIANT_NAMES}"
        )
    unknown = sorted(set(overrides) - set(MODEL_CONFIG_FIELDS))
    if unknown:
        raise ValueError(f"unknown model configuration fields: {unknown}")
    config = dict(MODEL_VARIANT_PRESETS[model_variant])
    for name, value in overrides.items():
        if value is not None:
            config[name] = value
    if config["branch_mode"] not in {"spa", "spe", "both"}:
        raise ValueError("branch_mode must be spa, spe or both")
    if config["fusion_mode"] not in {"sum", "mean", "softmax"}:
        raise ValueError("fusion_mode must be sum, mean or softmax")
    if config["branch_mode"] != "both" and config["fusion_mode"] != "sum":
        raise ValueError("single-branch models require fusion_mode=sum")
    if config["skip_scale"] not in {0, 1, 2}:
        raise ValueError("skip_scale must be 0, 1 or 2")
    if config["A_mode"] not in {"shared", "per_channel"}:
        raise ValueError("A_mode must be shared or per_channel")
    if config["norm_path"] not in {"bn", "gn"}:
        raise ValueError("norm_path must be bn or gn")
    if config["activation"] not in {"relu", "silu"}:
        raise ValueError("activation must be relu or silu")
    if config["head_dim"] not in {32, 64, 128}:
        raise ValueError("head_dim must be 32, 64 or 128")
    for name in ("hidden_dim", "token_num", "d_state"):
        if int(config[name]) <= 0:
            raise ValueError(f"{name} must be greater than zero")
        config[name] = int(config[name])
    config["skip_scale"] = int(config["skip_scale"])
    config["use_z"] = bool(config["use_z"])
    config["use_D"] = bool(config["use_D"])
    return {"model_variant": model_variant, **config}

DEFAULT_MODEL_CONFIG = {
    **resolve_model_config("current"),
    **MODEL_FIXED_CONFIG,
}
# Backward-compatible exported name. Runtime code uses args.model_config and
# never mutates this dictionary.
MODEL_CONFIG = dict(DEFAULT_MODEL_CONFIG)
SPLIT_TARGET_RATIO = (0.5, 0.2, 0.3)


def model_config_slug(config):
    """Use the exact same architecture directory key as FP32 training."""
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


FP32_CONFIG_SLUG = model_config_slug(DEFAULT_MODEL_CONFIG)


def current_fpga_simulator_compatible(config):
    """The checked-in integer simulator/RTL currently implements this only."""
    return all(
        config.get(key) == value
        for key, value in DEFAULT_MODEL_CONFIG.items()
    )


def d_path_simulator_compatible(config):
    return all(config.get(key) == value for key, value in DEFAULT_MODEL_CONFIG.items()
               if key != 'use_D') and isinstance(config.get('use_D'), bool)


def resolve_ssm_contract(value=None):
    contract = {'u_quantization': 'legacy', 'd_weight_bits': 8}
    if value is not None:
        if set(value) - set(contract):
            raise ValueError('Unknown QAT SSM contract fields')
        contract.update(value)
    if contract['u_quantization'] not in ('legacy', 'shared'):
        raise ValueError('u_quantization must be legacy or shared')
    if contract['d_weight_bits'] not in (8, 12, 16):
        raise ValueError('d_weight_bits must be 8, 12 or 16')
    return contract


def parse_seed_list(text):
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds cannot be empty")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds contains duplicates")
    return seeds


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("boolean values must be true or false")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Configuration-driven INT8 LSQ QAT for MambaHSI"
    )
    parser.add_argument("--dataset", choices=[*DATASET_ORDER, "all"], default="UP")
    parser.add_argument("--data_set_path", default="./data")
    parser.add_argument("--work_dir", default="./results")
    parser.add_argument("--exp_name", default="SPATIAL_SPLIT_3WAY_DENSE_QAT")
    parser.add_argument("--run_tag", default="int8_lsq_current_single")

    parser.add_argument("--fp32_dir", default="", help="Exact FP32 dataset/config directory")
    parser.add_argument("--fp32_work_dir", default="", help="Defaults to --work_dir")
    parser.add_argument("--fp32_exp_name", default="SPATIAL_SPLIT_3WAY_DENSE")
    parser.add_argument(
        "--fp32_run_tag",
        default="all_samples_sqrt_inverse_clip3_2000_nobias",
    )
    parser.add_argument(
        "--fp32_config_slug",
        default="",
        help=(
            "FP32 architecture subdirectory. Empty derives it from the resolved "
            "model configuration; normally do not set this manually."
        ),
    )
    parser.add_argument(
        "--pretrained_template",
        default="",
        help="Optional checkpoint template containing {dataset} and {seed}",
    )

    # Kept so the user's existing command still works.  These values are
    # validated against saved FP32 artifacts; the split is never regenerated.
    parser.add_argument("--tile_size", type=int, default=16)
    parser.add_argument("--fallback_tile_size", type=int, default=0)
    parser.add_argument("--split_strategy", choices=["blocks"], default="blocks")
    parser.add_argument("--split_block_size", type=int, default=16)
    parser.add_argument("--train_samples", type=int, default=2000)
    parser.add_argument("--min_train_samples", type=int, default=100)
    parser.add_argument("--pca_components", type=int, default=16)
    parser.add_argument(
        "--class_weight",
        choices=["none", "inverse", "sqrt_inverse"],
        default="sqrt_inverse",
    )

    parser.add_argument(
        "--model_variant",
        "--model-variant",
        choices=MODEL_VARIANT_NAMES,
        default="current",
    )
    parser.add_argument("--hidden_dim", "--hidden-dim", type=int, default=None)
    parser.add_argument(
        "--branch_mode",
        "--branch-mode",
        choices=["spa", "spe", "both"],
        default=None,
    )
    parser.add_argument(
        "--fusion_mode",
        "--fusion-mode",
        choices=["sum", "mean", "softmax"],
        default=None,
    )
    parser.add_argument(
        "--skip_scale",
        "--skip-scale",
        type=int,
        choices=[0, 1, 2],
        default=None,
    )
    parser.add_argument(
        "--use_z",
        "--use-z",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
    )
    parser.add_argument(
        "--use_D",
        "--use-D",
        dest="use_D",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
    )
    parser.add_argument(
        "--A_mode",
        "--a-mode",
        dest="A_mode",
        choices=["shared", "per_channel"],
        default=None,
    )
    parser.add_argument(
        "--norm_path",
        "--norm-path",
        choices=["bn", "gn"],
        default=None,
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "silu"],
        default=None,
    )
    parser.add_argument(
        "--head_dim",
        "--head-dim",
        type=int,
        choices=[32, 64, 128],
        default=None,
    )
    parser.add_argument("--token_num", "--token-num", type=int, default=None)
    parser.add_argument("--d_state", "--d-state", type=int, default=None)
    parser.add_argument(
        "--output_config_subdir",
        "--output-config-subdir",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Place each architecture in its own collision-safe QAT subdirectory.",
    )
    parser.add_argument("--dt-input-bits", type=int, choices=[8, 9, 10], default=9,
                        help="LSQ bit width at the low-rank dt_proj input")
    parser.add_argument("--dt-output-bits", type=int, choices=[6, 7, 8, 9, 10], default=8,
                        help="LSQ time-step output/address width; requires matching simulator format")
    parser.add_argument("--nbit", type=int, choices=[8], default=8)
    parser.add_argument('--ssm-u-quantization', choices=['legacy', 'shared'], default='shared',
                        help='New runs share quantized U across x_proj, recurrence and D; legacy reproduces old runs')
    parser.add_argument('--d-weight-bits', type=int, choices=[8, 12, 16], default=8)
    parser.add_argument('--d-init-policy', choices=['mean', 'max'], default='max',
                        help='D-only LSQ initialization; independent of ordinary weight initialization')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument(
        "--grad_clip",
        dest="grad_clip_norm",
        type=float,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--calibration_steps", type=int, default=100)
    parser.add_argument("--activation_scale_lr_multiplier", type=float, default=1.0)
    parser.add_argument("--dt_scale_lr_multiplier", type=float, default=1.0)
    parser.add_argument("--weight_scale_lr_multiplier", type=float, default=1.0)
    parser.add_argument("--d_scale_lr_multiplier", type=float, default=1.0)
    parser.add_argument("--freeze_bn_epoch", type=int, default=20)
    parser.add_argument("--freeze_lsq_epoch", type=int, default=-1,
                        help="Freeze learned activation/weight/D scales from this epoch; -1 keeps learning")
    parser.add_argument("--include_calibrated_baseline", action="store_true",
                        help="Allow epoch-0 calibrated QAT to compete on validation for best checkpoint")
    parser.add_argument("--training_diagnostics", action="store_true",
                        help="Record parameter/scale drift and quantization statistics on a fixed validation batch")
    parser.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument('--diagnose_initial_quantization', action='store_true',
                        help='Evaluate initial quantization policies on saved validation labels')
    parser.add_argument('--initial_diagnostics_only', action='store_true',
                        help='Stop after calibration/diagnosis; no training or test evaluation')
    parser.add_argument('--weight_init_validation_only', action='store_true',
                        help='Validate mean, patch-only max, all ordinary-weight max; no training/test evaluation')
    parser.add_argument('--weight_init_policy', choices=['mean','patch_max','all_weight_max'], default='mean',
                        help='After normal calibration, replace selected ordinary weight scales without recalibrating activations')
    parser.add_argument('--export_calibrated_only', action='store_true',
                        help='Export epoch0 plus matching test predictions for FPGA simulation; zero optimizer steps')
    parser.add_argument('--capture_oa_drop_pp', type=float, default=0.,
                        help='Save adjacent validation OA-drop checkpoint pairs; percentage points, 0 disables')
    parser.add_argument('--capture_max_events', type=int, default=3)
    parser.add_argument('--capture_start_epoch', type=int, default=25)
    parser.add_argument('--freeze_bn_affine', action='store_true',
                        help='Freeze BN gamma/beta from epoch 1; independent of running statistics')
    parser.add_argument('--weight_lr_multiplier', type=float, default=1.,
                        help='Ordinary parameter LR multiplier; excludes LSQ scales, D and BN affine')
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument(
        "--selection_metric", choices=["OA", "mAcc"], default="mAcc"
    )
    parser.add_argument(
        "--verify_bn_fold",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("0"))
    parser.add_argument("--device", default="cuda:0")
    parser.set_defaults(grad_clip_norm=1.0)
    return parser


def resolve_model_arguments(args):
    """Resolve one preset plus explicit overrides into the runtime contract."""
    overrides = {
        name: getattr(args, name)
        for name in MODEL_CONFIG_FIELDS
    }
    resolved = resolve_model_config(args.model_variant, **overrides)
    for name in MODEL_CONFIG_FIELDS:
        setattr(args, name, resolved[name])
    args.model_config = {
        **resolved,
        **MODEL_FIXED_CONFIG,
    }
    derived_slug = model_config_slug(args.model_config)
    if not args.fp32_config_slug:
        args.fp32_config_slug = derived_slug
    elif args.fp32_config_slug != derived_slug:
        raise ValueError(
            "--fp32_config_slug does not match the resolved architecture: "
            f"given={args.fp32_config_slug!r}, expected={derived_slug!r}. "
            "Remove the option to derive it automatically."
        )
    args.model_config_slug = derived_slug
    return args


def validate_args(args):
    fixed = {
        "tile_size": 16,
        "fallback_tile_size": 0,
        "split_strategy": "blocks",
        "split_block_size": 16,
        "train_samples": 2000,
        "min_train_samples": 100,
        "pca_components": 16,
    }
    errors = [
        f"--{name} must be {expected!r}, got {getattr(args, name)!r}"
        for name, expected in fixed.items()
        if getattr(args, name) != expected
    ]
    if errors:
        raise ValueError(
            "QAT must reuse the fixed 16x16 spatial/data protocol:\n"
            + "\n".join(errors)
        )
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("lr must be positive and weight_decay non-negative")
    if args.max_epoch <= 0 or args.eval_interval <= 0:
        raise ValueError("max_epoch/eval_interval must be positive")
    if args.grad_clip_norm < 0 or args.calibration_steps <= 0:
        raise ValueError("grad_clip_norm must be non-negative; calibration_steps positive")
    if any(getattr(args, key) <= 0 for key in (
            'activation_scale_lr_multiplier', 'dt_scale_lr_multiplier',
            'weight_scale_lr_multiplier', 'd_scale_lr_multiplier')):
        raise ValueError("LSQ learning-rate multipliers must be positive")
    if args.freeze_lsq_epoch < -1 or not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("freeze_lsq_epoch must be >= -1; min_lr_ratio must be in [0,1]")
    if (not math.isfinite(args.capture_oa_drop_pp) or args.capture_oa_drop_pp < 0
            or args.capture_max_events <= 0 or args.capture_start_epoch < 1
            or not math.isfinite(args.weight_lr_multiplier) or args.weight_lr_multiplier <= 0):
        raise ValueError('Invalid capture settings or weight_lr_multiplier')
    if args.capture_oa_drop_pp > 0 and args.eval_interval != 1:
        raise ValueError('Jump capture requires --eval_interval 1 for adjacent epochs')
    if args.initial_diagnostics_only:
        args.diagnose_initial_quantization = True
    if args.weight_init_validation_only and args.diagnose_initial_quantization:
        raise ValueError('Run weight-init comparison separately from initial quantization policy diagnostics')
    if (args.weight_init_validation_only or args.diagnose_initial_quantization) and (
            args.weight_init_policy != 'mean' or args.export_calibrated_only):
        raise ValueError('Run diagnostic-only policies separately from max-initialized training/export')
    if args.export_calibrated_only:
        args.max_epoch = 0
        args.include_calibrated_baseline = True
    return args


def resolve_device(text):
    if text.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(text)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_logger(path, name):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
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

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=convert)


def plot_loss(losses, path):
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(np.arange(1, len(losses) + 1), losses, color="tab:blue")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Masked dense cross-entropy")
    axis.set_title("16x16 dense INT8 LSQ QAT")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Saved FP32 artifact and dataset handling
# ---------------------------------------------------------------------------


def _read_mat_arrays(path):
    """Read ordinary or v7.3 MATLAB files without project-local helpers."""
    if sio is None:
        raise ImportError(
            "Reading MATLAB HSI datasets requires scipy. Install scipy in the "
            "Python environment used to launch this QAT script."
        )
    try:
        content = sio.loadmat(path)
        return {
            key: np.asarray(value)
            for key, value in content.items()
            if not key.startswith("__") and isinstance(value, np.ndarray)
        }
    except (NotImplementedError, ValueError, OSError):
        if h5py is None:
            raise RuntimeError(
                f"{path} is probably MATLAB v7.3; install h5py in this environment"
            )
        arrays = {}
        with h5py.File(path, "r") as handle:
            for key, value in handle.items():
                if isinstance(value, h5py.Dataset):
                    arrays[key] = np.asarray(value)
        return arrays


def _choose_mat_array(arrays, preferred_keys, ndim, path):
    for key in preferred_keys:
        if key in arrays and np.squeeze(arrays[key]).ndim == ndim:
            return np.asarray(arrays[key])
    candidates = [
        np.asarray(value)
        for value in arrays.values()
        if np.squeeze(value).ndim == ndim and np.issubdtype(value.dtype, np.number)
    ]
    if not candidates:
        raise KeyError(
            f"No numeric {ndim}D array found in {path}; keys={sorted(arrays)}"
        )
    return max(candidates, key=lambda value: value.size)


def _align_scene_axes(data, gt, dataset_name):
    data = np.squeeze(np.asarray(data))
    gt = np.squeeze(np.asarray(gt))
    if data.ndim != 3 or gt.ndim != 2:
        raise ValueError(f"{dataset_name}: data={data.shape}, gt={gt.shape}")
    for transpose_gt in (False, True):
        candidate_gt = gt.T if transpose_gt else gt
        for permutation in itertools.permutations(range(3)):
            candidate_data = data.transpose(permutation)
            if candidate_data.shape[:2] == candidate_gt.shape:
                return (
                    candidate_data.astype(np.float32, copy=False),
                    candidate_gt.astype(np.int64, copy=False),
                )
    raise ValueError(
        f"{dataset_name}: cannot align data={data.shape} with gt={gt.shape}"
    )


def load_dataset(dataset_name, data_root):
    config = DATASET_CONFIGS[dataset_name]
    root = Path(data_root).expanduser()
    candidates = [
        (root / config["data_file"], root / config["label_file"]),
        (root / Path(config["data_file"]).name, root / Path(config["label_file"]).name),
    ]
    # Prefer the established data-root layout, then accept a dataset directory.
    # Resolve the image and labels as a pair, never from different directories.
    for data_path, label_path in candidates:
        if data_path.is_file() and label_path.is_file():
            break
    else:
        attempted = "\n".join(f"  {data}\n  {label}" for data, label in candidates)
        raise FileNotFoundError(
            f"Missing {dataset_name} dataset file pair. Tried:\n{attempted}\n"
            "--data-path/--data_set_path accepts a data root or the dataset directory."
        )
    data = _choose_mat_array(
        _read_mat_arrays(data_path), config["data_keys"], 3, data_path
    )
    gt = _choose_mat_array(
        _read_mat_arrays(label_path), config["label_keys"], 2, label_path
    )
    data, gt = _align_scene_axes(data, gt, dataset_name)
    if not np.isfinite(data).all():
        raise ValueError(f"{dataset_name}: raw image contains NaN/Inf")
    foreground = np.unique(gt)
    foreground = foreground[foreground > 0]
    expected = np.arange(1, config["class_count"] + 1)
    if not np.array_equal(foreground, expected):
        raise ValueError(
            f"{dataset_name}: labels must be consecutive 1..C, got {foreground.tolist()}"
        )
    return data, gt, config["class_count"]


def resolve_fp32_dataset_dir(args, dataset_name):
    """Prefer the new config-slug layout, with one explicit legacy fallback."""
    if args.fp32_dir:
        direct = Path(args.fp32_dir).expanduser()
        if "{dataset}" in str(direct):
            direct = Path(str(direct).format(dataset=dataset_name))
        candidates = [direct]
    else:
        root = Path(args.fp32_work_dir or args.work_dir)
        base = root / args.fp32_exp_name / f"{dataset_name}_{args.fp32_run_tag}"
        candidates = [base / args.fp32_config_slug, base]

    required = ("train_only_preprocess.npz", "spatial_split_masks.npz")
    valid = [
        candidate
        for candidate in candidates
        if candidate.is_dir()
        and all((candidate / filename).exists() for filename in required)
    ]
    if not valid:
        attempted = "\n".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            "Could not locate an exact FP32 artifact directory. Tried:\n"
            f"{attempted}\nUse --fp32_dir to specify it directly."
        )
    return valid[0].resolve()


def transform_with_saved_preprocess(raw_data, preprocess_path):
    saved = np.load(preprocess_path)
    required = {
        "pca_mean",
        "pca_components",
        "explained_variance",
        "channel_min",
        "channel_max",
    }
    missing = sorted(required.difference(saved.files))
    if missing:
        raise KeyError(f"{preprocess_path} missing {missing}")
    mean = saved["pca_mean"].astype(np.float64)
    components = saved["pca_components"].astype(np.float64)
    variance = saved["explained_variance"].astype(np.float64)
    channel_min = saved["channel_min"].astype(np.float64)
    channel_max = saved["channel_max"].astype(np.float64)
    height, width, bands = raw_data.shape
    if mean.shape != (bands,) or components.shape[1] != bands:
        raise ValueError(
            f"Preprocess/data mismatch: bands={bands}, mean={mean.shape}, "
            f"components={components.shape}"
        )
    flat = raw_data.reshape(-1, bands).astype(np.float64, copy=False)
    transformed = (flat - mean) @ components.T
    transformed /= np.sqrt(np.maximum(variance, 1e-12))
    channel_range = np.maximum(channel_max - channel_min, 1e-12)
    normalized = np.clip(
        (transformed - channel_min) / channel_range, 0.0, 1.0
    )
    return normalized.reshape(height, width, -1).astype(np.float32)


def load_spatial_masks(fp32_dir, scene_shape):
    saved = np.load(fp32_dir / "spatial_split_masks.npz")
    masks = []
    for key in ("train_region", "val_region", "test_region"):
        if key not in saved.files:
            raise KeyError(f"spatial_split_masks.npz missing {key}")
        value = np.asarray(saved[key], dtype=bool)
        if value.shape != tuple(scene_shape):
            raise ValueError(f"{key}={value.shape}, expected={scene_shape}")
        masks.append(value)
    train_region, val_region, test_region = masks
    if np.any(train_region & val_region) or np.any(train_region & test_region):
        raise AssertionError("saved train region overlaps validation/test")
    if np.any(val_region & test_region):
        raise AssertionError("saved validation/test regions overlap")
    if not np.all(train_region | val_region | test_region):
        raise AssertionError("saved spatial split does not partition every raw pixel")
    return tuple(masks)


def make_tiles_from_saved_masks(masks, tile_size):
    """Rebuild the saved block list without rerunning stochastic splitting."""
    height, width = masks[0].shape
    tiles = [[] for _ in masks]
    for top in range(0, height, tile_size):
        bottom = min(top + tile_size, height)
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            ownership = [
                bool(mask[top:bottom, left:right].all()) for mask in masks
            ]
            if sum(ownership) != 1:
                fractions = [
                    float(mask[top:bottom, left:right].mean()) for mask in masks
                ]
                raise ValueError(
                    "Saved split is not aligned to the fixed 16x16 grid at "
                    f"top={top}, left={left}, fractions={fractions}."
                )
            split_index = ownership.index(True)
            tiles[split_index].append((top, bottom, left, right, split_index))
    return tuple(tiles)


def load_seed_label_maps(fp32_dir, seed, gt, masks, class_count, train_cap):
    path = fp32_dir / f"run_seed{seed}" / "sample_indices.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing exact FP32 sample indices: {path}")
    saved = np.load(path)
    keys = ("train_indices", "val_indices", "test_indices")
    if any(key not in saved.files for key in keys):
        raise KeyError(f"{path} must contain {keys}")
    indices = tuple(np.asarray(saved[key], dtype=np.int64) for key in keys)
    flat_gt = gt.reshape(-1)
    flat_masks = tuple(mask.reshape(-1) for mask in masks)
    label_maps = []
    for split_name, split_indices, region in zip(keys, indices, flat_masks):
        if split_indices.ndim != 1 or np.unique(split_indices).size != split_indices.size:
            raise ValueError(f"{path}: {split_name} is not a unique 1D index list")
        if np.any(split_indices < 0) or np.any(split_indices >= flat_gt.size):
            raise ValueError(f"{path}: {split_name} contains out-of-range indices")
        if np.any(~region[split_indices]) or np.any(flat_gt[split_indices] <= 0):
            raise ValueError(f"{path}: {split_name} leaves its saved spatial region")
        label_map = np.full(flat_gt.size, -1, dtype=np.int64)
        label_map[split_indices] = flat_gt[split_indices] - 1
        label_maps.append(label_map.reshape(gt.shape))

    train_counts = np.bincount(
        label_maps[0][label_maps[0] >= 0], minlength=class_count
    )
    available = np.asarray(
        [np.count_nonzero(masks[0] & (gt == label)) for label in range(1, class_count + 1)]
    )
    expected = np.minimum(available, train_cap)
    if not np.array_equal(train_counts, expected):
        raise ValueError(
            f"seed={seed}: saved train counts violate cap={train_cap}; "
            f"expected={expected.tolist()}, got={train_counts.tolist()}"
        )
    for split_index in (1, 2):
        expected_indices = np.flatnonzero(
            flat_masks[split_index] & (flat_gt > 0)
        )
        if not np.array_equal(np.sort(indices[split_index]), expected_indices):
            raise ValueError(
                f"seed={seed}: saved {keys[split_index]} is not the complete labeled split"
            )
    return (*label_maps, *indices)


def resolve_pretrained_path(args, fp32_dir, dataset_name, seed):
    if args.pretrained_template:
        return Path(
            args.pretrained_template.format(dataset=dataset_name, seed=seed)
        ).expanduser()
    return fp32_dir / f"run_seed{seed}" / "best_model.pth"


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a state_dict or contain state_dict")
    if checkpoint and all(str(key).startswith("module.") for key in checkpoint):
        checkpoint = {
            str(key)[len("module.") :]: value for key, value in checkpoint.items()
        }
    return checkpoint


def validate_saved_model_config(fp32_dir, seeds, expected_config):
    """Reject an ablation checkpoint before strict loading gives a vague error."""
    for seed in seeds:
        path = fp32_dir / f"run_seed{seed}" / "model_config.json"
        if not path.exists():
            if expected_config != DEFAULT_MODEL_CONFIG:
                raise FileNotFoundError(
                    f"{path} is required for a non-current architecture. "
                    "Refusing to infer a variant from tensor shapes alone."
                )
            continue  # Legacy current artifacts are still strict-loaded below.
        with path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
        missing = [key for key in MODEL_CONFIG_FIELDS if key not in saved]
        if missing:
            raise ValueError(f"{path} is missing architecture fields: {missing}")
        mismatches = {
            key: {"expected": value, "saved": saved.get(key)}
            for key, value in expected_config.items()
            if key in saved and saved.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"{path} does not match the requested model: {mismatches}"
            )


class DenseTileDataset(Dataset):
    def __init__(self, image, label_map, tiles, tile_size, require_label):
        self.image = np.asarray(image, dtype=np.float32)
        self.label_map = np.asarray(label_map, dtype=np.int64)
        self.tile_size = int(tile_size)
        self.tiles = [
            tile
            for tile in tiles
            if not require_label
            or np.any(self.label_map[tile[0] : tile[1], tile[2] : tile[3]] >= 0)
        ]

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        top, bottom, left, right, split_id = self.tiles[index]
        image = self.image[top:bottom, left:right]
        labels = self.label_map[top:bottom, left:right]
        valid_height, valid_width = image.shape[:2]
        pad_height = self.tile_size - valid_height
        pad_width = self.tile_size - valid_width
        if pad_height or pad_width:
            mode = "reflect" if min(valid_height, valid_width) > 1 else "edge"
            image = np.pad(
                image,
                ((0, pad_height), (0, pad_width), (0, 0)),
                mode=mode,
            )
            labels = np.pad(
                labels,
                ((0, pad_height), (0, pad_width)),
                mode="constant",
                constant_values=-1,
            )
        return (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(labels)),
            top,
            left,
            valid_height,
            valid_width,
            split_id,
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


# ---------------------------------------------------------------------------
# Embedded configurable MambaHSI (checkpoint-compatible with FP32 training)
# ---------------------------------------------------------------------------


_REFERENCE_WARNING_EMITTED = False


def selective_scan_backend():
    return (
        "mamba_ssm_optimized"
        if _optimized_selective_scan_fn is not None
        else "torch_reference"
    )


def _reference_selective_scan(u, delta, A, B, C, D=None):
    global _REFERENCE_WARNING_EMITTED
    if not _REFERENCE_WARNING_EMITTED:
        warnings.warn(
            "Using slow PyTorch selective scan: input is not on CUDA "
            "or the mamba_ssm CUDA extension is unavailable",
            RuntimeWarning,
        )
        _REFERENCE_WARNING_EMITTED = True
    delta = F.softplus(delta)
    batch_size, inner_dim, sequence_length = u.shape
    state_dim = A.shape[-1]
    state = u.new_zeros(batch_size, inner_dim, state_dim)
    A = A.to(dtype=u.dtype, device=u.device)
    outputs = []
    for index in range(sequence_length):
        delta_t = delta[:, :, index].unsqueeze(-1)
        input_t = u[:, :, index].unsqueeze(-1)
        B_t = B[:, :, index].to(u.dtype).unsqueeze(1)
        C_t = C[:, :, index].to(u.dtype).unsqueeze(1)
        dA = torch.exp(delta_t * A.unsqueeze(0))
        state = state * dA + input_t * (delta_t * B_t)
        output_t = torch.sum(state * C_t, dim=-1)
        if D is not None:
            output_t = output_t + u[:, :, index] * D.to(u.dtype).view(1, -1)
        outputs.append(output_t)
    return torch.stack(outputs, dim=-1)


def run_selective_scan(u, delta, A, B, C, D=None):
    # Importability alone does not make the CUDA kernel valid for CPU input.
    if u.is_cuda and _optimized_selective_scan_fn is not None:
        return _optimized_selective_scan_fn(
            u,
            delta,
            A,
            B,
            C,
            D=D,
            z=None,
            delta_bias=None,
            delta_softplus=True,
            return_last_state=False,
        )
    return _reference_selective_scan(u, delta, A, B, C, D=D)


def make_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation: {name}")


def valid_group_count(channels, requested_groups):
    groups = min(int(requested_groups), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def make_spatial_norm(norm_path, channels, group_num):
    if norm_path == "bn":
        return nn.BatchNorm2d(channels)
    if norm_path == "gn":
        return nn.GroupNorm(valid_group_count(channels, group_num), channels)
    raise ValueError(f"unsupported norm_path: {norm_path}")


class SharedSoftParams(nn.Module):
    """Unused but retained for strict compatibility with current checkpoints."""

    def __init__(self, sharpness=3.0, clip_value=7.0):
        super().__init__()
        self.n_segments = 3
        self.sharpness = float(sharpness)
        self.clip_value = float(clip_value)
        self.a = nn.Parameter(torch.tensor([-0.05, 0.2, 1.0]))
        self.b = nn.Parameter(torch.tensor([-0.35, 0.0, 0.0]))
        self.t = nn.Parameter(torch.tensor([-1.4, 0.0]))
        self.register_buffer(
            "fixed_ends", torch.tensor([-clip_value, clip_value], dtype=torch.float32)
        )


class CurrentMambaCore(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        use_z=False,
        use_D=False,
        A_mode="shared",
        norm_path="bn",
        activation="relu",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.expand * self.d_model
        self.dt_rank = math.ceil(self.d_model / 16)
        self.use_fast_path = False
        self.use_z = bool(use_z)
        self.use_D = bool(use_D)
        self.A_mode = A_mode
        self.norm_path = norm_path
        self.activation = activation

        projection_factor = 2 if self.use_z else 1
        self.in_proj = nn.Linear(
            self.d_model,
            self.d_inner * projection_factor,
            bias=False,
        )
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=True,
        )
        self.act = make_activation(activation)
        self.z_act = nn.SiLU()
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * self.d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inverse_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus_dt)
        self.dt_proj.bias._no_reinit = True

        base_A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        if self.A_mode == "shared":
            self.A_log_shared = nn.Parameter(torch.log(base_A))
            self.A_log_shared._no_weight_decay = True
        elif self.A_mode == "per_channel":
            expanded_A = base_A.unsqueeze(0).repeat(self.d_inner, 1)
            self.A_log = nn.Parameter(torch.log(expanded_A))
            self.A_log._no_weight_decay = True
        else:
            raise ValueError("A_mode must be shared or per_channel")
        if self.use_D:
            self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))
            self.D._no_weight_decay = True
        else:
            self.register_buffer("D", torch.zeros(self.d_inner, dtype=torch.float32))

        self.out_proj_linear = nn.Linear(self.d_inner, self.d_model, bias=False)
        if norm_path == "bn":
            self.out_proj_bn = nn.BatchNorm1d(self.d_model)
        elif norm_path == "gn":
            self.out_proj_bn = nn.Identity()
        else:
            raise ValueError("norm_path must be bn or gn")

    def continuous_A(self):
        if self.A_mode == "shared":
            A_log = self.A_log_shared.unsqueeze(0).expand(self.d_inner, -1)
        else:
            A_log = self.A_log
        return -torch.exp(A_log.float()).contiguous()

    def forward(self, hidden_states):
        if hidden_states.ndim != 3:
            raise ValueError("Mamba input must be [batch, length, channels]")
        batch_size, sequence_length, _ = hidden_states.shape
        projected = self.in_proj(hidden_states).transpose(1, 2).contiguous()
        if self.use_z:
            x, z = projected.chunk(2, dim=1)
        else:
            x = projected
            z = None
        x = self.act(self.conv1d(x)[..., :sequence_length])

        flat = x.transpose(1, 2).reshape(-1, self.d_inner)
        if getattr(self, '_qat_ssm_contract', {}).get('u_quantization') == 'shared':
            flat, s_u = self.x_proj.lsq_a(flat)
            x = flat.reshape(batch_size, sequence_length, self.d_inner).transpose(1, 2).contiguous()
            x_dbl = self.x_proj(flat, scale_x=s_u, input_is_quantized=True)
        else:
            x_dbl = self.x_proj(flat)
        dt_low_rank, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        if hasattr(self, "b_output_quant"):
            B = self.b_output_quant(B)
        if hasattr(self, "c_output_quant"):
            C = self.c_output_quant(C)

        # dt_proj already adds its bias.  selective_scan receives delta_bias=None,
        # so the bias is added exactly once in both FP32 and QAT.
        dt = self.dt_proj(dt_low_rank)
        if hasattr(self, "dt_output_quant"):
            dt = self.dt_output_quant(dt)
        dt = dt.reshape(batch_size, sequence_length, self.d_inner)
        dt = dt.transpose(1, 2).contiguous()
        B = B.reshape(batch_size, sequence_length, self.d_state)
        B = B.transpose(1, 2).contiguous()
        C = C.reshape(batch_size, sequence_length, self.d_state)
        C = C.transpose(1, 2).contiguous()

        D = self.D.float() if self.use_D else None
        if D is not None and hasattr(self, "d_weight_quant"):
            D = self.d_weight_quant(D)
        y = run_selective_scan(x, dt, self.continuous_A(), B, C, D=D)
        if z is not None:
            z = self.z_act(z)
            if hasattr(self, "z_gate_quant"):
                z = self.z_gate_quant(z)
            y = y * z
            if hasattr(self, "z_mul_output_quant"):
                y = self.z_mul_output_quant(y)
        y = y.transpose(1, 2).contiguous()
        output = self.out_proj_linear(y)
        output = output.transpose(1, 2).contiguous()
        output = self.out_proj_bn(output)
        return output.transpose(1, 2).contiguous()


class SpaMamba(nn.Module):
    def __init__(
        self,
        channels,
        d_state,
        use_z,
        use_D,
        A_mode,
        norm_path,
        activation,
        group_num=4,
    ):
        super().__init__()
        self.use_residual = False
        self.use_proj = True
        self.mamba = CurrentMambaCore(
            channels,
            d_state=d_state,
            use_z=use_z,
            use_D=use_D,
            A_mode=A_mode,
            norm_path=norm_path,
            activation=activation,
        )
        projection = []
        if norm_path == "gn":
            projection.append(
                nn.GroupNorm(valid_group_count(channels, group_num), channels)
            )
        projection.append(make_activation(activation))
        self.proj = nn.Sequential(*projection)

    def forward(self, x):
        batch, channels, height, width = x.shape
        sequence = x.permute(0, 2, 3, 1).contiguous()
        sequence = sequence.reshape(batch, height * width, channels)
        output = self.mamba(sequence)
        output = output.reshape(batch, height, width, channels)
        output = output.permute(0, 3, 1, 2).contiguous()
        return self.proj(output)


class SpeMamba(nn.Module):
    def __init__(
        self,
        channels,
        token_num,
        d_state,
        use_z,
        use_D,
        A_mode,
        norm_path,
        activation,
        group_num=4,
    ):
        super().__init__()
        self.use_residual = False
        self.use_proj = True
        self.input_channels = int(channels)
        self.token_num = int(token_num)
        self.group_channel_num = math.ceil(channels / self.token_num)
        self.channel_num = self.token_num * self.group_channel_num
        self.padded_channels = self.channel_num
        self.mamba = CurrentMambaCore(
            self.group_channel_num,
            d_state=d_state,
            use_z=use_z,
            use_D=use_D,
            A_mode=A_mode,
            norm_path=norm_path,
            activation=activation,
        )
        projection = []
        if norm_path == "gn":
            projection.append(
                nn.GroupNorm(
                    valid_group_count(self.channel_num, group_num),
                    self.channel_num,
                )
            )
        projection.append(make_activation(activation))
        self.proj = nn.Sequential(*projection)

    def forward(self, x):
        batch, channels, height, width = x.shape
        if channels != self.input_channels:
            raise ValueError(
                f"SpeMamba expected {self.input_channels} channels, got {channels}"
            )
        if channels < self.channel_num:
            padding = x.new_zeros(
                batch, self.channel_num - channels, height, width
            )
            x = torch.cat([x, padding], dim=1)
        sequence = x.permute(0, 2, 3, 1).contiguous()
        sequence = sequence.reshape(
            batch * height * width, self.token_num, self.group_channel_num
        )
        output = self.mamba(sequence)
        output = output.reshape(batch, height, width, self.channel_num)
        output = output.permute(0, 3, 1, 2).contiguous()
        return self.proj(output)[:, :channels]


class BothMamba(nn.Module):
    def __init__(
        self,
        channels,
        token_num,
        d_state,
        branch_mode,
        fusion_mode,
        skip_scale,
        use_z,
        use_D,
        A_mode,
        norm_path,
        activation,
        group_num=4,
    ):
        super().__init__()
        self.branch_mode = branch_mode
        self.fusion_mode = fusion_mode
        self.use_att = branch_mode == "both" and fusion_mode == "softmax"
        self.use_residual = True
        self.skip_scale = float(skip_scale)
        common = {
            "channels": channels,
            "d_state": d_state,
            "use_z": use_z,
            "use_D": use_D,
            "A_mode": A_mode,
            "norm_path": norm_path,
            "activation": activation,
            "group_num": group_num,
        }
        self.spa_mamba = (
            SpaMamba(**common) if branch_mode in {"spa", "both"} else None
        )
        self.spe_mamba = (
            SpeMamba(token_num=token_num, **common)
            if branch_mode in {"spe", "both"}
            else None
        )
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        if self.branch_mode == "spa":
            fused = self.spa_mamba(x)
            if hasattr(self, "spa_residual_quant"):
                fused = self.spa_residual_quant(fused)
        elif self.branch_mode == "spe":
            fused = self.spe_mamba(x)
            if hasattr(self, "spe_residual_quant"):
                fused = self.spe_residual_quant(fused)
        else:
            spa = self.spa_mamba(x)
            spe = self.spe_mamba(x)
            if hasattr(self, "spa_residual_quant"):
                spa = self.spa_residual_quant(spa)
            if hasattr(self, "spe_residual_quant"):
                spe = self.spe_residual_quant(spe)
            if self.fusion_mode == "sum":
                fused = spa + spe
            elif self.fusion_mode == "mean":
                fused = 0.5 * (spa + spe)
            elif self.fusion_mode == "softmax":
                weights = self.softmax(self.weights)
                if hasattr(self, "fusion_weight_quant"):
                    weights = self.fusion_weight_quant(weights)
                fused = weights[0] * spa + weights[1] * spe
            else:
                raise RuntimeError(f"unsupported fusion mode: {self.fusion_mode}")
        if hasattr(self, "fusion_quant"):
            fused = self.fusion_quant(fused)
        output = fused + self.skip_scale * x
        if hasattr(self, "block_output_quant"):
            output = self.block_output_quant(output)
        return output


class MambaHSI(nn.Module):
    def __init__(
        self,
        in_channels=16,
        hidden_dim=32,
        num_classes=9,
        branch_mode="both",
        fusion_mode="sum",
        skip_scale=2,
        use_z=False,
        use_D=False,
        A_mode="shared",
        norm_path="bn",
        activation="relu",
        head_dim=64,
        token_num=4,
        d_state=16,
        group_num=4,
    ):
        super().__init__()
        if branch_mode not in {"spa", "spe", "both"}:
            raise ValueError("branch_mode must be spa, spe or both")
        if fusion_mode not in {"sum", "mean", "softmax"}:
            raise ValueError("fusion_mode must be sum, mean or softmax")
        if branch_mode != "both" and fusion_mode != "sum":
            raise ValueError("single-branch models require fusion_mode=sum")
        self.mamba_type = branch_mode
        self.branch_mode = branch_mode
        self.fusion_mode = fusion_mode
        self.skip_scale = int(skip_scale)
        self.token_num = int(token_num)
        self.d_state = int(d_state)
        self.shared_params = SharedSoftParams()
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            make_spatial_norm(norm_path, hidden_dim, group_num),
            make_activation(activation),
        )
        block_kwargs = {
            "channels": hidden_dim,
            "token_num": token_num,
            "d_state": d_state,
            "branch_mode": branch_mode,
            "fusion_mode": fusion_mode,
            "skip_scale": skip_scale,
            "use_z": use_z,
            "use_D": use_D,
            "A_mode": A_mode,
            "norm_path": norm_path,
            "activation": activation,
            "group_num": group_num,
        }
        self.mamba = nn.Sequential(
            BothMamba(**block_kwargs),
            nn.AvgPool2d(kernel_size=2, stride=2),
            BothMamba(**block_kwargs),
            nn.AvgPool2d(kernel_size=2, stride=2),
            BothMamba(**block_kwargs),
        )
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, head_dim, kernel_size=1),
            make_spatial_norm(norm_path, head_dim, group_num),
            make_activation(activation),
            nn.Conv2d(head_dim, num_classes, kernel_size=1),
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("MambaHSI input must be [B,C,H,W]")
        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError("input height and width must be divisible by 4")
        x = self.patch_embedding(x)
        if hasattr(self, "patch_output_quant"):
            x = self.patch_output_quant(x)
        x = self.mamba(x)
        x = self.cls_head(x)
        if hasattr(self, "logits_quant"):
            x = self.logits_quant(x)
        return x


def build_configured_model(in_channels, class_count, config):
    model = MambaHSI(
        in_channels=in_channels,
        hidden_dim=config["hidden_dim"],
        num_classes=class_count,
        branch_mode=config["branch_mode"],
        fusion_mode=config["fusion_mode"],
        skip_scale=config["skip_scale"],
        use_z=config["use_z"],
        use_D=config["use_D"],
        A_mode=config["A_mode"],
        norm_path=config["norm_path"],
        activation=config["activation"],
        head_dim=config["head_dim"],
        token_num=config["token_num"],
        d_state=config["d_state"],
        group_num=config["group_num"],
    )
    model._resolved_model_config = dict(config)
    return model


def active_block_branches(block):
    branches = []
    if block.spa_mamba is not None:
        branches.append(("spa", block.spa_mamba))
    if block.spe_mamba is not None:
        branches.append(("spe", block.spe_mamba))
    return branches


def infer_model_config(model):
    """Recover the effective graph contract for API callers such as simulator."""
    if hasattr(model, "_resolved_model_config"):
        return dict(model._resolved_model_config)
    blocks = [layer for layer in model.mamba if isinstance(layer, BothMamba)]
    if not blocks:
        raise ValueError("model contains no configurable Mamba blocks")
    block = blocks[0]
    branches = active_block_branches(block)
    if not branches:
        raise ValueError("Mamba block contains no active branch")
    core = branches[0][1].mamba
    activation = "silu" if isinstance(core.act, nn.SiLU) else "relu"
    norm_path = (
        "gn"
        if any(isinstance(module, nn.GroupNorm) for module in model.modules())
        else "bn"
    )
    config = {
        "model_variant": "runtime",
        "hidden_dim": int(model.patch_embedding[0].out_channels),
        "branch_mode": block.branch_mode,
        "fusion_mode": block.fusion_mode,
        "skip_scale": int(block.skip_scale),
        "use_z": bool(core.use_z),
        "use_D": bool(core.use_D),
        "A_mode": core.A_mode,
        "norm_path": norm_path,
        "activation": activation,
        "head_dim": int(model.cls_head[0].out_channels),
        "token_num": int(model.token_num),
        "d_state": int(model.d_state),
        **MODEL_FIXED_CONFIG,
    }
    current_without_variant = {
        key: value
        for key, value in DEFAULT_MODEL_CONFIG.items()
        if key != "model_variant"
    }
    if all(config.get(key) == value for key, value in current_without_variant.items()):
        config["model_variant"] = "current"
    return config


def validate_fp32_model_contract(model, config):
    actual = infer_model_config(model)
    mismatches = {
        key: {"expected": config[key], "actual": actual.get(key)}
        for key in MODEL_CONFIG_FIELDS
        if actual.get(key) != config[key]
    }
    if mismatches:
        raise ValueError(f"constructed FP32 model contract mismatch: {mismatches}")
    blocks = [layer for layer in model.mamba if isinstance(layer, BothMamba)]
    if len(blocks) != config["block_count"]:
        raise ValueError(
            f"expected {config['block_count']} Mamba blocks, got {len(blocks)}"
        )
    for block_index, block in enumerate(blocks):
        if len(active_block_branches(block)) != (
            2 if config["branch_mode"] == "both" else 1
        ):
            raise ValueError(f"block{block_index} branch topology mismatch")
    return True


# ---------------------------------------------------------------------------
# Compact FPGA-aligned per-tensor LSQ implementation
# ---------------------------------------------------------------------------


class _LsqRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        del ctx
        return torch.sign(value) * torch.floor(torch.abs(value) + 0.5)

    @staticmethod
    def backward(ctx, gradient):
        del ctx
        return gradient.clone()


class _LsqQuantize(torch.autograd.Function):
    """Original LSQ forward/backward, restricted to FPGA per-tensor scale."""

    @staticmethod
    def forward(ctx, value, scale, gradient_scale, q_min, q_max):
        ctx.save_for_backward(value, scale)
        ctx.contract = gradient_scale, q_min, q_max
        integer = _LsqRound.apply((value / scale).clamp(q_min, q_max))
        return integer * scale

    @staticmethod
    def backward(ctx, gradient):
        value, scale = ctx.saved_tensors
        gradient_scale, q_min, q_max = ctx.contract
        normalized = value / scale
        smaller = (normalized < q_min).to(gradient.dtype)
        larger = (normalized > q_max).to(gradient.dtype)
        between = 1.0 - smaller - larger
        scale_gradient = (
            smaller * q_min
            + larger * q_max
            + between * _LsqRound.apply(normalized)
            - between * normalized
        ) * gradient * gradient_scale
        return (
            between * gradient,
            scale_gradient.sum().reshape(scale.shape),
            None,
            None,
            None,
        )


class LsqQuantizer4input(nn.Module):
    def __init__(
        self,
        bit,
        all_positive=False,
        per_channel=False,
        per_channel_num=None,
        weight_next=False,
        batch_init=100,
        init_strategy="mean",
    ):
        super().__init__()
        del per_channel_num, weight_next
        if per_channel:
            raise ValueError("FPGA deployment supports activation per-tensor only")
        self.a_bits = int(bit)
        self.all_positive = bool(all_positive)
        self.per_channel = False
        self.batch_init = int(batch_init)
        self.init_strategy = init_strategy
        self.Qn = 0 if self.all_positive else -(2 ** (self.a_bits - 1))
        self.Qp = (
            2**self.a_bits - 1
            if self.all_positive
            else 2 ** (self.a_bits - 1) - 1
        )
        self.s = nn.Parameter(torch.ones(1))
        self.init_state = 0
        self.g = 1.0

    @torch.no_grad()
    def _initialize_or_update(self, activation):
        if self.init_strategy == "max":
            current = activation.detach().abs().max() / max(self.Qp, 1)
        else:
            current = (
                activation.detach().abs().mean()
                * 2.0
                / math.sqrt(max(self.Qp, 1))
            )
        current = current.clamp(min=1e-8)
        if self.init_state == 0:
            self.s.fill_(current)
        else:
            self.s.copy_(0.9 * self.s + 0.1 * current)
        self.init_state += 1

    def forward(self, activation):
        if self.a_bits == 32:
            return activation, self.s[0]
        if self.init_state == 0:
            self.g = 1.0 / math.sqrt(max(activation.numel() * self.Qp, 1))
        if self.init_state < self.batch_init:
            self._initialize_or_update(activation)
        quantized = _LsqQuantize.apply(
            activation,
            self.s,
            self.g,
            self.Qn,
            self.Qp,
        )
        return quantized, self.s[0]


class LsqQuantizer4weight(nn.Module):
    def __init__(
        self,
        bit,
        all_positive=False,
        per_channel=False,
        per_channel_num=None,
        batch_init=100,
        init_strategy="mean",
    ):
        super().__init__()
        del per_channel_num
        if per_channel:
            raise ValueError("FPGA deployment supports weight per-tensor only")
        self.bit = int(bit)
        self.all_positive = bool(all_positive)
        self.per_channel = False
        self.batch_init = int(batch_init)
        if init_strategy not in ("mean", "max"):
            raise ValueError("weight init_strategy must be mean or max")
        self.init_strategy = init_strategy
        self.Qn = 0 if self.all_positive else -(2 ** (self.bit - 1))
        self.Qp = 2**self.bit - 1 if self.all_positive else 2 ** (self.bit - 1) - 1
        self.s = nn.Parameter(torch.ones(1))
        self.init_state = 0
        self.g = 1.0

    @torch.no_grad()
    def _initialize_or_update(self, weight):
        if self.init_strategy == "max":
            # D is concentrated near one: use its full signed code range.
            # For signed D8 this is exactly max(abs(D)) / 127.
            current = weight.detach().abs().max() / max(self.Qp, 1)
        else:
            current = (
                weight.detach().abs().mean()
                * 2.0
                / math.sqrt(max(self.Qp, 1))
            )
        current = current.clamp(min=1e-8)
        if self.init_state == 0:
            self.s.fill_(current)
        else:
            self.s.copy_(0.9 * self.s + 0.1 * current)
        self.init_state += 1

    def forward(self, weight):
        if self.bit == 32:
            return weight, self.s
        if self.init_state == 0:
            self.g = 1.0 / math.sqrt(max(weight.numel() * self.Qp, 1))
        if self.init_state < self.batch_init:
            self._initialize_or_update(weight)
        quantized = _LsqQuantize.apply(
            weight,
            self.s,
            self.g,
            self.Qn,
            self.Qp,
        )
        return quantized, self.s


BIAS_ACCUMULATOR_MIN = -(1 << 31)
BIAS_ACCUMULATOR_MAX = (1 << 31) - 1


class _SignedInt32BiasQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, scale):
        ctx.scale = scale
        return torch.clamp(
            torch.round(bias / scale),
            BIAS_ACCUMULATOR_MIN,
            BIAS_ACCUMULATOR_MAX,
        )

    @staticmethod
    def backward(ctx, gradient):
        return gradient.clone() / ctx.scale, None


def _quantize_accumulator_bias(bias, scale_x, weight_scale):
    if bias is None:
        return None
    if scale_x is None:
        return bias
    activation_scale = scale_x.reshape(-1)
    weight_scale = weight_scale.reshape(-1)
    if activation_scale.numel() != 1 or weight_scale.numel() != 1:
        raise RuntimeError("FPGA bias quantization requires scalar per-tensor scales")
    accumulator_scale = (activation_scale[0] * weight_scale[0]).detach()
    if not torch.isfinite(accumulator_scale) or accumulator_scale <= 0:
        raise FloatingPointError("invalid INT32 accumulator scale")
    # The original signed-INT32 bias path uses torch.round (ties-to-even),
    # unlike LSQ's half-away-from-zero rounding.
    integer = _SignedInt32BiasQuant.apply(bias, accumulator_scale)
    return integer * accumulator_scale


class QuanConv(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        nbit_w=8,
        nbit_a=8,
        stride=1,
        padding=0,
        norm=False,
        act=False,
        dilation=1,
        groups=1,
        bias=True,
        quan_input=True,
        input_per_channel=False,
    ):
        del input_per_channel
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.nbit_w = int(nbit_w)
        self.nbit_a = int(nbit_a)
        self.norm = bool(norm)
        self.act = bool(act)
        self.quan_input = bool(quan_input)
        self.lsq_w = LsqQuantizer4weight(self.nbit_w)
        self.quan_w = self.lsq_w
        if self.quan_input:
            self.lsq_a = LsqQuantizer4input(self.nbit_a)
            self.quan_a = self.lsq_a
        if self.norm:
            self.bn_weight = nn.Parameter(torch.ones(out_channels))
            self.bn_bias = nn.Parameter(torch.zeros(out_channels))
            self.register_buffer("bn_running_mean", torch.zeros(out_channels))
            self.register_buffer("bn_running_var", torch.ones(out_channels))
            self.momentum = 0.1
            self.bn_eps = 1e-5

    def forward(self, x, scale_x=None):
        if self.quan_input:
            x, scale_x = self.lsq_a(x)
        if self.norm:
            bn_scale = self.bn_weight / torch.sqrt(
                self.bn_running_var + self.bn_eps
            )
            folded_weight = self.weight * bn_scale.view(-1, 1, 1, 1)
            weight_q, weight_scale = self.lsq_w(folded_weight)
            if not self.training:
                source_bias = (
                    self.bias
                    if self.bias is not None
                    else torch.zeros_like(self.bn_running_mean)
                )
                fused_bias = (
                    (source_bias - self.bn_running_mean) * bn_scale + self.bn_bias
                )
                bias_q = _quantize_accumulator_bias(
                    fused_bias, scale_x, weight_scale
                )
                return F.conv2d(
                    x,
                    weight_q,
                    bias_q,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
            safe_scale = torch.where(
                bn_scale.abs() < 1e-8,
                torch.full_like(bn_scale, 1e-8),
                bn_scale,
            )
            unfolded_weight_q = weight_q / safe_scale.view(-1, 1, 1, 1)
            output = F.conv2d(
                x,
                unfolded_weight_q,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            return F.batch_norm(
                output,
                self.bn_running_mean,
                self.bn_running_var,
                self.bn_weight,
                self.bn_bias,
                True,
                self.momentum,
                self.bn_eps,
            )
        weight_q, weight_scale = self.lsq_w(self.weight)
        bias_q = _quantize_accumulator_bias(self.bias, scale_x, weight_scale)
        return F.conv2d(
            x,
            weight_q,
            bias_q,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuanLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        quan_input=True,
        nbit_w=8,
        nbit_a=8,
        norm=False,
        per_channel_w=False,
        init_strategy="mean",
    ):
        del per_channel_w
        super().__init__(in_features, out_features, bias)
        self.nbit_w = int(nbit_w)
        self.nbit_a = int(nbit_a)
        self.quan_input = bool(quan_input)
        self.norm = bool(norm)
        self.lsq_w = LsqQuantizer4weight(self.nbit_w)
        self.quan_w = self.lsq_w
        if self.quan_input:
            self.lsq_a = LsqQuantizer4input(
                self.nbit_a, init_strategy=init_strategy
            )
            self.quan_a = self.lsq_a
        if self.norm:
            self.bn_weight = nn.Parameter(torch.ones(out_features))
            self.bn_bias = nn.Parameter(torch.zeros(out_features))
            self.register_buffer("bn_running_mean", torch.zeros(out_features))
            self.register_buffer("bn_running_var", torch.ones(out_features))
            self.momentum = 0.1
            self.bn_eps = 1e-5

    def forward(self, x, scale_x=None, input_is_quantized=False):
        if input_is_quantized and scale_x is None:
            raise ValueError("Already quantized input requires its physical scale")
        if self.quan_input and not input_is_quantized:
            x, scale_x = self.lsq_a(x)
        if self.norm:
            bn_scale = self.bn_weight / torch.sqrt(
                self.bn_running_var + self.bn_eps
            )
            folded_weight = self.weight * bn_scale.view(-1, 1)
            weight_q, weight_scale = self.lsq_w(folded_weight)
            if not self.training:
                source_bias = (
                    self.bias
                    if self.bias is not None
                    else torch.zeros_like(self.bn_running_mean)
                )
                fused_bias = (
                    (source_bias - self.bn_running_mean) * bn_scale + self.bn_bias
                )
                bias_q = _quantize_accumulator_bias(
                    fused_bias, scale_x, weight_scale
                )
                return F.linear(x, weight_q, bias_q)
            safe_scale = torch.where(
                bn_scale.abs() < 1e-8,
                torch.full_like(bn_scale, 1e-8),
                bn_scale,
            )
            unfolded_weight_q = weight_q / safe_scale.view(-1, 1)
            output = F.linear(x, unfolded_weight_q, self.bias)
            if output.ndim == 2:
                return F.batch_norm(
                    output,
                    self.bn_running_mean,
                    self.bn_running_var,
                    self.bn_weight,
                    self.bn_bias,
                    True,
                    self.momentum,
                    self.bn_eps,
                )
            output = output.movedim(-1, 1)
            output = F.batch_norm(
                output,
                self.bn_running_mean,
                self.bn_running_var,
                self.bn_weight,
                self.bn_bias,
                True,
                self.momentum,
                self.bn_eps,
            )
            return output.movedim(1, -1)
        weight_q, weight_scale = self.lsq_w(self.weight)
        bias_q = _quantize_accumulator_bias(self.bias, scale_x, weight_scale)
        return F.linear(x, weight_q, bias_q)


class QuanConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        nbit_w=8,
        nbit_a=8,
        quan_input=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.nbit_w = int(nbit_w)
        self.nbit_a = int(nbit_a)
        self.quan_input = bool(quan_input)
        self.lsq_w = LsqQuantizer4weight(self.nbit_w)
        if self.quan_input:
            self.lsq_a = LsqQuantizer4input(self.nbit_a)

    def forward(self, x):
        scale_x = None
        if self.quan_input:
            x, scale_x = self.lsq_a(x)
        weight_q, weight_scale = self.lsq_w(self.weight)
        bias_q = _quantize_accumulator_bias(self.bias, scale_x, weight_scale)
        return F.conv1d(
            x,
            weight_q,
            bias_q,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ActivationFakeQuant(nn.Module):
    def __init__(self, bit=8, signed=True, init_strategy="mean"):
        super().__init__()
        self.bit = int(bit)
        self.signed = bool(signed)
        self.lsq_a = LsqQuantizer4input(
            bit=bit,
            all_positive=not signed,
            init_strategy=init_strategy,
        )

    def forward(self, x):
        return self.lsq_a(x)[0]


class WeightFakeQuant(nn.Module):
    """Per-tensor LSQ wrapper for standalone parameters such as SSM D."""

    def __init__(self, bit=8, all_positive=False, init_strategy="mean"):
        super().__init__()
        self.lsq_w = LsqQuantizer4weight(
            bit=bit,
            all_positive=all_positive,
            init_strategy=init_strategy,
        )

    def forward(self, weight):
        return self.lsq_w(weight)[0]


def _copy_bn_into_quant_layer(target, source):
    target.bn_weight.data.copy_(source.weight.data)
    target.bn_bias.data.copy_(source.bias.data)
    target.bn_running_mean.copy_(source.running_mean)
    target.bn_running_var.copy_(source.running_var)
    target.momentum = source.momentum if source.momentum is not None else 0.1
    target.bn_eps = source.eps


def _prepare_qat_children(module, nbit, prefix="", dt_input_bits=9):
    children = list(module.named_children())
    for index, (name, child) in enumerate(children):
        full_name = f"{prefix}.{name}" if prefix else name
        next_name = children[index + 1][0] if index + 1 < len(children) else None
        next_child = children[index + 1][1] if index + 1 < len(children) else None

        if isinstance(child, nn.Linear):
            has_bn = isinstance(next_child, nn.BatchNorm1d)
            is_dt_proj = full_name.endswith(".dt_proj")
            replacement = QuanLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                nbit_w=nbit,
                nbit_a=dt_input_bits if is_dt_proj else nbit,
                norm=has_bn,
                init_strategy="max" if is_dt_proj else "mean",
            )
            replacement.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                replacement.bias.data.copy_(child.bias.data)
            if has_bn:
                _copy_bn_into_quant_layer(replacement, next_child)
                setattr(module, next_name, nn.Identity())
            setattr(module, name, replacement)
            continue

        if isinstance(child, nn.Conv1d):
            replacement = QuanConv1d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                nbit_w=nbit,
                nbit_a=nbit,
            )
            replacement.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                replacement.bias.data.copy_(child.bias.data)
            setattr(module, name, replacement)
            continue

        if isinstance(child, nn.Conv2d):
            has_bn = isinstance(next_child, nn.BatchNorm2d)
            replacement = QuanConv(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                nbit_w=nbit,
                nbit_a=nbit,
                norm=has_bn,
            )
            replacement.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                replacement.bias.data.copy_(child.bias.data)
            if full_name == "patch_embedding.0":
                replacement.lsq_a = LsqQuantizer4input(
                    bit=8,
                    all_positive=True,
                    init_strategy="max",
                )
                replacement.quan_a = replacement.lsq_a
            if has_bn:
                _copy_bn_into_quant_layer(replacement, next_child)
                setattr(module, next_name, nn.Identity())
            setattr(module, name, replacement)
            continue

        _prepare_qat_children(child, nbit, prefix=full_name, dt_input_bits=dt_input_bits)


def add_fpga_fake_quant_nodes(model, nbit=8, dt_output_bits=8, d_weight_bits=8):
    model.patch_output_quant = ActivationFakeQuant(nbit, signed=True)
    model.logits_quant = ActivationFakeQuant(nbit, signed=True)
    for layer in model.mamba:
        if not isinstance(layer, BothMamba):
            continue
        layer.register_buffer(
            "_fpga_skip_scale", torch.tensor(layer.skip_scale, dtype=torch.float32)
        )
        if layer.spa_mamba is not None:
            layer.spa_residual_quant = ActivationFakeQuant(nbit, signed=True)
        if layer.spe_mamba is not None:
            layer.spe_residual_quant = ActivationFakeQuant(nbit, signed=True)
        layer.fusion_quant = ActivationFakeQuant(nbit, signed=True)
        layer.block_output_quant = ActivationFakeQuant(nbit, signed=True)
        if layer.branch_mode == "both" and layer.fusion_mode == "softmax":
            layer.fusion_weight_quant = ActivationFakeQuant(
                nbit,
                signed=False,
                init_strategy="max",
            )
        for _, branch in active_block_branches(layer):
            core = branch.mamba
            core.b_output_quant = ActivationFakeQuant(nbit, signed=True)
            core.c_output_quant = ActivationFakeQuant(nbit, signed=True)
            core.dt_output_quant = ActivationFakeQuant(
                dt_output_bits, signed=True, init_strategy="max"
            )
            if core.use_z:
                core.z_gate_quant = ActivationFakeQuant(
                    nbit,
                    signed=True,
                    init_strategy="max",
                )
                core.z_mul_output_quant = ActivationFakeQuant(nbit, signed=True)
            if core.use_D:
                core.d_weight_quant = WeightFakeQuant(
                    d_weight_bits, all_positive=False, init_strategy="max"
                )


def prepare_qat_model(model, nbit=8, model_config=None, numeric_config=None, ssm_contract=None):
    if hasattr(model, "patch_output_quant"):
        raise RuntimeError("prepare_qat_model was called twice")
    config = dict(model_config or infer_model_config(model))
    model._resolved_model_config = config
    model._qat_expected_fold_aware_bn = sum(
        isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
        for module in model.modules()
    )
    numeric = SSMNumericConfig.from_dict(numeric_config)
    model._ssm_numeric_config = numeric
    contract = resolve_ssm_contract(ssm_contract)
    model._qat_ssm_contract = contract
    _prepare_qat_children(model, nbit, dt_input_bits=numeric.dt_input_bits)
    add_fpga_fake_quant_nodes(model, nbit, dt_output_bits=numeric.dt_output_bits,
                             d_weight_bits=contract['d_weight_bits'])
    for module in model.modules():
        if isinstance(module, CurrentMambaCore):
            module._ssm_numeric_config = numeric
            module._qat_ssm_contract = contract
    validate_fpga_qat_contract(model, expect_bn=True, model_config=config)
    return model


def freeze_lsq_initialization(model):
    for module in model.modules():
        if isinstance(module, (LsqQuantizer4input, LsqQuantizer4weight)):
            module.init_state = max(module.init_state, module.batch_init)


def _fold_internal_qat_bn(module):
    if not isinstance(module, (QuanConv, QuanLinear)) or not module.norm:
        return False
    with torch.no_grad():
        source_bias = (
            module.bias
            if module.bias is not None
            else torch.zeros(
                module.weight.shape[0],
                dtype=module.weight.dtype,
                device=module.weight.device,
            )
        )
        bn_scale = module.bn_weight / torch.sqrt(
            module.bn_running_var + module.bn_eps
        )
        view_shape = [module.weight.shape[0]] + [1] * (module.weight.ndim - 1)
        fused_weight = module.weight * bn_scale.reshape(view_shape)
        fused_bias = (
            (source_bias - module.bn_running_mean) * bn_scale + module.bn_bias
        )
        module.weight.copy_(fused_weight)
        module.bias = nn.Parameter(fused_bias)
        module.norm = False
    return True


def fuse_qat_model_bns_for_deploy(model, validation_input=None):
    model.eval()
    freeze_lsq_initialization(model)
    before = None
    if validation_input is not None:
        with torch.no_grad():
            before = model(validation_input).detach()
    fused_names = []
    for name, module in model.named_modules():
        if _fold_internal_qat_bn(module):
            fused_names.append(name)
    model._bn_fused_for_deploy = True
    consistency = None
    if before is not None:
        with torch.no_grad():
            after = model(validation_input).detach()
        difference = (before.float() - after.float()).abs()
        consistency = {
            "allclose": bool(
                torch.allclose(before.float(), after.float(), rtol=1e-5, atol=1e-6)
            ),
            "cosine": float(
                F.cosine_similarity(
                    before.float().reshape(-1),
                    after.float().reshape(-1),
                    dim=0,
                ).item()
            ),
            "mae": float(difference.mean().item()),
            "max_abs_error": float(difference.max().item()),
        }
        model._bn_fold_consistency = consistency
        if not consistency["allclose"]:
            raise RuntimeError(f"BN folding consistency failed: {consistency}")
    return fused_names, consistency


def validate_fpga_qat_contract(model, expect_bn=True, model_config=None):
    config = dict(model_config or infer_model_config(model))
    numeric = getattr(model, "_ssm_numeric_config", SSMNumericConfig())
    external_bn = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
    ]
    if external_bn:
        raise RuntimeError(f"external BN remains after QAT conversion: {external_bn}")
    internal_bn = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (QuanConv, QuanLinear)) and module.norm
    ]
    expected = int(getattr(model, "_qat_expected_fold_aware_bn", 8))
    if expect_bn and len(internal_bn) != expected:
        raise RuntimeError(f"expected {expected} fold-aware BN, got {internal_bn}")
    if not expect_bn and internal_bn:
        raise RuntimeError(f"deploy model still contains fold-aware BN: {internal_bn}")
    quantizers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (LsqQuantizer4input, LsqQuantizer4weight))
    ]
    invalid = [
        name
        for name, module in quantizers
        if module.s.numel() != 1
        or not torch.isfinite(module.s.detach()).all()
        or module.s.detach().item() <= 0
    ]
    if invalid:
        raise RuntimeError(f"invalid per-tensor LSQ scales: {invalid}")
    patch_quant = model.patch_embedding[0].lsq_a
    if patch_quant.a_bits != 8 or not patch_quant.all_positive:
        raise RuntimeError("patch input must use UINT8 fake quantization")
    blocks = [layer for layer in model.mamba if isinstance(layer, BothMamba)]
    if len(blocks) != config["block_count"]:
        raise RuntimeError(
            f"expected {config['block_count']} Mamba blocks, got {len(blocks)}"
        )
    for block_index, block in enumerate(blocks):
        if block.branch_mode != config["branch_mode"]:
            raise RuntimeError(f"block{block_index} branch mode mismatch")
        if block.fusion_mode != config["fusion_mode"]:
            raise RuntimeError(f"block{block_index} fusion mode mismatch")
        if not math.isclose(
            float(block._fpga_skip_scale),
            float(config["skip_scale"]),
        ):
            raise RuntimeError(
                f"block{block_index} skip scale is not {config['skip_scale']}"
            )
        required_block_quantizers = ["fusion_quant", "block_output_quant"]
        if block.spa_mamba is not None:
            required_block_quantizers.append("spa_residual_quant")
        if block.spe_mamba is not None:
            required_block_quantizers.append("spe_residual_quant")
        if block.branch_mode == "both" and block.fusion_mode == "softmax":
            required_block_quantizers.append("fusion_weight_quant")
        for attr in required_block_quantizers:
            if not hasattr(block, attr):
                raise RuntimeError(f"block{block_index} missing {attr}")
        for branch_name, branch in active_block_branches(block):
            core = branch.mamba
            if not isinstance(core.dt_proj, QuanLinear) or core.dt_proj.nbit_a != numeric.dt_input_bits:
                raise RuntimeError(f"block{block_index}.{branch_name} dt input format mismatch")
            if core.dt_output_quant.lsq_a.a_bits != numeric.dt_output_bits:
                raise RuntimeError(f"block{block_index}.{branch_name} dt output format mismatch")
            if not isinstance(core.out_proj_linear, QuanLinear):
                raise RuntimeError(f"block{block_index}.{branch_name} out_proj not quantized")
            for attr in ("b_output_quant", "c_output_quant", "dt_output_quant"):
                if not hasattr(core, attr):
                    raise RuntimeError(f"block{block_index}.{branch_name} missing {attr}")
            if bool(core.use_z) != bool(config["use_z"]):
                raise RuntimeError(f"block{block_index}.{branch_name} z contract mismatch")
            if core.use_z:
                for attr in ("z_gate_quant", "z_mul_output_quant"):
                    if not hasattr(core, attr):
                        raise RuntimeError(
                            f"block{block_index}.{branch_name} missing {attr}"
                        )
            if bool(core.use_D) != bool(config["use_D"]):
                raise RuntimeError(f"block{block_index}.{branch_name} D contract mismatch")
            if core.use_D and not hasattr(core, "d_weight_quant"):
                raise RuntimeError(
                    f"block{block_index}.{branch_name} missing d_weight_quant"
                )
            if core.use_D:
                contract = getattr(model, '_qat_ssm_contract', resolve_ssm_contract())
                if core.D.shape != (core.d_inner,) or core.d_weight_quant.lsq_w.bit != contract['d_weight_bits']:
                    raise RuntimeError('D channel shape/quantization contract mismatch')
            if core.A_mode != config["A_mode"]:
                raise RuntimeError(f"block{block_index}.{branch_name} A mode mismatch")
    return True


def clamp_lsq_scales_positive(model, minimum=1e-8):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (LsqQuantizer4input, LsqQuantizer4weight)):
                module.s.clamp_(min=minimum)


def quantization_diagnostics(model):
    scales = {}
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, (LsqQuantizer4input, LsqQuantizer4weight)):
            scales[name] = float(module.s.detach().reshape(-1)[0].cpu())
        if isinstance(module, (QuanConv, QuanConv1d, QuanLinear)):
            scale = float(module.lsq_w.s.detach().reshape(-1)[0])
            qmin = -(2 ** (module.nbit_w - 1))
            qmax = 2 ** (module.nbit_w - 1) - 1
            raw = torch.round(module.weight.detach().float() / scale)
            clipped = raw.clamp(qmin, qmax)
            layers[name] = {
                "bit": module.nbit_w,
                "weight_scale": scale,
                "integer_min": int(clipped.min()),
                "integer_max": int(clipped.max()),
                "clip_ratio_percent": float(
                    ((raw < qmin) | (raw > qmax)).float().mean().item() * 100.0
                ),
            }
    return {"scales": scales, "layers": layers}


# ---------------------------------------------------------------------------
# Dense QAT training and evaluation
# ---------------------------------------------------------------------------


def dense_logits(model, image, tile_size):
    output = model(image)
    if output.ndim != 4:
        raise RuntimeError(f"model output must be BCHW, got {tuple(output.shape)}")
    return F.interpolate(
        output,
        size=(tile_size, tile_size),
        mode="bilinear",
        align_corners=True,
    )


def compute_class_weights(label_map, class_count, mode, device):
    valid = label_map[label_map >= 0]
    counts = np.bincount(valid, minlength=class_count).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"empty training class: {counts.tolist()}")
    if mode == "none":
        weights = np.ones(class_count, dtype=np.float64)
    elif mode == "inverse":
        weights = 1.0 / counts
    else:
        weights = 1.0 / np.sqrt(counts)
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device), counts


def cross_entropy_denominator(labels, class_weights):
    valid = labels[labels >= 0]
    if valid.numel() == 0:
        return 0.0
    if class_weights is None:
        return float(valid.numel())
    return float(class_weights[valid].sum().detach())


def bn_affine_parameters(model):
    result = []
    for module in model.modules():
        if isinstance(module, (QuanConv, QuanLinear)) and module.norm:
            result.extend((module.bn_weight, module.bn_bias))
        elif isinstance(module, nn.modules.batchnorm._BatchNorm) and module.affine:
            result.extend((module.weight, module.bias))
    return result


def freeze_bn_affine_parameters(model):
    for parameter in bn_affine_parameters(model):
        parameter.requires_grad_(False)
        parameter.grad = None


def make_qat_optimizer(model, args, logger):
    groups_by_name = {
        "dt_activation_scales": [],
        "activation_scales": [],
        "weight_scales": [],
        "d_weight_scales": [],
        "d_parameters": [],
        "weights_and_bn": [],
        "bn_affine": [],
    }
    bn_ids = {id(p) for p in bn_affine_parameters(model)}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith('.D') and getattr(parameter, '_no_weight_decay', False):
            groups_by_name['d_parameters'].append(parameter)
        elif '.d_weight_quant.lsq_w.s' in name:
            groups_by_name['d_weight_scales'].append(parameter)
        elif "dt_proj" in name and "lsq_a.s" in name:
            groups_by_name["dt_activation_scales"].append(parameter)
        elif "lsq_a.s" in name:
            groups_by_name["activation_scales"].append(parameter)
        elif "lsq_w.s" in name:
            groups_by_name["weight_scales"].append(parameter)
        elif id(parameter) in bn_ids:
            groups_by_name['bn_affine'].append(parameter)
        else:
            groups_by_name["weights_and_bn"].append(parameter)

    learning_rates = {
        "dt_activation_scales": args.lr * args.dt_scale_lr_multiplier,
        "activation_scales": args.lr * args.activation_scale_lr_multiplier,
        "weight_scales": args.lr * getattr(args, 'weight_scale_lr_multiplier', 1.0),
        "d_weight_scales": args.lr * getattr(args, 'd_scale_lr_multiplier', 1.0),
        "d_parameters": args.lr,
        # Keep the historical log key; BN affine has a separate group now.
        "weights_and_bn": args.lr * getattr(args, 'weight_lr_multiplier', 1.),
        "bn_affine": args.lr,
    }
    groups = []
    for name, parameters in groups_by_name.items():
        if not parameters:
            continue
        groups.append(
            {
                "params": parameters,
                "lr": learning_rates[name],
                "weight_decay": (
                    args.weight_decay if name in ("weights_and_bn", "bn_affine") else 0.0
                ),
                "name": name,
            }
        )
    optimizer = torch.optim.Adam(groups)
    logger.info(
        "QAT optimizer groups | %s",
        {
            name: {"parameters": len(parameters), "lr": learning_rates[name]}
            for name, parameters in groups_by_name.items()
        },
    )
    return optimizer


def set_qat_train_mode(model, freeze_bn):
    model.train()
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, (QuanConv, QuanLinear)) and module.norm:
                module.training = False


def freeze_learned_lsq_scales(model):
    """Freeze optimizer updates as well as calibration; leave weights/D trainable."""
    freeze_lsq_initialization(model)
    for module in model.modules():
        if isinstance(module, (LsqQuantizer4input, LsqQuantizer4weight)):
            module.s.requires_grad_(False)
            module.s.grad = None


def set_qat_epoch_lr(optimizer, epoch, max_epoch, schedule, minimum_ratio):
    # Apply before optimizer steps, so reported rates are those actually used.
    progress = (epoch - 1) / max(max_epoch - 1, 1)
    ratio = (1.0 if schedule == 'constant' else minimum_ratio +
             (1.0-minimum_ratio)*0.5*(1.0+math.cos(math.pi*progress)))
    for group in optimizer.param_groups:
        group.setdefault('qat_base_lr', group['lr'])
        group['lr'] = group['qat_base_lr'] * ratio


@torch.no_grad()
def fixed_batch_quantization_probe(model, inputs):
    """Read-only eval probe; statistics refer only to this fixed validation batch."""
    rows, handles = {}, []
    modes = [(m,m.training) for m in model.modules()]
    def hook(name):
        def collect(module, args, output):
            raw=args[0].detach()/module.s.detach()
            clipped=(raw<module.Qn)|(raw>module.Qp)
            row=dict(scale=float(module.s.detach().item()),count=raw.numel(),
                     clipping_ratio=float(clipped.float().mean()),
                     zero_ratio=float((output[0]==0).float().mean()))
            if '.d_weight_quant.' in name:
                codes=_LsqRound.apply(raw.clamp(module.Qn,module.Qp))
                row.update(unique_codes=int(codes.unique().numel()),
                           max_abs_quant_error=float((args[0]-output[0]).abs().max()))
            rows[name]=row
        return collect
    try:
        model.eval()
        for name,module in model.named_modules():
            if isinstance(module,(LsqQuantizer4input,LsqQuantizer4weight)):
                if module.init_state < module.batch_init:
                    raise RuntimeError('Probe requires completed LSQ calibration')
                handles.append(module.register_forward_hook(hook(name)))
        model(inputs)
    finally:
        for handle in handles:handle.remove()
        for module,mode in modes:module.training=mode
    return rows


def qat_drift_diagnostics(model, previous=None):
    """Changes since previous recorded evaluation, not necessarily one epoch."""
    snapshot={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    # named_parameters avoids double-counting quan_a/quan_w alias state_dict keys.
    groups={};scales=[]
    for name,parameter in model.named_parameters():
        if 'lsq_a.s' in name or 'lsq_w.s' in name:
            kind='lsq_scales'
        elif name.endswith('.D'):
            kind='D'
        elif 'bn_weight' in name or 'bn_bias' in name:
            kind='bn_affine'
        else:
            kind='other_parameters'
        now=snapshot[name]
        delta=0.0 if previous is None else float((now-previous[name]).abs().max())
        groups[kind]=max(groups.get(kind,0.0),delta)
        if kind=='lsq_scales':
            old=float(now.item()) if previous is None else float(previous[name].item())
            scales.append(dict(name=name,value=float(now.item()),trainable=parameter.requires_grad,
                               relative_change=(float(now.item())-old)/max(abs(old),1e-12)))
    bn_delta=max((float((value-previous[name]).abs().max()) for name,value in snapshot.items()
                  if 'bn_running_' in name),default=0.0) if previous is not None else 0.0
    return dict(max_abs_parameter_changes=groups,bn_running_max_abs_change=bn_delta,
                scales=scales),snapshot


@torch.no_grad()
def calibrate_lsq(model, loader, device, steps):
    model.eval()
    iterator = iter(loader)
    for _ in range(steps):
        try:
            image, *_ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            image, *_ = next(iterator)
        model(image.to(device, non_blocking=True))
    freeze_lsq_initialization(model)


def train_one_qat_epoch(
    model,
    loader,
    optimizer,
    class_weights,
    tile_size,
    device,
    grad_clip_norm,
    freeze_bn,
):
    set_qat_train_mode(model, freeze_bn)
    weighted_loss = 0.0
    denominator_sum = 0.0
    valid_pixels = 0
    for image, labels, *_ in loader:
        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = dense_logits(model, image, tile_size)
        loss = F.cross_entropy(
            logits,
            labels,
            weight=class_weights,
            ignore_index=-1,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite QAT loss: {float(loss.detach())}")
        loss.backward()
        if grad_clip_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"non-finite gradient norm: {float(norm)}")
        optimizer.step()
        clamp_lsq_scales_positive(model)
        denominator = cross_entropy_denominator(labels, class_weights)
        count = int((labels >= 0).sum())
        weighted_loss += float(loss.detach()) * denominator
        denominator_sum += denominator
        valid_pixels += count
    if valid_pixels == 0:
        raise RuntimeError("QAT epoch contained no supervised pixels")
    return weighted_loss / max(denominator_sum, 1e-12)


@torch.no_grad()
def stitch_prediction(model, loader, scene_shape, tile_size, device):
    model.eval()
    prediction = np.full(scene_shape, -1, dtype=np.int64)
    visits = np.zeros(scene_shape, dtype=np.uint8)
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
            if visits[top : top + height, left : left + width].any():
                raise AssertionError("dense output tiles overlap")
            prediction[top : top + height, left : left + width] = batch_prediction[
                item, :height, :width
            ]
            visits[top : top + height, left : left + width] = 1
    return prediction, visits


def evaluate_prediction(prediction, label_map, class_count):
    valid = label_map >= 0
    if np.any(prediction[valid] < 0):
        raise AssertionError("evaluation pixels are missing predictions")
    target = label_map[valid]
    predicted = prediction[valid]
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (target, predicted), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted_support = confusion.sum(axis=0).astype(np.float64)
    union = support + predicted_support - true_positive
    total = confusion.sum()
    per_class = np.divide(
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
    expected = (
        float((support * predicted_support).sum() / (total * total)) if total else 0.0
    )
    kappa = (oa - expected) / (1.0 - expected) if expected < 1.0 else 0.0
    return {
        "OA": oa,
        "mAcc": float(per_class.mean()),
        "Kappa": float(kappa),
        "mIoU": float(iou.mean()),
        "per_class_acc": per_class,
        "IoU": iou,
        "support": support.astype(np.int64),
        "confusion_matrix": confusion,
    }


def evaluate_model(model, loader, gt_shape, tile_size, device, label_map, classes):
    prediction, visits = stitch_prediction(
        model, loader, gt_shape, tile_size, device
    )
    return prediction, visits, evaluate_prediction(prediction, label_map, classes)


def make_seed_datasets_loaders(args, image, label_maps, tiles, seed):
    datasets = tuple(
        DenseTileDataset(
            image,
            label_map,
            split_tiles,
            args.tile_size,
            require_label=(split_index == 0),
        )
        for split_index, (label_map, split_tiles) in enumerate(zip(label_maps, tiles))
    )
    if len(datasets[0]) == 0:
        raise RuntimeError("no training tile contains a supervised pixel")
    loaders = (
        make_loader(datasets[0], args.batch_size, True, args.num_workers, seed),
        make_loader(datasets[1], args.eval_batch_size, False, args.num_workers, seed),
        make_loader(datasets[2], args.eval_batch_size, False, args.num_workers, seed),
    )
    return datasets, loaders


def model_report(model, optimizer, run_dir, logger, model_config):
    effective_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("shared_params.")
    )
    report = {
        "model_config": model_config,
        "qat_ssm_contract": model._qat_ssm_contract,
        "selective_scan_backend": selective_scan_backend(),
        "parameter_count_total": sum(p.numel() for p in model.parameters()),
        "parameter_count_effective": effective_parameters,
        "optimizer_groups": [
            {
                "name": group.get("name", "unnamed"),
                "lr": group["lr"],
                "parameter_tensors": len(group["params"]),
            }
            for group in optimizer.param_groups
        ],
        "model": str(model),
    }
    save_json(run_dir / "qat_model_report.json", report)
    logger.info(
        "Model config=%s | total_params=%d effective_params=%d | scan=%s",
        model_config,
        report["parameter_count_total"],
        effective_parameters,
        selective_scan_backend(),
    )


def build_fp32_model(in_channels, class_count, state, device, model_config):
    model = build_configured_model(in_channels, class_count, model_config)
    model.load_state_dict(state, strict=True)
    validate_fp32_model_contract(model, model_config)
    return model.to(device)


def run_seed(
    args,
    dataset_name,
    image,
    gt,
    class_count,
    masks,
    tiles,
    dataset_dir,
    fp32_dir,
    seed,
    device,
    logger,
):
    set_seed(seed)
    contract = resolve_ssm_contract(dict(u_quantization=args.ssm_u_quantization, d_weight_bits=args.d_weight_bits))
    numeric = SSMNumericConfig(dt_input_bits=args.dt_input_bits,
                               dt_output_bits=args.dt_output_bits)
    run_dir = dataset_dir / f"run_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    numeric_path = run_dir / "numeric_config.json"
    if any((run_dir / name).exists() for name in (
            "best_qat_foldaware.pth", "qat_training_history.jsonl", "qat_training_diagnostics.jsonl",
            "initial_quantization_diagnosis", "weight_init_validation")):
        raise FileExistsError(f"Refusing to overwrite an existing QAT run: {run_dir}; use a new run_tag")
    save_json(numeric_path, numeric.to_dict())
    labels_and_indices = load_seed_label_maps(
        fp32_dir,
        seed,
        gt,
        masks,
        class_count,
        args.train_samples,
    )
    train_label, val_label, test_label = labels_and_indices[:3]
    train_indices, val_indices, test_indices = labels_and_indices[3:]
    np.savez_compressed(
        run_dir / "sample_indices.npz",
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )
    datasets, loaders = make_seed_datasets_loaders(
        args,
        image,
        (train_label, val_label, test_label),
        tiles,
        seed,
    )
    train_dataset, val_dataset, test_dataset = datasets
    train_loader, val_loader, test_loader = loaders

    pretrained_path = resolve_pretrained_path(args, fp32_dir, dataset_name, seed)
    if not pretrained_path.exists():
        raise FileNotFoundError(f"Missing FP32 checkpoint: {pretrained_path}")
    fp32_state = extract_state_dict(torch.load(pretrained_path, map_location="cpu"))
    fp32_model = build_fp32_model(
        image.shape[2], class_count, fp32_state, device, args.model_config
    )
    _, fp32_val_visits, fp32_val_metrics = evaluate_model(
        fp32_model,
        val_loader,
        gt.shape,
        args.tile_size,
        device,
        val_label,
        class_count,
    )
    if not np.all(fp32_val_visits[masks[1]] == 1):
        raise AssertionError("FP32 validation region is not completely covered")
    del fp32_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = build_configured_model(
        image.shape[2], class_count, args.model_config
    )
    model.load_state_dict(fp32_state, strict=True)
    prepare_qat_model(
        model,
        nbit=args.nbit,
        model_config=args.model_config,
        numeric_config=numeric,
        ssm_contract=contract,
    )
    model.to(device)
    logger.info('QAT SSM contract=%s | D path software support=%s | D RTL implemented=False',
                contract, d_path_simulator_compatible(args.model_config))
    # Set D initialization before its first calibration forward. Deployment
    # restores the learned scale from the checkpoint instead of initializing it.
    for name, module in model.named_modules():
        if name.endswith('.d_weight_quant') and isinstance(module, WeightFakeQuant):
            module.lsq_w.init_strategy = args.d_init_policy
    calibrate_lsq(model, train_loader, device, args.calibration_steps)
    d_calibration_scales = {
        name: float(module.lsq_w.s.detach().cpu().item())
        for name, module in model.named_modules()
        if name.endswith('.d_weight_quant') and isinstance(module, WeightFakeQuant)
    }
    if d_calibration_scales:
        logger.info('D LSQ initialization=%s; post-calibration scales=%s',
                    args.d_init_policy, d_calibration_scales)
    validate_fpga_qat_contract(
        model,
        expect_bn=True,
        model_config=args.model_config,
    )
    if args.weight_init_validation_only:
        from qat_forensics import weight_init_validation
        report = weight_init_validation(sys.modules[__name__], model, args, val_loader,
            val_label, class_count, device, run_dir/'weight_init_validation', logger, fp32_val_metrics)
        return dict(seed=seed, mode='weight_init_validation_only',
                    report=str(run_dir/'weight_init_validation'/'weight_init_validation.json'),
                    cases={k: v['validation'] for k,v in report['cases'].items()})
    if args.diagnose_initial_quantization:
        from qat_forensics import initial_quantization_diagnosis
        initial_report = initial_quantization_diagnosis(sys.modules[__name__], model, fp32_state,
            args, val_loader, val_label, class_count, image.shape[2], device,
            run_dir/'initial_quantization_diagnosis', logger)
        if args.initial_diagnostics_only:
            return dict(seed=seed, mode='initial_diagnostics_only',
                        report=str(run_dir/'initial_quantization_diagnosis'/'initial_quantization.json'),
                        conversion_within_tolerance=initial_report['conversion_within_tolerance'])
        if not initial_report['conversion_within_tolerance'] or not initial_report['conversion_predictions_match']:
            raise RuntimeError('Conversion parity failed; inspect initial_quantization.json before training')
    weight_init_changes = []
    if args.weight_init_policy != 'mean':
        from qat_forensics import apply_weight_initialization
        weight_init_changes = apply_weight_initialization(sys.modules[__name__], model, args.weight_init_policy)
        logger.info('Applied weight initialization=%s; changed_scales=%d; activation/dt/D scales retained',
                    args.weight_init_policy, len(weight_init_changes))
    save_json(run_dir/'weight_initialization.json', dict(policy=args.weight_init_policy, changes=weight_init_changes,
              activation_calibration='original mean-weight calibration, no recalibration after scale replacement'))
    if args.freeze_bn_affine:
        freeze_bn_affine_parameters(model)
        logger.info('Freeze BN affine gamma/beta before optimizer construction')
    optimizer = make_qat_optimizer(model, args, logger)
    class_weights, class_counts = compute_class_weights(
        train_label, class_count, args.class_weight, device
    )
    logger.info(
        "seed=%d checkpoint=%s train_tiles=%d val_tiles=%d test_tiles=%d "
        "train_labels=%d val_labels=%d test_labels=%d class_counts=%s",
        seed,
        pretrained_path,
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        train_indices.size,
        val_indices.size,
        test_indices.size,
        class_counts.astype(int).tolist(),
    )
    if seed == args.seeds[0]:
        model_report(model, optimizer, run_dir, logger, args.model_config)

    best_score = -math.inf
    best_metrics = None
    best_epoch = 0
    best_path = run_dir / "best_qat_foldaware.pth"
    baseline_prediction, baseline_visits, baseline_metrics = evaluate_model(
        model, val_loader, gt.shape, args.tile_size, device, val_label, class_count)
    if not np.all(baseline_visits[masks[1]] == 1) or np.any(baseline_visits[masks[0]]) or np.any(baseline_visits[masks[2]]):
        raise AssertionError('Calibrated baseline crossed the saved validation split')
    save_json(run_dir / 'calibrated_baseline_validation.json', baseline_metrics)
    logger.info('Calibrated QAT seed=%d epoch=0 val_OA=%.6f val_mAcc=%.6f eligible_for_selection=%s',
                seed, baseline_metrics['OA'], baseline_metrics['mAcc'], args.include_calibrated_baseline)
    if args.include_calibrated_baseline:
        best_score = baseline_metrics[args.selection_metric]
        best_metrics = dict(baseline_metrics)
        torch.save(model.state_dict(), best_path)
    diagnostic_input = None
    previous_snapshot = None
    if args.training_diagnostics:
        # Direct dataset access avoids consuming the training loader/RNG order.
        diagnostic_input = torch.stack([val_dataset[i][0] for i in range(
            min(args.eval_batch_size, len(val_dataset)))]).to(device)
    history_path = run_dir / 'qat_training_history.jsonl'
    diagnostics_path = run_dir / 'qat_training_diagnostics.jsonl'

    def record_evaluation(epoch, loss, metrics, bn_frozen, scales_frozen):
        nonlocal previous_snapshot
        entry = dict(epoch=epoch, loss=loss, validation=metrics,
                     bn_statistics_frozen=bn_frozen, lsq_scales_frozen=scales_frozen,
                     bn_affine_frozen=args.freeze_bn_affine,
                     learning_rates={g['name']:g['lr'] for g in optimizer.param_groups})
        with history_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, default=lambda x: x.tolist() if hasattr(x,'tolist') else float(x))+'\n')
        if diagnostic_input is not None:
            drift, previous_snapshot = qat_drift_diagnostics(model, previous_snapshot)
            probe = fixed_batch_quantization_probe(model, diagnostic_input)
            drift.update(epoch=epoch, scope='First fixed validation batch; drift since previous recorded evaluation',
                         quantizers=probe)
            with diagnostics_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(drift)+'\n')
            worst = max(drift['scales'], key=lambda row:abs(row['relative_change']), default=None)
            if worst:
                logger.info('QAT diagnostics epoch=%d largest_scale_change=%s relative=%+.6g '
                            'BN_running_max_change=%.6g',epoch,worst['name'],worst['relative_change'],
                            drift['bn_running_max_abs_change'])

    record_evaluation(0, None, baseline_metrics, True, False)
    jump_capture = None
    if args.capture_oa_drop_pp > 0:
        from qat_forensics import JumpCapture
        jump_capture = JumpCapture(sys.modules[__name__], run_dir, args, class_count,
                                   image.shape[2], numeric, contract)
        jump_capture.observe(model, 0, baseline_metrics, baseline_prediction, val_dataset, val_label)
    losses = []
    scales_frozen = False
    start = time.perf_counter()
    for epoch in range(1, args.max_epoch + 1):
        set_qat_epoch_lr(optimizer, epoch, args.max_epoch, args.lr_schedule, args.min_lr_ratio)
        if not scales_frozen and args.freeze_lsq_epoch >= 0 and epoch >= max(args.freeze_lsq_epoch, 1):
            freeze_learned_lsq_scales(model)
            scales_frozen = True
            logger.info('Freeze learned LSQ scales at epoch=%d; model weights and D remain trainable',epoch)
        freeze_bn = args.freeze_bn_epoch >= 0 and epoch >= max(
            args.freeze_bn_epoch, 1
        )
        loss = train_one_qat_epoch(
            model,
            train_loader,
            optimizer,
            class_weights,
            args.tile_size,
            device,
            args.grad_clip_norm,
            freeze_bn,
        )
        losses.append(loss)
        if epoch == 1 or epoch % args.eval_interval == 0 or epoch == args.max_epoch:
            validation_prediction, visits, metrics = evaluate_model(
                model,
                val_loader,
                gt.shape,
                args.tile_size,
                device,
                val_label,
                class_count,
            )
            if not np.all(visits[masks[1]] == 1):
                raise AssertionError("validation region is not completely covered")
            if np.any(visits[masks[0]]) or np.any(visits[masks[2]]):
                raise AssertionError("validation tiles cross the saved split")
            score = metrics[args.selection_metric]
            logger.info(
                "QAT seed=%d epoch=%d/%d loss=%.6f val_OA=%.6f "
                "val_mAcc=%.6f select_%s=%.6f BN_frozen=%s LSQ_frozen=%s lr=%.8g",
                seed,
                epoch,
                args.max_epoch,
                loss,
                metrics["OA"],
                metrics["mAcc"],
                args.selection_metric,
                score,
                freeze_bn,
                scales_frozen,
                next(g['lr'] for g in optimizer.param_groups if g['name'] == 'weights_and_bn'),
            )
            record_evaluation(epoch, loss, metrics, freeze_bn, scales_frozen)
            if jump_capture is not None:
                jump_capture.observe(model, epoch, metrics, validation_prediction, val_dataset, val_label)
            if score >= best_score:
                best_score = score
                best_metrics = dict(metrics)
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
    train_seconds = time.perf_counter() - start
    if losses:
        plot_loss(losses, run_dir / "qat_train_loss_curve.png")
    if best_metrics is None or not best_path.exists():
        raise RuntimeError("no QAT checkpoint was selected on validation")

    model.load_state_dict(torch.load(best_path, map_location=device), strict=True)
    freeze_lsq_initialization(model)
    validate_fpga_qat_contract(
        model,
        expect_bn=True,
        model_config=args.model_config,
    )

    # Separate serialization/evaluation reproducibility from validation-to-test generalization.
    _, restored_visits, restored_metrics = evaluate_model(
        model, val_loader, gt.shape, args.tile_size, device, val_label, class_count)
    if not np.all(restored_visits[masks[1]] == 1):
        raise AssertionError('Reloaded best validation region is not covered')
    reload_match = np.array_equal(np.asarray(restored_metrics['confusion_matrix']),
                                  np.asarray(best_metrics['confusion_matrix']))
    save_json(run_dir / 'best_validation_recheck.json', dict(
        best_epoch=best_epoch, confusion_matrix_match=bool(reload_match),
        selected_metrics=best_metrics, reloaded_metrics=restored_metrics))
    if not reload_match:
        raise RuntimeError('Reloaded checkpoint does not reproduce selected validation confusion matrix; '
                           'inspect best_validation_recheck.json before attributing test loss to quantization')
    logger.info('Best checkpoint validation recheck: epoch=%d exact confusion matrix match=True',best_epoch)

    fp32_model = build_fp32_model(
        image.shape[2], class_count, fp32_state, device, args.model_config
    )
    fp32_prediction, fp32_visits, fp32_metrics = evaluate_model(
        fp32_model,
        test_loader,
        gt.shape,
        args.tile_size,
        device,
        test_label,
        class_count,
    )
    del fp32_model
    qat_prediction, qat_visits, qat_metrics = evaluate_model(
        model,
        test_loader,
        gt.shape,
        args.tile_size,
        device,
        test_label,
        class_count,
    )
    if not np.all(fp32_visits[masks[2]] == 1) or not np.all(
        qat_visits[masks[2]] == 1
    ):
        raise AssertionError("test region is not completely covered")
    if np.any(qat_visits[masks[0]]) or np.any(qat_visits[masks[1]]):
        raise AssertionError("test tiles cross the saved split")

    deploy_model = build_configured_model(
        image.shape[2], class_count, args.model_config
    )
    deploy_model.load_state_dict(fp32_state, strict=True)
    prepare_qat_model(
        deploy_model,
        nbit=args.nbit,
        model_config=args.model_config,
        numeric_config=numeric,
        ssm_contract=contract,
    )
    deploy_model.load_state_dict(torch.load(best_path, map_location="cpu"), strict=True)
    deploy_model.to(device).eval()
    freeze_lsq_initialization(deploy_model)
    probe = next(iter(train_loader))[0][:1].to(device) if args.verify_bn_fold else None
    fused_names, fold_consistency = fuse_qat_model_bns_for_deploy(
        deploy_model, validation_input=probe
    )
    validate_fpga_qat_contract(
        deploy_model,
        expect_bn=False,
        model_config=args.model_config,
    )
    deploy_prediction, deploy_visits, deploy_metrics = evaluate_model(
        deploy_model,
        test_loader,
        gt.shape,
        args.tile_size,
        device,
        test_label,
        class_count,
    )
    if not np.all(deploy_visits[masks[2]] == 1):
        raise AssertionError("deploy test region is not completely covered")
    evaluated = test_label >= 0
    fold_prediction_match = float(
        np.mean(qat_prediction[evaluated] == deploy_prediction[evaluated])
    )
    if args.verify_bn_fold and fold_prediction_match < 1.0:
        raise RuntimeError(
            f"physical BN folding changed test predictions: {fold_prediction_match:.8f}"
        )

    torch.save(deploy_model.state_dict(), run_dir / "best_qat_deploy_fused.pth")
    np.save(run_dir / "fp32_test_prediction.npy", fp32_prediction)
    np.save(run_dir / "qat_foldaware_test_prediction.npy", qat_prediction)
    np.save(run_dir / "qat_deploy_test_prediction.npy", deploy_prediction)
    save_json(
        run_dir / "quantization_diagnostics.json",
        quantization_diagnostics(deploy_model),
    )
    result = {
        "dataset": dataset_name,
        "seed": seed,
        "weight_init_policy": args.weight_init_policy,
        "weight_init_changes": weight_init_changes,
        "export_calibrated_only": args.export_calibrated_only,
        "optimization_steps": len(train_loader)*args.max_epoch,
        "numeric_config": numeric.to_dict(),
        "qat_ssm_contract": contract,
        "d_weight_initialization": (
            ('max_abs/qmax' if args.d_init_policy == 'max' else '2*mean_abs/sqrt(qmax)')
            if args.model_config['use_D'] else None),
        "d_scales_after_calibration": d_calibration_scales,
        "d_init_policy": args.d_init_policy,
        "calibrated_baseline_validation": baseline_metrics,
        "include_calibrated_baseline": args.include_calibrated_baseline,
        "best_validation_recheck_match": bool(reload_match),
        "freeze_lsq_epoch": args.freeze_lsq_epoch,
        "lr_schedule": args.lr_schedule,
        "min_lr_ratio": args.min_lr_ratio,
        "scale_lr_multipliers": dict(activation=args.activation_scale_lr_multiplier,
            dt=args.dt_scale_lr_multiplier,weight=args.weight_scale_lr_multiplier,D=args.d_scale_lr_multiplier),
        "training_diagnostics": args.training_diagnostics,
        "d_path_simulator_compatible": d_path_simulator_compatible(args.model_config),
        "rtl_baseline_compatible": numeric.rtl_baseline_compatible and current_fpga_simulator_compatible(args.model_config),
        "qat_recurrence": "floating selective scan; only deployment boundaries are trained; internal error ablation is evaluated by simulator",
        "protocol": (
            "exact saved raw-pixel-disjoint 16x16 three-way split; exact saved "
            "per-seed labels; exact saved train-only PCA/minmax; configuration-"
            "matched MambaHSI; per-tensor INT8 LSQ QAT"
        ),
        "model_config": args.model_config,
        "model_config_slug": args.model_config_slug,
        "current_fpga_simulator_compatible": current_fpga_simulator_compatible(
            args.model_config
        ),
        "source_fp32_dir": str(fp32_dir),
        "pretrained_fp32_path": str(pretrained_path),
        "qat_weight_path": str(best_path),
        "deploy_weight_path": str(run_dir / "best_qat_deploy_fused.pth"),
        "split_strategy": "blocks",
        "split_target_ratio": SPLIT_TARGET_RATIO,
        "split_block_size": args.split_block_size,
        "tile_size": args.tile_size,
        "train_label_count": int(train_indices.size),
        "val_label_count": int(val_indices.size),
        "test_label_count": int(test_indices.size),
        "train_class_counts": class_counts.astype(int),
        "class_weight_mode": args.class_weight,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "train_batches_per_epoch": len(train_loader),
        "calibration_steps": args.calibration_steps,
        "optimizer": "Adam",
        "initial_lr": args.lr,
        "final_lr": float(next(g['lr'] for g in optimizer.param_groups if g['name'] == 'weights_and_bn')),
        "final_learning_rates": {g['name']: float(g['lr']) for g in optimizer.param_groups},
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "freeze_bn_epoch": args.freeze_bn_epoch,
        "freeze_bn_affine": args.freeze_bn_affine,
        "weight_lr_multiplier": args.weight_lr_multiplier,
        "jump_capture": dict(threshold_pp=args.capture_oa_drop_pp, max_events=args.capture_max_events,
                             start_epoch=args.capture_start_epoch,
                             events=jump_capture.events if jump_capture is not None else []),
        "residual_contract": [
            {
                "block_index": block_index,
                "block_use_residual": True,
                "block_skip_scale": float(args.skip_scale),
                "spa_use_residual": False,
                "spe_use_residual": False,
            }
            for block_index in range(args.model_config["block_count"])
        ],
        "max_epoch": args.max_epoch,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_val_score": best_score,
        "best_val_OA": best_metrics["OA"],
        "best_val_mAcc": best_metrics["mAcc"],
        "fp32_val_OA": fp32_val_metrics["OA"],
        "fp32_val_mAcc": fp32_val_metrics["mAcc"],
        "fp32_test_OA": fp32_metrics["OA"],
        "fp32_test_mAcc": fp32_metrics["mAcc"],
        "fp32_test_Kappa": fp32_metrics["Kappa"],
        "fp32_test_mIoU": fp32_metrics["mIoU"],
        "qat_foldaware_test_OA": qat_metrics["OA"],
        "qat_foldaware_test_mAcc": qat_metrics["mAcc"],
        "qat_deploy_test_OA": deploy_metrics["OA"],
        "qat_deploy_test_mAcc": deploy_metrics["mAcc"],
        "qat_deploy_test_Kappa": deploy_metrics["Kappa"],
        "qat_deploy_test_mIoU": deploy_metrics["mIoU"],
        "qat_test_OA": deploy_metrics["OA"],
        "qat_test_mAcc": deploy_metrics["mAcc"],
        "qat_test_Kappa": deploy_metrics["Kappa"],
        "qat_test_mIoU": deploy_metrics["mIoU"],
        "qat_test_per_class_acc": deploy_metrics["per_class_acc"],
        "qat_test_IoU": deploy_metrics["IoU"],
        "qat_test_support": deploy_metrics["support"],
        "qat_confusion_matrix": deploy_metrics["confusion_matrix"],
        "OA_loss_percentage_points": (
            fp32_metrics["OA"] - deploy_metrics["OA"]
        )
        * 100.0,
        "mAcc_loss_percentage_points": (
            fp32_metrics["mAcc"] - deploy_metrics["mAcc"]
        )
        * 100.0,
        "bn_fold_test_prediction_match": fold_prediction_match,
        "bn_fold_probe_consistency": fold_consistency,
        "folded_bn_layers": fused_names,
        "train_seconds": train_seconds,
    }
    save_json(run_dir / "result.json", result)
    logger.info(
        "FINAL QAT seed=%d FP32_OA=%.6f QAT_OA=%.6f OA_loss_pp=%.4f "
        "FP32_mAcc=%.6f QAT_mAcc=%.6f best_epoch=%d fold_match=%.6f",
        seed,
        fp32_metrics["OA"],
        deploy_metrics["OA"],
        result["OA_loss_percentage_points"],
        fp32_metrics["mAcc"],
        deploy_metrics["mAcc"],
        best_epoch,
        fold_prediction_match,
    )
    return result


def summarize_results(dataset_name, args, fp32_dir, results):
    summary = {
        "dataset": dataset_name,
        "runs": len(results),
        "seeds": [int(result["seed"]) for result in results],
        "split_strategy": "blocks",
        "split_target_ratio": list(SPLIT_TARGET_RATIO),
        "tile_size": args.tile_size,
        "split_block_size": args.split_block_size,
        "train_samples_per_class": args.train_samples,
        "train_samples_per_class_cap": args.train_samples,
        "min_train_samples_per_class": args.min_train_samples,
        "train_label_mode": "per_class_cap_without_replacement",
        "source_fp32_dir": str(fp32_dir),
        "model_config": args.model_config,
        "model_config_slug": args.model_config_slug,
        "current_fpga_simulator_compatible": current_fpga_simulator_compatible(
            args.model_config
        ),
    }
    metric_pairs = (
        ("fp32_test_OA", "FP32_OA"),
        ("fp32_test_mAcc", "FP32_mAcc"),
        ("fp32_test_Kappa", "FP32_Kappa"),
        ("fp32_test_mIoU", "FP32_mIoU"),
        ("qat_test_OA", "QAT_OA"),
        ("qat_test_mAcc", "QAT_mAcc"),
        ("qat_test_Kappa", "QAT_Kappa"),
        ("qat_test_mIoU", "QAT_mIoU"),
        ("OA_loss_percentage_points", "OA_loss_pp"),
        ("mAcc_loss_percentage_points", "mAcc_loss_pp"),
    )
    for result_key, summary_key in metric_pairs:
        values = np.asarray([result[result_key] for result in results], dtype=np.float64)
        if not result_key.endswith("percentage_points"):
            values = values * 100.0
        summary[f"{summary_key}_mean"] = float(values.mean())
        summary[f"{summary_key}_std"] = float(
            values.std(ddof=1) if values.size > 1 else 0.0
        )
        summary[f"{summary_key}_values"] = values
    return summary


def run_dataset(args, dataset_name, device):
    raw_data, gt, class_count = load_dataset(dataset_name, args.data_set_path)
    fp32_dir = resolve_fp32_dataset_dir(args, dataset_name)
    validate_saved_model_config(fp32_dir, args.seeds, args.model_config)
    image = transform_with_saved_preprocess(
        raw_data,
        fp32_dir / "train_only_preprocess.npz",
    )
    if image.shape[:2] != gt.shape or image.shape[2] != args.pca_components:
        raise ValueError(
            f"Saved preprocessing produced {image.shape}; expected "
            f"{(*gt.shape, args.pca_components)}"
        )

    masks = load_spatial_masks(fp32_dir, gt.shape)
    tiles = make_tiles_from_saved_masks(masks, args.tile_size)
    region_names = ("train", "validation", "test")
    region_counts = {
        name: [
            int(np.count_nonzero(mask & (gt == class_id)))
            for class_id in range(1, class_count + 1)
        ]
        for name, mask in zip(region_names, masks)
    }
    for name in region_names:
        if min(region_counts[name]) <= 0:
            raise ValueError(
                f"Saved {name} region has an empty class: {region_counts[name]}"
            )
    if min(region_counts["train"]) < args.min_train_samples:
        raise ValueError(
            "Saved training region violates --min_train_samples: "
            f"counts={region_counts['train']}"
        )

    dataset_dir = (
        Path(args.work_dir)
        / args.exp_name
        / f"{dataset_name}_{args.run_tag}"
    )
    if args.output_config_subdir:
        dataset_dir = dataset_dir / args.model_config_slug
    for seed in args.seeds:
        existing = dataset_dir / f'run_seed{seed}' / 'best_qat_foldaware.pth'
        if existing.exists():
            raise FileExistsError(f'Refusing to overwrite {existing}; choose a new --run_tag')
    dataset_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(
        dataset_dir / "train_qat.log",
        f"qat_single_{dataset_name}_{time.time_ns()}",
    )
    logger.info(
        "dataset=%s raw=%s gt=%s preprocessed=%s range=[%.8f, %.8f]",
        dataset_name,
        raw_data.shape,
        gt.shape,
        image.shape,
        float(image.min()),
        float(image.max()),
    )
    logger.info("FP32 artifacts=%s", fp32_dir)
    logger.info(
        "Exact saved split | tiles train=%d val=%d test=%d | class_counts=%s",
        len(tiles[0]),
        len(tiles[1]),
        len(tiles[2]),
        region_counts,
    )
    logger.info("Resolved QAT model config=%s", args.model_config)
    logger.info(
        "Current fixed FPGA simulator compatible=%s",
        current_fpga_simulator_compatible(args.model_config),
    )

    np.savez_compressed(
        dataset_dir / "spatial_split_masks.npz",
        train_region=masks[0],
        val_region=masks[1],
        test_region=masks[2],
    )
    save_json(
        dataset_dir / "qat_config.json",
        {
            "arguments": vars(args),
            "model_config": args.model_config,
            "model_config_slug": args.model_config_slug,
            "current_fpga_simulator_compatible": (
                current_fpga_simulator_compatible(args.model_config)
            ),
            "source_fp32_dir": str(fp32_dir),
            "selective_scan_backend": selective_scan_backend(),
            "region_class_counts": region_counts,
            "tile_counts": {
                name: len(split_tiles)
                for name, split_tiles in zip(region_names, tiles)
            },
        },
    )

    results = [
        run_seed(
            args,
            dataset_name,
            image,
            gt,
            class_count,
            masks,
            tiles,
            dataset_dir,
            fp32_dir,
            seed,
            device,
            logger,
        )
        for seed in args.seeds
    ]
    if args.initial_diagnostics_only or args.weight_init_validation_only:
        summary = dict(dataset=dataset_name, mode=('weight_init_validation_only' if args.weight_init_validation_only
                                                  else 'initial_diagnostics_only'), runs=results)
        save_json(dataset_dir / ('weight_init_validation_summary.json' if args.weight_init_validation_only
                                 else 'initial_diagnostics_summary.json'), summary)
        return summary
    summary = summarize_results(dataset_name, args, fp32_dir, results)
    save_json(dataset_dir / "summary.json", summary)
    logger.info(
        "SUMMARY dataset=%s FP32_OA=%.4f+/-%.4f QAT_OA=%.4f+/-%.4f "
        "OA_loss_pp=%.4f+/-%.4f",
        dataset_name,
        summary["FP32_OA_mean"],
        summary["FP32_OA_std"],
        summary["QAT_OA_mean"],
        summary["QAT_OA_std"],
        summary["OA_loss_pp_mean"],
        summary["OA_loss_pp_std"],
    )
    return summary


def main():
    args = build_parser().parse_args()
    args = resolve_model_arguments(args)
    args = validate_args(args)
    device = resolve_device(args.device)
    selected = list(DATASET_ORDER) if args.dataset == "all" else [args.dataset]
    summaries = [run_dataset(args, name, device) for name in selected]
    output = (
        Path(args.work_dir)
        / args.exp_name
        / (
            f"all_datasets_summary_{args.run_tag}_"
            f"{args.model_config_slug}.json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, summaries)
    print(f"All {'validation-only' if args.initial_diagnostics_only or args.weight_init_validation_only else 'QAT'} runs completed. Summary saved to: {output}")


if __name__ == "__main__":
    main()
