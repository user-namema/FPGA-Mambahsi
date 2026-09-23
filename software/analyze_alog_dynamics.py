#!/usr/bin/env python3
"""Extract learned selective-SSM dynamics from a trained MambaHSI run.

The script deliberately reuses the immutable artifacts saved by
``train_mambahsi_spatial_split_128_dense.py``:

* ``run_seed*/model_config.json`` and ``best_model.pth`` restore the model;
* ``spatial_split.json`` selects the original train/validation/test tiles;
* ``train_only_preprocess.npz`` applies the original training-only PCA and
  normalization without refitting either transform.

It reports three distinct objects that must not be conflated:

1. the learned continuous pole rates ``lambda = exp(A_log)``;
2. the input-dependent positive step sizes ``delta = softplus(dt_proj(...))``;
3. the discrete retention and e-fold memory length
   ``Abar = exp(-delta * lambda)``, ``L = 1 / (delta * lambda)``.

For a channel-shared A basis, every channel has its own delta and recurrent
state, but the same lambda vector.  For a per-channel A model, lambda has shape
``[d_inner, d_state]``.
"""

import argparse
import csv
import glob
import hashlib
import copy
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from mambahsi_ablation_model import MambaHSI, selective_scan_backend


MODEL_ARGUMENTS = (
    "hidden_dim",
    "branch_mode",
    "fusion_mode",
    "skip_scale",
    "use_z",
    "use_D",
    "A_mode",
    "norm_path",
    "activation",
    "head_dim",
    "token_num",
    "d_state",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Extract learned A_log spectra, dynamic delta statistics and "
            "effective memory horizons from a trained MambaHSI checkpoint."
        )
    )
    parser.add_argument(
        "--run-dir",
        action="append", default=[],
        help="FP32 or QAT run directory; repeat for multiple runs.",
    )
    parser.add_argument('--run-glob', action='append', default=[], help='Quoted glob for run directories')
    parser.add_argument('--artifact-dir', help='Relocated FP32 directory with saved PCA and spatial_split.json')
    parser.add_argument('--checkpoint', help='Override checkpoint path (single-run only)')
    parser.add_argument('--dataset', help='Validate dataset identity, or filter runs selected by glob')
    parser.add_argument('--coefficient-fraction-bits', type=int, nargs='+', default=[16, 20, 24])
    parser.add_argument('--retention-lags', type=int, nargs='+', default=[1, 4, 16, 64, 256])
    parser.add_argument(
        "--data-path",
        default="./data",
        help="Dataset root resolved by the QAT dataset loader (UP/HongHu/HanChuan/Houston).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="test",
        help="Frozen spatial split used to estimate dynamic delta statistics.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=0,
        help=(
            "0 analyzes every tile. A positive value selects that many tiles "
            "at evenly spaced indices for a deterministic pilot analysis."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda, cuda:0, cpu, or another torch device.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=512,
        help="Number of logarithmic bins used for streaming delta quantiles.",
    )
    parser.add_argument(
        "--delta-log10-min",
        type=float,
        default=-6.0,
        help="Lower log10(delta) edge of the streaming histogram.",
    )
    parser.add_argument(
        "--delta-log10-max",
        type=float,
        default=2.0,
        help="Upper log10(delta) edge of the streaming histogram.",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Read A_log and checkpoint statistics without loading image data.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Default: <run-dir>/learned_dynamics_<split>."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip optional PDF/PNG summary plots.",
    )
    return parser


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(text):
    if text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Use --device cpu only for a small static/pilot analysis."
        )
    return torch.device(text)


def load_state_dict(path, device):
    try:
        value = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        value = torch.load(path, map_location=device)
    if isinstance(value, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = value.get(key)
            if isinstance(candidate, dict):
                value = candidate
                break
    if not isinstance(value, dict):
        raise TypeError("Checkpoint does not contain a state_dict mapping.")
    if value and all(str(key).startswith("module.") for key in value):
        value = {str(key)[7:]: tensor for key, tensor in value.items()}
    return value


def locate_artifact_dir(run_dir, override=None):
    if override:
        root = Path(override).expanduser().resolve()
    elif (run_dir.parent / 'spatial_split.json').exists():
        root = run_dir.parent
    else:
        result = read_json(run_dir / 'result.json')
        root = Path(result.get('source_fp32_dir', ''))
        if not root.is_dir():
            raise FileNotFoundError('QAT source FP32 artifacts moved; pass --artifact-dir')
    for name in ('spatial_split.json',):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    return root


def assert_model_device(model, device):
    """Check every registered tensor, including LSQ scales and skip buffers."""
    expected = torch.device(device)
    if expected.type == 'cuda' and expected.index is None:
        expected = torch.device('cuda', torch.cuda.current_device())
    misplaced = [name + '=' + str(t.device)
                 for name, t in list(model.named_parameters()) + list(model.named_buffers())
                 if t.device != expected]
    if misplaced:
        raise RuntimeError('Restored model must be entirely on %s; misplaced: %s' %
                           (expected, ', '.join(misplaced[:20])))


def restore_model(run_dir, device, checkpoint=None, artifact_dir=None):
    is_qat = (run_dir / 'best_qat_foldaware.pth').is_file()
    if is_qat:
        from train_mambahsi_spatial_split_dense_qat import (
            build_configured_model, prepare_qat_model, freeze_lsq_initialization,
            fuse_qat_model_bns_for_deploy, DATASET_CONFIGS)
        result = read_json(run_dir / 'result.json')
        source = locate_artifact_dir(run_dir, artifact_dir)
        source_config = source / f'run_seed{result["seed"]}' / 'model_config.json'
        config = read_json(source_config) if source_config.exists() else dict(result['model_config'])
        config.update(result.get('model_config', {}))
        config.setdefault('in_channels', 16)
        config.setdefault('num_classes', DATASET_CONFIGS[result['dataset']]['class_count'])
        model = build_configured_model(config['in_channels'], config['num_classes'], config)
        prepare_qat_model(model, model_config=config, numeric_config=result.get('numeric_config'),
                          ssm_contract=result.get('qat_ssm_contract'))
        # prepare_qat_model creates new LSQ parameters/buffers on CPU. Move the
        # complete prepared graph, not only the original floating model.
        model.to(device)
        weight_path = Path(checkpoint) if checkpoint else run_dir / 'best_qat_foldaware.pth'
        if 'deploy_fused' in weight_path.name:
            raise ValueError('Use foldaware QAT checkpoint; the analyzer folds BN after strict loading')
        model.load_state_dict(load_state_dict(weight_path, device), strict=True)
        freeze_lsq_initialization(model)
        fuse_qat_model_bns_for_deploy(model)
    else:
        config = read_json(run_dir / 'model_config.json')
        kwargs = {name: config[name] for name in MODEL_ARGUMENTS}
        # Newer variable-depth checkpoints preserve their block count.
        if config.get('block_count', 3) != 3:
            raise ValueError('FP32 analyzer model supports three blocks; use a matching model implementation for other depths')
        model = MambaHSI(in_channels=int(config['in_channels']),
                         num_classes=int(config['num_classes']), **kwargs).to(device)
        weight_path = Path(checkpoint) if checkpoint else run_dir / 'best_model.pth'
        model.load_state_dict(load_state_dict(weight_path, device), strict=True)
    # BN fusion can replace parameters/modules; normalize placement again.
    model.to(device).eval()
    assert_model_device(model, device)
    model._analysis_checkpoint_path = str(weight_path.resolve())
    model._analysis_checkpoint_kind = 'QAT' if is_qat else 'FP32'
    return model, config


def parse_seed(run_dir):
    match = re.search(r"run_seed(\d+)$", run_dir.name)
    return int(match.group(1)) if match else None


def iter_mamba_cores(model):
    block_index = 0
    for module in model.mamba:
        if not hasattr(module, "branch_mode"):
            continue
        for branch_name, attribute in (
            ("spa", "spa_mamba"),
            ("spe", "spe_mamba"),
        ):
            branch = getattr(module, attribute, None)
            if branch is not None:
                yield (
                    "block{}_{}".format(block_index, branch_name),
                    block_index,
                    branch_name,
                    branch.mamba,
                )
        block_index += 1


def relative_frobenius(matrix, approximation):
    denominator = float(np.linalg.norm(matrix))
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm(matrix - approximation) / denominator)


def extract_static_dynamics(model, run_metadata):
    spectrum_rows = []
    core_summaries = []
    core_arrays = {}
    total_a_parameters = 0

    for core_name, block_index, branch_name, core in iter_mamba_cores(model):
        if hasattr(core, "A_log_shared"):
            theta_vector = (
                core.A_log_shared.detach().float().cpu().numpy().astype(np.float64)
            )
            theta_matrix = np.repeat(
                theta_vector.reshape(1, -1), int(core.d_inner), axis=0
            )
            stored_theta = theta_vector.reshape(1, -1)
            stored_channels = ["shared"]
            mode = "shared"
            parameter_count = int(theta_vector.size)
        elif hasattr(core, "A_log"):
            theta_matrix = (
                core.A_log.detach().float().cpu().numpy().astype(np.float64)
            )
            stored_theta = theta_matrix
            stored_channels = list(range(theta_matrix.shape[0]))
            mode = "per_channel"
            parameter_count = int(theta_matrix.size)
        else:
            raise AttributeError("{} has neither A_log_shared nor A_log".format(core_name))

        total_a_parameters += parameter_count
        lambda_matrix = np.exp(theta_matrix)
        row_shared_lambda = np.mean(lambda_matrix, axis=0, keepdims=True)
        row_shared_lambda_matrix = np.repeat(
            row_shared_lambda, lambda_matrix.shape[0], axis=0
        )
        row_shared_theta = np.mean(theta_matrix, axis=0, keepdims=True)
        row_shared_theta_matrix = np.repeat(
            row_shared_theta, theta_matrix.shape[0], axis=0
        )
        singular_values = np.linalg.svd(lambda_matrix, compute_uv=False)
        singular_energy = singular_values * singular_values
        first_rank_energy = float(
            singular_energy[0] / max(float(np.sum(singular_energy)), 1e-30)
        )

        for stored_index, channel in enumerate(stored_channels):
            theta_values = stored_theta[stored_index]
            for state_index, theta in enumerate(theta_values):
                rate = math.exp(float(theta))
                spectrum_rows.append(
                    {
                        **run_metadata,
                        "core": core_name,
                        "block": block_index,
                        "branch": branch_name,
                        "A_mode": mode,
                        "channel": channel,
                        "state": state_index,
                        "A_log": float(theta),
                        "lambda": rate,
                        "continuous_tau": 1.0 / rate,
                    }
                )

        per_state_mean = np.mean(lambda_matrix, axis=0)
        per_state_std = np.std(lambda_matrix, axis=0)
        per_state_cv = per_state_std / np.maximum(np.abs(per_state_mean), 1e-30)
        core_summary = {
            "core": core_name,
            "block": block_index,
            "branch": branch_name,
            "A_mode": mode,
            "d_inner": int(core.d_inner),
            "d_state": int(core.d_state),
            "stored_A_parameter_count": parameter_count,
            "lambda_min": float(lambda_matrix.min()),
            "lambda_max": float(lambda_matrix.max()),
            "lambda_mean": float(lambda_matrix.mean()),
            "mean_statewise_channel_cv": float(per_state_cv.mean()),
            "max_statewise_channel_cv": float(per_state_cv.max()),
            "row_shared_relative_fro_error_lambda": relative_frobenius(
                lambda_matrix, row_shared_lambda_matrix
            ),
            "row_shared_relative_fro_error_A_log": relative_frobenius(
                theta_matrix, row_shared_theta_matrix
            ),
            "best_rank1_energy_ratio_lambda": first_rank_energy,
        }
        core_summaries.append(core_summary)
        core_arrays[core_name] = {
            "theta": theta_matrix,
            "lambda": lambda_matrix,
            "block": block_index,
            "branch": branch_name,
            "A_mode": mode,
        }

    return spectrum_rows, core_summaries, core_arrays, total_a_parameters


class FrozenTileDataset(Dataset):
    def __init__(self, image, tiles, tile_size):
        self.image = np.asarray(image, dtype=np.float32)
        self.tiles = list(tiles)
        self.tile_size = int(tile_size)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        top, bottom, left, right = self.tiles[index]
        tile = self.image[top:bottom, left:right]
        valid_height, valid_width = tile.shape[:2]
        pad_height = self.tile_size - valid_height
        pad_width = self.tile_size - valid_width
        if pad_height or pad_width:
            pad_mode = "reflect" if min(valid_height, valid_width) > 1 else "edge"
            tile = np.pad(
                tile,
                ((0, pad_height), (0, pad_width), (0, 0)),
                mode=pad_mode,
            )
        tile = np.ascontiguousarray(tile.transpose(2, 0, 1))
        return torch.from_numpy(tile)


def apply_saved_preprocess(raw_data, preprocess_path):
    # Share the float64 transform and output cast used by the QAT evaluation.
    from train_mambahsi_spatial_split_dense_qat import transform_with_saved_preprocess
    return transform_with_saved_preprocess(raw_data, preprocess_path)


def split_tiles(split_info, split_name):
    tile_size = int(split_info["effective_tile_size"])
    selected = []
    for block in split_info["blocks"]:
        block_split = block["split"]
        if split_name != "all" and block_split != split_name:
            continue
        region_top = int(block["top"])
        region_bottom = int(block["bottom"])
        region_left = int(block["left"])
        region_right = int(block["right"])
        for top in range(region_top, region_bottom, tile_size):
            bottom = min(top + tile_size, region_bottom)
            for left in range(region_left, region_right, tile_size):
                right = min(left + tile_size, region_right)
                selected.append((top, bottom, left, right))
    if not selected:
        raise RuntimeError("No {} tiles were found in spatial_split.json".format(split_name))
    return selected, tile_size


class RetentionAccumulator:
    """Streaming actual discrete coefficients and within-sequence retention.

    All counts weight visited coefficients/windows, not ROM addresses. Projection
    error compares row-mean lambda at fixed delta, without state alignment claims.
    """
    def __init__(self, pole_rates, fraction_bits, lags):
        self.poles = torch.as_tensor(pole_rates, dtype=torch.float64)
        self.fraction_bits = sorted(set(fraction_bits))
        self.lags = sorted(set(lags))
        self.sequence_shape = None
        self.stats = {}

    def _add(self, kind, lag, bits, values, reference=None, ones=0):
        key = (kind, lag, bits)
        row = self.stats.setdefault(key, dict(kind=kind, lag=lag, fraction_bits=bits,
            count=0, sum=0., sum_sq=0., minimum=1., maximum=0., rounded_one_count=0,
            abs_error_sum=0., squared_error=0., reference_squared=0.))
        row['count'] += values.numel()
        row['sum'] += float(values.sum())
        row['sum_sq'] += float(values.square().sum())
        row['minimum'] = min(row['minimum'], float(values.min()))
        row['maximum'] = max(row['maximum'], float(values.max()))
        row['rounded_one_count'] += int(ones)
        if reference is not None:
            diff = values - reference
            row['abs_error_sum'] += float(diff.abs().sum())
            row['squared_error'] += float(diff.square().sum())
            row['reference_squared'] += float(reference.square().sum())

    def update(self, dt_output):
        if isinstance(dt_output, (tuple, list)):
            dt_output = dt_output[0]
        if self.sequence_shape is None:
            raise RuntimeError('Missing core pre-hook sequence shape')
        batch, length = self.sequence_shape
        channels = self.poles.shape[0]
        dt = dt_output.detach().double().reshape(batch, length, channels)
        if not torch.isfinite(dt).all():
            raise FloatingPointError('Nonfinite dt in retention analysis')
        delta = F.softplus(dt)
        poles = self.poles.to(delta.device)
        if not torch.isfinite(poles).all() or torch.any(poles <= 0):
            raise FloatingPointError('Invalid continuous pole rates')
        log_a = -delta[..., None] * poles[None, None]
        actual = torch.exp(log_a)
        self._add('actual_Abar', 1, 0, actual)
        projected_log = -delta[..., None] * poles.mean(0)[None, None, None]
        self._add('row_mean_shared_projection_Abar', 1, 0, torch.exp(projected_log), actual)
        for bits in self.fraction_bits:
            codes = torch.floor(actual * 2.**bits + .5)
            self._add('rounded_Abar', 1, bits, codes * 2.**(-bits), actual,
                      (codes == 1 << bits).sum())
        # No windows cross the batch/pixel axis. For Spe length=4, no lag>4 is reported.
        zero = torch.zeros_like(log_a[:, :1])
        cumulative = torch.cat([zero, log_a.cumsum(1)], 1)
        projected_cumulative = torch.cat([zero, projected_log.cumsum(1)], 1)
        for lag in self.lags:
            if lag > length:
                continue
            retained = torch.exp(cumulative[:, lag:] - cumulative[:, :-lag])
            projected = torch.exp(projected_cumulative[:, lag:] - projected_cumulative[:, :-lag])
            self._add('actual_window_retention', lag, 0, retained)
            self._add('row_mean_shared_projection_retention', lag, 0, projected, retained)

    def rows(self):
        result = []
        for source in self.stats.values():
            row = dict(source)
            count = row['count']
            row['mean'] = row['sum'] / count
            row['std'] = math.sqrt(max(0., row['sum_sq'] / count - row['mean']**2))
            row['rounded_one_fraction'] = row['rounded_one_count'] / count if row['kind'] == 'rounded_Abar' else None
            row['mae'] = row['abs_error_sum'] / count if row['reference_squared'] > 0 else None
            row['relative_l2'] = math.sqrt(row['squared_error'] / row['reference_squared']) if row['reference_squared'] > 0 else None
            result.append(row)
        return result


class DeltaAccumulator:
    def __init__(self, channels, bins, log10_min, log10_max, device):
        if bins <= 1:
            raise ValueError("hist-bins must be greater than one")
        if log10_max <= log10_min:
            raise ValueError("delta-log10-max must exceed delta-log10-min")
        self.channels = int(channels)
        self.bins = int(bins)
        self.log10_min = float(log10_min)
        self.log10_max = float(log10_max)
        self.device = device
        self.count = torch.zeros(self.channels, dtype=torch.int64, device=device)
        self.nonfinite = torch.zeros(
            self.channels, dtype=torch.int64, device=device
        )
        self.below = torch.zeros(self.channels, dtype=torch.int64, device=device)
        self.above = torch.zeros(self.channels, dtype=torch.int64, device=device)
        self.sum = torch.zeros(self.channels, dtype=torch.float64, device=device)
        self.sum_sq = torch.zeros(
            self.channels, dtype=torch.float64, device=device
        )
        self.minimum = torch.full(
            (self.channels,), float("inf"), dtype=torch.float32, device=device
        )
        self.maximum = torch.full(
            (self.channels,), -float("inf"), dtype=torch.float32, device=device
        )
        self.histogram = torch.zeros(
            (self.channels, self.bins), dtype=torch.int64, device=device
        )

    def update(self, dt_output):
        if isinstance(dt_output, (tuple, list)):
            dt_output = dt_output[0]
        values = F.softplus(dt_output.detach().float()).reshape(-1, self.channels)
        finite = torch.isfinite(values)
        zeros = torch.zeros((), dtype=values.dtype, device=values.device)
        finite_values = torch.where(finite, values, zeros)

        self.count += finite.sum(dim=0)
        self.nonfinite += (~finite).sum(dim=0)
        self.sum += finite_values.double().sum(dim=0)
        self.sum_sq += (finite_values.double() ** 2).sum(dim=0)

        positive_infinity = torch.full_like(values, float("inf"))
        negative_infinity = torch.full_like(values, -float("inf"))
        self.minimum = torch.minimum(
            self.minimum,
            torch.where(finite, values, positive_infinity).min(dim=0).values,
        )
        self.maximum = torch.maximum(
            self.maximum,
            torch.where(finite, values, negative_infinity).max(dim=0).values,
        )

        low_value = 10.0 ** self.log10_min
        high_value = 10.0 ** self.log10_max
        self.below += (finite & (values < low_value)).sum(dim=0)
        self.above += (finite & (values > high_value)).sum(dim=0)

        clipped = values.clamp(min=low_value, max=high_value)
        scaled = (
            (torch.log10(clipped) - self.log10_min)
            / (self.log10_max - self.log10_min)
            * self.bins
        )
        bin_index = torch.floor(scaled).to(torch.int64)
        bin_index.clamp_(0, self.bins - 1)
        channel_index = torch.arange(
            self.channels, device=values.device, dtype=torch.int64
        ).view(1, self.channels)
        combined = channel_index * self.bins + bin_index
        combined = combined.expand_as(values)[finite]
        counts = torch.bincount(
            combined,
            minlength=self.channels * self.bins,
        ).reshape(self.channels, self.bins)
        self.histogram += counts

    def to_numpy(self):
        return {
            "count": self.count.cpu().numpy(),
            "nonfinite": self.nonfinite.cpu().numpy(),
            "below": self.below.cpu().numpy(),
            "above": self.above.cpu().numpy(),
            "sum": self.sum.cpu().numpy(),
            "sum_sq": self.sum_sq.cpu().numpy(),
            "minimum": self.minimum.cpu().numpy(),
            "maximum": self.maximum.cpu().numpy(),
            "histogram": self.histogram.cpu().numpy(),
            "log10_min": self.log10_min,
            "log10_max": self.log10_max,
            "bins": self.bins,
        }


def histogram_quantile(histogram, quantile, log10_min, log10_max):
    total = int(np.sum(histogram))
    if total <= 0:
        return float("nan")
    target = quantile * max(total - 1, 0)
    index = int(np.searchsorted(np.cumsum(histogram), target, side="right"))
    index = min(max(index, 0), histogram.size - 1)
    width = (log10_max - log10_min) / histogram.size
    return float(10.0 ** (log10_min + (index + 0.5) * width))


def summarize_delta(core_name, accumulator, run_metadata):
    arrays = accumulator.to_numpy()
    rows = []
    for channel in range(accumulator.channels):
        count = int(arrays["count"][channel])
        if count:
            mean = float(arrays["sum"][channel] / count)
            variance = max(
                float(arrays["sum_sq"][channel] / count - mean * mean),
                0.0,
            )
            std = math.sqrt(variance)
        else:
            mean = float("nan")
            std = float("nan")
        histogram = arrays["histogram"][channel]
        quantiles = {
            "delta_p01": histogram_quantile(
                histogram, 0.01, arrays["log10_min"], arrays["log10_max"]
            ),
            "delta_p10": histogram_quantile(
                histogram, 0.10, arrays["log10_min"], arrays["log10_max"]
            ),
            "delta_p50": histogram_quantile(
                histogram, 0.50, arrays["log10_min"], arrays["log10_max"]
            ),
            "delta_p90": histogram_quantile(
                histogram, 0.90, arrays["log10_min"], arrays["log10_max"]
            ),
            "delta_p99": histogram_quantile(
                histogram, 0.99, arrays["log10_min"], arrays["log10_max"]
            ),
        }
        rows.append(
            {
                **run_metadata,
                "core": core_name,
                "channel": channel,
                "count": count,
                "nonfinite_count": int(arrays["nonfinite"][channel]),
                "below_histogram_count": int(arrays["below"][channel]),
                "above_histogram_count": int(arrays["above"][channel]),
                "delta_mean": mean,
                "delta_std": std,
                "delta_min": float(arrays["minimum"][channel]),
                "delta_max": float(arrays["maximum"][channel]),
                **quantiles,
            }
        )
    return rows, arrays


def make_memory_horizon_rows(delta_rows, core_arrays, run_metadata):
    rows = []
    by_core_channel = {
        (row["core"], int(row["channel"])): row for row in delta_rows
    }
    for core_name, values in core_arrays.items():
        rates = values["lambda"]
        for channel in range(rates.shape[0]):
            delta = by_core_channel[(core_name, channel)]
            d10 = float(delta["delta_p10"])
            d50 = float(delta["delta_p50"])
            d90 = float(delta["delta_p90"])
            for state, rate in enumerate(rates[channel]):
                rate = float(rate)
                rows.append(
                    {
                        **run_metadata,
                        "core": core_name,
                        "block": values["block"],
                        "branch": values["branch"],
                        "A_mode": values["A_mode"],
                        "channel": channel,
                        "state": state,
                        "lambda": rate,
                        "delta_p10": d10,
                        "delta_p50": d50,
                        "delta_p90": d90,
                        # L=1/(delta*lambda) reverses the delta quantiles.
                        "horizon_p10_steps": 1.0 / max(d90 * rate, 1e-30),
                        "horizon_p50_steps": 1.0 / max(d50 * rate, 1e-30),
                        "horizon_p90_steps": 1.0 / max(d10 * rate, 1e-30),
                        "Abar_at_delta_p50": math.exp(-d50 * rate),
                    }
                )
    return rows


def load_frozen_dataset(dataset_name, data_path, dataset_dir, split_name):
    # Import lazily so --static-only does not require scipy or dataset files.

    from train_mambahsi_spatial_split_dense_qat import load_dataset
    raw_data, _, _ = load_dataset(dataset_name, str(data_path))
    raw_data = np.asarray(raw_data)
    if raw_data.ndim != 3:
        raise ValueError("Expected HxWxBands data, got {}".format(raw_data.shape))
    image = apply_saved_preprocess(
        raw_data,
        dataset_dir / "train_only_preprocess.npz",
    )
    split_info = read_json(dataset_dir / "spatial_split.json")
    tiles, tile_size = split_tiles(split_info, split_name)
    return FrozenTileDataset(image, tiles, tile_size), tile_size


def optional_plots(output_dir, core_arrays, delta_arrays, memory_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; CSV/JSON outputs were still written.")
        return

    names = list(core_arrays)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(names), 2)))

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for color, name in zip(colors, names):
        rates = core_arrays[name]["lambda"]
        states = np.arange(1, rates.shape[1] + 1)
        median = np.median(rates, axis=0)
        low = np.quantile(rates, 0.10, axis=0)
        high = np.quantile(rates, 0.90, axis=0)
        axis.plot(states, median, marker="o", ms=3, lw=1.4, label=name, color=color)
        if rates.shape[0] > 1 and not np.allclose(low, high):
            axis.fill_between(states, low, high, color=color, alpha=0.12)
    axis.set_yscale("log")
    axis.set_xlabel("State index")
    axis.set_ylabel(r"Learned pole rate $\lambda=\exp(A_{\log})$")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(ncol=2, fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "lambda_spectrum.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "lambda_spectrum.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    if delta_arrays:
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        for color, name in zip(colors, names):
            arrays = delta_arrays[name]
            histogram = arrays["histogram"].sum(axis=0).astype(np.float64)
            if histogram.sum() == 0:
                continue
            histogram /= histogram.sum()
            edges = np.linspace(
                arrays["log10_min"], arrays["log10_max"], arrays["bins"] + 1
            )
            centers = 10.0 ** (0.5 * (edges[:-1] + edges[1:]))
            axis.plot(centers, histogram, lw=1.4, label=name, color=color)
        axis.set_xscale("log")
        axis.set_xlabel(r"Input-conditioned step size $\Delta$")
        axis.set_ylabel("Probability mass per log bin")
        axis.grid(True, which="both", alpha=0.22)
        axis.legend(ncol=2, fontsize=8, frameon=False)
        figure.tight_layout()
        figure.savefig(output_dir / "delta_distribution.pdf", bbox_inches="tight")
        figure.savefig(
            output_dir / "delta_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close(figure)

    if memory_rows:
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        for color, name in zip(colors, names):
            selected = [row for row in memory_rows if row["core"] == name]
            state_count = core_arrays[name]["lambda"].shape[1]
            matrix = np.full((core_arrays[name]["lambda"].shape[0], state_count), np.nan)
            for row in selected:
                matrix[int(row["channel"]), int(row["state"])] = row[
                    "horizon_p50_steps"
                ]
            states = np.arange(1, state_count + 1)
            median = np.nanmedian(matrix, axis=0)
            low = np.nanquantile(matrix, 0.10, axis=0)
            high = np.nanquantile(matrix, 0.90, axis=0)
            axis.plot(states, median, marker="o", ms=3, lw=1.4, label=name, color=color)
            axis.fill_between(states, low, high, color=color, alpha=0.12)
        axis.set_yscale("log")
        axis.set_xlabel("State index")
        axis.set_ylabel("Median e-fold memory horizon (sequence steps)")
        axis.grid(True, which="both", alpha=0.22)
        axis.legend(ncol=2, fontsize=8, frameon=False)
        figure.tight_layout()
        figure.savefig(output_dir / "memory_horizon.pdf", bbox_inches="tight")
        figure.savefig(output_dir / "memory_horizon.png", dpi=300, bbox_inches="tight")
        plt.close(figure)


def analyze_run(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    dataset_dir = locate_artifact_dir(run_dir, args.artifact_dir)
    split_info = read_json(dataset_dir / "spatial_split.json")
    dataset_name = str(split_info["dataset"])
    if args.dataset and dataset_name != args.dataset:
        raise ValueError(f"Dataset mismatch: expected {args.dataset}, got {dataset_name}")
    seed = parse_seed(run_dir)
    device = resolve_device(args.device)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / "learned_dynamics_{}".format(args.split)
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite analysis output: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = restore_model(run_dir, device, args.checkpoint, args.artifact_dir)
    run_metadata = {
        "dataset": dataset_name,
        "seed": seed,
        "run_dir": str(run_dir),
        "variant": config.get("model_variant", "unknown"),
    }
    spectrum_rows, core_summaries, core_arrays, total_a_parameters = (
        extract_static_dynamics(model, run_metadata)
    )

    spectrum_fields = [
        "dataset",
        "seed",
        "run_dir",
        "variant",
        "core",
        "block",
        "branch",
        "A_mode",
        "channel",
        "state",
        "A_log",
        "lambda",
        "continuous_tau",
    ]
    write_csv(output_dir / "alog_spectrum.csv", spectrum_rows, spectrum_fields)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    compatibility_parameters = sum(
        parameter.numel() for parameter in model.shared_params.parameters()
    )
    summary = {
        **run_metadata,
        "checkpoint": model._analysis_checkpoint_path,
        "checkpoint_kind": model._analysis_checkpoint_kind,
        "checkpoint_sha256": hashlib.sha256(Path(model._analysis_checkpoint_path).read_bytes()).hexdigest(),
        "model_config": config,
        "selective_scan_backend": __import__(type(model).__module__, fromlist=["selective_scan_backend"]).selective_scan_backend(),
        "analysis_split": args.split,
        "total_parameters": int(total_parameters),
        "effective_forward_parameters": int(
            total_parameters - compatibility_parameters
        ),
        "total_A_log_parameters": int(total_a_parameters),
        "core_static_dynamics": core_summaries,
        "dynamic_analysis_performed": not args.static_only,
    }

    delta_rows = []
    delta_arrays = {}
    memory_rows = []
    if not args.static_only:
        dataset, tile_size = load_frozen_dataset(
            dataset_name,
            Path(args.data_path),
            dataset_dir,
            args.split,
        )
        original_tile_count = len(dataset)
        if args.max_tiles > 0 and args.max_tiles < original_tile_count:
            indices = np.linspace(
                0,
                original_tile_count - 1,
                args.max_tiles,
                dtype=np.int64,
            )
            dataset = Subset(dataset, indices.tolist())
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        accumulators = {}
        retention_accumulators = {}
        handles = []
        for core_name, _, _, core in iter_mamba_cores(model):
            accumulator = DeltaAccumulator(
                channels=int(core.d_inner),
                bins=args.hist_bins,
                log10_min=args.delta_log10_min,
                log10_max=args.delta_log10_max,
                device=device,
            )
            accumulators[core_name] = accumulator

            retention = RetentionAccumulator(core_arrays[core_name]['lambda'],
                                             args.coefficient_fraction_bits, args.retention_lags)
            retention_accumulators[core_name] = retention

            def capture_shape(_module, inputs, target=retention):
                target.sequence_shape = tuple(inputs[0].shape[:2])

            def capture(_module, _inputs, output, target=accumulator, dynamics=retention):
                target.update(output)
                dynamics.update(output)

            handles.append(core.register_forward_pre_hook(capture_shape))
            # QAT dt_proj output is not yet the boundary-quantized time step.
            boundary = getattr(core, 'dt_output_quant', core.dt_proj)
            handles.append(boundary.register_forward_hook(capture))

        processed = 0
        try:
            with torch.inference_mode():
                for batch_index, images in enumerate(loader):
                    images = images.to(
                        device,
                        dtype=torch.float32,
                        non_blocking=device.type == "cuda",
                    )
                    model(images)
                    processed += int(images.shape[0])
                    if (batch_index + 1) % 25 == 0 or processed == len(dataset):
                        print(
                            "dynamic forward: {}/{} tiles".format(
                                processed, len(dataset)
                            )
                        )
        finally:
            for handle in handles:
                handle.remove()

        for core_name, accumulator in accumulators.items():
            rows, arrays = summarize_delta(
                core_name, accumulator, run_metadata
            )
            delta_rows.extend(rows)
            delta_arrays[core_name] = arrays

        retention_rows = [dict(**run_metadata, core=name, **row)
                          for name, accumulator in retention_accumulators.items()
                          for row in accumulator.rows()]
        if retention_rows:
            write_csv(output_dir / 'discrete_retention_stats.csv', retention_rows, list(retention_rows[0]))
        summary['discrete_retention_definition'] = (
            'Actual sequential windows exp(-lambda*sum(delta)); no cross-pixel/tile windows. '
            'Shared projection uses channel-mean lambda and identical delta, not retraining.')
        delta_fields = [
            "dataset",
            "seed",
            "run_dir",
            "variant",
            "core",
            "channel",
            "count",
            "nonfinite_count",
            "below_histogram_count",
            "above_histogram_count",
            "delta_mean",
            "delta_std",
            "delta_min",
            "delta_max",
            "delta_p01",
            "delta_p10",
            "delta_p50",
            "delta_p90",
            "delta_p99",
        ]
        write_csv(
            output_dir / "delta_channel_stats.csv", delta_rows, delta_fields
        )
        memory_rows = make_memory_horizon_rows(
            delta_rows, core_arrays, run_metadata
        )
        memory_fields = [
            "dataset",
            "seed",
            "run_dir",
            "variant",
            "core",
            "block",
            "branch",
            "A_mode",
            "channel",
            "state",
            "lambda",
            "delta_p10",
            "delta_p50",
            "delta_p90",
            "horizon_p10_steps",
            "horizon_p50_steps",
            "horizon_p90_steps",
            "Abar_at_delta_p50",
        ]
        write_csv(
            output_dir / "memory_horizon_stats.csv",
            memory_rows,
            memory_fields,
        )
        histogram_payload = {}
        for core_name, arrays in delta_arrays.items():
            safe_name = core_name.replace("/", "_")
            histogram_payload[safe_name + "_histogram"] = arrays["histogram"]
            histogram_payload[safe_name + "_log10_edges"] = np.linspace(
                arrays["log10_min"],
                arrays["log10_max"],
                arrays["bins"] + 1,
            )
        np.savez_compressed(
            output_dir / "delta_histograms.npz", **histogram_payload
        )
        summary.update(
            {
                "tile_size": int(tile_size),
                "available_tiles_in_split": int(original_tile_count),
                "analyzed_tiles": int(len(dataset)),
                "delta_histogram_bins": int(args.hist_bins),
                "delta_histogram_log10_range": [
                    float(args.delta_log10_min),
                    float(args.delta_log10_max),
                ],
                "quantile_method": (
                    "approximate quantiles from a streaming log10 histogram"
                ),
                "memory_horizon_definition": "L=1/(delta*exp(A_log))",
            }
        )

    write_json(output_dir / "analysis_summary.json", summary)
    if not args.no_plots:
        optional_plots(output_dir, core_arrays, delta_arrays, memory_rows)

    print("Analysis finished: {}".format(output_dir))
    print("A_log parameters: {}".format(total_a_parameters))
    if not args.static_only:
        print(
            "Dynamic delta statistics were estimated on {} {} tile(s).".format(
                len(dataset), args.split
            )
        )

    return summary


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.max_tiles < 0:
        parser.error('batch-size must be positive and max-tiles nonnegative')
    if any(f < 1 or f > 30 for f in args.coefficient_fraction_bits) or any(l < 1 for l in args.retention_lags):
        parser.error('coefficient fractional bits must be 1..30; lags must be positive')
    explicit = {Path(p).expanduser().resolve() for p in args.run_dir}
    selected = set(explicit)
    for pattern in args.run_glob:
        matches = glob.glob(str(Path(pattern).expanduser()), recursive=True)
        if not matches:
            parser.error(f'No runs match {pattern}')
        for p in matches:
            run = Path(p).resolve()
            if args.dataset and run not in explicit:
                root = locate_artifact_dir(run, args.artifact_dir)
                if read_json(root / 'spatial_split.json')['dataset'] != args.dataset:
                    continue
            selected.add(run)
    if not selected:
        parser.error('Provide --run-dir or --run-glob matching at least one run')
    if len(selected) > 1 and (args.checkpoint or args.artifact_dir):
        parser.error('--checkpoint and --artifact-dir overrides require a single run')
    summaries = []
    if len(selected) > 1 and not args.output_dir:
        parser.error('Multi-run analysis requires --output-dir for an unambiguous aggregate')
    root = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if len(selected) > 1 and root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to mix old aggregate results: {root}')
    for run in sorted(selected):
        one = copy.copy(args)
        one.run_dir = str(run)
        if len(selected) > 1:
            digest = hashlib.sha256(str(run).encode()).hexdigest()[:10]
            dataset_name = read_json(locate_artifact_dir(run) / 'spatial_split.json')['dataset']
            one.output_dir = str(root / f'{dataset_name}_{run.parent.name}_{run.name}_{digest}')
        summary = analyze_run(one)
        summaries.append(summary)
    if len(selected) > 1:
        write_json(root / 'multi_run_summary.json', summaries)
        rows = [dict(dataset=s['dataset'], seed=s['seed'], variant=s['variant'],
                     run_dir=s['run_dir'], **core) for s in summaries for core in s['core_static_dynamics']]
        write_csv(root / 'multi_run_pole_summary.csv', rows, list(rows[0]))
        dynamic_rows = []
        for child in sorted(root.glob('*/discrete_retention_stats.csv')):
            with child.open(newline='') as handle:
                dynamic_rows.extend(csv.DictReader(handle))
        if dynamic_rows:
            write_csv(root / 'multi_run_retention_summary.csv', dynamic_rows, list(dynamic_rows[0]))


if __name__ == "__main__":
    main()
