"""Configurable MambaHSI models for controlled architecture ablations.

The ``current`` preset reproduces the architecture used by the deployment
pipeline: two branches, sum fusion, a ``2*x`` block skip, no z gate, no D
term, shared A, BatchNorm, ReLU and a 64-channel classification head.

The ``original_safe`` preset restores the main choices from the original
MambaHSI implementation while keeping the spatial branch batch-safe.  The
model deliberately keeps all ablation switches in one implementation so that
data splitting, training and evaluation do not change between variants.
"""

import math
import warnings
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn as _optimized_selective_scan_fn,
    )
except ImportError:
    _optimized_selective_scan_fn = None


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
    # Alias kept for concise experiment commands.  It intentionally uses the
    # batch-safe Spa tokenization rather than reproducing the original bug.
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
    # ``custom`` starts from the current deployment model.  Every field may
    # then be overridden explicitly from the command line.
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
    """Resolve a preset and explicit CLI overrides into one validated dict."""
    if model_variant not in MODEL_VARIANT_PRESETS:
        choices = ", ".join(MODEL_VARIANT_NAMES)
        raise ValueError(
            "unknown model_variant={!r}; choose one of {}".format(
                model_variant,
                choices,
            )
        )

    unknown = sorted(set(overrides) - set(MODEL_CONFIG_FIELDS))
    if unknown:
        raise ValueError("unknown model configuration fields: {}".format(unknown))

    config = deepcopy(MODEL_VARIANT_PRESETS[model_variant])
    for name, value in overrides.items():
        if value is not None:
            config[name] = value

    if config["branch_mode"] not in {"spa", "spe", "both"}:
        raise ValueError("branch_mode must be spa, spe or both")
    if config["fusion_mode"] not in {"sum", "mean", "softmax"}:
        raise ValueError("fusion_mode must be sum, mean or softmax")
    if config["branch_mode"] != "both" and config["fusion_mode"] != "sum":
        raise ValueError(
            "fusion_mode is only meaningful for branch_mode=both; "
            "use fusion_mode=sum for a single-branch ablation"
        )
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
    if int(config["hidden_dim"]) <= 0:
        raise ValueError("hidden_dim must be greater than zero")
    if int(config["token_num"]) <= 0:
        raise ValueError("token_num must be greater than zero")
    if int(config["d_state"]) <= 0:
        raise ValueError("d_state must be greater than zero")

    config["skip_scale"] = int(config["skip_scale"])
    config["hidden_dim"] = int(config["hidden_dim"])
    config["head_dim"] = int(config["head_dim"])
    config["token_num"] = int(config["token_num"])
    config["d_state"] = int(config["d_state"])
    config["use_z"] = bool(config["use_z"])
    config["use_D"] = bool(config["use_D"])
    return {"model_variant": model_variant, **config}


def selective_scan_backend():
    if _optimized_selective_scan_fn is None:
        return "torch_reference"
    return "mamba_ssm_optimized"


_REFERENCE_WARNING_EMITTED = False


def _reference_selective_scan(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    delta_bias=None,
    delta_softplus=True,
):
    """Differentiable fallback for non-CUDA input or an unavailable extension.

    It follows the recurrent selective-scan equations and is intentionally
    simple.  Production GPU experiments should report the optimized backend.
    """
    global _REFERENCE_WARNING_EMITTED
    if not _REFERENCE_WARNING_EMITTED:
        warnings.warn(
            "Using the slow PyTorch selective-scan reference: input is not "
            "on CUDA or the mamba_ssm CUDA extension is unavailable.",
            RuntimeWarning,
        )
        _REFERENCE_WARNING_EMITTED = True

    if delta_bias is not None:
        delta = delta + delta_bias.to(delta.dtype).view(1, -1, 1)
    if delta_softplus:
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
        dB = delta_t * B_t
        state = state * dA + input_t * dB
        output_t = torch.sum(state * C_t, dim=-1)
        if D is not None:
            output_t = output_t + u[:, :, index] * D.to(u.dtype).view(1, -1)
        outputs.append(output_t)

    return torch.stack(outputs, dim=-1)


def run_selective_scan(u, delta, A, B, C, D, delta_bias):
    if u.is_cuda and _optimized_selective_scan_fn is not None:
        return _optimized_selective_scan_fn(
            u,
            delta,
            A,
            B,
            C,
            D=D,
            z=None,
            delta_bias=delta_bias,
            delta_softplus=True,
            return_last_state=False,
        )
    return _reference_selective_scan(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        delta_bias=delta_bias,
        delta_softplus=True,
    )


def make_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError("unsupported activation: {}".format(name))


def valid_group_count(channels, requested_groups):
    groups = min(int(requested_groups), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def make_spatial_norm(norm_path, channels, group_num):
    if norm_path == "bn":
        return nn.BatchNorm2d(channels)
    return nn.GroupNorm(valid_group_count(channels, group_num), channels)


class SharedSoftParams(nn.Module):
    """Checkpoint-compatible placeholder retained from the deployment model.

    The current MambaHSI source registers these tensors but never consumes
    them in its effective forward graph.  Keeping the names is necessary for
    strict loading by the existing QAT and FPGA scripts.  They remain equally
    present in every variant, so they do not bias cross-variant comparisons.
    """

    def __init__(self, sharpness=3.0, clip_value=7.0):
        super().__init__()
        self.n_segments = 3
        self.sharpness = float(sharpness)
        self.clip_value = float(clip_value)
        self.a = nn.Parameter(
            torch.tensor([-0.05, 0.2, 1.0], dtype=torch.float32)
        )
        self.b = nn.Parameter(
            torch.tensor([-0.35, 0.0, 0.0], dtype=torch.float32)
        )
        self.t = nn.Parameter(
            torch.tensor([-1.4, 0.0], dtype=torch.float32)
        )
        self.register_buffer(
            "fixed_ends",
            torch.tensor([-clip_value, clip_value], dtype=torch.float32),
        )


class ConfigurableMamba(nn.Module):
    """Mamba core with independent z, D, A, normalization and activation axes."""

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
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.expand * self.d_model
        self.dt_rank = (
            math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        )
        self.use_z = bool(use_z)
        self.use_D = bool(use_D)
        self.A_mode = A_mode
        self.norm_path = norm_path
        self.activation = activation

        projection_factor = 2 if self.use_z else 1
        self.in_proj = nn.Linear(
            self.d_model,
            self.d_inner * projection_factor,
            bias=bias,
        )
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=conv_bias,
        )
        self.act = make_activation(activation)
        # z is the native Mamba SiLU gate.  Keeping it fixed makes --use_z
        # independent from the separate ReLU/SiLU activation-path ablation.
        self.z_act = nn.SiLU()
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inverse_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus_dt)
        self.dt_proj.bias._no_reinit = True

        base_A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        if A_mode == "shared":
            self.A_log_shared = nn.Parameter(torch.log(base_A))
            self.A_log_shared._no_weight_decay = True
        elif A_mode == "per_channel":
            expanded_A = base_A.unsqueeze(0).repeat(self.d_inner, 1)
            self.A_log = nn.Parameter(torch.log(expanded_A))
            self.A_log._no_weight_decay = True
        else:
            raise ValueError("A_mode must be shared or per_channel")

        if self.use_D:
            self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))
            self.D._no_weight_decay = True
        else:
            # Retain the key used by the current deployment checkpoint while
            # ensuring the disabled D term is never passed into selective scan.
            self.register_buffer(
                "D",
                torch.zeros(self.d_inner, dtype=torch.float32),
            )

        # These names match the current deployment/QAT checkpoint contract.
        self.out_proj_linear = nn.Linear(self.d_inner, self.d_model, bias=bias)
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
            raise ValueError(
                "Mamba input must have shape [batch, length, channels], got {}".format(
                    tuple(hidden_states.shape)
                )
            )
        sequence_length = hidden_states.shape[1]
        projected = self.in_proj(hidden_states).transpose(1, 2).contiguous()
        if self.use_z:
            x, z = projected.chunk(2, dim=1)
        else:
            x = projected
            z = None

        x = self.conv1d(x)[..., :sequence_length]
        x = self.act(x)

        x_dbl = self.x_proj(x.transpose(1, 2).reshape(-1, self.d_inner))
        dt_low_rank, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        if hasattr(self, "b_output_quant"):
            B = self.b_output_quant(B)
        if hasattr(self, "c_output_quant"):
            C = self.c_output_quant(C)

        # Add dt_proj.bias exactly once here, matching the current QAT/FPGA
        # contract where the quantized dt_proj output already contains bias.
        # selective_scan therefore receives delta_bias=None below.
        dt = self.dt_proj(dt_low_rank)
        if hasattr(self, "dt_output_quant"):
            dt = self.dt_output_quant(dt)
        batch_size = hidden_states.shape[0]
        dt = dt.reshape(batch_size, sequence_length, self.d_inner)
        dt = dt.transpose(1, 2).contiguous()
        B = B.reshape(batch_size, sequence_length, self.d_state)
        B = B.transpose(1, 2).contiguous()
        C = C.reshape(batch_size, sequence_length, self.d_state)
        C = C.transpose(1, 2).contiguous()

        D = self.D.float() if self.use_D else None
        if D is not None and hasattr(self, "d_weight_quant"):
            D = self.d_weight_quant(D)
        y = run_selective_scan(
            x,
            dt,
            self.continuous_A(),
            B,
            C,
            D,
            None,
        )
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


def make_branch_projection(channels, norm_path, activation, group_num):
    layers = []
    if norm_path == "gn":
        layers.append(
            nn.GroupNorm(
                valid_group_count(channels, group_num),
                channels,
            )
        )
    layers.append(make_activation(activation))
    return nn.Sequential(*layers)


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
        # Compatibility attributes used by the current QAT/FPGA tooling.
        # Branch-local residuals remain disabled for every controlled ablation;
        # the only residual is applied once in BothMamba.
        self.use_residual = False
        self.use_proj = True
        self.mamba = ConfigurableMamba(
            d_model=channels,
            d_state=d_state,
            use_z=use_z,
            use_D=use_D,
            A_mode=A_mode,
            norm_path=norm_path,
            activation=activation,
        )
        self.proj = make_branch_projection(
            channels,
            norm_path,
            activation,
            group_num,
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        sequence = x.permute(0, 2, 3, 1).contiguous()
        sequence = sequence.reshape(batch_size, height * width, channels)
        sequence = self.mamba(sequence)
        output = sequence.reshape(batch_size, height, width, channels)
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
        self.group_channel_num = math.ceil(channels / token_num)
        self.padded_channels = self.token_num * self.group_channel_num
        # Legacy name retained so the existing current-model FPGA simulator
        # can consume the configurable current preset without source forks.
        self.channel_num = self.padded_channels
        self.mamba = ConfigurableMamba(
            d_model=self.group_channel_num,
            d_state=d_state,
            use_z=use_z,
            use_D=use_D,
            A_mode=A_mode,
            norm_path=norm_path,
            activation=activation,
        )
        self.proj = make_branch_projection(
            self.padded_channels,
            norm_path,
            activation,
            group_num,
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        if channels != self.input_channels:
            raise ValueError(
                "SpeMamba expected {} channels, got {}".format(
                    self.input_channels,
                    channels,
                )
            )
        if channels < self.padded_channels:
            padding = x.new_zeros(
                batch_size,
                self.padded_channels - channels,
                height,
                width,
            )
            x = torch.cat([x, padding], dim=1)

        sequence = x.permute(0, 2, 3, 1).contiguous()
        sequence = sequence.reshape(
            batch_size * height * width,
            self.token_num,
            self.group_channel_num,
        )
        sequence = self.mamba(sequence)
        output = sequence.reshape(
            batch_size,
            height,
            width,
            self.padded_channels,
        )
        output = output.permute(0, 3, 1, 2).contiguous()
        output = self.proj(output)
        return output[:, : self.input_channels]


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
        self.skip_scale = float(skip_scale)
        self.use_att = branch_mode == "both" and fusion_mode == "softmax"
        self.use_residual = True

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
        if branch_mode in {"spa", "both"}:
            self.spa_mamba = SpaMamba(**common)
        else:
            self.spa_mamba = None
        if branch_mode in {"spe", "both"}:
            self.spe_mamba = SpeMamba(token_num=token_num, **common)
        else:
            self.spe_mamba = None

        if branch_mode == "both" and fusion_mode == "softmax":
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
            spa_output = self.spa_mamba(x)
            spe_output = self.spe_mamba(x)
            if hasattr(self, "spa_residual_quant"):
                spa_output = self.spa_residual_quant(spa_output)
            if hasattr(self, "spe_residual_quant"):
                spe_output = self.spe_residual_quant(spe_output)
            if self.fusion_mode == "sum":
                fused = spa_output + spe_output
            elif self.fusion_mode == "mean":
                fused = 0.5 * (spa_output + spe_output)
            elif self.fusion_mode == "softmax":
                weights = self.softmax(self.weights)
                if hasattr(self, "fusion_weight_quant"):
                    weights = self.fusion_weight_quant(weights)
                fused = weights[0] * spa_output + weights[1] * spe_output
            else:
                raise RuntimeError(
                    "unsupported fusion mode: {}".format(self.fusion_mode)
                )
        if hasattr(self, "fusion_quant"):
            fused = self.fusion_quant(fused)
        output = fused + self.skip_scale * x
        if hasattr(self, "block_output_quant"):
            output = self.block_output_quant(output)
        return output


class MambaHSI(nn.Module):
    def __init__(
        self,
        in_channels=128,
        hidden_dim=64,
        num_classes=10,
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
            raise ValueError(
                "single-branch models require fusion_mode=sum as a placeholder"
            )
        if skip_scale not in {0, 1, 2}:
            raise ValueError("skip_scale must be 0, 1 or 2")
        if int(hidden_dim) <= 0 or int(token_num) <= 0 or int(d_state) <= 0:
            raise ValueError("hidden_dim, token_num and d_state must be positive")

        self.mamba_type = branch_mode
        self.branch_mode = branch_mode
        self.fusion_mode = fusion_mode
        self.skip_scale = int(skip_scale)
        self.token_num = int(token_num)
        self.d_state = int(d_state)

        self.shared_params = SharedSoftParams(
            sharpness=3.0,
            clip_value=7.0,
        )
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
            raise ValueError("MambaHSI input must have shape [B, C, H, W]")
        if x.shape[-2] % 4 != 0 or x.shape[-1] % 4 != 0:
            raise ValueError("input height and width must both be divisible by 4")
        x = self.patch_embedding(x)
        if hasattr(self, "patch_output_quant"):
            x = self.patch_output_quant(x)
        x = self.mamba(x)
        x = self.cls_head(x)
        if hasattr(self, "logits_quant"):
            x = self.logits_quant(x)
        return x
