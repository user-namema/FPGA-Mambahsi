"""Numerical contracts and same-input SSM error replay (no FPGA timing claims).

Default arithmetic is UQ1.24 A, unsigned 19-bit K, signed32/Q24 H,
HA0 once after a common-domain sum. Non-integer source-isolation modes are
explicit diagnostic counterfactuals, not synthesizable RTL implementations.
"""
from dataclasses import asdict, dataclass, replace
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SSMNumericConfig:
    dt_input_bits: int = 9
    dt_output_bits: int = 8
    a_fraction_bits: int = 24
    k_bits: int = 19
    k_preferred_fraction_bits: int = 24
    state_bits: int = 32
    state_fraction_bits: int = 24
    rounding: str = 'single'
    error_source: str = 'all'
    accumulator_bits: int = 65
    coefficient_backend: str = "exact"
    pwl_segments: int = 32
    d_coefficient_bits: int = 27
    d_max_left_shift: int = 24

    def __post_init__(self):
        for name, lo, hi in [('dt_input_bits', 2, 16), ('dt_output_bits', 2, 12),
                             ('a_fraction_bits', 1, 30), ('k_bits', 2, 30),
                             ('k_preferred_fraction_bits', 0, 30),
                             ('state_bits', 2, 32), ('state_fraction_bits', 0, 30),
                             ('accumulator_bits', 2, 127), ('pwl_segments', 2, 4096),
                             ('d_coefficient_bits', 2, 32), ('d_max_left_shift', 0, 40)]:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not lo <= value <= hi:
                raise ValueError(f'{name} must be an integer in [{lo},{hi}]')
        if self.coefficient_backend not in ('exact', 'pwl'):
            raise ValueError('coefficient_backend must be exact or pwl')
        if self.rounding not in ('single', 'separate'):
            raise ValueError('rounding must be single or separate')
        if self.error_source not in ('all', 'none', 'a-only', 'k-only', 'state-only', 'd-only'):
            raise ValueError('invalid error_source')
        if self.error_source != 'all' and self.rounding != 'single':
            raise ValueError('source isolation uses single writeback; compare separate under all')

    @classmethod
    def from_dict(cls, value=None):
        return value if isinstance(value, cls) else cls(**(value or {}))

    def to_dict(self):
        return asdict(self)

    @property
    def dt_min(self):
        return -(1 << (self.dt_output_bits - 1))

    @property
    def dt_count(self):
        return 1 << self.dt_output_bits

    @property
    def state_min(self):
        return -(1 << (self.state_bits - 1))

    @property
    def state_max(self):
        return (1 << (self.state_bits - 1)) - 1

    @property
    def rtl_baseline_compatible(self):
        return self == SSMNumericConfig()


def add_numeric_arguments(parser, boundary_defaults=True):
    defaults = SSMNumericConfig()
    for name in defaults.to_dict():
        if name in ('rounding', 'error_source', 'coefficient_backend'):
            choices = {'rounding': ('single','separate'), 'error_source': ('all','none','a-only','k-only','state-only','d-only'), 'coefficient_backend': ('exact','pwl')}[name]
            parser.add_argument('--ssm-' + name.replace('_', '-'), choices=choices,
                                default=getattr(defaults, name))
        else:
            flag = '--' + name.replace('_', '-') if name.startswith('dt_') else '--ssm-' + name.replace('_', '-')
            default = getattr(defaults, name)
            if not boundary_defaults and name.startswith('dt_'):
                default = None
            parser.add_argument(flag, type=int, default=default)


def config_from_args(args, checkpoint_config=None):
    saved = SSMNumericConfig.from_dict(checkpoint_config)
    values = {}
    for name in saved.to_dict():
        dest = name if name.startswith('dt_') else 'ssm_' + name
        value = getattr(args, dest, None)
        values[name] = getattr(saved, name) if value is None else value
    return SSMNumericConfig(**values)


def round_ha0(x):
    return torch.sign(x) * torch.floor(torch.abs(x) + .5)


def choose_fraction(values, bits, preferred):
    if not torch.isfinite(values).all() or torch.any(values < 0):
        raise ValueError('K must be finite and nonnegative')
    for frac in range(preferred, -1, -1):
        if torch.all(round_ha0(values * 2.**frac) <= (1 << bits) - 1):
            return frac
    raise OverflowError('K cannot be represented even at fraction_bits=0')


def certify_update_range(a_codes, k_codes, k_fraction, config):
    """Python integer interval proof over every address and all INT8 B/U/H.

    Includes common-sum, absolute-value and HA0 bias headroom. The proof is
    deliberately conservative across states, but never uses sampled inputs.
    """
    c = SSMNumericConfig.from_dict(config)
    shift = c.a_fraction_bits + c.state_fraction_bits - int(k_fraction)
    if shift < 0:
        raise ValueError('K precision exceeds exact common accumulator domain')
    aa = a_codes.detach().cpu().numpy().astype(object)
    kk = k_codes.detach().cpu().numpy().astype(object)
    low, high = 0, 0
    for a, k in zip(aa, kk):
        amax = int(max(a))
        lo = amax * c.state_min + ((int(k) * (-128 * 127)) << shift)
        hi = amax * c.state_max + ((int(k) * (-128 * -128)) << shift)
        low, high = min(low, lo), max(high, hi)
    bias = 1 << (c.a_fraction_bits - 1)
    magnitude_with_bias = max(abs(low), abs(high)) + bias
    required = magnitude_with_bias.bit_length() + 1
    if required > c.accumulator_bits:
        raise OverflowError(f'SSM update needs {required} signed bits including HA0, '
                            f'configured {c.accumulator_bits}; f={k_fraction}, shift={shift}')
    return dict(min_sum=low, max_sum=high, rounding_bias=bias,
                required_signed_bits=required, configured_bits=c.accumulator_bits,
                native_int64_safe=required <= 64, injection_left_shift=shift,
                proof='all addresses; full signed state and INT8 B/U domains')


def _pwl_evaluate(x, func, segments):
    """Uniform-segment interpolation over a frozen compiler-known input range.

    This is a numerical baseline, not an emulation of any published SFU's RTL.
    """
    lo, hi = float(x.min()), float(x.max())
    if hi == lo:
        return func(x)
    points = torch.linspace(lo, hi, segments + 1, dtype=torch.float64)
    values = func(points)
    position = ((x - lo) / (hi - lo) * segments).clamp(0, segments)
    index = position.floor().long().clamp(max=segments - 1)
    fraction = position - index
    return values[index] + fraction * (values[index + 1] - values[index])


def compile_coefficients(theta, s_dt, s_b, s_u, config=None):
    c = SSMNumericConfig.from_dict(config)
    scales = [float(torch.as_tensor(v).detach().cpu()) for v in (s_dt, s_b, s_u)]
    if any(not np.isfinite(v) or v <= 0 for v in scales):
        raise ValueError('SSM scales must be finite and positive')
    theta = torch.as_tensor(theta).detach().cpu().double().reshape(-1)
    if theta.numel() == 0 or not torch.isfinite(theta).all():
        raise ValueError('pole parameters must be nonempty and finite')
    rates = torch.exp(theta)
    if not torch.isfinite(rates).all() or torch.any(rates <= 0):
        raise ValueError('pole rates must be finite and strictly positive')
    codes = torch.arange(c.dt_min, c.dt_min + c.dt_count, dtype=torch.float64)
    delta = F.softplus(codes * scales[0])
    a = torch.exp(-delta[:, None] * torch.exp(theta)[None, :])
    k = delta * scales[1] * scales[2]
    if not torch.isfinite(a).all() or not torch.isfinite(k).all():
        raise ValueError('Nonfinite compiled coefficients')
    approximation_a, approximation_k = a, k
    if c.coefficient_backend == 'pwl':
        approximate_delta = _pwl_evaluate(codes * scales[0], F.softplus, c.pwl_segments)
        exp_input = -approximate_delta[:, None] * torch.exp(theta)[None, :]
        approximation_a = _pwl_evaluate(exp_input, torch.exp, c.pwl_segments).clamp(0., 1.)
        approximation_k = approximate_delta * scales[1] * scales[2]
    aq = round_ha0(approximation_a * 2.**c.a_fraction_bits).to(torch.int64)
    f = choose_fraction(approximation_k, c.k_bits, min(c.k_preferred_fraction_bits,
                                       c.a_fraction_bits + c.state_fraction_bits))
    kq = round_ha0(approximation_k * 2.**f).to(torch.int64)
    certificate = certify_update_range(aq, kq, f, c)
    coefficient_error = {}
    for name, actual, reference in [('A', aq.double() * 2.**(-c.a_fraction_bits), a),
                                     ('K', kq.double() * 2.**(-f), k)]:
        difference = (actual - reference).abs()
        coefficient_error[name + '_mae'] = float(difference.mean())
        coefficient_error[name + '_max_abs_error'] = float(difference.max())
    return dict(a=aq, k=kq, a_float=a, k_float=k, k_fraction_bits=f,
                coefficient_error=coefficient_error,
                range_certificate=certificate, config=c.to_dict(),
                logical_rom_bits=c.dt_count * (theta.numel() * (c.a_fraction_bits + 1) + c.k_bits))


def _shift_python(value, shift):
    value = int(value)
    if shift <= 0:
        return value << -shift
    return (-1 if value < 0 else 1) * ((abs(value) + (1 << (shift - 1))) >> shift)


def _shift_tensor(x, shift):
    if shift <= 0:
        return x << -shift
    mag = (x.abs() + (1 << (shift - 1))) >> shift
    return torch.where(x < 0, -mag, mag)


def integer_update(h, a, kbu, f, config, native_safe=True):
    c = SSMNumericConfig.from_dict(config)
    common_shift = c.a_fraction_bits + c.state_fraction_bits - f
    if native_safe:
        pa, pu = a * h, kbu << common_shift
        raw = (_shift_tensor(pa + pu, c.a_fraction_bits) if c.rounding == 'single'
               else _shift_tensor(pa, c.a_fraction_bits) + _shift_tensor(pu, c.a_fraction_bits))
    else:
        # Rare 65-bit cases use arbitrary precision, never a silent int64 wrap.
        pa = a.cpu().numpy().astype(object) * h.cpu().numpy().astype(object)
        pu = kbu.cpu().numpy().astype(object) * (1 << common_shift)
        if c.rounding == 'single':
            wide = np.vectorize(lambda v: _shift_python(v, c.a_fraction_bits), otypes=[object])(pa + pu)
        else:
            rnd = np.vectorize(lambda v: _shift_python(v, c.a_fraction_bits), otypes=[object])
            wide = rnd(pa) + rnd(pu)
        # Keep saturation statistics before converting the clipped state.
        count = int(np.count_nonzero((wide < c.state_min) | (wide > c.state_max)))
        clipped = np.clip(wide, c.state_min, c.state_max).astype(np.int64)
        return torch.from_numpy(clipped).to(h.device), count
    count = int(((raw < c.state_min) | (raw > c.state_max)).sum())
    return raw.clamp(c.state_min, c.state_max), count


def error_stats(actual, reference):
    diff = (actual.double() - reference.double()).reshape(-1)
    ref = reference.double().reshape(-1)
    return dict(count=diff.numel(), abs_sum=float(diff.abs().sum()),
                squared_error=float((diff * diff).sum()), reference_squared=float((ref * ref).sum()),
                max_abs_error=float(diff.abs().max()) if diff.numel() else 0.)


def finalize_stats(row):
    row = dict(row)
    row['mae'] = row['abs_sum'] / max(row['count'], 1)
    row['relative_l2'] = ((row['squared_error'] / row['reference_squared']) ** .5
                          if row['reference_squared'] > 0 else None)
    return row


class ErrorRecorder:
    def __init__(self):
        self.rows = {}

    def add(self, core, time, actual, reference, saturation=0):
        key = (core, time)
        new = error_stats(actual, reference)
        if key not in self.rows:
            self.rows[key] = dict(core=core, time=time, saturation_count=0,
                                  count=0, abs_sum=0., squared_error=0., reference_squared=0., max_abs_error=0.)
        row = self.rows[key]
        for k in ('count', 'abs_sum', 'squared_error', 'reference_squared'):
            row[k] += new[k]
        row['max_abs_error'] = max(row['max_abs_error'], new['max_abs_error'])
        row['saturation_count'] += saturation

    def write(self, path):
        rows = [finalize_stats(row) for row in self.rows.values()]
        if rows:
            with Path(path).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return rows


def scan_codes(u, dt, b, readout, tables, config=None, scales=(1., 1.),
               recorder=None, core='core', d_tables=None, components=None):
    """u[B,K,L], dt[B,L,K], b/readout[B,N,L]. Returns codes or physical Y.

    all -> exact integer readout codes with physical scale s_C*2^-FH.
    other modes -> float64 physical readout, with exactly the same input codes.
    Local reference uses the SAME u/dt/b/c input and full-precision coefficients.
    """
    c = SSMNumericConfig.from_dict(config or tables['config'])
    device = u.device
    u, dt, b, readout = [x.to(torch.int64) for x in (u, dt, b, readout)]
    batch, channels, length = u.shape
    states = tables['a'].shape[1]
    if dt.shape != (batch, length, channels) or b.shape != (batch, states, length) or readout.shape != b.shape:
        raise ValueError('inconsistent scan tensor shapes')
    for x in (u, b, readout):
        if torch.any(x < -128) or torch.any(x > 127):
            raise ValueError('U/B/C must contain signed INT8 codes')
    if torch.any(dt < c.dt_min) or torch.any(dt >= c.dt_min + c.dt_count):
        raise ValueError('dt code outside configured address domain')
    if (1 << (c.state_bits - 1)) * 128 * states > torch.iinfo(torch.int64).max:
        raise OverflowError('readout cannot fit int64 container')
    f = tables['k_fraction_bits']
    aq, kq = tables['a'].to(device), tables['k'].to(device)
    af, kf = tables['a_float'].to(device), tables['k_float'].to(device)
    h = torch.zeros((batch, channels, states), device=device, dtype=torch.int64 if c.error_source == 'all' else torch.float64)
    href = torch.zeros_like(h, dtype=torch.float64)
    outputs = []
    s_c = float(scales[1])
    if c.error_source == 'd-only' and d_tables is None:
        raise ValueError('d-only requires a D-enabled checkpoint/input capture')
    if d_tables is not None:
        if len(d_tables['coefficient']) != channels:
            raise ValueError('D must have one coefficient per SSM channel')
        if d_tables['state_fraction_bits'] != c.state_fraction_bits or d_tables['s_c'] != s_c:
            raise ValueError('D table scale does not match readout scale')
        if d_tables['state_bits'] != c.state_bits or d_tables['state_count'] != states:
            raise ValueError('D range certificate does not match state geometry')
        d_integer = d_tables['coefficient'].to(device)
        d_physical_per_code = d_tables['d_quant'].to(device) * d_tables['s_u']
    c_outputs, d_outputs = [], []
    for t in range(length):
        addr = dt[:, t] - c.dt_min
        bu = b[:, :, t].unsqueeze(1) * u[:, :, t].unsqueeze(-1)
        saturation = 0
        if c.error_source == 'all':
            h, saturation = integer_update(h, aq[addr], kq[addr].unsqueeze(-1) * bu,
                                           f, c, tables['range_certificate']['native_int64_safe'])
            physical_h = h.double() * 2.**(-c.state_fraction_bits)
            y = (h * readout[:, :, t].unsqueeze(1)).sum(-1)
        else:
            a = aq[addr].double() * 2.**(-c.a_fraction_bits) if c.error_source == 'a-only' else af[addr]
            k = kq[addr].double() * 2.**(-f) if c.error_source == 'k-only' else kf[addr]
            h = a * h + k.unsqueeze(-1) * bu
            if c.error_source == 'state-only':
                raw = round_ha0(h * 2.**c.state_fraction_bits)
                saturation = int(((raw < c.state_min) | (raw > c.state_max)).sum())
                h = raw.clamp(c.state_min, c.state_max) * 2.**(-c.state_fraction_bits)
            physical_h = h
            y = (h * readout[:, :, t].unsqueeze(1) * s_c).sum(-1)
        c_value = y
        if d_tables is not None:
            d_ref = d_physical_per_code.unsqueeze(0) * u[:, :, t]
            d_codes = (d_integer.unsqueeze(0) * u[:, :, t]) << d_tables['left_shift']
            if c.error_source == 'all':
                d_value = d_codes
            elif c.error_source == 'd-only':
                d_value = d_codes.double() * s_c * 2.**(-c.state_fraction_bits)
            else:
                d_value = d_ref
            y = y + d_value
            c_outputs.append(c_value)
            d_outputs.append(d_value)
        if recorder is not None:
            href = af[addr] * href + kf[addr].unsqueeze(-1) * bu
            recorder.add(core + '/state', t, physical_h, href, saturation)
            actual_y = y.double() * s_c * 2.**(-c.state_fraction_bits) if c.error_source == 'all' else y
            ref_y = (href * readout[:, :, t].unsqueeze(1) * s_c).sum(-1)
            if d_tables is not None:
                physical_c = c_value.double() * s_c * 2.**(-c.state_fraction_bits) if c.error_source == 'all' else c_value
                physical_d = d_value.double() * s_c * 2.**(-c.state_fraction_bits) if c.error_source == 'all' else d_value
                recorder.add(core + '/c_readout', t, physical_c, ref_y)
                recorder.add(core + '/d_path', t, physical_d, d_ref)
                ref_y = ref_y + d_ref
            recorder.add(core + '/readout', t, actual_y, ref_y)
        outputs.append(y)
    result = torch.stack(outputs, dim=-1)
    if components is not None and d_tables is not None:
        components.update(c=torch.stack(c_outputs, dim=-1), d=torch.stack(d_outputs, dim=-1), total=result)
    return result
