"""Frozen QAT-D compiler. It changes the output readout only, never the state.

qD is trained by LSQ. M_D is a separate, scale-folded hardware coefficient.
All range proofs use Python integers and cover the full signed INT8 U/C domain.
"""
import math
import torch
from ssm_error_ablation import SSMNumericConfig, round_ha0


def compile_d_path(d_quant, s_u, s_c, state_count, config=None):
    c = SSMNumericConfig.from_dict(config)
    d = torch.as_tensor(d_quant).detach().cpu().double()
    su, sc = float(s_u), float(s_c)
    if d.ndim != 1 or not d.numel() or not torch.isfinite(d).all():
        raise ValueError('D must be a finite, nonempty channel vector')
    if not all(math.isfinite(v) and v > 0 for v in (su, sc)):
        raise ValueError('D compilation requires positive finite U/C scales')
    if not isinstance(state_count, int) or state_count < 1:
        raise ValueError('Invalid state count')
    target = d * su / sc * 2.**c.state_fraction_bits
    if not torch.isfinite(target).all():
        raise OverflowError('Nonfinite folded D coefficients')
    lo, hi = -(1 << (c.d_coefficient_bits-1)), (1 << (c.d_coefficient_bits-1))-1
    for shift in range(c.d_max_left_shift + 1):
        values = round_ha0(target / 2.**shift)
        if torch.all((values >= lo) & (values <= hi)):
            coefficients = values.to(torch.int64)
            break
    else:
        raise OverflowError('D cannot fit configured coefficient width/left-shift budget')
    # Exact extrema for each independent C*H product, with conservative C/D sum.
    cp = [h*v for h in (c.state_min, c.state_max) for v in (-128,127)]
    c_low, c_high = state_count*min(cp), state_count*max(cp)
    d_bounds = [(min(int(m)*-128, int(m)*127) << shift,
                 max(int(m)*-128, int(m)*127) << shift) for m in coefficients]
    low = min(c_low + a for a, _ in d_bounds)
    high = max(c_high + b for _, b in d_bounds)
    # Existing RTL transports signed48 into its signed48 x signed16 requantizer.
    if low < -(1 << 47) or high > (1 << 47)-1:
        raise OverflowError('C+D readout exceeds existing signed48 interface')
    for dl, dh in d_bounds:
        if dl < -(1 << 63) or dh > (1 << 63)-1:
            raise OverflowError('Shifted D product exceeds software int64')
    physical_step = sc * 2.**(shift-c.state_fraction_bits)
    deployed = coefficients.double() * physical_step / su
    error = (deployed-d).abs()
    certificate = dict(c_min=c_low, c_max=c_high, total_min=low, total_max=high,
        output_bits=48, coefficient_bits=c.d_coefficient_bits,
        product_bits=c.d_coefficient_bits+8, left_shift=shift,
        coefficient_min=int(coefficients.min()), coefficient_max=int(coefficients.max()),
        coefficient_count=d.numel(), coefficient_rom_bits=d.numel()*c.d_coefficient_bits,
        fold_D_max_abs_error=float(error.max()), fold_D_mae=float(error.mean()),
        max_output_error_full_u_domain=float(error.max())*su*128,
        reference='frozen dequantized QAT D; excludes LSQ error relative to original D',
        proof='full signed H, INT8 C/U domains; conservative independent C+D bounds')
    return dict(coefficient=coefficients, left_shift=shift, d_quant=d,
                s_u=su, s_c=sc, state_fraction_bits=c.state_fraction_bits,
                state_bits=c.state_bits, state_count=state_count,
                range_certificate=certificate)


def certify_d_requant(d_tables, multiplier, shift):
    """Check signed64 multiply and HA0-bias/left-shift intermediates."""
    cert = d_tables['range_certificate']
    m = int(multiplier)
    products = [cert['total_min']*m, cert['total_max']*m]
    magnitude = max(map(abs, products))
    if shift > 0:
        magnitude += 1 << (int(shift)-1)
    elif shift < 0:
        magnitude <<= -int(shift)
    if magnitude > (1 << 63)-1:
        raise OverflowError('C+D requantization intermediate exceeds int64')
    return dict(max_abs_intermediate=magnitude, multiplier=m, shift=int(shift), carrier_bits=64)
