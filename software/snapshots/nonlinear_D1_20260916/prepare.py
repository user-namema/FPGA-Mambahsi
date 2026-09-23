"""Freeze v5 coefficients and generate independent arithmetic reference vectors.

No torch/Vivado dependency. Reads the supplied exported decimal coefficient tables.
Online decay constants are recovered from intersected quantization intervals when
the original A_log checkpoint is unavailable, then checked against ALL 256 codes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q = 1 << 30
EXP_C = [round((-math.log(2)) ** k / math.factorial(k) * Q) for k in range(11)]
REC_C = [round(Q / 3)] * 19
LOG_C = [round(Q / (2*k+1)) for k in range(9)]
ENDPOINTS = [round(2 ** (-k/8) * Q) for k in range(9)]


def poly(x, coeffs):
    y = coeffs[-1]
    for c in coeffs[-2::-1]:
        y = ((y * x) >> 30) + c
        assert -(1 << 31) <= y < (1 << 31)
    return y


def exp_neg(x24, method):
    if x24 >= 32 * (1 << 24):
        return 0
    # w represents positive x*log2(e), in Q30.
    w = (x24 * (round(math.log2(math.e)*Q) if method == 0 else 23*Q//16)) >> 24
    n, f = w >> 30, w & (Q-1)
    if method == 0:
        y = max(0, poly(f, EXP_C))
    else:
        seg, local = f >> 27, f & ((1 << 27)-1)
        y = ENDPOINTS[seg] - ((ENDPOINTS[seg]-ENDPOINTS[seg+1])*local >> 27)
    return y >> n if n < 32 else 0


def online(qdt, sd, decay, scale, kf, method):
    x24 = (abs(qdt)*sd + 32) >> 6
    y = exp_neg(x24, method)
    if method == 0:
        r = ((Q-y) * round(Q/3)) >> 30
        recip = poly(r, REC_C)
        z = y * recip >> 30
        z2 = z*z >> 30
        log1p = (2*z*poly(z2, LOG_C)) >> 30
    else:
        log1p = y
    delta = ((log1p + 32) >> 6) + (x24 if qdt > 0 else 0)
    a = []
    for d in decay:
        mag = (delta*d + (1 << 23)) >> 24
        a.append(min(1 << 24, (exp_neg(mag, method)+32) >> 6))
    k = (delta*scale + (1 << (63-kf))) >> (64-kf)
    return a, min((1 << 19)-1, k)


def integers(path):
    result = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            result.extend(int(x) for x in line.split())
    return result


def packed(a, k):
    return sum(int(x) << (25*i) for i, x in enumerate(a)) | (int(k) << 400)


def emit_mem(path, words):
    path.write_text(''.join(f'{w:0105x}\n' for w in words), encoding='ascii')


def svpack(values, bits):
    return f"{bits*len(values)}'h" + ''.join(f'{v & ((1<<bits)-1):0{bits//4}x}' for v in reversed(values))


def prepare(source, decay_json=None):
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    supplied_decay = json.loads(decay_json.read_text()) if decay_json else None
    report = {'schema': 1, 'source': str(source), 'cores': [],
              'online_constant_provenance': ('User-supplied positive decay constants from '+str(decay_json)) if decay_json else 'Quantization-interval reconstruction, not original checkpoint A_log. Full 256-code equivalence is mandatory.',
              'n0': 'Online Q30 degree-10 exp2 polynomial, degree-18 reciprocal, degree-8 atanh log polynomial. NOT FP32 vendor IP.',
              'n1': 'FastMamba equations (3),(6) adapted to this contract; 23/16 log2(e), 8-segment endpoint interpolation. NOT author RTL.'}
    tops = '`timescale 1ns/1ps\n'
    test_tops = '`timescale 1ns/1ps\n'
    for block in range(3):
        for branch in ['spa', 'spe']:
            name = f'blk{block}_{branch}'
            mp = source / f'{name}_lut_manifest.json'
            m = json.loads(mp.read_text(encoding='utf-8'))
            flat = integers(source / f'{name}_Abar_uq1_24.txt')
            kf = m['K_fraction_bits']
            kvals = integers(source / f'{name}_K_q{kf}_u19.txt')
            assert len(flat) == 4096 and len(kvals) == 256
            delta = [math.log1p(math.exp((i-128)*m['s_dt'])) for i in range(256)]
            decay = []
            for state in range(16):
                lower, upper = 0., float('inf')
                for i, dd in enumerate(delta):
                    code = flat[16*i+state]
                    upper_y = min(1., (code+.5)/(1 << 24))
                    lower = max(lower, -math.log(upper_y)/dd)
                    if code > 0:
                        upper = min(upper, -math.log((code-.5)/(1 << 24))/dd)
                if not lower < upper:
                    raise RuntimeError(f'{name} state {state}: no compatible decay interval')
                d = float(supplied_decay[name][state]) if supplied_decay else (lower+upper)/2
                assert all(math.floor(math.exp(-dd*d)*(1 << 24)+.5) == flat[16*i+state]
                           for i, dd in enumerate(delta))
                decay.append(d)
            assert all(math.floor(dd*m['s_B']*m['s_u']*(1<<kf)+.5) == kvals[i]
                       for i, dd in enumerate(delta))
            sd = round(m['s_dt'] * Q)
            dq = [round(x*(1 << 24)) for x in decay]
            scale = round(m['s_B']*m['s_u']*(1 << 40))
            assert max(dq) < (1 << 48) and scale < (1 << 48)
            assert max(abs(q)*sd for q in [-128,127]) < (1 << 48)
            gold = [packed(flat[16*i:16*i+16], kvals[i]) for i in range(256)]
            emit_mem(data / f'{name}_n3.mem', gold)
            stats = {}
            for method in [0,1]:
                vals = [online(i-128, sd, dq, scale, kf, method) for i in range(256)]
                words = [packed(a,k) for a,k in vals]
                emit_mem(data / f'{name}_n{method}.mem', words)
                ae = [abs(a[j]-flat[i*16+j]) for i,(a,k) in enumerate(vals) for j in range(16)]
                ke = [abs(k-kvals[i]) for i,(a,k) in enumerate(vals)]
                stats[f'N{method}'] = {'A_max_LSB_error': max(ae), 'A_mismatch_count': sum(x!=0 for x in ae),
                    'K_max_LSB_error': max(ke), 'K_mismatch_count': sum(x!=0 for x in ke)}
            constants = {'SDT_Q30': sd, 'SBSU_Q40': scale, 'DECAY_Q24': svpack(dq,48),
                         'K_FRAC': kf, 'CORE_NAME': name}
            args = f".SDT_Q30(32'd{sd}), .SBSU_Q40(48'd{scale}), .DECAY_Q24({svpack(dq,48)}), .K_FRAC({kf})"
            for method in [0,1,3]:
                tops += f'''\nmodule nl_n{method}_{name} (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(.METHOD({method}), {args}, .ROM_FILE("{name}_n3.mem")) u_core (
        .clk(clk),.rst(rst),.in_valid(in_valid),.q_dt(q_dt),
        .out_valid(out_valid),.out_address(out_address),.out_coeff(out_coeff));
endmodule
'''
            test_tops += f'''\nmodule tb_{name};
    nl_tb #(.CORE("{name}"), {args}) u_tb();
endmodule
'''
            (data / f'{name}_params.json').write_text(json.dumps(constants,indent=2)+'\n')
            entry = {'name':name, 's_dt':m['s_dt'], 's_B':m['s_B'], 's_u':m['s_u'],
                     'K_fraction_bits':kf, 'decay_equivalent':decay, 'float_reconstruction_exact':True,
                     'reference_A_count':4096, 'reference_K_count':256, 'error':stats,
                     'files':{}}
            for p in sorted(data.glob(f'{name}_*')):
                entry['files'][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
            report['cores'].append(entry)
    (data / 'manifest.json').write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8')
    header = '// Generated arithmetic coefficients, NOT dt-indexed lookup tables.\n'
    for name, c in [('EXP_COEFF',EXP_C),('REC_COEFF',REC_C),('LOG_COEFF',LOG_C),('PWL_ENDPOINT',ENDPOINTS)]:
        header += f'`define {name} {svpack(c,32)}\n'
    (ROOT / 'rtl' / 'math_constants.vh').write_text(header,encoding='ascii')
    (ROOT / 'rtl' / 'generated_tops.sv').write_text(tops,encoding='ascii')
    (ROOT / 'sim' / 'generated_tb_tops.sv').write_text(test_tops,encoding='ascii')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=ROOT.parent/'UP_nobias_seed0_v5_kfold/first_tile_layers/ssm_luts')
    ap.add_argument('--decay-json',type=Path,help='Optional {core_name: [16 positive exp(A_log) values]} from original checkpoint')
    args = ap.parse_args()
    prepare(args.source,args.decay_json)
