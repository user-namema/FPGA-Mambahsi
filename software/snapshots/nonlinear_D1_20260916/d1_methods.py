"""One coefficient contract for the D1 software and OOC RTL generators."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import torch
import prepare
import n2_model
from ssm_error_ablation import certify_update_range, SSMNumericConfig

CHECKPOINT = 'd113c6123eb3234ea2a6b8fe13640c02b82e235e469e4337ec01c291844daf33'
INPUT_SHA256 = '5e830f01222de08bbffd59d391f73d8130972deee529c75f7506071edf7c752f'
LABEL_SHA256 = '28f8b8f34b34f2f58ba67a2fe1fa0de20fd3fae80fb8209a5ebb6e44e5481354'
CORES = [f'blk{b}_{s}' for b in range(3) for s in ('spa', 'spe')]
FIT = json.loads((Path(__file__).parent / 'n2_config.json').read_text(encoding='utf-8-sig'))['fit']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coefficient_words(path):
    """Decode all 256 419-bit words, independent of text newline encoding.

    No masking/truncation: malformed, missing, extra or over-width words fail.
    This package uses plain hex MEMs, not comments or @address directives.
    """
    path = Path(path)
    tokens = path.read_text(encoding='utf-8-sig').split()
    if len(tokens) != 256:
        raise ValueError(f'{path}: expected 256 coefficient words, got {len(tokens)}')
    words = []
    for address, token in enumerate(tokens):
        if not re.fullmatch(r'[0-9a-fA-F]+', token):
            raise ValueError(f'{path}: invalid hexadecimal word at address={address}')
        word = int(token, 16)
        if word >= (1 << 419):
            raise ValueError(f'{path}: coefficient exceeds 419 bits at address={address}')
        words.append(word)
    return words


def check_coefficient_files(actual_path, expected_path, context='coefficients'):
    actual = coefficient_words(actual_path)
    expected = coefficient_words(expected_path)
    for address, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            fields = []
            mask = (1 << 25)-1
            for state in range(16):
                a_got, a_want = (got >> (25*state)) & mask, (want >> (25*state)) & mask
                if a_got != a_want:
                    fields.append(f'A[{state}]={a_got} expected={a_want}')
            if got >> 400 != want >> 400:
                fields.append(f'K={got >> 400} expected={want >> 400}')
            raise AssertionError(
                f'{context}: software/RTL coefficient vectors differ at '
                f'address={address} signed_dt={address-128}: ' + '; '.join(fields))
    return len(actual)

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(x):
        if isinstance(x, (np.ndarray, torch.Tensor)):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=default,
                               allow_nan=False), encoding='utf-8')

def constants(theta, sdt, sb, su, fraction):
    decay = torch.exp(torch.as_tensor(theta).detach().float().cpu().double()).tolist()
    result = dict(s_dt=float(sdt), s_B=float(sb), s_u=float(su),
                  SDT_Q30=round(float(sdt) * 2**30),
                  SBSU_Q40=round(float(sb) * float(su) * 2**40),
                  DECAY_Q24=[round(x * 2**24) for x in decay],
                  decay_from_checkpoint=decay, K_fraction_bits=int(fraction))
    if (len(decay) != 16 or not 0 <= result['SDT_Q30'] < 2**32
            or not 0 <= result['SBSU_Q40'] < 2**48
            or any(not 0 <= x < 2**48 for x in result['DECAY_Q24'])
            or not 0 <= fraction <= 24):
        raise ValueError('Constants exceed the fixed comparison RTL format')
    return result

def variants(base, params, device='cpu'):
    c = SSMNumericConfig.from_dict(base['config'])
    if (c.dt_min != -128 or c.dt_count != 256 or c.a_fraction_bits != 24
            or c.k_bits != 19 or c.state_fraction_bits != 24
            or c.state_bits != 32 or c.rounding != 'single'
            or c.error_source != 'all' or c.coefficient_backend != 'exact'):
        raise ValueError('D1 comparison requires INT8 dt, A25/K19, Q24 state, exact integer ROM baseline')
    result = {'n3': base}
    for method in ('n0', 'n1', 'n2'):
        args = (params['SDT_Q30'], params['DECAY_Q24'], params['SBSU_Q40'], params['K_fraction_bits'])
        values = [n2_model.online(q, *args, FIT) if method == 'n2'
                  else prepare.online(q, *args, int(method[1])) for q in range(-128, 128)]
        a = torch.tensor([x[0] for x in values], dtype=torch.int64, device=device)
        k = torch.tensor([x[1] for x in values], dtype=torch.int64, device=device)
        result[method] = dict(base, a=a, k=k,
            range_certificate=certify_update_range(a.cpu(), k.cpu(), params['K_fraction_bits'], c))
    return result

def export_tables(directory, core, tables, params, d_table):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    params = dict(params, d_coefficient=d_table['coefficient'].cpu().tolist(),
                  d_left_shift=int(d_table['left_shift']),
                  d_range_certificate=d_table['range_certificate'])
    params['coefficient_errors_vs_n3'] = {
        method: {key: finished(stats(tables['n3'][key], table[key])) for key in ('a','k')}
        for method,table in tables.items()}
    dump(directory / f'{core}_parameters.json', params)
    for method, table in tables.items():
        aa, kk = table['a'].cpu().tolist(), table['k'].cpu().tolist()
        words = [prepare.packed(a, k) for a, k in zip(aa, kk)]
        (directory / f'{core}_{method}.mem').write_text(
            ''.join(f'{v:0105x}\n' for v in words), encoding='ascii')
        np.savez(directory / f'{core}_{method}.npz', Abar=np.array(aa), K=np.array(kk),
                 K_FRAC=params['K_fraction_bits'])
    return params

def stats(reference, actual):
    a = torch.as_tensor(actual).detach().cpu().double().reshape(-1)
    r = torch.as_tensor(reference).detach().cpu().double().reshape(-1)
    if r.shape != a.shape or not torch.isfinite(a).all() or not torch.isfinite(r).all():
        raise ValueError('Incompatible/nonfinite comparison')
    e = a-r
    return dict(count=r.numel(), mismatches=int(torch.count_nonzero(e)),
                abs_sum=float(e.abs().sum()), squared_error=float((e*e).sum()),
                reference_squared=float((r*r).sum()), max_abs_error=float(e.abs().max()) if e.numel() else 0.)

def merge(target, row):
    for key, value in row.items():
        target[key] = max(target.get(key, 0), value) if key == 'max_abs_error' else target.get(key, 0)+value

def finished(row):
    return dict(row, MAE=row['abs_sum']/max(1,row['count']),
                RMSE=(row['squared_error']/max(1,row['count']))**.5,
                relative_L2=(row['squared_error']/row['reference_squared'])**.5 if row['reference_squared'] else None)
