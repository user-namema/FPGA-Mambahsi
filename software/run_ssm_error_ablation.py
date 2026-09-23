#!/usr/bin/env python3
"""Paired local replay on captured SSM inputs; no data/model retraining required."""
import argparse
from dataclasses import replace
import glob
import json
from pathlib import Path

import numpy as np
import torch
from ssm_error_ablation import SSMNumericConfig, compile_coefficients, scan_codes, ErrorRecorder
from ssm_d_path import compile_d_path


def configurations(base, suite, has_d=False, k_bits_grid=None, k_fraction_grid=None):
    if suite == 'requant':
        return [('none_ideal', replace(base, error_source='none', rounding='single', readout_requantization='ideal')),
                ('all_ideal', replace(base, error_source='all', rounding='single', readout_requantization='ideal')),
                ('all_hardware', replace(base, error_source='all', rounding='single', readout_requantization='hardware'))]
    if suite == 'k-precision':
        return [(f'K_b{bits}_fmax{frac}', replace(base, k_bits=bits,
                    k_preferred_fraction_bits=frac, error_source='all', rounding='single',
                    readout_requantization='auto'))
                for bits in dict.fromkeys(k_bits_grid or (19,21,23))
                for frac in dict.fromkeys(k_fraction_grid or (24,26,28))]
    if suite == 'nonlinear':
        return [('exact', replace(base, coefficient_backend='exact'))] + [
            (f'pwl_{n}', replace(base, coefficient_backend='pwl', pwl_segments=n)) for n in (8,16,32,64)]
    if suite == 'sources':
        return [(source, replace(base, error_source=source, rounding='single', readout_requantization='auto'))
                for source in (('none', 'a-only', 'k-only', 'state-only', 'd-only', 'all') if has_d
                               else ('none', 'a-only', 'k-only', 'state-only', 'all'))] + [
                    ('separate', replace(base, error_source='all', rounding='separate', readout_requantization='auto'))]
    result = [('baseline', replace(base, error_source='all', rounding='single'))]
    for fa in (16, 20):
        result.append((f'A_f{fa}', replace(base, a_fraction_bits=fa)))
    for bits, frac in ((32, 16), (32, 20), (24, 24), (28, 24), (28, 20), (24, 16)):
        result.append((f'H_b{bits}_f{frac}', replace(base, state_bits=bits, state_fraction_bits=frac)))
    for fk in (16, 20):
        result.append((f'K_fmax{fk}', replace(base, k_preferred_fraction_bits=fk)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-glob', action='append', required=True)
    parser.add_argument('--suite', choices=['sources', 'widths', 'nonlinear', 'k-precision'], default='sources')
    parser.add_argument('--k-bits-grid', type=int, nargs='+', default=[19,21,23])
    parser.add_argument('--k-fraction-grid', type=int, nargs='+', default=[24,26,28])
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    files = sorted({Path(p).resolve() for pattern in args.inputs_glob for p in glob.glob(pattern, recursive=True)})
    if not files:
        parser.error('no replay inputs match --inputs-glob')
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'refusing to mix with existing output: {out}')
    out.mkdir(parents=True, exist_ok=True)
    recorders, configs, inputs, certificates = {}, {}, [], []
    boundary = None
    d_presence = None
    with torch.inference_mode():
        for path in files:
            with np.load(path, allow_pickle=False) as saved:
                c = SSMNumericConfig.from_dict(json.loads(str(saved['numeric_config'])))
                current = (c.dt_input_bits, c.dt_output_bits)
                if boundary is not None and boundary != current:
                    raise ValueError('Do not pool inputs from different dt quantization boundaries')
                boundary = current
                tensors = [torch.from_numpy(saved[k].copy()) for k in ('u', 'dt', 'b', 'c')]
                has_d = 'd_quant' in saved
                if d_presence is not None and has_d != d_presence:
                    raise ValueError('Do not pool D0 and D1 captures')
                d_presence = has_d
                for label, config in configurations(c, args.suite, has_d=has_d,
                        k_bits_grid=args.k_bits_grid, k_fraction_grid=args.k_fraction_grid):
                    if label in configs and configs[label] != config.to_dict():
                        raise ValueError('Do not pool replay inputs from different numeric base configurations')
                    recorder = recorders.setdefault(label, ErrorRecorder())
                    configs[label] = config.to_dict()
                    tables = compile_coefficients(saved['theta'], saved['s_dt'], saved['s_b'], saved['s_u'], config)
                    d_tables = (compile_d_path(saved['d_quant'], float(saved['s_u']), float(saved['s_c']),
                                              tensors[2].shape[1], config) if has_d else None)
                    scan_codes(*tensors, tables, config, scales=(float(saved['s_b']), float(saved['s_c'])),
                               recorder=recorder, core=str(saved['core']), d_tables=d_tables)
                    certificates.append(dict(input=str(path), variant=label, core=str(saved['core']),
                        k_fraction_bits=tables['k_fraction_bits'], logical_rom_bits=tables['logical_rom_bits'],
                        k_bits=config.k_bits, k_preferred_fraction_bits=config.k_preferred_fraction_bits,
                        d_certificate=None if d_tables is None else d_tables['range_certificate'],
                        coefficient_error=tables['coefficient_error'], **tables['range_certificate']))
                inputs.append(str(path))
    for label, recorder in recorders.items():
        recorder.write(out / f'{label}_error_by_position.csv')
    import csv
    with (out / 'coefficient_summary.csv').open('w', newline='') as stream:
        rows = [dict(input=r['input'], variant=r['variant'], core=r['core'],
                     k_bits=r['k_bits'], k_preferred_fraction_bits=r['k_preferred_fraction_bits'],
                     k_fraction_bits=r['k_fraction_bits'], logical_rom_bits=r['logical_rom_bits'],
                     **r['coefficient_error']) for r in certificates]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out / 'replay_manifest.json').write_text(json.dumps(dict(
        input_files=inputs, suite=args.suite, configurations=configs,
        range_certificates=certificates,
        scope='same captured U/dt/B/C per variant; local replay only; no end-to-end OA or FPGA PPA',
        error_reference='float64 coefficients and recurrence on the same quantized inputs; D reference is frozen LSQ D, d-only isolates folded coefficient error'), indent=2))
    print(out)


if __name__ == '__main__':
    main()
