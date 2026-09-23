#!/usr/bin/env python3
"""Run the frozen UP D1 N0-N3 comparison with its recorded source and checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'software/snapshots/nonlinear_D1_20260916'
CHECKPOINT = 'd113c6123eb3234ea2a6b8fe13640c02b82e235e469e4337ec01c291844daf33'
# Reuse the tested logical-device mapping; this module imports only stdlib.
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'software'))
from run_fpga_qat_eval1_four_datasets import device_environment

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def verify_sources():
    for name, expected in json.loads((SNAPSHOT / 'source_hashes.json').read_text()).items():
        if digest(SNAPSHOT / name) != expected: raise ValueError('Frozen source/reference changed: ' + name)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qat-run-dir', default=os.environ.get('N0N3_QAT_RUN_DIR'))
    p.add_argument('--fp32-dir', default=os.environ.get('N0N3_FP32_DIR'))
    p.add_argument('--data-path', default=os.environ.get('DATA_ROOT', './data'))
    p.add_argument('--device', default=os.environ.get('DEVICE', 'cuda:0'))
    p.add_argument('--output-dir', default='./results/20_nonlinear_D1')
    p.add_argument('--trace-tiles', type=int, default=5)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if not a.qat_run_dir: p.error('--qat-run-dir or N0N3_QAT_RUN_DIR is required (historical D1 checkpoint)')
    if a.trace_tiles < -1: p.error('--trace-tiles must be -1 or nonnegative')
    verify_sources()
    device, env = device_environment(a.device, os.environ)
    qat = Path(a.qat_run_dir).expanduser().resolve()
    output = Path(a.output_dir).expanduser().resolve()
    command = [sys.executable, '-u', str(SNAPSHOT / 'both_FPGA_nonlinear_D1.py'),
        '--qat-run-dir', str(qat), '--dataset', 'UP', '--seed', '0',
        '--data-path', str(Path(a.data_path).expanduser().resolve()), '--device', device,
        '--nonlinear-backend', 'all', '--trace-tiles', str(a.trace_tiles), '--output-dir', str(output)]
    if a.fp32_dir: command += ['--fp32-dir', str(Path(a.fp32_dir).expanduser().resolve())]
    prefix = ('CUDA_VISIBLE_DEVICES=' + shlex.quote(env['CUDA_VISIBLE_DEVICES']) + ' ') if 'CUDA_VISIBLE_DEVICES' in env else ''
    print(prefix + shlex.join(command), flush=True)
    if a.dry_run: return
    if output.exists(): raise FileExistsError('Use a new output directory: ' + str(output))
    if digest(qat / 'best_qat_foldaware.pth') != CHECKPOINT:
        raise ValueError('Wrong checkpoint: this experiment requires historical D1 d113c612..., not latest eval1 weights')
    subprocess.run(command, cwd=str(SNAPSHOT), env=env, check=True)

if __name__ == '__main__': main()
