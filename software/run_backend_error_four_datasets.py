#!/usr/bin/env python3
"""Validate/reuse eval1 D1 checkpoints; sequential N0-N3 jobs with safe resume.

This entry point imports stdlib only. --dry-run never starts training/inference.
"""
import argparse
import csv
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

from run_fpga_qat_eval1_four_datasets import DATASETS, CONFIG, validate_qat, device_environment, sha, read, save

ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = (
    'backend_error_four_datasets.py', 'backend_error_metrics.py', 'run_backend_error_four_datasets.py',
    'run_fpga_qat_eval1_four_datasets.py', 'both_FPGA_single_qat_source.py',
    'train_mambahsi_spatial_split_dense_qat.py', 'mambahsi_ablation_model.py',
    'ssm_error_ablation.py', 'ssm_d_path.py', 'qat_forensics.py',
    'snapshots/nonlinear_D1_20260916/d1_methods.py', 'snapshots/nonlinear_D1_20260916/prepare.py',
    'snapshots/nonlinear_D1_20260916/n2_model.py', 'snapshots/nonlinear_D1_20260916/n2_config.json',
)


def verify_output(job):
    output = Path(job['output'])
    base = read(output/'fpga_simulation_result.json')
    summary = read(output/'nonlinear_experiment/accuracy_propagation_summary.json')
    for r in (base, summary):
        key = 'qat_checkpoint_sha256' if r is base else 'checkpoint_sha256'
        if (r['dataset'], r['seed'], r[key]) != (job['dataset'], job['seed'], job['checkpoint_sha256']):
            raise ValueError('Output identity mismatch: '+str(output))
        if r['numeric_config'] != job['numeric_config']:
            raise ValueError('Output numeric configuration differs from trained checkpoint')
    if base['saved_qat_prediction_match'] != 1 or base['qat_batch1_vs_saved_batch_match'] != 1:
        raise ValueError('QAT reference mismatch')
    if base['ssm_execution_mode'] != 'exact-integer' or summary['status'] != 'COMPLETE':
        raise ValueError('Incomplete or non-integer run')
    if set(summary['methods']) != {'n0','n1','n2','n3'} or len(summary['coefficients']) != 6:
        raise ValueError('Missing methods/cores')
    if not summary['n3_float_postprocess_matches_base'] or summary['tiles'] != base['tile_count']:
        raise ValueError('N3 baseline or scene coverage differs')
    record = output/'JOB_COMPLETE.json'
    if record.exists():
        saved = read(record)
        if saved['fingerprint'] != job['fingerprint']:
            raise ValueError('Completion marker fingerprint mismatch')
        for name, expected in saved['output_hashes'].items():
            if sha(output/name) != expected:
                raise ValueError('Completed output modified: '+name)
    elif job.get('status') == 'complete':
        raise ValueError('Completed job is missing JOB_COMPLETE.json')
    return summary


def collect(out, jobs):
    rows = []
    for job in jobs:
        if job['status'] != 'complete':
            continue
        s = verify_output(job)
        for method, r in s['methods'].items():
            for scope in ('scene','labeled','train','validation','test'):
                x = r['scopes'][scope]['integer_vs_n3']
                q = r['scopes'][scope]['integer_vs_qat']
                rows.append(dict(dataset=job['dataset'], seed=job['seed'], method=method, scope=scope,
                    count=x['count'], disagreements=x['mismatches'], disagreement_fraction=x['mismatch_fraction'],
                    OA=x['actual_OA'], n3_OA=x['reference_OA'],
                    correct_to_wrong=x['reference_correct_actual_wrong'], wrong_to_correct=x['reference_wrong_actual_correct'],
                    both_wrong_changed=x['both_wrong_changed'], qat_disagreements=q['mismatches'],
                    logit_MAE=r['logits_vs_n3']['MAE'], logit_RMSE=r['logits_vs_n3']['RMSE'],
                    logit_physical_MAE=r['logits_physical_vs_n3']['MAE'],
                    logit_physical_RMSE=r['logits_physical_vs_n3']['RMSE'],
                    logit_max_abs=r['logits_vs_n3']['max_abs_error'], logit_mismatched_bits=r['mismatched_bits'],
                    interpolation_disagreements=r['scopes'][scope]['integer_vs_float_postprocess_mismatches']))
    if rows:
        with (out/'per_seed_backend_summary.csv').open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    groups = {}
    for ds in DATASETS:
        for method in ('n3','n0','n1','n2'):
            for scope in ('scene','test'):
                rr = [r for r in rows if (r['dataset'],r['method'],r['scope']) == (ds,method,scope)]
                if not rr:
                    continue
                group = dict(n_completed=len(rr), seeds=[r['seed'] for r in rr])
                for key in ('disagreement_fraction','OA','correct_to_wrong','wrong_to_correct','logit_MAE','logit_RMSE','logit_physical_MAE','logit_physical_RMSE'):
                    vals = [r[key] for r in rr if r[key] is not None]
                    group[key] = dict(mean=statistics.mean(vals) if vals else None,
                                      sample_std=statistics.stdev(vals) if len(vals)>1 else None)
                groups[f'{ds}/{method}/{scope}'] = group
    save(out/'dataset_backend_summary.json', dict(
        scope='Per-dataset seed mean/std; label disagreement is not ground-truth error; OA uses labeled pixels',
        requested=len(jobs), completed=sum(j['status']=='complete' for j in jobs), groups=groups))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qat-root', default='./qat_eval1_4datasets')
    p.add_argument('--fp32-root', help='Optional relocated FP32 root; same layout as eval1 runner')
    p.add_argument('--jobs-json', help='Optional explicit list of dataset, seed, qat_run_dir, fp32_dir')
    p.add_argument('--data-path', default='./data')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    p.add_argument('--device', default='cuda')
    p.add_argument('--trace-tiles', type=int, default=5)
    p.add_argument('--output-dir', default='./backend_error_D1_4datasets')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--continue-on-error', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(',')]
    if not seeds or min(seeds)<0 or len(set(seeds))!=len(seeds) or len(set(a.datasets))!=len(a.datasets) or a.trace_tiles < -1:
        p.error('Invalid/duplicate dataset, seeds or trace limit')
    device, env = device_environment(a.device, os.environ)
    env.setdefault('MPLBACKEND', 'Agg')
    out, data, qatroot = [Path(x).expanduser().resolve() for x in (a.output_dir,a.data_path,a.qat_root)]
    if not data.is_dir():
        raise FileNotFoundError(data)
    sources = {name:sha(ROOT/name) for name in SOURCE_NAMES}
    sources.update({str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'utils').glob('*.py'))})
    # A once-per-invocation content identity, independent of CUDA and filesystem mtime.
    raw = {str(f.relative_to(data)):sha(f) for f in sorted(data.rglob('*'))
           if f.is_file() and f.suffix.lower() in ('.mat','.npy','.npz','.hdr','.raw','.dat','.tif','.tiff')}
    if not raw:
        raise ValueError('No raw dataset files found under '+str(data))
    explicit = read(a.jobs_json) if a.jobs_json else None
    jobs = []
    for ds in a.datasets:
        for seed in seeds:
            if explicit is not None:
                selected = [r for r in explicit if (r['dataset'],r['seed']) == (ds,seed)]
                if len(selected) != 1:
                    raise ValueError(f'jobs-json must contain exactly one {ds}/seed{seed}')
                spec = selected[0]
                path = Path(spec['qat_run_dir']).expanduser().resolve()/'result.json'
                r = read(path)
                fp = Path(spec['fp32_dir']).expanduser().resolve()
            else:
                found = [(f, read(f)) for f in (qatroot/ds).rglob('result.json')]
                selected = [(f,r) for f,r in found if (r.get('dataset'),r.get('seed')) == (ds,seed)]
                if len(selected) != 1:
                    raise ValueError(f'{ds}/seed{seed}: expected one result.json under {qatroot/ds}, found {len(selected)}; use --jobs-json for explicit paths')
                path, r = selected[0]
                default = str(Path(a.fp32_root)/(ds+'_all_samples_sqrt_inverse_clip3_2000_nobias')/CONFIG) if a.fp32_root else r['source_fp32_dir']
                fp = Path(os.environ.get('FP32_'+ds.upper(), default)).expanduser().resolve()
            if (r.get('dataset'),r.get('seed')) != (ds,seed):
                raise ValueError('Explicit job metadata mismatch')
            validate_qat(r, path)
            required = dict(dt_input_bits=9,dt_output_bits=8,a_fraction_bits=24,k_bits=19,
                            state_bits=32,state_fraction_bits=24,rounding='single',
                            coefficient_backend='exact',error_source='all')
            if any(r.get('numeric_config',{}).get(k) != v for k,v in required.items()):
                raise ValueError('Requires trained DT9/DT8 A25/K19 Q24 contract: '+str(path))
            protected = (path.parent, fp, data)
            if any(out == x or out in x.parents or x in out.parents for x in protected):
                raise ValueError('Output and input directories must not contain one another')
            files = [path,path.with_name('best_qat_foldaware.pth'),path.with_name('sample_indices.npz'),
                     path.with_name('qat_deploy_test_prediction.npy'),fp/'train_only_preprocess.npz',
                     fp/'spatial_split_masks.npz',fp/'spatial_split.json',fp/f'run_seed{seed}'/'sample_indices.npz']
            hashes = {str(f):sha(f) for f in files}  # fail before launching any job
            identity = dict(dataset=ds,seed=seed,source_hashes=sources,input_hashes=hashes,raw_data_hashes=raw,
                            numeric_config=r['numeric_config'],trace_tiles=a.trace_tiles,device=device,
                            cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),python=sys.executable)
            fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            command = [sys.executable,'-u',str(ROOT/'backend_error_four_datasets.py'),'--qat-run-dir',str(path.parent),
                       '--fp32-dir',str(fp),'--dataset',ds,'--seed',str(seed),'--data-path',str(data),
                       '--device',device,'--trace-tiles',str(a.trace_tiles)]
            jobs.append(dict(dataset=ds,seed=seed,numeric_config=r['numeric_config'],identity=identity,
                             checkpoint_sha256=hashes[str(files[1])],fingerprint=fingerprint,command=command,status='pending'))
    if a.dry_run:
        for j in jobs:
            print(shlex.join(j['command']+['--output-dir',str(out/j['dataset']/f'seed{j["seed"]}'/'attempt001')]))
        print(f'PREFLIGHT PASS: {len(jobs)} checkpoints; code/data/preprocess/split hashes recorded in memory; no inference started.')
        return 0
    manifest = out/'manifest.json'
    if out.exists() and any(out.iterdir()):
        if not a.resume:
            raise FileExistsError('Use --resume or a new output directory')
        old = read(manifest)
        if [(j['dataset'],j['seed'],j['fingerprint']) for j in old] != [(j['dataset'],j['seed'],j['fingerprint']) for j in jobs]:
            raise ValueError('Resume inputs/code/options changed; use a new output directory')
        jobs = old
    out.mkdir(parents=True, exist_ok=True)
    save(manifest,jobs)
    failures = []
    for i,j in enumerate(jobs,1):
        if j['status'] == 'complete':
            verify_output(j)
            print(f'SKIP complete {j["dataset"]}/seed{j["seed"]}',flush=True)
            continue
        folder = out/j['dataset']/f'seed{j["seed"]}'
        folder.mkdir(parents=True,exist_ok=True)
        n = 1
        while (folder/f'attempt{n:03d}').exists():
            n += 1
        target = folder/f'attempt{n:03d}'
        target.mkdir()
        log = folder/f'attempt{n:03d}.log'
        j.update(status='running',output=str(target),log=str(log))
        save(manifest,jobs)
        print(f'[{i}/{len(jobs)}] {j["dataset"]}/seed{j["seed"]} N3/N0/N1/N2 -> {log}',flush=True)
        try:
            with log.open('w',encoding='utf-8') as f:
                proc = subprocess.run(j['command']+['--output-dir',str(target)],cwd=str(ROOT),env=env,stdout=f,stderr=subprocess.STDOUT)
            if proc.returncode:
                raise RuntimeError(f'Worker exited {proc.returncode}')
            verify_output(j)
            outputs = {str(f.relative_to(target)):sha(f) for f in sorted(target.rglob('*'))
                       if f.is_file() and f.suffix in ('.json','.npy','.npz','.mem')}
            save(target/'JOB_COMPLETE.json',dict(fingerprint=j['fingerprint'],output_hashes=outputs))
            j.update(status='complete'); j.pop('error',None)
        except (Exception,KeyboardInterrupt) as exc:
            j.update(status='failed',error=str(exc))
            failures.append(f'{j["dataset"]}/seed{j["seed"]}')
            save(manifest,jobs)
            if log.exists():
                with log.open(errors='replace') as f: print(''.join(deque(f,maxlen=40)))
            if isinstance(exc,KeyboardInterrupt) or not a.continue_on_error:
                raise RuntimeError('Outputs preserved; use identical command plus --resume. '+str(log)) from exc
        save(manifest,jobs)
        collect(out,jobs)
    collect(out,jobs)
    print(f'FINISHED: {sum(j["status"]=="complete" for j in jobs)}/{len(jobs)} complete; failures={failures}',flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
