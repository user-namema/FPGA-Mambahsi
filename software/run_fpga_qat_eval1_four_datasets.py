#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run existing FPGA simulator on eval1 QAT artifacts; Python 3.8+, stdlib only."""
import argparse
from collections import deque
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

DATASETS = ('UP', 'HanChuan', 'HongHu', 'Houston')
CONFIG = 'current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16'
SOURCES = ('both_FPGA_single_qat_source.py', 'train_mambahsi_spatial_split_dense_qat.py',
           'ssm_error_ablation.py', 'ssm_d_path.py', 'qat_forensics.py')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p, obj):
    p = Path(p)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def device_environment(device, original):
    env = dict(original)
    if device in ('cuda', 'cpu'):
        return device, env
    if not re.fullmatch(r'cuda:[0-9]+', device):
        raise ValueError('DEVICE must be cpu, cuda or cuda:N')
    index = int(device.split(':')[1])
    visible = env.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        devices = [x.strip() for x in visible.split(',') if x.strip()]
        if index >= len(devices) or devices[index] == '-1':
            raise ValueError('Requested logical GPU is outside CUDA_VISIBLE_DEVICES')
        selected = devices[index]
    else:
        selected = str(index)
    env['CUDA_VISIBLE_DEVICES'] = selected
    return 'cuda', env


def find_project(requested):
    if requested:
        return Path(requested).expanduser().resolve()
    here = Path(__file__).resolve().parent
    for root in (Path.cwd(), Path.cwd().parent, here, here.parent):
        if (root / SOURCES[0]).is_file():
            return root.resolve()
    raise FileNotFoundError('Use --project-root to specify the MambaHSI directory')


def validate_qat(r, p):
    expected = dict(batch_size=32, eval_batch_size=1, weight_init_policy='patch_max',
                    d_init_policy='mean', freeze_bn_epoch=20, freeze_lsq_epoch=20)
    for k, v in expected.items():
        if r.get(k) != v:
            raise ValueError('%s: %s=%r; expected %r' % (p, k, r.get(k), v))
    if r.get('best_validation_recheck_match') is not True or r.get('bn_fold_test_prediction_match') != 1:
        raise ValueError('QAT checkpoint reproduction failed: ' + str(p))
    n = r.get('numeric_config', {})
    if n.get('error_source') != 'all' or n.get('readout_requantization', 'auto') not in ('auto', 'hardware'):
        raise ValueError('Expected full integer SSM with hardware readout: ' + str(p))
    if r.get('model_config', {}).get('use_D') is not True:
        raise ValueError('Expected D1 model: ' + str(p))


def verify_result(result, job):
    if (result.get('dataset'), result.get('seed')) != (job['dataset'], job['seed']):
        raise ValueError('Simulation dataset/seed mismatch')
    if result.get('qat_reference_batch_size') != 1 or result.get('saved_qat_prediction_match') != 1:
        raise ValueError('Saved QAT batch1 prediction reproduction failed')
    if result.get('qat_checkpoint_sha256') != job['checkpoint_sha256']:
        raise ValueError('Simulation checkpoint hash mismatch')
    if result.get('numeric_config') != job['numeric_config'] or result.get('ssm_execution_mode') != 'exact-integer':
        raise ValueError('Unexpected numeric configuration or non-integer execution')


def summarize(out, jobs):
    rows = []
    for j in jobs:
        if j.get('status') != 'complete':
            continue
        r = read(Path(j['output']) / 'fpga_simulation_result.json')
        verify_result(r, j)
        row = dict(dataset=j['dataset'], seed=j['seed'], best_epoch=r['qat_best_epoch'])
        for stage in ('qat_saved_batch', 'qat_direct', 'staged_fp', 'int8_hw'):
            for metric in ('OA', 'mAcc', 'Kappa'):
                row[stage + '_' + metric] = r['metrics'][stage][metric] * (1 if metric == 'Kappa' else 100)
        row['QAT_to_INT8_loss_pp'] = row['qat_direct_OA'] - row['int8_hw_OA']
        row['reference_path_loss_pp'] = row['qat_direct_OA'] - row['staged_fp_OA']
        row['saved_vs_direct_agreement_pct'] = 100 * r['qat_batch1_vs_saved_batch_match']
        row['QAT_vs_INT8_agreement_pct'] = 100 * r['test_agreements']['qat_direct_vs_int8']['rate']
        row['output'] = j['output']
        rows.append(row)
    if rows:
        with (out / 'per_seed_summary.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    grouped = {}
    for ds in DATASETS:
        rr = [r for r in rows if r['dataset'] == ds]
        if rr:
            grouped[ds] = {'n_completed': len(rr), 'seeds': [r['seed'] for r in rr]}
            for k in ('qat_direct_OA', 'int8_hw_OA', 'int8_hw_mAcc', 'int8_hw_Kappa', 'QAT_to_INT8_loss_pp'):
                v = [r[k] for r in rr]
                grouped[ds][k] = dict(mean=statistics.mean(v), std=statistics.stdev(v) if len(v)>1 else None)
    save(out / 'dataset_summary.json', grouped)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root')
    p.add_argument('--qat-root', default='./qat_eval1_4datasets')
    p.add_argument('--fp32-root', help='Optional relocated FP32 root; dataset/config folders are inferred')
    p.add_argument('--data-path', default='./data')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output-dir', default='./fpga_qat_eval1_4datasets')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--report-dt-inputs', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Check QAT metadata and print commands; no simulation/output writes')
    a = p.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(',')]
    if not seeds or min(seeds)<0 or len(set(seeds))!=len(seeds) or len(set(a.datasets))!=len(a.datasets):
        p.error('Invalid or duplicate seeds/datasets')
    project = find_project(a.project_root)
    device, env = device_environment(a.device, os.environ)
    qatroot, out, data = [Path(x).expanduser().resolve() for x in (a.qat_root, a.output_dir, a.data_path)]
    if qatroot == out or qatroot in out.parents:
        p.error('Output must be outside QAT input directory')
    for name in SOURCES:
        if not (project/name).is_file():
            raise FileNotFoundError(project/name)
    source_hashes = {name: sha(project/name) for name in SOURCES}
    jobs = []
    for ds in a.datasets:
        found = {s: [] for s in seeds}
        for path in (qatroot/ds).rglob('result.json'):
            r = read(path)
            if r.get('dataset') == ds and r.get('seed') in found:
                found[r['seed']].append((path, r))
        for seed in seeds:
            if len(found[seed]) != 1:
                raise ValueError('%s seed%d: expected one result.json under %s, found %d' % (ds,seed,qatroot/ds,len(found[seed])))
            path, r = found[seed][0]; validate_qat(r, path)
            default = str(Path(a.fp32_root)/(ds+'_all_samples_sqrt_inverse_clip3_2000_nobias')/CONFIG) if a.fp32_root else r['source_fp32_dir']
            fp = Path(os.environ.get('FP32_'+ds.upper(), default)).expanduser().resolve()
            files = [path, path.with_name('best_qat_foldaware.pth'), path.with_name('sample_indices.npz'), path.with_name('qat_deploy_test_prediction.npy')]
            fpfiles = [fp/'train_only_preprocess.npz',fp/'spatial_split_masks.npz',fp/'spatial_split.json',fp/('run_seed%d'%seed)/'sample_indices.npz']
            for f in files + ([] if a.dry_run else fpfiles):
                if not f.is_file(): raise FileNotFoundError(f)
            if not a.dry_run and not data.is_dir(): raise FileNotFoundError(data)
            checkpoint_hash = sha(files[1])
            identity = dict(dataset=ds, seed=seed, qat_result_sha256=sha(path),
                checkpoint_sha256=checkpoint_hash, source_hashes=source_hashes,
                input_hashes={str(f):sha(f) for f in files[2:]+([] if a.dry_run else fpfiles)},
                fp32_dir=str(fp), data_path=str(data), device=device,
                report_dt_inputs=a.report_dt_inputs,
                cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'), python=sys.executable)
            fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            cmd=[sys.executable,'-u',str(project/SOURCES[0]),'--qat-run-dir',str(path.parent),'--fp32-dir',str(fp),
                 '--dataset',ds,'--seed',str(seed),'--data-path',str(data),'--device',device,
                 '--qat-reference-batch-size','1','--report-ssm-errors','--ssm-analysis-split','test']
            if a.report_dt_inputs:
                cmd.append('--report-dt-inputs')
            jobs.append(dict(dataset=ds,seed=seed,command=cmd,identity=identity,fingerprint=fingerprint,
                             numeric_config=r['numeric_config'],checkpoint_sha256=checkpoint_hash,status='pending'))
    if a.dry_run:
        for j in jobs:
            print(json.dumps(dict(dataset=j['dataset'],seed=j['seed'],CUDA_VISIBLE_DEVICES=env.get('CUDA_VISIBLE_DEVICES'),
                 command=j['command']+['--output-dir',str(out/j['dataset']/('seed%d'%j['seed'])/'attempt001')]),ensure_ascii=False))
        print('Validated %d QAT runs. Dry run skips FP32/data availability checks.'%len(jobs));return
    manifest=out/'manifest.json'
    if out.exists() and any(out.iterdir()):
        if not a.resume: raise FileExistsError('Use --resume or a new output directory: '+str(out))
        previous=read(manifest)
        if [(j['dataset'],j['seed'],j['fingerprint']) for j in previous] != [(j['dataset'],j['seed'],j['fingerprint']) for j in jobs]:
            raise ValueError('Resume inputs/code/options differ; use the original settings or a new output directory')
        jobs=previous
    out.mkdir(parents=True,exist_ok=True);save(manifest,jobs)
    for i,j in enumerate(jobs,1):
        if j['status']=='complete':
            verify_result(read(Path(j['output'])/'fpga_simulation_result.json'),j)
            print('Skip complete: %s seed%d'%(j['dataset'],j['seed']),flush=True);continue
        base=out/j['dataset']/('seed%d'%j['seed']);base.mkdir(parents=True,exist_ok=True)
        attempt=1
        while (base/('attempt%03d'%attempt)).exists(): attempt+=1
        target=base/('attempt%03d'%attempt);target.mkdir()
        log=base/('attempt%03d.log'%attempt)
        j.update(status='running',output=str(target),log=str(log));save(manifest,jobs)
        print('[%d/%d] %s seed%d -> %s'%(i,len(jobs),j['dataset'],j['seed'],log),flush=True)
        try:
            with log.open('w') as f:
                proc=subprocess.run(j['command']+['--output-dir',str(target)],cwd=str(project),env=env,stdout=f,stderr=subprocess.STDOUT)
            if proc.returncode: raise RuntimeError('Simulator exited %d'%proc.returncode)
            verify_result(read(target/'fpga_simulation_result.json'),j)
            j['status']='complete';j.pop('error',None);save(manifest,jobs);summarize(out,jobs)
        except (Exception, KeyboardInterrupt) as exc:
            j.update(status='failed',error=str(exc));save(manifest,jobs)
            if log.exists():
                with log.open(errors='replace') as f: print(''.join(deque(f,maxlen=60)))
            raise RuntimeError('Failed; outputs preserved. Retry same command with --resume. Log: '+str(log)) from exc
    summarize(out,jobs)
    print('Completed %d runs; summaries: %s'%(len(jobs),out),flush=True)

if __name__=='__main__':
    main()
