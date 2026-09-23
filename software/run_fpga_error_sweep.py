#!/usr/bin/env python3
"""Run isolated full-scene simulator variants; logs and all results stay separate."""
import argparse
from collections import deque
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import numpy as np

from ssm_error_ablation import SSMNumericConfig
from run_ssm_error_ablation import configurations


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qat-run-dir',action='append',required=True)
    p.add_argument('--data-path',required=True)
    p.add_argument('--fp32-dir',help='Optional relocated FP32 artifacts, single run only')
    p.add_argument('--suite',choices=['sources','widths','nonlinear','both','all','requant','k-precision'],default='sources')
    p.add_argument('--k-bits-grid',type=int,nargs='+',default=[19,21,23])
    p.add_argument('--k-fraction-grid',type=int,nargs='+',default=[24,26,28])
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda','auto'])
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--analysis-split',choices=['train','validation','test','all'],default='test')
    args=p.parse_args(argv)
    if args.fp32_dir and len(args.qat_run_dir)!=1: p.error('--fp32-dir requires a single QAT run')
    out=Path(args.output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Use a fresh output directory: {out}')
    out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for value in args.qat_run_dir:
        run=Path(value).expanduser().resolve()
        result=json.loads((run/'result.json').read_text())
        base=SSMNumericConfig.from_dict(result.get('numeric_config'))
        variants={}
        suites = ['sources','widths','nonlinear','requant','k-precision'] if args.suite=='all' else (['sources','widths'] if args.suite=='both' else [args.suite])
        for suite in suites:
            variants.update(configurations(base,suite,has_d=result.get('model_config', {}).get('use_D',False),
                                           k_bits_grid=args.k_bits_grid,k_fraction_grid=args.k_fraction_grid))
        digest=hashlib.sha256(str(run).encode()).hexdigest()[:8]
        key=f'{result["dataset"]}_seed{result["seed"]}_{digest}'
        for label,c in variants.items():
            dest=out/key/label
            cmd=[sys.executable,str(Path(__file__).with_name('both_FPGA_single_qat_source.py').resolve()),
                 '--qat-run-dir',str(run),'--dataset',result['dataset'],'--seed',str(result['seed']),
                 '--data-path',str(Path(args.data_path).expanduser().resolve()),'--device',args.device,
                 '--output-dir',str(dest),'--report-ssm-errors','--ssm-analysis-split',args.analysis_split]
            if args.fp32_dir: cmd+=['--fp32-dir',str(Path(args.fp32_dir).expanduser().resolve())]
            for field, value in c.to_dict().items():
                flag='--'+field.replace('_','-') if field.startswith('dt_') else '--ssm-'+field.replace('_','-')
                cmd += [flag,str(value)]
            jobs.append(dict(run=str(run),dataset=result['dataset'],seed=result['seed'],variant=label,
                             numeric_config=c.to_dict(),output=str(dest),command=cmd))
    manifest=out/'sweep_manifest.json'
    manifest.write_text(json.dumps(dict(dry_run=args.dry_run,jobs=jobs),indent=2))
    (out/'commands.sh').write_text('#!/bin/sh\nset -eu\n'+'\n'.join(shlex.join(j['command']) for j in jobs)+'\n')
    if args.dry_run:
        print(f'{len(jobs)} jobs planned at {manifest}; no experiments launched')
        return
    summary_rows=[]
    coefficient_rows=[]
    for index, job in enumerate(jobs, 1):
        dest=Path(job['output']);dest.parent.mkdir(parents=True,exist_ok=True)
        log_path=dest.parent/(dest.name+'.log')
        job.update(status='running',completed=False,log_file=str(log_path))
        manifest.write_text(json.dumps(dict(dry_run=False,jobs=jobs),indent=2))
        print(f'[{index}/{len(jobs)}] {job["dataset"]} seed={job["seed"]} '
              f'variant={job["variant"]}\nLog: {log_path}',flush=True)
        try:
            with log_path.open('w',encoding='utf-8') as log:
                proc=subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT,check=False)
            returncode=proc.returncode
        except OSError as exc:
            returncode=1
            job['launch_error']=str(exc)
            with log_path.open('a',encoding='utf-8') as log:
                log.write(f'\nUnable to launch simulator: {exc}\n')
        job['returncode']=returncode
        if returncode:
            job['status']='failed'
            manifest.write_text(json.dumps(dict(dry_run=False,jobs=jobs),indent=2))
            with log_path.open(encoding='utf-8',errors='replace') as log:
                tail=''.join(deque(log,maxlen=80))
            print(f'\nSimulator failed: {job["variant"]} (exit {returncode})\n'
                  f'Log: {log_path}\n--- Last 80 log lines ---\n{tail}\n'
                  'Correct the error above and use a new --output-dir; existing results are preserved.',
                  file=sys.stderr,flush=True)
            raise SystemExit(returncode if returncode > 0 else 1)
        result=json.loads((dest/'fpga_simulation_result.json').read_text())
        metric=result['metrics']['int8_hw']
        summary_rows.append(dict(dataset=job['dataset'],seed=job['seed'],variant=job['variant'],
            execution_mode=result['ssm_execution_mode'],OA=metric['OA'],mAcc=metric['mAcc'],
            readout_requantization=result['ssm_readout_requantization'],
            k_bits=result['numeric_config']['k_bits'],
            k_preferred_fraction_bits=result['numeric_config']['k_preferred_fraction_bits'],
            Kappa=metric['Kappa'],mIoU=metric['mIoU'],
            OA_loss_pp=100*(result['metrics']['qat_direct']['OA']-metric['OA']),
            logical_rom_bits=result['ssm_rom_total_packed_bits'],
            d_path_rom_bits=result.get('d_path_rom_bits',0)))
        for core in result['ssm_roms']:
            coefficient_rows.append(dict(dataset=job['dataset'],seed=job['seed'],run=job['run'],variant=job['variant'],
                core=core['core'],k_bits=core['K_logical_bits'],k_fraction_bits=core['K_fraction_bits'],
                k_preferred_fraction_bits=result['numeric_config']['k_preferred_fraction_bits'],
                logical_rom_bits=core['packed_bits'],**core['coefficient_error']))
        with (out/'coefficient_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(coefficient_rows[0]));writer.writeheader();writer.writerows(coefficient_rows)
        with (out/'sweep_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary_rows[0]));writer.writeheader();writer.writerows(summary_rows)
        job['completed']=True
        job['status']='completed'
        job['result_file']=str(dest/'fpga_simulation_result.json')
        manifest.write_text(json.dumps(dict(dry_run=False,jobs=jobs),indent=2))
    pairs=[]
    for run in dict.fromkeys(j['run'] for j in jobs):
        variants={j['variant']:j for j in jobs if j['run']==run}
        if 'all_ideal' not in variants or 'all_hardware' not in variants:
            continue
        ideal_path=Path(variants['all_ideal']['output']);hw_path=Path(variants['all_hardware']['output'])
        ideal=json.loads((ideal_path/'fpga_simulation_result.json').read_text())
        hw=json.loads((hw_path/'fpga_simulation_result.json').read_text())
        assert ideal['qat_checkpoint_sha256']==hw['qat_checkpoint_sha256']
        with np.load(Path(run)/'sample_indices.npz',allow_pickle=False) as saved:
            ix=saved['test_indices']
        ip=np.load(ideal_path/'int8_hw_prediction.npy',allow_pickle=False).ravel()[ix]
        hp=np.load(hw_path/'int8_hw_prediction.npy',allow_pickle=False).ravel()[ix]
        pairs.append(dict(run=run,reference='all_ideal',actual='all_hardware',
            scope='Only SSM output requantizer implementation changes; downstream effects included.',
            test_prediction_changes=int(np.count_nonzero(ip!=hp)),test_pixels=int(ix.size),
            OA_loss_pp=100*(ideal['metrics']['int8_hw']['OA']-hw['metrics']['int8_hw']['OA']),
            mAcc_loss_pp=100*(ideal['metrics']['int8_hw']['mAcc']-hw['metrics']['int8_hw']['mAcc'])))
    if pairs:
        (out/'requant_comparison.json').write_text(json.dumps(pairs,indent=2))
    print(f'Completed {len(jobs)} jobs. Results linked from {manifest}')


if __name__=='__main__': main()
