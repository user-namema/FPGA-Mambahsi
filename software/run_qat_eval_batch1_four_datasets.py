#!/usr/bin/env python3
"""Fresh QAT: training B32, validation/selection/test/deploy B1. Python >=3.8."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

DATASETS = ('UP', 'HanChuan', 'HongHu', 'Houston')
CONFIG = 'current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16'

def build_command(script, dataset, fp32, root, args):
    return [sys.executable, '-u', str(script), '--dataset', dataset,
        '--data_set_path', str(Path(args.data_path).resolve()), '--fp32_dir', str(fp32),
        '--work_dir', str(root / 'models'), '--run_tag', 'patch_max_D_mean_freeze20_eval1',
        '--seeds', args.seeds, '--device', args.device,
        '--weight_init_policy', 'patch_max', '--d-init-policy', 'mean',
        '--use_D', 'true', '--use_z', 'false', '--ssm-u-quantization', 'shared', '--d-weight-bits', '8',
        '--dt-input-bits', '9', '--dt-output-bits', '8',
        '--batch_size', '32', '--eval_batch_size', '1', '--calibration_steps', '100',
        '--freeze_bn_epoch', '20', '--freeze_lsq_epoch', '20',
        '--lr', '1e-5', '--lr_schedule', 'constant', '--max_epoch', str(args.max_epoch),
        '--eval_interval', '1', '--selection_metric', 'mAcc', '--include_calibrated_baseline',
        '--training_diagnostics', '--capture_oa_drop_pp', '5', '--capture_start_epoch', '1']

def validate_results(root, seeds):
    rows = []
    for p in sorted(root.glob('models/**/run_seed*/result.json')):
        r = json.loads(p.read_text())
        if r.get('batch_size') != 32 or r.get('eval_batch_size') != 1:
            raise RuntimeError('Unexpected train/evaluation batch: ' + str(p))
        if not r.get('best_validation_recheck_match') or r.get('bn_fold_test_prediction_match') != 1.0:
            raise RuntimeError('Checkpoint reload or BN fold reproduction failed: ' + str(p))
        rows.append(dict(dataset=r['dataset'], seed=r['seed'], result_path=str(p),
            qat_run_dir=str(p.parent), best_epoch=r['best_epoch'],
            train_batch_size=r['batch_size'], eval_batch_size=r['eval_batch_size'],
            fp32_OA=r['fp32_test_OA'], qat_OA=r['qat_test_OA'], qat_mAcc=r['qat_test_mAcc']))
    if sorted(r['seed'] for r in rows) != sorted(seeds):
        raise RuntimeError('Missing or duplicate completed seeds: ' + str(root))
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--fp32-root', default='./results/SPATIAL_SPLIT_3WAY_DENSE')
    p.add_argument('--fp32-tag', default='all_samples_sqrt_inverse_clip3_2000_nobias')
    p.add_argument('--data-path', default='./data')
    p.add_argument('--output-dir', default='./qat_patch_max_D_mean_freeze20_eval1_4datasets')
    p.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-epoch', type=int, default=100)
    p.add_argument('--dry-run', action='store_true', help='Print commands without reading model/data or writing output')
    args=p.parse_args()
    seeds=[int(s) for s in args.seeds.split(',')]
    if not seeds or len(set(seeds))!=len(seeds) or min(seeds)<0 or args.max_epoch<1:
        p.error('Invalid seeds or max-epoch')
    if len(set(args.datasets))!=len(args.datasets): p.error('Duplicate datasets')
    args.seeds=','.join(map(str,seeds))
    script=Path(__file__).resolve().with_name('train_mambahsi_spatial_split_dense_qat.py')
    output=Path(args.output_dir).resolve()
    jobs=[]
    for ds in args.datasets:
        fp32=Path(os.environ.get('FP32_'+ds.upper(), str(Path(args.fp32_root)/(ds+'_'+args.fp32_tag)/CONFIG))).resolve()
        root=output/ds
        cmd=build_command(script,ds,fp32,root,args)
        jobs.append((ds,fp32,root,cmd))
        if args.dry_run:
            print(json.dumps(dict(dataset=ds,command=cmd),ensure_ascii=False));continue
        for f in ['train_only_preprocess.npz','spatial_split_masks.npz','spatial_split.json']+['run_seed%d/best_model.pth'%s for s in seeds]:
            if not (fp32/f).is_file(): raise FileNotFoundError(str(fp32/f))
        if root.exists(): raise FileExistsError('Results preserved; choose a new output directory: '+str(root))
    if args.dry_run:return
    if not script.is_file():raise FileNotFoundError(str(script))
    if not Path(args.data_path).is_dir():raise FileNotFoundError(args.data_path)
    allrows=[]
    for ds,fp32,root,cmd in jobs:
        root.mkdir(parents=True)
        (root/'launch.json').write_text(json.dumps(dict(command=cmd,train_batch=32,eval_batch=1,
            script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),freeze_unit='epoch',
            selection='validation mAcc, batch=1; calibrated epoch0 eligible'),indent=2))
        print('Running %s; log: %s'%(ds,root/'model.log'),flush=True)
        with (root/'model.log').open('w') as log:
            proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:
            print('\n'.join((root/'model.log').read_text(errors='replace').splitlines()[-60:]))
            raise RuntimeError('QAT failed; results preserved: '+str(root))
        rows=validate_results(root,seeds);allrows.extend(rows)
        (root/'completed_runs.json').write_text(json.dumps(rows,indent=2))
        (output/'completed_runs.json').write_text(json.dumps(allrows,indent=2))
        print('Verified evaluation batch=1 for %s (%d seeds)'%(ds,len(rows)),flush=True)

if __name__=='__main__':main()
