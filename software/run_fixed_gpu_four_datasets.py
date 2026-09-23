"""Discover the same patch_max/D_mean/freeze20 QAT artifacts as FPGA simulation."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


def discover(dataset, seed):
    default = './patch_max_D_mean_freeze_10seeds/freeze20/models' if dataset=='UP' else str(Path(os.environ.get('QAT_OTHER_ROOT','./patch_max_D_mean_freeze20_other3_10seeds'))/dataset/'models')
    root = Path(os.environ.get('QAT_'+dataset.upper()+'_ROOT',default)).expanduser()
    found=[]
    for p in root.rglob('result.json'):
        r=json.loads(p.read_text())
        if r.get('dataset')==dataset and r.get('seed')==seed: found.append((p,r))
    if len(found)!=1: raise ValueError('%s seed%d: expected one QAT result under %s, found %d'%(dataset,seed,root,len(found)))
    p,r=found[0]
    for key,value in dict(weight_init_policy='patch_max',d_init_policy='mean',freeze_bn_epoch=20,freeze_lsq_epoch=20).items():
        if r.get(key)!=value: raise ValueError('%s: %s differs from freeze20 protocol'%(p,key))
    fp=Path(os.environ.get('FP32_'+dataset.upper(),r['source_fp32_dir'])).expanduser()
    for artifact in ['train_only_preprocess.npz','spatial_split_masks.npz','spatial_split.json']:
        if not (fp/artifact).is_file(): raise FileNotFoundError(fp/artifact)
    if not (p.parent/'best_qat_foldaware.pth').is_file(): raise FileNotFoundError(p.parent/'best_qat_foldaware.pth')
    return p.parent.resolve(),fp.resolve()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',default=['UP','HanChuan','HongHu','Houston'],choices=['UP','HanChuan','HongHu','Houston'])
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seeds',default='0')
    parser.add_argument('--batch-sizes',default='1,2,4,8,16,32,64')
    parser.add_argument('--data-path',default='./data')
    parser.add_argument('--output-dir',default='./gpu_fixed_4datasets')
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--warmup-steps',type=int,default=5)
    parser.add_argument('--single-tile-trials',type=int,default=20)
    parser.add_argument('--measure-h2d',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    a=parser.parse_args()
    seeds=[int(x) for x in a.seeds.split(',')]
    if len(seeds)!=len(set(seeds)) or len(a.datasets)!=len(set(a.datasets)): raise ValueError('Duplicate seeds/datasets')
    root=Path(a.output_dir).resolve(); here=Path(__file__).resolve().parent
    jobs=[]
    for ds in a.datasets:
        for seed in seeds:
            qat,fp=discover(ds,seed)
            dest=root/ds/('seed%d'%seed)
            cmd=[sys.executable,str(here/'run_current_a_gpu_experiments.py'),'--task','gpu','--datasets',ds,
                 '--seeds',str(seed),'--device',a.device,'--model-kind','fixed','--qat-template',str(qat),
                 '--fp32-template',str(fp),'--data-path',a.data_path,'--batch-sizes',a.batch_sizes,
                 '--repeats',str(a.repeats),'--warmup-steps',str(a.warmup_steps),
                 '--single-tile-trials',str(a.single_tile_trials),'--output-dir',str(dest),'--resume']
            if a.measure_h2d: cmd.append('--measure-h2d')
            if a.dry_run: cmd.append('--dry-run')
            jobs.append((dest,cmd))
    # Discover and validate all checkpoints before running expensive jobs.
    for dest,cmd in jobs: subprocess.run(cmd,check=True)
    if not a.dry_run:
        rows=[]
        for dest,_ in jobs:
            with (dest/'gpu_batch_summary.csv').open() as f: rows.extend(csv.DictReader(f))
        root.mkdir(parents=True,exist_ok=True)
        with (root/'gpu_fixed_batch_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print('Saved:',root/'gpu_fixed_batch_summary.csv')


if __name__=='__main__': main()
