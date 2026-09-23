"""Paired validation-led QAT stability diagnostics; never select by test accuracy."""
import argparse
from collections import deque
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import statistics


def variants():
    return {
        'control_bn20': ['--freeze_bn_epoch','20'],
        'bn1': ['--freeze_bn_epoch','1'],
        'bn1_fixed_scales': ['--freeze_bn_epoch','1','--freeze_lsq_epoch','1'],
        'bn1_slow_scales': ['--freeze_bn_epoch','1',
            '--activation_scale_lr_multiplier','0.1','--dt_scale_lr_multiplier','0.1',
            '--weight_scale_lr_multiplier','0.1','--d_scale_lr_multiplier','0.1'],
        'bn1_low_lr_cosine': ['--freeze_bn_epoch','1','--lr','1e-6',
                              '--lr_schedule','cosine','--min_lr_ratio','0.1'],
        'fixed_low_weight_lr': ['--freeze_bn_epoch','1','--freeze_lsq_epoch','1',
                                '--weight_lr_multiplier','0.1'],
        'fixed_freeze_bn_affine': ['--freeze_bn_epoch','1','--freeze_lsq_epoch','1',
                                   '--freeze_bn_affine'],
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fp32-dir',required=True)
    p.add_argument('--data-path',required=True)
    p.add_argument('--dataset',default='UP')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seeds',default='0,6')
    p.add_argument('--variants',nargs='+',choices=list(variants()),default=list(variants())[:4])
    p.add_argument('--max-epoch',type=int,default=100)
    p.add_argument('--eval-interval',type=int,default=1)
    p.add_argument('--calibration-steps',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--eval-batch-size',type=int,default=8)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--capture-oa-drop-pp',type=float,default=0.)
    p.add_argument('--capture-max-events',type=int,default=3)
    p.add_argument('--capture-start-epoch',type=int,default=25)
    args=p.parse_args(argv)
    if (not math.isfinite(args.capture_oa_drop_pp) or args.capture_oa_drop_pp < 0
            or args.capture_max_events <= 0 or args.capture_start_epoch < 1):
        p.error('Invalid capture threshold/start/event limit')
    if args.capture_oa_drop_pp > 0 and args.eval_interval != 1:
        p.error('Capture requires --eval-interval 1')
    out=Path(args.output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh output directory: '+str(out))
    if any(v<=0 for v in [args.max_epoch,args.eval_interval,args.calibration_steps,args.batch_size,args.eval_batch_size]):
        p.error('Epoch, calibration, evaluation interval and batch sizes must be positive')
    seeds=[int(s) for s in args.seeds.split(',')]
    if len(set(seeds))!=len(seeds) or any(s<0 for s in seeds):
        p.error('Use unique nonnegative seeds separated by commas')
    out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for name in dict.fromkeys(args.variants):
        work=out/name/'training'
        command=[sys.executable,str(Path(__file__).with_name('train_mambahsi_spatial_split_dense_qat.py').resolve()),
            '--dataset',args.dataset,'--data_set_path',str(Path(args.data_path).expanduser().resolve()),
            '--fp32_dir',str(Path(args.fp32_dir).expanduser().resolve()),'--work_dir',str(work),
            '--run_tag',name,'--seeds',args.seeds,'--device',args.device,
            '--use_D','true','--use_z','false','--ssm-u-quantization','shared','--d-weight-bits','8',
            '--dt-input-bits','9','--dt-output-bits','8','--lr','1e-5','--lr_schedule','constant',
            '--max_epoch',str(args.max_epoch),'--eval_interval',str(args.eval_interval),
            '--calibration_steps',str(args.calibration_steps),'--batch_size',str(args.batch_size),
            '--eval_batch_size',str(args.eval_batch_size),'--selection_metric','mAcc',
            '--training_diagnostics','--include_calibrated_baseline',
            '--capture_oa_drop_pp',str(args.capture_oa_drop_pp),
            '--capture_max_events',str(args.capture_max_events),
            '--capture_start_epoch',str(args.capture_start_epoch)]+variants()[name]
        jobs.append(dict(variant=name,command=command,work_dir=str(work),status='planned'))
    manifest=out/'stability_manifest.json'
    def save():
        manifest.write_text(json.dumps(dict(dry_run=args.dry_run,jobs=jobs,
            protocol='D1/shared-U/D8 max init/dt9-out8; validation mAcc selection includes epoch 0 in every arm; test metrics are descriptive only'),indent=2))
    save()
    if args.dry_run:
        print(f'{len(jobs)} variant jobs, {len(jobs)*len(seeds)} training runs planned: {manifest}')
        return
    rows=[]
    for job in jobs:
        root=Path(job['work_dir']).parent;root.mkdir(parents=True,exist_ok=True)
        log=root/'training.log';job.update(status='running',log=str(log));save()
        print(f"[QAT stability] {job['variant']} -> {log}",flush=True)
        with log.open('w') as stream:
            code=subprocess.run(job['command'],stdout=stream,stderr=subprocess.STDOUT).returncode
        job['returncode']=code
        if code:
            job['status']='failed';save()
            with log.open() as stream:print(''.join(deque(stream,maxlen=80)),file=sys.stderr)
            raise SystemExit(code if code>0 else 1)
        paths=sorted(Path(job['work_dir']).glob('**/run_seed*/result.json'))
        if len(paths)!=len(seeds):
            job['status']='failed_missing_results';save()
            raise RuntimeError('Incomplete result files: '+str(root))
        job['results']=[str(path) for path in paths]
        for path in paths:
            result=json.loads(path.read_text())
            history=[json.loads(line) for line in (path.parent/'qat_training_history.jsonl').read_text().splitlines()]
            trained=[r['validation']['OA'] for r in history if r['epoch']>0]
            late=[r['validation']['OA'] for r in history if r['epoch']>=25]
            rows.append(dict(variant=job['variant'],seed=result['seed'],result_path=str(path),
                baseline_val_OA=result['calibrated_baseline_validation']['OA'],
                best_epoch=result['best_epoch'],best_val_OA=result['best_val_OA'],best_val_mAcc=result['best_val_mAcc'],
                observed_val_OA_range_pp=100*(max(trained)-min(trained)),
                post25_val_OA_range_pp=100*(max(late)-min(late)) if late else '',
                post25_val_OA_std_pp=100*statistics.pstdev(late) if late else '',
                post25_max_adjacent_drop_pp=100*max([0.]+[a-b for a,b in zip(late,late[1:])]),
                post25_max_adjacent_change_pp=100*max([0.]+[abs(a-b) for a,b in zip(late,late[1:])]),
                eval_interval=args.eval_interval,
                captured_events=len(result.get('jump_capture',{}).get('events',[])),
                freeze_bn_affine=result.get('freeze_bn_affine',False),
                weight_lr_multiplier=result.get('weight_lr_multiplier',1.),
                reload_validation_match=result['best_validation_recheck_match']))
        with (out/'validation_summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        job['status']='complete';save()
    print('Finished: '+str(out/'validation_summary.csv'))


if __name__=='__main__':main()
