"""Run calibration export + FPGA, then validation-selected QAT + FPGA, for both max policies."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys


def simulator_device(device, environment=None):
    env=dict(os.environ if environment is None else environment)
    if device in ('cpu','cuda'):
        return device,env
    if device.startswith('cuda:') and device[5:].isdigit():
        index=int(device[5:])
        visible=env.get('CUDA_VISIBLE_DEVICES')
        if visible is not None:
            devices=[x.strip() for x in visible.split(',') if x.strip()]
            if index>=len(devices):raise ValueError('CUDA index outside CUDA_VISIBLE_DEVICES')
            env['CUDA_VISIBLE_DEVICES']=devices[index]
        else:
            env['CUDA_VISIBLE_DEVICES']=str(index)
        return 'cuda',env
    raise ValueError('Use cpu, cuda, or cuda:N')


def fresh_path(path):
    candidate=path
    i=1
    while candidate.exists():
        candidate=path.with_name(path.name+'_retry'+str(i));i+=1
    return candidate


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fp32-dir',required=True)
    p.add_argument('--data-path',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--dataset',default='UP')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seeds',default='0,6')
    p.add_argument('--stage',choices=['all','calibrated','qat'],default='all')
    p.add_argument('--max-epoch',type=int,default=20)
    p.add_argument('--calibration-steps',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--eval-batch-size',type=int,default=8)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true',help='Continue same manifest/settings, reuse complete model exports')
    args=p.parse_args(argv)
    sim_device,sim_env=simulator_device(args.device)
    seeds=[int(v) for v in args.seeds.split(',')]
    if not seeds or len(set(seeds))!=len(seeds) or min(seeds)<0:
        p.error('Use unique nonnegative seeds')
    if min(args.max_epoch,args.calibration_steps,args.batch_size,args.eval_batch_size)<=0:
        p.error('Epoch/calibration/batch sizes must be positive')
    out=Path(args.output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError('Use a fresh output directory: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parent
    fp=str(Path(args.fp32_dir).expanduser().resolve())
    data=str(Path(args.data_path).expanduser().resolve())
    stages=['calibrated','qat'] if args.stage=='all' else [args.stage]
    jobs=[]
    for stage in stages:
        for policy in ['patch_max','all_weight_max']:
            root=out/stage/policy
            command=[sys.executable,str(source/'train_mambahsi_spatial_split_dense_qat.py'),
                '--dataset',args.dataset,'--data_set_path',data,'--fp32_dir',fp,
                '--work_dir',str(root/'models'),'--run_tag',stage+'_'+policy,
                '--seeds',args.seeds,'--device',args.device,'--weight_init_policy',policy,
                '--use_D','true','--use_z','false','--ssm-u-quantization','shared','--d-weight-bits','8',
                '--dt-input-bits','9','--dt-output-bits','8','--batch_size',str(args.batch_size),
                '--eval_batch_size',str(args.eval_batch_size),'--calibration_steps',str(args.calibration_steps),
                '--freeze_bn_epoch','1','--freeze_lsq_epoch','1','--selection_metric','mAcc',
                '--include_calibrated_baseline','--lr','1e-5','--max_epoch',str(args.max_epoch),'--eval_interval','1']
            command += ['--export_calibrated_only'] if stage=='calibrated' else [
                '--training_diagnostics','--capture_oa_drop_pp','5','--capture_start_epoch','1']
            jobs.append(dict(stage=stage,policy=policy,root=str(root),command=command,status='planned',simulations=[]))
    manifest=out/'pipeline_manifest.json'
    if args.resume:
        if not manifest.exists():raise FileNotFoundError('Resume requires pipeline_manifest.json')
        previous=json.loads(manifest.read_text())
        if previous.get('dry_run'):raise ValueError('Cannot resume a dry-run plan')
        old=previous['jobs']
        if [j['command'] for j in old]!=[j['command'] for j in jobs]:
            raise ValueError('Resume settings differ; use the original command and add --resume')
        jobs=old
    def save():
        manifest.write_text(json.dumps(dict(dry_run=args.dry_run,jobs=jobs,
            protocol='Both policies; original activation calibration; fixed LSQ and BN stats; validation mAcc selection including epoch0. Test/FPGA scores never select checkpoints.'),indent=2))
    def run(command,log,env=None):
        if args.resume:log=fresh_path(log)
        log.parent.mkdir(parents=True,exist_ok=True)
        print('Running -> '+str(log),flush=True)
        with log.open('w') as stream:
            code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env).returncode
        if code:
            with log.open() as stream:print(''.join(deque(stream,maxlen=80)),file=sys.stderr)
            raise RuntimeError('Failed; outputs preserved: '+str(log))
    save()
    if args.dry_run:
        print(f'{len(jobs)} model jobs; {len(jobs)*len(seeds)} FPGA simulations planned: {manifest}')
        return
    # No silent CPU fallback for expensive scientific GPU experiments.
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; fix environment or explicitly choose --device cpu')
    summary=[]
    for job in jobs:
        root=Path(job['root']);job['status']='running';save()
        try:
            existing=sorted((root/'models').rglob('result.json'))
            if args.resume and existing:
                if len(existing)!=len(seeds):raise RuntimeError('Partial model export; refusing to overwrite existing seeds')
                for path in existing:
                    for name in ('best_qat_foldaware.pth','best_qat_deploy_fused.pth','sample_indices.npz','qat_deploy_test_prediction.npy'):
                        if not path.with_name(name).exists():raise RuntimeError('Incomplete exported model: '+str(path.parent))
                print('Reusing model exports: '+str(root/'models'),flush=True)
            else:
                if args.resume and (root/'models').exists() and any((root/'models').rglob('*.pth')):
                    raise RuntimeError('Incomplete training checkpoint; refusing to overwrite')
                run(job['command'],root/'model.log')
            paths=sorted((root/'models').rglob('result.json'))
            if len(paths)!=len(seeds):raise RuntimeError('Incomplete model outputs')
            results=[(path,json.loads(path.read_text())) for path in paths]
            if sorted(r['seed'] for _,r in results)!=sorted(seeds):raise RuntimeError('Seed set mismatch')
            for path,r in results:
                if r['weight_init_policy']!=job['policy']:raise RuntimeError('Initialization mismatch')
                if job['stage']=='calibrated' and (r['optimization_steps']!=0 or r['best_epoch']!=0):
                    raise RuntimeError('Calibration-only export unexpectedly trained')
                simout=root/'simulations'/('seed'+str(r['seed']))
                completed=next((s for s in reversed(job['simulations']) if s['seed']==r['seed']
                                and s['status']=='complete' and Path(s.get('result','')).is_file()),None) if args.resume else None
                if completed:
                    simout=Path(completed['output_dir'])
                elif args.resume:
                    simout=fresh_path(simout)
                cmd=[sys.executable,str(source/'both_FPGA_single_qat_source.py'),
                    '--qat-run-dir',str(path.parent),'--fp32-dir',fp,'--data-path',data,
                    '--dataset',args.dataset,'--seed',str(r['seed']),'--device',sim_device,
                    '--output-dir',str(simout),'--report-ssm-errors','--ssm-analysis-split','test']
                sim=completed or dict(seed=r['seed'],command=cmd,output_dir=str(simout),status='running',
                    cuda_visible_devices=sim_env.get('CUDA_VISIBLE_DEVICES'))
                if not completed:
                    job['simulations'].append(sim);save()
                    run(cmd,root/('sim_seed'+str(r['seed'])+'.log'),env=sim_env)
                simresult=simout/'fpga_simulation_result.json'
                if not simresult.exists():raise RuntimeError('Missing FPGA result')
                sim['status']='complete';sim['result']=str(simresult);save()
                summary.append(dict(stage=job['stage'],policy=job['policy'],seed=r['seed'],
                    best_epoch=r['best_epoch'],optimization_steps=r['optimization_steps'],
                    initial_validation=r['calibrated_baseline_validation'],best_val_mAcc=r['best_val_mAcc'],
                    model_test_OA=r['qat_test_OA'],model_result=str(path),
                    fpga_result=json.loads(simresult.read_text()),fpga_result_path=str(simresult)))
                (out/'pipeline_results.json').write_text(json.dumps(summary,indent=2))
            job['status']='complete';save()
        except Exception:
            for sim in job['simulations']:
                if sim['status']=='running':sim['status']='failed'
            job['status']='failed';save();raise
    print('Completed: '+str(out/'pipeline_results.json'))


if __name__=='__main__':main()
