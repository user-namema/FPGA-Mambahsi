#!/usr/bin/env python3
"""Numbered reproduction entry points; no PyTorch import during command planning."""
import argparse
from collections import deque
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOFTWARE=ROOT/'software'
sys.path.insert(0,str(SOFTWARE))
# Planning must not create a bytecode cache beside the frozen source snapshot.
_bytecode_setting=sys.dont_write_bytecode
sys.dont_write_bytecode=True
from run_fpga_qat_eval1_four_datasets import device_environment
sys.dont_write_bytecode=_bytecode_setting
CONFIG='current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16'
DATASETS=['UP','HanChuan','HongHu','Houston']
ARCHITECTURE_CASES=[
 '00_baseline_current','01_reference_original_matched','02_reference_original_full',
 '05_fusion_mean','06_fusion_softmax','07_skip_scale_1','08_skip_scale_0',
 '09_restore_z','10_restore_D','12_norm_gn_path','13_activation_silu',
 '14_head_dim_32','15_head_dim_128','16_token_num_2','17_token_num_8',
 '18_d_state_8','19_d_state_32','20_norm_gn_activation_silu','21_restore_z_D',
]
STABILITY_CASES=['control_bn20','bn1','bn1_fixed_scales','bn1_slow_scales',
                 'fixed_low_weight_lr','fixed_freeze_bn_affine']
RESUMABLE={'03_shared_a','06_max_init','11_gpu_fp32','12_gpu_fixed',
           '14_fpga_eval1','15_dt_inputs'}
STAGES={
 '01_fp32_current':'Train final D1 FP32 models',
 '02_architecture':'Historical architecture selection (D0 reference)',
 '03_shared_a':'Paired shared/per-channel A training and dynamics',
 '04_initialization':'Full-validation, zero-training quantization diagnostics',
 '05_stability':'BN/scales/weight learning-rate single-factor QAT',
 '06_max_init':'patch_max/all_weight_max calibration + QAT + simulation',
 '07_freeze':'UP patch_max + D_mean freeze1/freeze20 comparison',
 '08_qat_eval8':'Historical four-dataset QAT, saved evaluation B8',
 '09_fpga_eval8':'Integer simulation of historical B8-selected QAT',
 '10_error_sources':'SSM error-source/full-scene suites',
 '11_gpu_fp32':'FP32 GPU multi-batch timing',
 '12_gpu_fixed':'Fixed arithmetic GPU reference timing',
 '13_qat_eval1':'Four-dataset QAT, selection/evaluation B1',
 '14_fpga_eval1':'Integer simulation of eval1-selected QAT',
 '15_dt_inputs':'Read-only dt input range and quantization probes',
 '16_gpu_power':'UP GPU power and energy',
 '17_replay_jump':'Replay a captured QAT validation jump',
 '18_local_ssm':'Replay captured U/dt/B/C inputs locally',
 '19_batch_diagnosis':'Trace same-tile B1/B8 numerical differences',
}

def merge_fixed_summaries(out,datasets,seeds):
 """Merge requested jobs once, keeping the historical CSV metric definitions."""
 rows=[];fields=None;seen=set()
 identity=('dataset','seed','batch_size','model_kind','scope','clock')
 for ds in datasets:
  for seed in seeds:
   source=out/ds/('seed%d'%seed)/'gpu_batch_summary.csv'
   with source.open(newline='',encoding='utf-8-sig') as stream:
    reader=csv.DictReader(stream)
    if fields is None:fields=reader.fieldnames
    if not fields or reader.fieldnames!=fields or any(k not in fields for k in identity):
     raise ValueError('Incompatible GPU summary columns: '+str(source))
    count=0
    for row in reader:
     if row['dataset']!=ds or row['seed']!=str(seed) or row['model_kind']!='fixed':
      raise ValueError('GPU summary identity mismatch: '+str(source))
     key=tuple(row[k] for k in identity)
     if key in seen:raise ValueError('Duplicate GPU summary record: '+str(key))
     seen.add(key);rows.append(row);count+=1
    if not count:raise ValueError('Empty GPU summary: '+str(source))
 target=out/'gpu_fixed_batch_summary.csv'
 temporary=target.with_suffix('.csv.tmp')
 with temporary.open('w',newline='',encoding='utf-8') as stream:
  writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 temporary.replace(target)
 return target

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('stage',choices=list(STAGES))
 p.add_argument('--datasets',nargs='+',choices=DATASETS)
 p.add_argument('--seeds',default=os.getenv('SEEDS'))
 p.add_argument('--device',default=os.getenv('DEVICE','cuda:0'))
 p.add_argument('--data-root',default=os.getenv('DATA_ROOT',str(ROOT/'data')))
 p.add_argument('--output-root',default=os.getenv('OUTPUT_ROOT'))
 p.add_argument('--fp32-root',default=os.getenv('FP32_ROOT',str(ROOT/'results/SPATIAL_SPLIT_3WAY_DENSE')))
 p.add_argument('--fp32-template',default=os.getenv('FP32_TEMPLATE'))
 p.add_argument('--qat-template',default=os.getenv('QAT_TEMPLATE'))
 p.add_argument('--qat-root',default=os.getenv('QAT_ROOT',str(ROOT/'results/qat_eval1_4datasets')))
 p.add_argument('--pair-root',default=os.getenv('PAIR_ROOT',str(ROOT/'results/03_shared_a')))
 p.add_argument('--batch-sizes',default=os.getenv('BATCH_SIZES','1,2,4,8,16,32,64'))
 p.add_argument('--cases',nargs='+',help='02: architecture file stems; 05: stability variants')
 p.add_argument('--phase',choices=['all','train','analyze'],default='all')
 p.add_argument('--init-kind',choices=['weights','groups'],default='weights')
 p.add_argument('--suite',choices=['sources','widths','nonlinear','requant','k-precision'],default='sources')
 p.add_argument('--event-dir',default=os.getenv('EVENT_DIR'))
 p.add_argument('--prediction-dir',default=os.getenv('PREDICTION_DIR'))
 p.add_argument('--inputs-glob',default=os.getenv('INPUTS_GLOB'))
 p.add_argument('--mode',choices=['native','tf32-off','reference-tf32-off'],default='native')
 p.add_argument('--resume',action='store_true',default=os.getenv('RESUME')=='1')
 p.add_argument('--allow-display-processes',action='store_true',default=os.getenv('ALLOW_DISPLAY_PROCESSES')=='1',help='Stage16: allow graphics-only Xorg/Xwayland; other GPU work remains rejected')
 p.add_argument('--dry-run',action='store_true',default=os.getenv('DRY_RUN')=='1')
 a=p.parse_args(argv);s=a.stage
 if a.resume and s not in RESUMABLE:
  p.error('--resume is unsupported for this stage; use a fresh --output-root')
 if a.cases and s not in ('02_architecture','05_stability'):
  p.error('--cases is only available for stages 02 and 05')
 try:
  device_environment(a.device,os.environ)
 except ValueError as e:p.error(str(e))
 if s in ('11_gpu_fp32','12_gpu_fixed','16_gpu_power') and a.device=='cpu':
  p.error('GPU timing and power stages require --device cuda or cuda:N')
 if s in ('11_gpu_fp32','12_gpu_fixed','16_gpu_power'):
  try:batches=[int(x) for x in a.batch_sizes.split(',')]
  except ValueError:p.error('--batch-sizes must contain comma-separated positive integers')
  if not batches or min(batches)<1 or len(set(batches))!=len(batches):
   p.error('--batch-sizes must contain unique positive integers')
 diagnostic=s in ['04_initialization','05_stability','06_max_init','07_freeze','10_error_sources','15_dt_inputs','16_gpu_power','17_replay_jump','18_local_ssm','19_batch_diagnosis']
 datasets=a.datasets or (['UP'] if diagnostic else DATASETS)
 seeds=a.seeds or ('0' if s in ['10_error_sources','11_gpu_fp32','12_gpu_fixed','15_dt_inputs','16_gpu_power','17_replay_jump','18_local_ssm','19_batch_diagnosis'] else ('0,6' if s in ['04_initialization','05_stability','06_max_init'] else ','.join(map(str,range(10)))))
 try:seedlist=[int(x) for x in seeds.split(',')]
 except ValueError:p.error('--seeds must contain comma-separated nonnegative integers')
 if not seedlist or min(seedlist)<0 or len(set(seedlist))!=len(seedlist) or len(set(datasets))!=len(datasets):p.error('Invalid duplicate/negative seeds or datasets')
 seeds=','.join(map(str,seedlist))
 if s in ['17_replay_jump','18_local_ssm','19_batch_diagnosis','16_gpu_power'] and (datasets!=['UP'] or len(seedlist)!=1):p.error('This entry accepts UP and one seed; use underlying Python CLI for other cases')
 defaultout={'01_fp32_current':ROOT/'results','08_qat_eval8':ROOT/'results/qat_eval8_4datasets','13_qat_eval1':ROOT/'results/qat_eval1_4datasets'}.get(s,ROOT/'results'/s)
 out=Path(a.output_root or defaultout).expanduser().resolve();data=str(Path(a.data_root).expanduser().resolve())
 # Resolve inputs before subprocesses change cwd to the repository root.
 for name in ('fp32_root','qat_root','pair_root','event_dir','prediction_dir','inputs_glob'):
  if getattr(a,name) is not None:setattr(a,name,str(Path(getattr(a,name)).expanduser().resolve()))
 fp_template=str(Path(a.fp32_template or str(Path(a.fp32_root)/('{dataset}_all_samples_sqrt_inverse_clip3_2000_nobias')/CONFIG)).expanduser().resolve())
 fp=lambda ds: str(Path(fp_template.format(dataset=ds)).expanduser().resolve())
 defaultqat=str(ROOT/'results/qat_eval8_4datasets/{dataset}/models/SPATIAL_SPLIT_3WAY_DENSE_QAT/{dataset}_patch_max_D_mean_freeze20'/CONFIG/'run_seed{seed}')
 qt=lambda ds,seed: str(Path((a.qat_template or defaultqat).format(dataset=ds,seed=seed)).expanduser().resolve())
 jobs=[]
 def add(label,script,args,device_map=False):
  env=dict(os.environ)
  if device_map:
   mapped,env=device_environment(a.device,env)
   args=[mapped if x=='__SIM_DEVICE__' else x for x in args]
  jobs.append((label,[sys.executable,'-u',str(SOFTWARE/script)]+[str(x) for x in args],env))
 def qatargs(ds,dest,tag,eval_batch=8,dinit='mean'):
  return ['--dataset',ds,'--data_set_path',data,'--fp32_dir',fp(ds),'--work_dir',dest,'--run_tag',tag,'--seeds',seeds,'--device',a.device,'--weight_init_policy','patch_max','--d-init-policy',dinit,'--use_D','true','--use_z','false','--ssm-u-quantization','shared','--d-weight-bits','8','--dt-input-bits','9','--dt-output-bits','8','--batch_size','32','--eval_batch_size',eval_batch,'--calibration_steps','100','--freeze_bn_epoch','20','--freeze_lsq_epoch','20','--lr','1e-5','--lr_schedule','constant','--max_epoch','100','--eval_interval','1','--selection_metric','mAcc','--include_calibrated_baseline','--training_diagnostics','--capture_oa_drop_pp','5','--capture_start_epoch','1']
 if s in ['01_fp32_current','02_architecture']:
  cases=['10_restore_D'] if s=='01_fp32_current' else (a.cases or ARCHITECTURE_CASES)
  for ds in datasets:
   for case in cases:
    script=SOFTWARE/'ablation_scripts'/(case+'.sh')
    if not script.is_file() or case.startswith('_'):p.error('Unknown included architecture case: '+case)
    cmd=['bash',str(script),'--dataset',ds,'--data_set_path',data,'--work_dir',str(out),'--seeds',seeds,'--device',a.device]
    jobs.append((ds+'_'+case,cmd,dict(os.environ,PYTHON_BIN=sys.executable)))
 elif s=='03_shared_a':
  common=['--datasets']+datasets+['--seeds',seeds,'--device',a.device,'--data-path',data]
  if a.phase!='analyze':add('train','run_current_a_gpu_experiments.py',['--task','train-a']+common+['--fp32-template',fp_template,'--max-epoch','400','--output-dir',out]+(['--resume'] if a.resume else []))
  if a.phase!='train':add('analyze','run_current_a_gpu_experiments.py',['--task','analyze-a']+common+['--pair-root',a.pair_root if a.phase=='analyze' else out,'--max-tiles','0','--analysis-batch-size','8','--output-dir',str(out)+'_dynamics']+(['--resume'] if a.resume else []))
 elif s=='04_initialization':
  for ds in datasets:
   add(ds,'train_mambahsi_spatial_split_dense_qat.py',qatargs(ds,out/ds/'models','init_'+a.init_kind,dinit='max')+['--weight_init_policy','mean','--weight_init_validation_only' if a.init_kind=='weights' else '--initial_diagnostics_only'])
 elif s=='05_stability':
  for ds in datasets:add(ds,'run_qat_stability_sweep.py',['--fp32-dir',fp(ds),'--data-path',data,'--dataset',ds,'--device',a.device,'--seeds',seeds,'--max-epoch','100','--eval-interval','1','--variants']+(a.cases or STABILITY_CASES)+['--output-dir',out/ds])
 elif s=='06_max_init':
  for ds in datasets:add(ds,'run_max_init_qat_fpga.py',['--fp32-dir',fp(ds),'--data-path',data,'--dataset',ds,'--device',a.device,'--seeds',seeds,'--stage','all','--max-epoch','20','--output-dir',out/ds]+(['--resume'] if a.resume else []))
 elif s in ['07_freeze','08_qat_eval8']:
  for ds in datasets:
   for epoch in ([1,20] if s=='07_freeze' else [20]):
    dest=out/ds/('freeze%d'%epoch)/'models' if s=='07_freeze' else out/ds/'models'
    add(ds+'_freeze%d'%epoch,'train_mambahsi_spatial_split_dense_qat.py',qatargs(ds,dest,'patch_max_D_mean_freeze%d'%epoch)+['--freeze_bn_epoch',epoch,'--freeze_lsq_epoch',epoch])
 elif s in ['09_fpga_eval8','10_error_sources','12_gpu_fixed','19_batch_diagnosis']:
  for ds in datasets:
   for seed in seedlist:
    base=['--qat-run-dir',qt(ds,seed),'--fp32-dir',fp(ds),'--data-path',data,'--device','__SIM_DEVICE__']
    if s=='09_fpga_eval8':add(ds+str(seed),'both_FPGA_single_qat_source.py',base+['--dataset',ds,'--seed',seed,'--report-ssm-errors','--output-dir',out/ds/('seed%d'%seed)],True)
    elif s=='10_error_sources':add(ds+str(seed),'run_fpga_error_sweep.py',base+['--suite',a.suite,'--output-dir',out/ds/('seed%d'%seed)/a.suite],True)
    elif s=='19_batch_diagnosis':
     if not a.prediction_dir:p.error('--prediction-dir is required')
     add('trace','diagnose_qat_batch.py',base+['--prediction-dir',a.prediction_dir,'--save-traces','--output-dir',out],True)
    else:add(ds+str(seed),'run_current_a_gpu_experiments.py',['--task','gpu','--datasets',ds,'--seeds',seed,'--device',a.device,'--data-path',data,'--model-kind','fixed','--fp32-template',fp(ds),'--qat-template',qt(ds,seed),'--batch-sizes',a.batch_sizes,'--repeats','5','--warmup-steps','5','--single-tile-trials','20','--output-dir',out/ds/('seed%d'%seed)]+(['--resume'] if a.resume else []))
 elif s=='11_gpu_fp32':add('gpu','run_current_a_gpu_experiments.py',['--task','gpu','--datasets']+datasets+['--seeds',seeds,'--device',a.device,'--data-path',data,'--fp32-template',fp_template,'--model-kind','fp32','--batch-sizes',a.batch_sizes,'--repeats','20','--warmup-steps','100','--single-tile-trials','200','--output-dir',out]+(['--resume'] if a.resume else []))
 elif s=='13_qat_eval1':
  args=['--datasets']+datasets+['--seeds',seeds,'--device',a.device,'--data-path',data,'--fp32-root',a.fp32_root,'--max-epoch','100','--output-dir',out]
  add('qat','run_qat_eval_batch1_four_datasets.py',args)
  for ds in datasets:jobs[-1][2]['FP32_'+ds.upper()]=fp(ds)
 elif s in ['14_fpga_eval1','15_dt_inputs']:
  args=['--project-root',SOFTWARE,'--qat-root',a.qat_root,'--fp32-root',a.fp32_root,'--data-path',data,'--device',a.device,'--datasets']+datasets+['--seeds',seeds,'--output-dir',out]
  if a.resume:args+=['--resume']
  if s=='15_dt_inputs':args+=['--report-dt-inputs']
  add('fpga','run_fpga_qat_eval1_four_datasets.py',args)
  env=jobs[-1][2]
  for ds in datasets:env['FP32_'+ds.upper()]=fp(ds)
 elif s=='16_gpu_power':add('power','benchmark_up_gpu_power.py',['--fp32-dir',fp('UP'),'--data-path',data,'--device',a.device,'--seed',seedlist[0],'--model-kind','fp32','--batch-sizes',a.batch_sizes,'--seconds','30','--warmup-seconds','10','--idle-seconds','5','--repeats','3','--sample-ms','100','--output-dir',out]+(['--allow-display-processes'] if a.allow_display_processes else []))
 elif s=='17_replay_jump':
  if not a.event_dir:p.error('--event-dir is required')
  add('replay','replay_qat_jump.py',['--event-dir',a.event_dir,'--device',a.device,'--mode',a.mode,'--output-dir',out])
 elif s=='18_local_ssm':
  if not a.inputs_glob:p.error('--inputs-glob is required')
  if a.suite=='requant':p.error('Use stage10 for end-to-end requantization')
  add('local','run_ssm_error_ablation.py',['--inputs-glob',a.inputs_glob,'--suite',a.suite,'--output-dir',out])
 if a.dry_run:
  for label,cmd,env in jobs:
   overrides=[k+'='+shlex.quote(env[k]) for k in ('CUDA_VISIBLE_DEVICES','FP32_UP','FP32_HANCHUAN','FP32_HONGHU','FP32_HOUSTON','PYTHON_BIN') if k in env and (k=='CUDA_VISIBLE_DEVICES' or env[k]!=os.environ.get(k))]
   print(label+': '+' '.join(overrides+[shlex.quote(str(x)) for x in cmd]))
  return
 logs=ROOT/'results/launch_logs'/s;logs.mkdir(parents=True,exist_ok=True)
 for label,cmd,env in jobs:
  log=logs/(label+'.log');i=1
  while log.exists():log=logs/(label+'.attempt%d.log'%i);i+=1
  print('Running '+label+'; log: '+str(log),flush=True)
  with log.open('w') as f:
   f.write(json.dumps(cmd)+'\n');f.flush()
   result=subprocess.run(cmd,cwd=str(ROOT),env=env,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:
   with log.open(errors='replace') as f:print(''.join(deque(f,maxlen=60)))
   raise SystemExit('Failed; results preserved: '+str(log))
 if s=='12_gpu_fixed':print('Saved: '+str(merge_fixed_summaries(out,datasets,seedlist)))
 print('Completed stage '+s)

if __name__=='__main__':main()
