#!/usr/bin/env python3
"""UP steady inference power; reuses the existing scene/model/tile GPU benchmark."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from gpu_power_monitor import NVMLDevice,PowerSampler,energy_summary,integrate_power


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fp32-dir',required=True)
    p.add_argument('--qat-run-dir')
    p.add_argument('--model-kind',choices=['fp32','qat','fixed'],default='fp32')
    p.add_argument('--model-path')
    p.add_argument('--data-path',default='./data')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-sizes',default='1,2,4,8,16,32,64')
    p.add_argument('--scopes',nargs='+',choices=['model_only','full_gpu_pipeline'],default=['model_only','full_gpu_pipeline'])
    p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--warmup-seconds',type=float,default=10)
    p.add_argument('--idle-seconds',type=float,default=5)
    p.add_argument('--sample-ms',type=float,default=100)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output-dir',default='./gpu_power_UP')
    p.add_argument('--allow-tf32',action='store_true')
    p.add_argument('--allow-reference-scan',action='store_true')
    p.add_argument('--allow-other-processes',action='store_true',help='Keep results explicitly marked contaminated/unchecked')
    p.add_argument('--allow-display-processes',action='store_true',
                   help='Allow graphics-only Xorg/Xwayland, still reject other compute/graphics tasks; record display background')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();a.dataset='UP';a.hidden_dim=None;a.token_num=None
    a.batch_sizes=[int(b) for b in a.batch_sizes.split(',')]
    if not a.batch_sizes or min(a.batch_sizes)<1 or len(set(a.batch_sizes))!=len(a.batch_sizes):p.error('Invalid batch-sizes')
    if a.seconds<5 or a.warmup_seconds<1 or a.idle_seconds<1 or a.sample_ms<20 or a.repeats<1:p.error('Require seconds>=5, warmup/idle>=1, sample-ms>=20, repeats>=1')
    if a.model_kind!='fp32' and not a.qat_run_dir:p.error('--qat-run-dir required for qat/fixed')
    return a


def save_samples(path,rows,t0):
    fields=list(rows[0])+['relative_s']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows(dict(r,relative_s=r['monotonic_s']-t0) for r in rows)


def check_processes(nv,allow,allow_display=False):
    state=nv.other_processes()
    permitted=set(state.get('display_pids',[])) if allow_display else set()
    # Never waive a process that is also reported as a compute process.
    permitted-=set(state.get('compute_pids',[]))
    state['allowed_display_pids']=sorted(permitted)
    state['blocking_pids']=sorted(set(state['other_pids'])-permitted)
    state['process_policy']='allow_other' if allow else ('allow_display_only' if allow_display else 'strict')
    if (state['blocking_pids'] or state['unavailable']) and not allow:
        raise RuntimeError('GPU must be exclusive and process-query supported: '+json.dumps(state)+
                           '; graphics-only Xorg/Xwayland may be allowed with --allow-display-processes; '
                           'other jobs require an idle GPU or explicit --allow-other-processes (marked in report)')
    return state


def process_boundary_summary(before,after):
    states=(before,after)
    verified=all(not s['unavailable'] for s in states)
    exclusive=verified and all(not s['other_pids'] for s in states)
    compute_clear=verified and all('compute_pids' in s and not s['compute_pids'] for s in states)
    display=any(s.get('display_pids') for s in states)
    display_only=verified and display and all(
        set(s['other_pids'])<=set(s.get('display_pids',[])) and not s.get('compute_pids') for s in states)
    condition='exclusive' if exclusive else ('display_background_only' if display_only else 'shared_or_unverified')
    return dict(exclusive_at_boundaries=exclusive,
                no_other_compute_at_boundaries=compute_clear,
                display_present_at_boundaries=display,
                process_condition=condition)


def main():
    a=parse_args()
    if a.dry_run:
        print(json.dumps(vars(a),indent=2));return
    import numpy as np
    import torch
    import benchmark_gpu_batch1_dense16 as bench
    root=Path(a.output_dir).resolve()
    if root.exists():raise FileExistsError('Choose a new output directory; preserved: '+str(root))
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; check driver and nvidia-smi')
    device=torch.device(a.device)
    if device.type!='cuda':raise ValueError('--device must be cuda')
    torch.cuda.set_device(device)
    torch.empty(1,device=device);torch.cuda.synchronize(device)
    torch.backends.cuda.matmul.allow_tf32=a.allow_tf32
    torch.backends.cudnn.allow_tf32=a.allow_tf32
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(a.seed)
    nv=NVMLDevice(torch.cuda.current_device())
    try:
        initial_processes=check_processes(nv,a.allow_other_processes,a.allow_display_processes)
        print('[GPU process policy] '+json.dumps(dict(nvml=nv.metadata,processes=initial_processes)),flush=True)
        run,meta,checkpoint,fp32=bench.resolve_artifacts(a);a.fp32_dir=str(fp32)
        image,gt,classes,blocks,prep=bench.load_scene_and_blocks(a,fp32)
        net,_,_=bench.load_deploy_model(a,meta,checkpoint,image,blocks,classes,device)
        net.eval()
        cpu=bench.build_cpu_tiles(image,blocks);tile_count=len(cpu)
        root.mkdir(parents=True)
        manifest=dict(arguments=vars(a),nvml=nv.metadata,initial_processes=initial_processes,
            torch_version=torch.__version__,cuda_version=torch.version.cuda,
            checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            config=getattr(net,'_benchmark_config',getattr(net,'config',None)),
            backend=getattr(net,'_benchmark_scan_backend','fixed_gpu_reference' if a.model_kind=='fixed' else 'unknown'),
            tile_count=tile_count,scene_shape=list(gt.shape),preprocess_seconds_excluded=prep,
            protocol='GPU-resident input; complete-scene loop, synchronize after each scene; no PCA/H2D/disk in measurement',
            power_note='NVML telemetry includes device idle/background power; on RTX4090 power usage is ~1s averaged, polling faster does not improve sensor resolution',
            execution='FP32 CUDA' if a.model_kind=='fp32' else ('QAT fake-quant; not native INT8' if a.model_kind=='qat' else 'FP64/INT64 fixed simulation; not native INT8'))
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2,default=bench.json_default))
        trials=[]
        with torch.inference_mode():
            if a.model_kind=='fixed':
                indices=np.linspace(0,tile_count-1,3,dtype=int)
                verification=net.verify(torch.cat([cpu[i]['tile'] for i in indices]).to(device))
                (root/'fixed_verification.json').write_text(json.dumps(verification,default=bench.json_default,indent=2))
            for batch_size in a.batch_sizes:
                batches=bench.preload_gpu_batches(bench.make_tile_batches(cpu,batch_size),device)
                if a.model_kind=='fixed':
                    net.verify_batch_lanes(batches[0]['tiles']);net.verify_batch_lanes(batches[-1]['tiles'])
                for scope in a.scopes:
                    fn=(lambda:bench.run_model_only(net,batches)) if scope=='model_only' else (
                        lambda:bench.run_full_pipeline(net,batches,gt.shape,device,False))
                    for repeat in range(a.repeats):
                        prefix='batch%d_%s_repeat%d'%(batch_size,scope,repeat)
                        print('Measuring '+prefix+' on '+nv.metadata['uuid'],flush=True)
                        before=check_processes(nv,a.allow_other_processes,a.allow_display_processes)
                        torch.cuda.synchronize(device)
                        time.sleep(2.0)  # let the ~1s NVML power average drain after prior work
                        idle=PowerSampler(nv,a.sample_ms/1000).start()
                        it0=time.perf_counter();time.sleep(a.idle_seconds);it1=time.perf_counter()
                        idle_rows=idle.stop();idle_w=integrate_power(idle_rows,it0,it1)/(it1-it0)
                        save_samples(root/(prefix+'_idle.csv'),idle_rows,it0)
                        # Warm both full and tail batch shapes, then sustain full scenes.
                        net(batches[0]['tiles']);net(batches[-1]['tiles']);torch.cuda.synchronize(device)
                        wt=time.perf_counter()
                        while time.perf_counter()-wt<a.warmup_seconds:
                            fn();torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        sampler=PowerSampler(nv,a.sample_ms/1000).start()
                        try:
                            torch.cuda.synchronize(device)
                            e0=nv.energy();t0=time.perf_counter();scenes=0
                            while time.perf_counter()-t0<a.seconds:
                                fn();torch.cuda.synchronize(device);scenes+=1
                            t1=time.perf_counter();e1=nv.energy()
                        finally:rows=sampler.stop()
                        save_samples(root/(prefix+'_power.csv'),rows,t0)
                        after=check_processes(nv,a.allow_other_processes,a.allow_display_processes)
                        r=energy_summary(rows,t0,t1,e0,e1,scenes*tile_count)
                        r.update(batch_size=batch_size,scope=scope,repeat=repeat,scenes=scenes,
                            idle_power_w=idle_w,idle_settle_seconds=2.0,incremental_energy_j=r['energy_j']-idle_w*r['duration_s'],
                            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(device),
                            processes_before=before,processes_after=after,
                            energy_unavailable_reason=nv.energy_unavailable_reason,
                            power_csv=prefix+'_power.csv')
                        r.update(process_boundary_summary(before,after))
                        (root/(prefix+'.json')).write_text(json.dumps(r,indent=2))
                        trials.append(r);(root/'trials.json').write_text(json.dumps(trials,indent=2))
                        print(' %.2f W | %.1f tile/s | %.2f uJ/tile | %s'%(r['mean_power_w'],r['tiles_per_s'],r['microjoules_per_tile'],r['energy_method']),flush=True)
                del batches
        fields=['batch_size','scope','repeats','mean_power_w','sd_power_w','tiles_per_s','microjoules_per_tile','sd_microjoules_per_tile','all_exclusive_at_boundaries','all_no_other_compute_at_boundaries','any_display_present_at_boundaries','process_conditions','energy_methods']
        with (root/'power_summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for b in a.batch_sizes:
                for scope in a.scopes:
                    rr=[r for r in trials if r['batch_size']==b and r['scope']==scope]
                    w.writerow(dict(batch_size=b,scope=scope,repeats=len(rr),
                        mean_power_w=statistics.mean(r['mean_power_w'] for r in rr),
                        sd_power_w=statistics.stdev(r['mean_power_w'] for r in rr) if len(rr)>1 else 0,
                        tiles_per_s=statistics.mean(r['tiles_per_s'] for r in rr),
                        microjoules_per_tile=statistics.mean(r['microjoules_per_tile'] for r in rr),
                        sd_microjoules_per_tile=statistics.stdev(r['microjoules_per_tile'] for r in rr) if len(rr)>1 else 0,
                        all_exclusive_at_boundaries=all(r['exclusive_at_boundaries'] for r in rr),
                        all_no_other_compute_at_boundaries=all(r['no_other_compute_at_boundaries'] for r in rr),
                        any_display_present_at_boundaries=any(r['display_present_at_boundaries'] for r in rr),
                        process_conditions=','.join(sorted(set(r['process_condition'] for r in rr))),
                        energy_methods=','.join(sorted(set(r['energy_method'] for r in rr)))))
        print('Saved: '+str(root/'power_summary.csv'))
    finally:nv.close()

if __name__=='__main__':main()
