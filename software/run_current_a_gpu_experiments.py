"""Current four-dataset A ablation and multi-batch GPU benchmarks (Python 3.8+).
Run --help or --dry-run without importing PyTorch.
"""
import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

DATASETS = ['UP', 'HanChuan', 'HongHu', 'Houston']
CONFIG = 'current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16'
DEFAULT = './results/SPATIAL_SPLIT_3WAY_DENSE/{dataset}_all_samples_sqrt_inverse_clip3_2000_nobias/' + CONFIG


def numbers(value):
    result = [int(x) for x in value.split(',')]
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise argparse.ArgumentTypeError('Expected unique nonnegative integers separated by commas')
    return result


def save_csv(path, rows):
    if not rows: return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize_pairs(root, datasets, seeds):
    rows, aggregate = [], []
    for ds in datasets:
        manifests = [json.loads((root/ds/m/'pair_protocol.json').read_text()) for m in ['shared','per_channel']]
        if manifests[0]['source_hashes'] != manifests[1]['source_hashes'] or manifests[0]['max_epoch'] != manifests[1]['max_epoch']:
            raise ValueError('Pair protocols differ: '+ds)
        results = {m:[] for m in ['shared','per_channel']}
        for seed in seeds:
            pair=[]
            for mode in results:
                run = root/ds/mode/('run_seed%d'%seed)
                result = json.loads((run/'result.json').read_text())
                if result['dataset'] != ds or int(result['seed']) != seed or result['model_config']['A_mode'] != mode or not result['model_config']['use_D']:
                    raise ValueError('Pair identity mismatch: '+str(run))
                pair.append(result); results[mode].append(result)
            c0,c1 = [dict(x['model_config']) for x in pair]
            c0.pop('A_mode'); c1.pop('A_mode')
            if c0 != c1: raise ValueError('Architecture differs beyond A_mode')
            row=dict(dataset=ds, seed=seed)
            for metric in ['OA','mAcc','Kappa']:
                row[metric+'_shared_percent'] = pair[0]['test_'+metric]*100
                row[metric+'_per_channel_percent'] = pair[1]['test_'+metric]*100
                row[metric+'_cost_shared_pp'] = (pair[1]['test_'+metric]-pair[0]['test_'+metric])*100
            rows.append(row)
        stats={'dataset':ds,'seeds':len(seeds)}
        for mode,items in results.items():
            for metric in ['OA','mAcc','Kappa']:
                values=[x['test_'+metric]*100 for x in items]
                stats[mode+'_'+metric+'_mean']=statistics.mean(values)
                stats[mode+'_'+metric+'_sample_std']=statistics.stdev(values) if len(values)>1 else None
        differences=[x['OA_cost_shared_pp'] for x in rows if x['dataset']==ds]
        stats['paired_OA_cost_shared_pp_mean']=statistics.mean(differences)
        stats['paired_OA_cost_shared_pp_sample_std']=statistics.stdev(differences) if len(differences)>1 else None
        aggregate.append(stats)
    save_csv(root/'paired_accuracy_by_seed.csv',rows)
    save_csv(root/'paired_accuracy_summary.csv',aggregate)
    (root/'paired_accuracy_summary.json').write_text(json.dumps(aggregate,indent=2))
    return aggregate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True, choices=['train-a','analyze-a','summarize-a','gpu'])
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    p.add_argument('--seeds', type=numbers, default=numbers('0,1,2,3,4,5,6,7,8,9'))
    p.add_argument('--fp32-template', default=DEFAULT, help='Directory template with {dataset}; no shell glob')
    p.add_argument('--qat-template', help='QAT run template with {dataset} and {seed}, required for --model-kind qat')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--pair-root', default='./a_shared_current_4datasets', help='train-a output root for analyze/summarize')
    p.add_argument('--data-path', default='./data')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-epoch', type=int, default=400)
    p.add_argument('--max-tiles', type=int, default=0, help='A analysis: 0=all test tiles')
    p.add_argument('--static-only', action='store_true')
    p.add_argument('--batch-sizes', type=numbers, default=numbers('1,2,4,8,16,32,64'))
    p.add_argument('--analysis-batch-size', type=int, default=8)
    p.add_argument('--model-kind', choices=['fp32','qat','fixed'], default='fp32')
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--single-tile-trials', type=int, default=200)
    p.add_argument('--measure-h2d', action='store_true')
    p.add_argument('--allow-tf32', action='store_true')
    p.add_argument('--allow-reference-scan', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a=p.parse_args()
    if min(a.batch_sizes)<1 or a.analysis_batch_size<1 or a.max_tiles<0 or a.max_epoch<1:
        p.error('Invalid batch/epoch/tile count')
    if a.task=='gpu' and a.model_kind in ('qat','fixed') and not a.qat_template:
        p.error('--qat-template is required for QAT timing')
    root=Path(a.output_dir).resolve();pair=Path(a.pair_root).resolve();here=Path(__file__).resolve().parent
    if a.task=='summarize-a':
        summarize_pairs(pair,a.datasets,a.seeds);print('Saved summaries:',pair);return
    jobs=[]
    for ds in a.datasets:
        fp=Path(a.fp32_template.format(dataset=ds)).expanduser().resolve()
        if a.task=='train-a':
            for mode in ['shared','per_channel']:
                dest=root/ds/mode
                cmd=[sys.executable,str(here/'train_shared_a_frozen_pair.py'),'--source-dir',str(fp),'--output-dir',str(dest),'--a-mode',mode,'--seeds',','.join(map(str,a.seeds)),'--device',a.device,'--data-path',a.data_path,'--max-epoch',str(a.max_epoch)]
                if a.resume:cmd+=['--resume']
                jobs.append((ds+'/'+mode,cmd))
        elif a.task=='analyze-a':
            for mode in ['shared','per_channel']:
                for seed in a.seeds:
                    dest=root/ds/mode/('seed%d'%seed)
                    cmd=[sys.executable,str(here/'analyze_alog_dynamics.py'),'--run-dir',str(pair/ds/mode/('run_seed%d'%seed)),'--dataset',ds,'--data-path',a.data_path,'--device',a.device,'--split','test','--batch-size',str(a.analysis_batch_size),'--max-tiles',str(a.max_tiles),'--coefficient-fraction-bits','16','20','24','--no-plots','--output-dir',str(dest)]
                    if a.static_only:cmd+=['--static-only']
                    jobs.append((ds+'/'+mode+'/seed%d'%seed,cmd))
        else:
            for seed in a.seeds:
                for batch in a.batch_sizes:
                    dest=root/ds/('seed%d'%seed)/('batch%d'%batch)
                    cmd=[sys.executable,str(here/'benchmark_gpu_batch1_dense16.py'),'--fp32-dir',str(fp),'--dataset',ds,'--seed',str(seed),'--data-path',a.data_path,'--device',a.device,'--model-kind',a.model_kind,'--batch-size',str(batch),'--repeats',str(a.repeats),'--warmup-steps',str(a.warmup_steps),'--single-tile-trials',str(a.single_tile_trials if batch==1 else 0),'--output-dir',str(dest)]
                    if a.qat_template:cmd+=['--qat-run-dir',a.qat_template.format(dataset=ds,seed=seed)]
                    if a.model_kind=='fixed':cmd+=['--single-tile-warmup',str(a.warmup_steps)]
                    for name in ['measure_h2d','allow_tf32','allow_reference_scan']:
                        if getattr(a,name):cmd+=['--'+name.replace('_','-')]
                    jobs.append((ds+'/seed%d/batch%d'%(seed,batch),cmd))
    if a.dry_run:
        import shlex
        for _,cmd in jobs:print(' '.join(shlex.quote(x) for x in cmd))
        return
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for key,cmd in jobs:
        log=root/'logs'/(key.replace('/','_')+'.log');log.parent.mkdir(exist_ok=True)
        done=log.with_suffix('.done.json');digest=hashlib.sha256(json.dumps(cmd).encode()).hexdigest()
        # --resume itself does not change the scientific command signature.
        digest=hashlib.sha256(json.dumps([x for x in cmd if x!='--resume']).encode()).hexdigest()
        if a.resume and done.exists() and json.loads(done.read_text()).get('command_sha256')==digest:
            print('Completed:',key);continue
        if log.exists() and not (a.resume and a.task=='train-a'):
            raise FileExistsError('Existing log/output preserved: '+str(log))
        print('Running:',key,'Log:',log,flush=True)
        with log.open('a' if a.resume else 'w') as f:
            f.write(json.dumps(cmd)+'\n');f.flush()
            result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            print('\n'.join(log.read_text(errors='replace').splitlines()[-40:]),file=sys.stderr)
            raise RuntimeError('Failed: '+key+'; completed runs preserved; see '+str(log))
        done.write_text(json.dumps(dict(command_sha256=digest,command=cmd),indent=2))
    if a.task=='train-a':summarize_pairs(root,a.datasets,a.seeds)
    elif a.task=='analyze-a':
        for ds in a.datasets:
            for mode in ['shared','per_channel']:
                for seed in a.seeds:
                    summary=json.loads((root/ds/mode/('seed%d'%seed)/'analysis_summary.json').read_text())
                    for core in summary['core_static_dynamics']:
                        rows.append(dict(core,dataset=ds,A_mode=mode,seed=seed))
        # JSON preserves nested fields; CSV contains scalar pole metrics.
        (root/'pole_metrics_all_runs.json').write_text(json.dumps(rows,indent=2))
        scalar=[{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in rows]
        save_csv(root/'pole_metrics_all_runs.csv',scalar)
        aggregate=[]
        for ds in a.datasets:
            for mode in ['shared','per_channel']:
                cores=sorted({x['core'] for x in rows if x['dataset']==ds and x['A_mode']==mode})
                for core in cores:
                    group=[x for x in rows if x['dataset']==ds and x['A_mode']==mode and x['core']==core]
                    record=dict(dataset=ds,A_mode=mode,core=core,seeds=len(group))
                    for metric in ['row_shared_relative_fro_error_lambda','row_shared_relative_fro_error_A_log','best_rank1_energy_ratio_lambda']:
                        values=[x[metric]*100 for x in group]
                        record[metric+'_percent_mean']=statistics.mean(values)
                        record[metric+'_percent_sample_std']=statistics.stdev(values) if len(values)>1 else None
                    aggregate.append(record)
        save_csv(root/'pole_metrics_summary.csv',aggregate)
    else:
        for ds in a.datasets:
            for seed in a.seeds:
                for batch in a.batch_sizes:
                    files=list((root/ds/('seed%d'%seed)/('batch%d'%batch)).glob('gpu_batch*_benchmark_*.json'))
                    if len(files)!=1:raise ValueError('Expected one benchmark report: '+str(files))
                    report=json.loads(files[0].read_text())
                    for scope in ['model_only','full_gpu_pipeline','pipeline_with_h2d']:
                        values=report.get(scope)
                        if not values:continue
                        for clock in ['wall_clock','cuda_events']:
                            v=values.get(clock)
                            if v:rows.append(dict(dataset=ds,seed=seed,batch_size=batch,model_kind=a.model_kind,backend=report['selective_scan_backend'],scope=scope,clock=clock,median_ms_per_scene=v['median_ms_per_scene'],amortized_ms_per_tile=v['median_ms_per_tile'],tiles_per_second=v['median_tiles_per_second'],p95_ms_per_scene=v['p95_ms_per_scene'],checkpoint_sha256=report['checkpoint_sha256']))
        save_csv(root/'gpu_batch_summary.csv',rows)
    print('Finished:',root)


if __name__=='__main__':main()
