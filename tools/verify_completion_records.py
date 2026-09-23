#!/usr/bin/env python3
"""Check the final received records; --check-arrays additionally requires NumPy."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]

def require(ok, message):
    if not ok: raise ValueError(message)

def close(a, b):
    require(math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-8), 'Numeric mismatch: %s vs %s' % (a,b))

def verify(root=ROOT, check_arrays=False):
    e = root / 'evidence/completion_20260923'
    provenance = json.loads((e / 'provenance.json').read_text())
    for row in provenance['files']:
        p = (root / row['destination']).resolve()
        require(root.resolve() in p.parents, 'Path outside repository')
        require(hashlib.sha256(p.read_bytes()).hexdigest() == row['published_sha256'], 'Hash mismatch: '+row['destination'])
    reports = {}
    for p in (e / 'gpu_fp32').rglob('gpu_batch*_benchmark_*.json'):
        o = json.loads(p.read_text()); key = (o['dataset'],o['seed'],o['batch_size'])
        require(key not in reports and o['model_kind']=='fp32', 'Duplicate or non-FP32 report')
        reports[key] = o
        for scope in ('model_only','full_gpu_pipeline'):
            for clock in ('wall_clock','cuda_events'):
                t=o[scope][clock]; v=t['samples_ms']
                require(len(v)==t['repeats']==20, 'Incomplete timing samples')
                close(statistics.mean(v),t['mean_ms_per_scene'])
                close(statistics.median(v),t['median_ms_per_scene'])
                close(t['median_ms_per_scene']/o['tile_count'],t['median_ms_per_tile'])
                close(1000*o['tile_count']/t['median_ms_per_scene'],t['median_tiles_per_second'])
    require(set(reports)=={(d,0,b) for d in ('UP','HanChuan','HongHu','Houston') for b in (1,2,4,8,16,32,64)}, 'Incomplete FP32 batch grid')
    for p in (e/'gpu_fp32/gpu_batch_summary.csv',root/'evidence/gpu_fp32_batch_summary.csv'):
        with p.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        require(len(rows)==112,'FP32 table rows')
        for row in rows:
            o=reports[row['dataset'],int(row['seed']),int(row['batch_size'])];t=o[row['scope']][row['clock']]
            require(row['checkpoint_sha256']==o['checkpoint_sha256'],'FP32 checkpoint mismatch')
            for a,b in [('median_ms_per_scene','median_ms_per_scene'),('amortized_ms_per_tile','median_ms_per_tile'),('tiles_per_second','median_tiles_per_second'),('p95_ms_per_scene','p95_ms_per_scene')]:close(row[a],t[b])
    folder=e/'nonlinear_D1';s=json.loads((folder/'accuracy_propagation_summary.json').read_text())
    require(s['status']=='COMPLETE' and s['tiles']==858 and s['trace_tiles']==5,'D1 coverage')
    snapshot=root/'software/snapshots/nonlinear_D1_20260916'
    for name,expected in s['source_sha256'].items():
        require(hashlib.sha256((snapshot/name).read_bytes()).hexdigest()==expected,'D1 source mismatch: '+name)
    require(json.loads((snapshot/'n2_config.json').read_text())['fit']==s['n2_fit'],'N2 fit mismatch')
    for p in (folder/'coefficients').glob('*.mem'):
        ref=snapshot/'hardware_reference'/p.name
        require([int(x,16) for x in p.read_text().split()]==[int(x,16) for x in ref.read_text().split()], 'Coefficient mismatch: '+p.name)
    dt=json.loads((e/'dt_input_UP_seed0/dt_input_diagnostics.json').read_text())
    require(len(dt['rows'])==18 and dt['split']=='test','DT rows/scope')
    require({(r['core'],r['variant']) for r in dt['rows']}=={('blk%d_%s'%(b,k),v) for b in range(3) for k in ('spa','spe') for v in ('current','int8_same_scale','int8_wider_scale')},'DT coverage')
    for r in dt['rows']:close(100*r['clipped_count']/r['count'],r['clipped_pct'])
    if check_arrays:
        import numpy as np
        base=np.load(folder/'n3_tile_logits_int8.npy',allow_pickle=False)
        labels=np.load(folder/'n3_prediction_rtl_integer.npy',allow_pickle=False)
        anchor=json.loads((folder/'D1_board_anchor_check.json').read_text())
        require(hashlib.sha256(np.ascontiguousarray(labels,dtype=np.uint8).tobytes()).hexdigest()==anchor['labels_actual']==anchor['labels_expected'],'N3 label hash')
        for method in ('n0','n1','n2','n3'):
            values=np.load(folder/(method+'_tile_logits_int8.npy'),allow_pickle=False)
            pred=np.load(folder/(method+'_prediction_rtl_integer.npy'),allow_pickle=False)
            delta=values.astype(np.float64)-base.astype(np.float64);entry=s['methods'][method]
            require(int(np.count_nonzero(delta))==entry['logits_vs_n3']['mismatches'],'Logit mismatch count')
            close(float(np.abs(delta).mean()),entry['logits_vs_n3']['MAE'])
            require(int(np.count_nonzero(pred!=labels))==entry['agreement']['all']['mismatches'],'Prediction mismatch count')
    return dict(published_files=len(provenance['files']),fp32_reports=28,fp32_summary_rows=112,D1_tiles=858,D1_source_files=10,dt_rows=18,arrays_checked=check_arrays)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-arrays',action='store_true')
    print(json.dumps(verify(check_arrays=p.parse_args().check_arrays),indent=2))
