#!/usr/bin/env python3
"""Verify bundled server evidence and recompute aggregate results (standard library only)."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def close(a, b, message):
    require(math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-8), message)

def verify(root=ROOT):
    e = root / 'evidence/server_update_20260923'
    provenance = json.loads((e / 'provenance.json').read_text())
    for item in provenance['files']:
        p = (root / item['destination']).resolve()
        require(root.resolve() in p.parents, 'Path outside repository')
        require(hashlib.sha256(p.read_bytes()).hexdigest() == item['published_sha256'], 'Hash mismatch: ' + item['destination'])
    f = e / 'fpga_eval1'
    jobs = json.loads((f / 'manifest.json').read_text())
    expected = {(d, n) for d in ('UP', 'HanChuan', 'HongHu', 'Houston') for n in range(10)}
    require(len(jobs) == 40 and {(j['dataset'], j['seed']) for j in jobs} == expected, 'Incomplete or duplicate FPGA runs')
    reports = {}
    for job in jobs:
        d, n = job['dataset'], job['seed']
        r = json.loads((f / d / ('seed' + str(n)) / 'fpga_simulation_result.json').read_text())
        require(job['status'] == 'complete', 'Incomplete job')
        require((r['dataset'], r['seed']) == (d, n), 'Run identity mismatch')
        require(r['qat_checkpoint_sha256'] == job['checkpoint_sha256'] == job['identity']['checkpoint_sha256'], 'Checkpoint mismatch')
        require(r['numeric_config'] == job['numeric_config'], 'Numeric configuration mismatch')
        require(r['qat_reference_batch_size'] == 1 and r['saved_qat_prediction_match'] == 1 and r['qat_batch1_vs_saved_batch_match'] == 1, 'Saved QAT parity mismatch')
        hashes = json.loads((root / 'software/snapshots/eval1_server_20260923/source_hashes.json').read_text())
        for name, digest in job['identity']['source_hashes'].items():
            require(hashes[name] == digest, 'Run source mismatch: ' + name)
        reports[d, n] = r
    summary = json.loads((f / 'dataset_summary.json').read_text())
    for d, g in summary.items():
        m = [reports[d, n]['metrics'] for n in range(10)]
        values = {'qat_direct_OA': [100*x['qat_direct']['OA'] for x in m],
                  'int8_hw_OA': [100*x['int8_hw']['OA'] for x in m],
                  'int8_hw_mAcc': [100*x['int8_hw']['mAcc'] for x in m],
                  'int8_hw_Kappa': [x['int8_hw']['Kappa'] for x in m],
                  'QAT_to_INT8_loss_pp': [100*(x['qat_direct']['OA']-x['int8_hw']['OA']) for x in m]}
        for k, v in values.items():
            close(statistics.mean(v), g[k]['mean'], d + ' ' + k + ' mean')
            close(statistics.stdev(v), g[k]['std'], d + ' ' + k + ' sample SD')
    power = e / 'gpu_power_UP'
    trials = [json.loads(p.read_text()) for p in power.glob('batch*_repeat*.json')]
    keys = {(t['batch_size'], t['scope'], t['repeat']) for t in trials}
    require(len(trials) == 42 and keys == {(b, s, n) for b in (1,2,4,8,16,32,64) for s in ('model_only','full_gpu_pipeline') for n in range(3)}, 'Incomplete power trials')
    for t in trials:
        require(t['energy_method'] == 'nvml_energy_counter', 'Unexpected energy method')
        close(t['energy_j'], t['energy_counter_end_j']-t['energy_counter_start_j'], 'Counter energy')
        close(t['mean_power_w'], t['energy_j']/t['duration_s'], 'Mean power')
        close(t['tiles_per_s'], t['tiles']/t['duration_s'], 'Throughput')
        close(t['microjoules_per_tile'], 1e6*t['energy_j']/t['tiles'], 'Energy/tile')
        require(t['process_condition'] == 'display_background_only' and t['no_other_compute_at_boundaries'] and not t['exclusive_at_boundaries'], 'Incorrect process classification')
        for key in ('processes_before','processes_after'):
            st = t[key]
            require(not st['compute_pids'] and not st['blocking_pids'] and not st['unavailable'], 'Other compute task or unavailable query')
            require(set(st['other_pids']) == set(st['allowed_display_pids']), 'Unclassified process')
    with (power / 'power_summary.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 14, 'Power summary count')
    for row in rows:
        group = [t for t in trials if t['batch_size'] == int(row['batch_size']) and t['scope'] == row['scope']]
        require(len(group) == int(row['repeats']) == 3, 'Power repeats')
        for field in ('mean_power_w','tiles_per_s','microjoules_per_tile'):
            close(row[field], statistics.mean(t[field] for t in group), 'Power summary: ' + field)
    return {'verified_published_files': len(provenance['files']), 'fpga_eval1_runs': 40, 'power_trials': 42, 'power_settings': 14}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    print(json.dumps(verify(p.parse_args().root), indent=2))

if __name__ == '__main__':
    main()
