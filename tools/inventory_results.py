#!/usr/bin/env python3
"""Index the author's existing result directories without loading checkpoints.

Usage: python tools/inventory_results.py --workspace /path/to/archive --output-dir /tmp/index
Counts are evidence files, not an assertion of independent samples or successful reruns.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

DATASETS = ('UP', 'HanChuan', 'HongHu', 'Houston')
SPECS = [
    ('fpga_eval1', 'FPGA模拟和GPU测速和缺失的记录', 'fpga_simulation_result.json'),
    ('gpu_power_UP', 'FPGA模拟和GPU测速和缺失的记录/gpu_power_UP_gpu0_display_v2', 'batch*_repeat*.json'),
    ('shared_a_training', 'a_shared_current_4datasets', 'result.json'),
    ('shared_a_dynamics', 'a_shared_current_4datasets_dynamics', 'analysis_summary.json'),
    ('qat_eval8_UP', 'freeze20', 'result.json'),
    ('qat_eval8_other3', 'patch_max_D_mean_freeze20_other3_10seeds', 'result.json'),
    ('freeze1_UP', 'freeze1', 'result.json'),
    ('fpga_eval8', 'fpga_patch_max_D_mean_freeze20_4datasets', 'fpga_simulation_result.json'),
    ('qat_eval1', 'qat_eval1_4datasets', 'result.json'),
    ('d1_fullscene_sources', 'UP_seed0_8ba0296d', 'fpga_simulation_result.json'),
    ('gpu_fixed', 'gpu_fixed_4datasets_v2', 'gpu_batch*_benchmark_*.json'),
    ('weight_init_validation', 'qat_weight_init_validation_seed0_6', 'weight_init_validation.json'),
    ('max_init_qat', 'max_init_qat_fpga_seed0_6', 'result.json'),
    ('max_init_fpga', 'max_init_qat_fpga_seed0_6', 'fpga_simulation_result.json'),
    ('stability_bn', 'qat_stability_bn_seed0_6', 'result.json'),
    ('stability_scales', 'qat_stability_scales_seed0_6', 'result.json'),
    ('stability_single_factor', 'qat_single_factor_seed0_6', 'result.json'),
]

def inventory(workspace):
    records = []
    def add(group, path):
        raw = path.read_bytes()
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            obj = {}
        records.append(dict(experiment=group, source=str(path.relative_to(workspace)),
            sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
            dataset=obj.get('dataset', ''), seed=obj.get('seed', ''),
            eval_batch_size=obj.get('eval_batch_size', ''),
            model_config_slug=obj.get('model_config_slug', ''),
            test_OA=obj.get('test_OA', ''), qat_test_OA=obj.get('qat_test_OA', '')))
    for ds in DATASETS:
        root = workspace / (ds + '_all_samples_sqrt_inverse_clip3_2000_nobias')
        for path in sorted(root.rglob('result.json')):
            config = path.parent.parent.name
            if not config.startswith(('current_', 'original_safe')):
                continue
            if '_br-both_' not in config or (config.startswith('current_') and '_A-per_channel_' in config):
                continue
            add('architecture_19', path)
    for group, dirname, pattern in SPECS:
        for path in sorted((workspace / dirname).rglob(pattern)):
            add(group, path)
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows = inventory(args.workspace.resolve())
    if not rows:
        parser.error('No matching result files; check --workspace')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'reference_run_index.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = sorted(set(row['experiment'] for row in rows))
    counts = {group: sum(row['experiment'] == group for row in rows) for group in groups}
    (args.output_dir / 'reference_run_counts.json').write_text(json.dumps(counts, indent=2) + '\n')
    print(json.dumps(counts, indent=2))

if __name__ == '__main__':
    main()
