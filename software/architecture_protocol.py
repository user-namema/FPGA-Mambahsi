"""Explicit FP32 training commands, without a Bash dependency.

The options match the included ablation_scripts protocol.  Keep the historical
19-case paper selection in experiments/run.py; the two additional single-factor
cases remain available through --cases.
"""

COMMON_ARGS = [
    '--exp_name', 'SPATIAL_SPLIT_3WAY_DENSE',
    '--run_tag', 'all_samples_sqrt_inverse_clip3_2000_nobias',
    '--tile_size', '16', '--fallback_tile_size', '0',
    '--split_strategy', 'blocks', '--split_axis', 'auto',
    '--split_block_size', '16', '--split_trials', '30000', '--split_seed', '2026',
    '--train_samples', '2000', '--min_train_samples', '100',
    '--val_samples', '10', '--min_val_samples', '50', '--min_test_samples', '100',
    '--class_weight', 'sqrt_inverse', '--pca_components', '16',
    '--batch_size', '32', '--eval_batch_size', '8',
    '--lr', '1e-3', '--weight_decay', '0.0', '--max_epoch', '400',
    '--eval_interval', '5', '--grad_clip_norm', '3.0',
    '--gradient_diagnostics', 'true', '--diagnostic_batch_interval', '1',
    '--scheduler', 'step', '--scheduler_step_size', '20',
    '--scheduler_gamma', '0.9', '--scheduler_patience', '3',
    '--scheduler_min_lr', '1e-6', '--early_stopping_patience', '12',
    '--early_stopping_min_delta', '1e-4', '--selection_metric', 'mAcc',
    '--num_workers', '0', '--record_computecost', 'false',
]

CURRENT_MODEL = {
    'model_variant': 'current', 'hidden_dim': '32', 'branch_mode': 'both',
    'fusion_mode': 'sum', 'skip_scale': '2', 'use_z': 'false', 'use_D': 'false',
    'A_mode': 'shared', 'norm_path': 'bn', 'activation': 'relu',
    'head_dim': '64', 'token_num': '4', 'd_state': '16',
}
MATCHED_MODEL = dict(CURRENT_MODEL, model_variant='original_safe_matched',
                     fusion_mode='softmax', use_z='true', use_D='true',
                     A_mode='per_channel', norm_path='gn', activation='silu')
FULL_MODEL = dict(MATCHED_MODEL, model_variant='original_safe', hidden_dim='64',
                  head_dim='128')

CASE_OVERRIDES = {
    '00_baseline_current': {},
    '01_reference_original_matched': MATCHED_MODEL,
    '02_reference_original_full': FULL_MODEL,
    '03_branch_spa_only': {'branch_mode': 'spa'},
    '05_fusion_mean': {'fusion_mode': 'mean'},
    '06_fusion_softmax': {'fusion_mode': 'softmax'},
    '07_skip_scale_1': {'skip_scale': '1'},
    '08_skip_scale_0': {'skip_scale': '0'},
    '09_restore_z': {'use_z': 'true'},
    '10_restore_D': {'use_D': 'true'},
    '11_A_per_channel': {'A_mode': 'per_channel'},
    '12_norm_gn_path': {'norm_path': 'gn'},
    '13_activation_silu': {'activation': 'silu'},
    '14_head_dim_32': {'head_dim': '32'},
    '15_head_dim_128': {'head_dim': '128'},
    '16_token_num_2': {'token_num': '2'},
    '17_token_num_8': {'token_num': '8'},
    '18_d_state_8': {'d_state': '8'},
    '19_d_state_32': {'d_state': '32'},
    '20_norm_gn_activation_silu': {'norm_path': 'gn', 'activation': 'silu'},
    '21_restore_z_D': {'use_z': 'true', 'use_D': 'true'},
}


def training_arguments(case, dataset, data_root, output_root, seeds, device):
    """Return a complete configurable argument vector for the FP32 trainer."""
    if case not in CASE_OVERRIDES:
        raise ValueError('Unknown included architecture case: ' + case)
    model = dict(CURRENT_MODEL, **CASE_OVERRIDES[case])
    args = list(COMMON_ARGS)
    for name, value in model.items():
        args.extend(['--' + name, value])
    args.extend(['--dataset', dataset, '--data_set_path', str(data_root),
                 '--work_dir', str(output_root), '--seeds', seeds, '--device', device])
    return args
