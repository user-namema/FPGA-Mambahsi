#!/usr/bin/env bash

# Shared, frozen protocol for every MambaHSI architecture ablation.
# Source this file from an experiment script; do not execute it directly.

set -euo pipefail

ABLATION_SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"
PROJECT_ROOT="$(
    cd "${ABLATION_SCRIPT_DIR}/.."
    pwd
)"

cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_mambahsi_spatial_split_128_dense.py}"

# These arguments explicitly freeze the user's validated UP protocol.  Values
# that were implicit defaults in the original command are written out here so
# a future parser-default change cannot silently alter the ablation protocol.
COMMON_TRAIN_ARGS=(
    --dataset UP
    --data_set_path ./data
    --work_dir ./results
    --exp_name SPATIAL_SPLIT_3WAY_DENSE
    --run_tag all_samples_sqrt_inverse_clip3_2000_nobias

    --tile_size 16
    --fallback_tile_size 0
    --split_strategy blocks
    --split_axis auto
    --split_block_size 16
    --split_trials 30000
    --split_seed 2026

    --train_samples 2000
    --min_train_samples 100
    --val_samples 10
    --min_val_samples 50
    --min_test_samples 100

    --class_weight sqrt_inverse
    --pca_components 16
    --batch_size 32
    --eval_batch_size 8
    --lr 1e-3
    --weight_decay 0.0
    --max_epoch 400
    --eval_interval 5
    --grad_clip_norm 3.0
    --gradient_diagnostics true
    --diagnostic_batch_interval 1

    --scheduler step
    --scheduler_step_size 20
    --scheduler_gamma 0.9
    --scheduler_patience 3
    --scheduler_min_lr 1e-6
    --early_stopping_patience 12
    --early_stopping_min_delta 1e-4
    --selection_metric mAcc

    --num_workers 0
    --seeds 0,1,2,3,4,5,6,7,8,9
    --device cuda:0
    --record_computecost false
)

# Exact current deployment baseline.  Experiment-specific arguments are placed
# after this array, so a repeated option in an experiment script overrides only
# the named factor while every other model field remains frozen.
CURRENT_MODEL_ARGS=(
    --model_variant current
    --hidden_dim 32
    --branch_mode both
    --fusion_mode sum
    --skip_scale 2
    --use_z false
    --use_D false
    --A_mode shared
    --norm_path bn
    --activation relu
    --head_dim 64
    --token_num 4
    --d_state 16
)

# Composite reference matching current capacity.  This changes several axes at
# once and must not be interpreted as a single-factor causal ablation.
ORIGINAL_MATCHED_MODEL_ARGS=(
    --model_variant original_safe_matched
    --hidden_dim 32
    --branch_mode both
    --fusion_mode softmax
    --skip_scale 2
    --use_z true
    --use_D true
    --A_mode per_channel
    --norm_path gn
    --activation silu
    --head_dim 64
    --token_num 4
    --d_state 16
)

# Batch-safe reproduction of the original model capacity.
ORIGINAL_FULL_MODEL_ARGS=(
    --model_variant original_safe
    --hidden_dim 64
    --branch_mode both
    --fusion_mode softmax
    --skip_scale 2
    --use_z true
    --use_D true
    --A_mode per_channel
    --norm_path gn
    --activation silu
    --head_dim 128
    --token_num 4
    --d_state 16
)

print_command() {
    local argument
    printf 'Command:'
    for argument in "$@"; do
        printf ' %q' "${argument}"
    done
    printf '\n'
}

run_ablation() {
    local -a command=(
        "${PYTHON_BIN}"
        "${TRAIN_SCRIPT}"
        "${COMMON_TRAIN_ARGS[@]}"
        "$@"
    )

    print_command "${command[@]}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        return 0
    fi
    "${command[@]}"
}

run_current_ablation() {
    run_ablation "${CURRENT_MODEL_ARGS[@]}" "$@"
}

run_original_matched_reference() {
    run_ablation "${ORIGINAL_MATCHED_MODEL_ARGS[@]}" "$@"
}

run_original_full_reference() {
    run_ablation "${ORIGINAL_FULL_MODEL_ARGS[@]}" "$@"
}

