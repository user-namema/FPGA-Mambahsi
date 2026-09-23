#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# E3b: increase the classification-head width from 64 to 128.
run_current_ablation \
    --head_dim 128 \
    "$@"
