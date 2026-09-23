#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# E3a: reduce the classification-head width from 64 to 32.
run_current_ablation \
    --head_dim 32 \
    "$@"
