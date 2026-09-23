#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# E6: complete the BN/GN x ReLU/SiLU 2x2 path comparison.
run_current_ablation \
    --norm_path gn \
    --activation silu \
    "$@"
