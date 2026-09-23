#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# E2: replace explicit ReLU activations with SiLU; keep the BN path.
run_current_ablation \
    --activation silu \
    "$@"
