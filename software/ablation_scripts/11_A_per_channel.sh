#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# S7: replace one shared A vector with per-inner-channel A parameters.
run_current_ablation \
    --A_mode per_channel \
    "$@"
