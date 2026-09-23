#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# E5b: increase the SSM state dimension from 16 to 32.
run_current_ablation \
    --d_state 32 \
    "$@"
