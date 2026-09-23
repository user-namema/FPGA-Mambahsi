#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# E5a: reduce the SSM state dimension from 16 to 8.
run_current_ablation \
    --d_state 8 \
    "$@"
