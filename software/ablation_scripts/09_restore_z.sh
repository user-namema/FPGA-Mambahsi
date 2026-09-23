#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# S5: restore only the native SiLU z gate.
run_current_ablation \
    --use_z true \
    "$@"
