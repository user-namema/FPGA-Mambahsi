#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# S1: remove Spe; retain the current block-level 2*x shortcut.
run_current_ablation \
    --branch_mode spa \
    "$@"
