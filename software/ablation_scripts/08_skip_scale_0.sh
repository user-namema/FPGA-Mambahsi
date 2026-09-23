#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# S4b: remove the block-level shortcut.
run_current_ablation \
    --skip_scale 0 \
    "$@"
