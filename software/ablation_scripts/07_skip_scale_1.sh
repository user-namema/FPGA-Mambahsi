#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# S4a: conventional block shortcut fusion + x.
run_current_ablation \
    --skip_scale 1 \
    "$@"
