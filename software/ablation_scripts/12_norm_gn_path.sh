#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# E1: replace the complete BN path with the complete GN path; keep ReLU.
run_current_ablation \
    --norm_path gn \
    "$@"
