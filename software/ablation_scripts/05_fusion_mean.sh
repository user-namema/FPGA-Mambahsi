#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# S3a: replace direct branch sum with an unlearned arithmetic mean.
run_current_ablation \
    --fusion_mode mean \
    "$@"
