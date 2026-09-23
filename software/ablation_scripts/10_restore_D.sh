#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# S6: restore only the trainable SSM D*x term.
run_current_ablation \
    --use_D true \
    "$@"
