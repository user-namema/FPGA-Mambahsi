#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# E7: test the interaction of the two removed native Mamba paths.
run_current_ablation \
    --use_z true \
    --use_D true \
    "$@"
