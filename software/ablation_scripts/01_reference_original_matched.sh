#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Sref-matched: original architecture choices at hidden32/head64.
run_original_matched_reference "$@"
