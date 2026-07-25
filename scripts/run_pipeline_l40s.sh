#!/usr/bin/env bash
# Deprecated alias — pipeline targets NVIDIA T4, not L40S.
echo "NOTE: run_pipeline_l40s.sh redirects to run_pipeline_t4.sh (T4 presets)."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_pipeline_t4.sh" "$@"
