#!/usr/bin/env bash
set -euo pipefail
DIR="${1:-$HOME/models}"
LOG_DIR="${2:-$HOME/hailo_bench_logs}"
mkdir -p "$LOG_DIR"
shopt -s nullglob
hefs=("$DIR"/*.hef)
((${#hefs[@]})) || { echo "No HEF files found in $DIR" >&2; exit 1; }
for hef in "${hefs[@]}"; do
  name=$(basename "$hef" .hef)
  echo "========== $name =========="
  hailortcli benchmark "$hef" 2>&1 | tee "$LOG_DIR/${name}_benchmark.txt"
done
echo "Logs saved in $LOG_DIR"
