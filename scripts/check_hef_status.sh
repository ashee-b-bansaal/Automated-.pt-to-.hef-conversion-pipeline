#!/usr/bin/env bash
set -euo pipefail
DIR="${1:-$HOME/models}"
shift || true
models=("$@")
if ((${#models[@]} == 0)); then
  find "$DIR" -maxdepth 1 -type f -name '*.hef' -printf '%TY-%Tm-%Td %TH:%TM  %s bytes  %f\n' | sort
  exit 0
fi
for model in "${models[@]}"; do
  file="$DIR/${model}.hef"
  if [[ -s "$file" ]]; then
    printf 'OK      %-30s %s\n' "$model" "$(du -h "$file" | cut -f1)"
  else
    printf 'MISSING %s\n' "$model"
  fi
done
