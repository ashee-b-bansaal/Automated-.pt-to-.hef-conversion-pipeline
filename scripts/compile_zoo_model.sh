#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Compile a known Hailo Model Zoo recipe, optionally substituting a custom ONNX.

Usage:
  compile_zoo_model.sh --model MODEL --model-zoo-root DIR --calib-dir DIR [options]

Options:
  --onnx FILE        Custom ONNX checkpoint. Omit to use Model Zoo pretrained weights.
  --classes N        Custom class count when supported by the recipe.
  --hw-arch ARCH     hailo8 (default), hailo8l, etc.
  --output-dir DIR   Default: ./outputs
  --log-dir DIR      Default: OUTPUT_DIR/logs

Important: the custom ONNX must match the architecture expected by the recipe.
A ResNet-50 recipe is not a universal recipe for every classifier.
EOF
}

MODEL=""; MODEL_ZOO_ROOT=""; CALIB_DIR=""; ONNX=""; CLASSES=""; HW_ARCH="hailo8"
OUTPUT_DIR="$PWD/outputs"; LOG_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --model-zoo-root) MODEL_ZOO_ROOT="$2"; shift 2;;
    --calib-dir) CALIB_DIR="$2"; shift 2;;
    --onnx) ONNX="$2"; shift 2;;
    --classes) CLASSES="$2"; shift 2;;
    --hw-arch) HW_ARCH="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

[[ -n "$MODEL" && -n "$MODEL_ZOO_ROOT" && -n "$CALIB_DIR" ]] || { usage; exit 2; }
command -v hailomz >/dev/null || { echo "ERROR: hailomz is not available. Activate the Hailo DFC environment." >&2; exit 1; }
[[ -d "$MODEL_ZOO_ROOT" ]] || { echo "ERROR: Model Zoo not found: $MODEL_ZOO_ROOT" >&2; exit 1; }
[[ -d "$CALIB_DIR" ]] || { echo "ERROR: calibration directory not found: $CALIB_DIR" >&2; exit 1; }
[[ -z "$ONNX" || -f "$ONNX" ]] || { echo "ERROR: ONNX not found: $ONNX" >&2; exit 1; }

YAML=$(find "$MODEL_ZOO_ROOT/hailo_model_zoo/cfg/networks" -maxdepth 1 -name "${MODEL}.yaml" | head -n 1)
[[ -n "$YAML" ]] || { echo "ERROR: no installed Model Zoo recipe named $MODEL" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
LOG_DIR=${LOG_DIR:-$OUTPUT_DIR/logs}
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${MODEL}.log"

cmd=(hailomz compile --yaml "$YAML" --calib-path "$CALIB_DIR" --hw-arch "$HW_ARCH")
[[ -z "$ONNX" ]] || cmd+=(--ckpt "$ONNX")
[[ -z "$CLASSES" ]] || cmd+=(--classes "$CLASSES")

printf 'Running:'; printf ' %q' "${cmd[@]}"; echo
set +e
(cd "$MODEL_ZOO_ROOT" && "${cmd[@]}") 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
[[ $status -eq 0 ]] || { echo "FAILED. See $LOG" >&2; exit "$status"; }

HEF=$(find "$MODEL_ZOO_ROOT" -maxdepth 1 -type f -name "${MODEL}.hef" -print -quit)
if [[ -z "$HEF" ]]; then
  HEF=$(find "$MODEL_ZOO_ROOT" -maxdepth 1 -type f -name '*.hef' -printf '%T@ %p\n' \
    | sort -n | tail -n 1 | cut -d' ' -f2-)
fi
[[ -n "$HEF" && -f "$HEF" ]] || { echo "Compile returned success, but no HEF was found." >&2; exit 1; }
cp -f "$HEF" "$OUTPUT_DIR/"
echo "SUCCESS: $OUTPUT_DIR/$(basename "$HEF")"
ls -lh "$OUTPUT_DIR/$(basename "$HEF")"
