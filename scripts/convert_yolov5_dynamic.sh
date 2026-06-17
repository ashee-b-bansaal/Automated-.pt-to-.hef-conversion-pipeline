#!/usr/bin/env bash
# Dynamic YOLOv5 -> Hailo HEF converter.
# Preserved from the working project pipeline; see docs/TROUBLESHOOTING.md.
# Key idea:
#   - Do NOT hardcode conv IDs.
#   - For each model, let Hailo parse ONNX and generate HAR.
#   - Read the HAR to discover actual YOLO detection-head conv layers.
#   - Patch that model's NMS JSON dynamically.
#   - Re-run compile.

set -u
set -o pipefail

MODELS_STRING="${MODELS:-yolov5s yolov5m}"
read -r -a MODEL_LIST <<< "$MODELS_STRING"

IMG="${IMG:-640}"
CLASSES="${CLASSES:-80}"
HEAD_CHANNELS="${HEAD_CHANNELS:-255}"   # COCO YOLOv5: 3 anchors * (80 classes + 5)
HW_ARCH="${HW_ARCH:-hailo8}"

MODEL_ZOO_ROOT="${MODEL_ZOO_ROOT:-$HOME/Video_streaming/hailo_model_zoo}"
YOLOV5_DIR="${YOLOV5_DIR:-$HOME/hailo_clean/yolov5}"
CALIB="${CALIB:-$HOME/coco_calib_images}"
OUT_DIR="${OUT_DIR:-$HOME/yolov5_hefs}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "=== Config ==="
echo "MODEL_ZOO_ROOT=$MODEL_ZOO_ROOT"
echo "YOLOV5_DIR=$YOLOV5_DIR"
echo "CALIB=$CALIB"
echo "OUT_DIR=$OUT_DIR"
echo "MODELS=${MODEL_LIST[*]}"
echo "HEAD_CHANNELS=$HEAD_CHANNELS"
echo

command -v hailomz >/dev/null 2>&1 || { echo "ERROR: hailomz not found. Activate hailo_env first."; exit 1; }
[ -d "$MODEL_ZOO_ROOT" ] || { echo "ERROR: no Model Zoo folder: $MODEL_ZOO_ROOT"; exit 1; }
[ -d "$CALIB" ] || { echo "ERROR: no calibration folder: $CALIB"; exit 1; }

# Hailo Model Zoo v2.18 workaround: torch_infer.py may auto-import and fail.
if [ -f "$MODEL_ZOO_ROOT/hailo_model_zoo/core/infer/torch_infer.py" ]; then
    echo "Disabling torch_infer.py auto-import workaround..."
    mv "$MODEL_ZOO_ROOT/hailo_model_zoo/core/infer/torch_infer.py" \
       "$MODEL_ZOO_ROOT/hailo_model_zoo/core/infer/torch_infer.py.disabled"
fi

if [ ! -d "$YOLOV5_DIR" ]; then
    mkdir -p "$(dirname "$YOLOV5_DIR")"
    git clone https://github.com/ultralytics/yolov5.git "$YOLOV5_DIR"
fi

detect_heads_from_har() {
    local har="$1"
    local model="$2"
    local channels="$3"

    python - "$har" "$model" "$channels" 2>/dev/null <<'PY'
import re
import sys
from hailo_sdk_client import ClientRunner

har, model, channels = sys.argv[1], sys.argv[2], int(sys.argv[3])

runner = ClientRunner(har=har)
hn = runner.get_hn()
layers = hn.get("layers", {})

def contains_channels(obj, target):
    if isinstance(obj, int):
        return obj == target
    if isinstance(obj, float):
        return int(obj) == target
    if isinstance(obj, str):
        return obj == str(target)
    if isinstance(obj, dict):
        return any(contains_channels(v, target) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(contains_channels(v, target) for v in obj)
    return False

candidates = []
pattern = re.compile(rf"^{re.escape(model)}/conv[0-9]+$")
for name, layer in layers.items():
    name = str(name)
    if pattern.match(name) and contains_channels(layer, channels):
        candidates.append(name)

# Detection heads are the final three convs with YOLO head channel count.
heads = candidates[-3:]

if len(heads) != 3:
    print(f"ERROR: found {len(heads)} heads for {model}, candidates={candidates}", file=sys.stderr)
    sys.exit(2)

for h in heads:
    print(h)
PY
}

patch_nms_with_heads() {
    local nms_json="$1"
    shift
    python - "$nms_json" "$@" <<'PY'
import json
import re
import sys
from pathlib import Path

p = Path(sys.argv[1])
heads = [x.strip() for x in sys.argv[2:] if re.match(r"^yolov5[a-z0-9_/-]*/conv[0-9]+$", x.strip())]

if len(heads) != 3:
    raise SystemExit(f"ERROR: Need exactly 3 valid heads, got {heads}")

data = json.loads(p.read_text())
entries = []

def walk(o):
    if isinstance(o, dict):
        if "encoded_layer" in o:
            entries.append(o)
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for x in o:
            walk(x)

walk(data)

if len(entries) != 3:
    print(f"WARNING: expected 3 encoded_layer entries but found {len(entries)}. Patching first 3.")

if len(entries) < 3:
    raise SystemExit(f"ERROR: Not enough encoded_layer entries in {p}")

print("Patching NMS encoded_layer fields dynamically:")
for i, head in enumerate(heads):
    old = entries[i].get("encoded_layer")
    entries[i]["encoded_layer"] = head
    print(f"  {i}: {old} -> {head}")

p.write_text(json.dumps(data, indent=4))
PY
}

compile_one() {
    local model="$1"
    echo
    echo "=============================="
    echo "Converting $model"
    echo "=============================="

    local onnx="$YOLOV5_DIR/${model}.onnx"
    local yaml alls nms har hef log1 log2
    yaml="$(find "$MODEL_ZOO_ROOT/hailo_model_zoo/cfg/networks" -name "${model}.yaml" | head -n 1)"
    alls="$(find "$MODEL_ZOO_ROOT/hailo_model_zoo/cfg/alls" -name "${model}.alls" | head -n 1)"
    nms="$(find "$MODEL_ZOO_ROOT/hailo_model_zoo/cfg/postprocess_config" -name "${model}_nms_config.json" | head -n 1)"
    har="$MODEL_ZOO_ROOT/${model}.har"
    hef="$MODEL_ZOO_ROOT/${model}.hef"
    log1="$LOG_DIR/${model}_compile_first.log"
    log2="$LOG_DIR/${model}_compile_second.log"

    if [ -z "$yaml" ] || [ -z "$alls" ] || [ -z "$nms" ]; then
        echo "SKIP $model: missing Model Zoo config(s)"
        echo "  yaml=$yaml"
        echo "  alls=$alls"
        echo "  nms=$nms"
        return 2
    fi

    echo "YAML=$yaml"
    echo "ALLS=$alls"
    echo "NMS=$nms"

    # Reset only this model's scripts/config before each attempt.
    (cd "$MODEL_ZOO_ROOT" && git checkout -- "$nms" "$alls" 2>/dev/null || true)

    if [ ! -f "$onnx" ]; then
        echo "Exporting ${model}.pt -> ONNX..."
        (cd "$YOLOV5_DIR" && python export.py --weights "${model}.pt" --img "$IMG" --batch 1 --include onnx --opset 11 --simplify)
    else
        echo "ONNX exists: $onnx"
    fi

    echo "First compile attempt..."
    set +e
    (cd "$MODEL_ZOO_ROOT" && hailomz compile --ckpt "$onnx" --calib-path "$CALIB" --yaml "$yaml" --hw-arch "$HW_ARCH" --classes "$CLASSES") 2>&1 | tee "$log1"
    local status=${PIPESTATUS[0]}
    set -e

    if [ "$status" -ne 0 ]; then
        echo "First compile failed. Discovering this model's true heads from HAR..."
        if [ ! -f "$har" ]; then
            echo "ERROR: HAR was not created, cannot dynamically patch NMS."
            return 1
        fi

        mapfile -t heads < <(detect_heads_from_har "$har" "$model" "$HEAD_CHANNELS" | grep -E "^${model}/conv[0-9]+$")

        echo "Discovered heads for $model: ${heads[*]}"
        if [ "${#heads[@]}" -ne 3 ]; then
            echo "ERROR: dynamic detection failed for $model."
            return 1
        fi

        patch_nms_with_heads "$nms" "${heads[0]}" "${heads[1]}" "${heads[2]}"
        echo "NMS after dynamic patch:"
        grep -n "encoded_layer" "$nms" || true

        echo "Second compile attempt..."
        set +e
        (cd "$MODEL_ZOO_ROOT" && hailomz compile --ckpt "$onnx" --calib-path "$CALIB" --yaml "$yaml" --hw-arch "$HW_ARCH" --classes "$CLASSES") 2>&1 | tee "$log2"
        status=${PIPESTATUS[0]}
        set -e

        if [ "$status" -ne 0 ]; then
            echo "ERROR: $model failed after dynamic NMS patch. See $log2"
            return 1
        fi
    fi

    if [ ! -f "$hef" ]; then
        echo "ERROR: compile succeeded but expected HEF missing: $hef"
        return 1
    fi

    cp "$hef" "$OUT_DIR/"
    echo "SUCCESS: $model -> $OUT_DIR/${model}.hef"
    ls -lh "$OUT_DIR/${model}.hef"
}

SUCCESS=()
SKIPPED=()
FAILED=()

for model in "${MODEL_LIST[@]}"; do
    compile_one "$model"
    code=$?
    if [ "$code" -eq 0 ]; then
        SUCCESS+=("$model")
    elif [ "$code" -eq 2 ]; then
        SKIPPED+=("$model")
    else
        FAILED+=("$model")
    fi
done

echo
echo "=============================="
echo "Summary"
echo "=============================="
echo "Successful: ${SUCCESS[*]:-none}"
echo "Skipped:    ${SKIPPED[*]:-none}"
echo "Failed:     ${FAILED[*]:-none}"
echo
echo "HEFs:"
ls -lh "$OUT_DIR"/*.hef 2>/dev/null || true
