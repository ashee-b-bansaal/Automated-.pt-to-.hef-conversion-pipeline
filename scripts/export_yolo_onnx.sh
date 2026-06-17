#!/usr/bin/env bash
set -euo pipefail
usage() {
  cat <<'EOF'
Export YOLO .pt weights to static ONNX.

Current Ultralytics models:
  export_yolo_onnx.sh --weights best.pt --imgsz 640

Legacy YOLOv5 repo:
  export_yolo_onnx.sh --weights yolov5s.pt --imgsz 640 --yolov5-dir ~/yolov5
EOF
}
WEIGHTS=""; IMGSZ=640; OPSET=11; YOLOV5_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights) WEIGHTS="$2"; shift 2;;
    --imgsz) IMGSZ="$2"; shift 2;;
    --opset) OPSET="$2"; shift 2;;
    --yolov5-dir) YOLOV5_DIR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -f "$WEIGHTS" ]] || { echo "Weights not found: $WEIGHTS" >&2; exit 1; }
if [[ -n "$YOLOV5_DIR" ]]; then
  [[ -f "$YOLOV5_DIR/export.py" ]] || { echo "export.py not found in $YOLOV5_DIR" >&2; exit 1; }
  python "$YOLOV5_DIR/export.py" --weights "$WEIGHTS" --img "$IMGSZ" --batch 1 --include onnx --opset "$OPSET"
else
  command -v yolo >/dev/null || { echo "Ultralytics 'yolo' command not found. Install ultralytics in the export environment." >&2; exit 1; }
  yolo export model="$WEIGHTS" format=onnx imgsz="$IMGSZ" opset="$OPSET" dynamic=False simplify=False batch=1
fi
