# Hailo-8 M.2: Software Suite (Self-Extract) Guide

This guide is a second, standalone flow for converting custom `.pt` models to
`.hef` on Hailo-8/Hailo-8L using the Hailo Software Suite **self-extract**
installation path (not Docker).

It uses project-local paths only.

## 0) Set project paths

```bash
export PROJ="$HOME/Documents/Automated-.pt-to-.hef-conversion-pipeline"
export SUITE_ROOT="$PROJ/hailo_sw_suite"
export MZ_ROOT="$PROJ/hailo_model_zoo"
export EXPORT_ENV="$PROJ/.envs/model_export_env"
export HAILO_ENV="$PROJ/.envs/hailo_env"
cd "$PROJ"
```

## 1) Install Software Suite (self-extract)

1. Download the Hailo Software Suite package for Hailo-8/8L from Developer Zone.
2. Place the self-extract installer file in `$PROJ`.
3. Extract into project directory:

```bash
chmod +x "$PROJ/<HAILO_SUITE_INSTALLER>.run"
"$PROJ/<HAILO_SUITE_INSTALLER>.run" --target "$SUITE_ROOT"
```

After extraction, inspect installer docs/scripts:

```bash
ls -la "$SUITE_ROOT"
```

Run the suite's install/setup script from that folder (script names vary by release).

## 2) Create export environment (Python 3.10)

```bash
python3.10 -m venv "$EXPORT_ENV"
source "$EXPORT_ENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$PROJ/requirements-export.txt"
```

If you export YOLO models, install Ultralytics in this export env:

```bash
python -m pip install ultralytics
```

## 3) Create/activate Hailo compile environment

Create your Hailo compile env according to the Software Suite release notes and
install instructions. If the suite ships wheel packages, install them here:

```bash
python3.10 -m venv "$HAILO_ENV"
source "$HAILO_ENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install "$PROJ/hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl"
python -m pip install "$PROJ/hailort-4.23.0-cp310-cp310-linux_x86_64.whl"
```

Install Model Zoo v2.x inside project:

```bash
git clone https://github.com/hailo-ai/hailo_model_zoo.git "$MZ_ROOT"
cd "$MZ_ROOT"
git checkout v2.18
python -m pip install -e . --no-build-isolation
cd "$PROJ"
```

Verify:

```bash
which hailo
which hailomz
hailo --version
hailomz --version
python -c "import hailo_sdk_client; print('hailo_sdk_client OK')"
```

## 4) Health check

```bash
source "$HAILO_ENV/bin/activate"
"$PROJ/hmct" doctor --model-zoo-root "$MZ_ROOT" --calib-dir "$PROJ/calibration_images"
```

## 5) Export `.pt` to ONNX

### A) Fine-tuned classifier (example: ResNet/ResNeXt family)

```bash
source "$EXPORT_ENV/bin/activate"
cd "$PROJ"

"$PROJ/hmct" export-classifier \
  --arch resnet50 \
  --checkpoint "$PROJ/resnet50_best.pt" \
  --num-classes 2 \
  --image-size 224 \
  --output "$PROJ/models/resnet50_finetuned.onnx"

"$PROJ/scripts/check_onnx.py" "$PROJ/models/resnet50_finetuned.onnx"
```

If you get layer size mismatch errors, your checkpoint architecture does not
match `--arch`. Use the correct architecture (for example `resnext50_32x4d`)
and correct `--num-classes`.

### B) Custom YOLOv11 `.pt`

```bash
source "$EXPORT_ENV/bin/activate"
cd "$PROJ"

"$PROJ/scripts/export_yolo_onnx.sh" --weights "$PROJ/yolov11_custom.pt" --imgsz 640
"$PROJ/scripts/check_onnx.py" "$PROJ/yolov11_custom.onnx"
```

## 6) Compile ONNX to HEF (Hailo-8 M.2)

Activate compile env:

```bash
source "$HAILO_ENV/bin/activate"
cd "$PROJ"
```

### A) Model Zoo recipe flow (recommended when matching recipe exists)

```bash
"$PROJ/scripts/compile_zoo_model.sh" \
  --model yolov11s \
  --onnx "$PROJ/yolov11_custom.onnx" \
  --classes 1 \
  --model-zoo-root "$MZ_ROOT" \
  --calib-dir "$PROJ/calibration_images" \
  --hw-arch hailo8 \
  --output-dir "$PROJ/hailo_outputs/yolov11_custom"
```

### B) Direct DFC flow (BYOM)

Prepare calibration NPY:

```bash
python "$PROJ/scripts/prepare_calibration.py" \
  --images "$PROJ/calibration_images" \
  --output "$PROJ/calib/calib_224.npy" \
  --height 224 --width 224 --count 1024 --center-crop
```

Compile:

```bash
python "$PROJ/scripts/compile_direct_dfc.py" \
  --onnx "$PROJ/models/custom.onnx" \
  --network-name custom_model \
  --calib-npy "$PROJ/calib/calib_224.npy" \
  --input-name input --input-shape 1 3 224 224 \
  --hw-arch hailo8 \
  --output-dir "$PROJ/hailo_outputs/custom_model"
```

## 7) Expected outputs

HEF files are generated under:

```text
$PROJ/hailo_outputs/<model_name>/*.hef
```

## 8) Common blockers

- `libhailort.so... cannot open shared object file`
  - Runtime libs missing from linker path. Install matching HailoRT runtime from
    the same suite release and ensure environment setup exports required lib paths.
- `GLIBC_2.29 not found`
  - Host OS too old for that suite build. Use a newer compile host/container or
    a suite release compatible with your OS.
- `No module named hailo_sdk_client`
  - Wrong environment active. Compile must run in Hailo env, not export env.
