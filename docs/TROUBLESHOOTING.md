# Troubleshooting

## Package conflicts

The project previously broke after `onnxscript` / `onnxslim` upgraded NumPy,
ONNX, protobuf, and `ml-dtypes` inside the Hailo environment. Keep export tools
in a separate virtual environment. Record a working environment with:

```bash
python -m pip freeze > hailo-environment-lock.txt
```

For the previous Hailo-8 setup, the working stack included DFC 3.33.1,
HailoRT 4.23.0, NumPy 1.26.4, ONNX 1.16.0, protobuf 3.20.3,
`ml-dtypes` 0.4.1, and pyparsing 2.4.7. Treat those as a historical lock,
not universal versions for every Software Suite release.

## GPU appears but Hailo falls back to CPU

`nvidia-smi` and `tf.config.list_physical_devices('GPU')` are insufficient.
Test an actual TensorFlow Conv2D through `hmct doctor --test-gpu`. The earlier
failure `No DNN in stream executor` meant CUDA visibility worked but cuDNN did
not.

## Calibration set too small

Optimization scripts can require a fixed number of samples. YOLO26 previously
required 1024 images and rejected a 500-image set. Count files before launching:

```bash
find "$CALIB_DIR" -type f | wc -l
```

Use deployment-like images. Labels are not required for calibration, but are
required to measure post-quantization accuracy.

## YOLOv5 NMS layer mismatch

Different YOLOv5 exports can assign different convolution node IDs. The old
NMS JSON referenced nonexistent layers. `convert_yolov5_dynamic.sh` reads the
parsed HAR, finds the final three detection convolutions, patches the JSON, and
retries. Do not hardcode `conv47`, `conv54`, etc. across variants.

## Shell variables were ignored

Variables assigned on separate lines are not inherited by a child process
unless exported. Use either:

```bash
CALIB=/data/calib MODELS="yolov8n yolov8s" ./script.sh
```

or:

```bash
export CALIB=/data/calib
export MODELS="yolov8n yolov8s"
./script.sh
```

## Compile was interrupted overnight

HEF generation is atomic enough to check by existence and nonzero size, but
always review per-model logs. Use `check_hef_status.sh` and rerun only missing
models. For long runs, use `tmux` and save logs with `tee`.

## Model Zoo branch mismatch

For Hailo-8/Hailo-8L, use a Model Zoo **v2.x** version paired with DFC **v3.x**.
The current master branch targets newer Hailo-10/Hailo-15 devices. Prefer the
Model Zoo bundled with the exact Hailo Software Suite release.

## `hmct` commands fail with "No such file or directory"

`hmct` is in this repository root. Run commands from the toolkit directory:

```bash
cd /path/to/Automated-.pt-to-.hef-conversion-pipeline
./hmct ...
```

If `source /model_export_env/bin/activate` fails, the leading `/` is usually
wrong. Use `~/model_export_env/bin/activate` or an explicit absolute path.

## `compile_direct_dfc.py`: `No module named hailo_sdk_client`

This typically means the export environment is active during compile. Compile
must run in the Hailo environment only.

```bash
deactivate 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hailo310
python -c "import hailo_sdk_client; print('OK')"
```

## `libhailort.so.4.23.0` cannot be opened

`hailo_platform` found its Python extension, but the runtime shared library is
not on the dynamic linker path.

1. Locate runtime libs:

```bash
find "$CONDA_PREFIX" -type f \( -name "libhailort.so*" -o -name "*pyhailort*.so" \) -print
```

2. If `libhailort.so.*` exists, export its directory:

```bash
export LD_LIBRARY_PATH="/path/to/libhailort_dir:${LD_LIBRARY_PATH:-}"
```

3. If `libhailort.so.*` is missing entirely, install the matching HailoRT
runtime package from the same Hailo Software Suite release.

## `GLIBC_2.29 not found` for `HSim.so`

This is an OS compatibility issue, not a Python package conflict. The DFC/HSim
binary requires a newer glibc than the host provides.

Check host libc:

```bash
ldd --version
```

If host glibc is older than required (for example 2.28 vs required 2.29), use:

- a newer compile host/container, or
- a Hailo release built for that OS/libc baseline.

`pip`, `uv`, and `conda` cannot safely "upgrade system glibc" on a shared node.

## Checkpoint loads with massive size mismatches

If most backbone layers mismatch, the checkpoint architecture differs from the
requested exporter architecture (for example `resnext50_32x4d` vs `resnet50`).
Use the correct `--arch`.

Also match `--num-classes` to the trained head shape. For example:

- `fc.weight` shape `[1, 2048]` -> `--num-classes 1`
- `fc.weight` shape `[2, 2048]` -> `--num-classes 2`

## TorchScript ONNX export fails with `torch.export` errors

ScriptModule exports can fail under the new exporter path. Force legacy export
mode in custom scripts:

```python
torch.onnx.export(model, x, out_path, opset_version=17, dynamo=False)
```

If your scripted model is already a wrapper module, export it directly instead
of wrapping again for tracing.
