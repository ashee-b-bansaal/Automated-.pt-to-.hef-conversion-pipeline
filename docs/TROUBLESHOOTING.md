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
