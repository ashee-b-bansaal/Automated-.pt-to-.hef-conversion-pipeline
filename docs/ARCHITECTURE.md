# Toolchain architecture

## Two isolated environments

### 1. Export environment

Contains PyTorch, torchvision, Ultralytics, OpenCLIP, and ONNX export tools.
Its job ends at a static `.onnx` file.

### 2. Hailo compiler environment

Contains the exact mutually compatible versions of the Dataflow Compiler,
Hailo Model Zoo, TensorFlow, ONNX, NumPy, protobuf, and related packages.
Do not install arbitrary packages into this environment.

## Conversion products

```text
.pt/.pth
  -> ONNX export
.onnx
  -> parse
parsed.har
  -> optimize and INT8 calibration
optimized.har
  -> hardware mapping and compile
.hef
  -> HailoRT on Raspberry Pi / host
```

## Three supported paths

1. **Pretrained Model Zoo model:** use the installed recipe and Zoo checkpoint.
2. **Fine-tuned known architecture:** export ONNX and compile against the exact
   corresponding Zoo recipe when graph compatibility is preserved.
3. **Bring Your Own Model:** use `compile_direct_dfc.py`; choose parser end nodes,
   calibration preprocessing, ALLS instructions, and postprocessing manually.

Model Zoo presence means Hailo has supplied a tested recipe. It does **not** mean
models outside the Zoo are categorically impossible. Those models simply move
from path 1/2 to path 3 and may fail because of unsupported operators, graph
boundaries, quantization sensitivity, or insufficient Hailo resources.
