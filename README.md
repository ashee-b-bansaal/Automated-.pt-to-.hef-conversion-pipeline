# Hailo Edge Model Toolkit

A reproducible toolkit for taking vision models from PyTorch/ONNX to Hailo
Executable Format (`.hef`), then benchmarking and running them on Raspberry Pi
5 + Hailo-8/Hailo-8L.

It supports four practical workflows:

1. **Existing Hailo Model Zoo models** — YOLO, ResNet, ViT, MobileNet, etc.
2. **Fine-tuned known architectures** — for example a custom ResNet-50
   checkpoint with a different classifier head.
3. **Custom YOLO models** — export `.pt -> .onnx`, then reuse the matching Zoo
   recipe; includes dynamic YOLOv5 NMS repair.
4. **Bring Your Own ONNX model** — direct DFC parse/optimize/compile when no
   tested Model Zoo recipe exists.

> **Compatibility warning:** for Hailo-8 and Hailo-8L, use a Hailo Model Zoo
> **v2.x** checkout paired with Dataflow Compiler **v3.x**. The current Model
> Zoo master branch targets newer Hailo-10/Hailo-15 devices. The safest setup is
> the Model Zoo and DFC shipped in the same Hailo Software Suite release.

Suggested toolchain: 

Hailo hardware:       Hailo-8 M.2
Target arch:          hailo8
Python:               3.10
OS for compiling:     WSL Ubuntu 22.04/ Linux machine
Dataflow Compiler:    3.33.1
HailoRT:              4.23.0
Hailo Model Zoo:      v2.18

## The most important rule

Keep **two separate Python environments**:

```text
model_export_env: PyTorch / torchvision / Ultralytics / OpenCLIP / ONNX export
hailo_env:        Dataflow Compiler / Model Zoo / pinned compiler dependencies
```

Do not install Ultralytics, `onnxscript`, `onnxslim`, or random upgrades inside
`hailo_env`. The compiler stack is version-sensitive.

## Repository map

```text
hmct                                  main helper CLI
hailo_model_toolkit/                  environment, checkpoint, and Zoo helpers
scripts/export_yolo_onnx.sh           YOLO .pt -> static ONNX
scripts/convert_yolov5_dynamic.sh     YOLOv5 NMS-aware conversion
scripts/compile_zoo_model.sh          known Zoo model or matching custom ONNX
scripts/compile_direct_dfc.py         arbitrary ONNX BYOM path
scripts/prepare_calibration.py        image folder -> NHWC calibration array
scripts/run_classification.py         simple HEF classification inference
scripts/benchmark_hefs.sh             benchmark all HEFs on target device
docs/MODEL_ZOO_MODELS.md              Hailo-8 model inventory and local scanner
```

## 1. Prepare the export environment

```bash
python3.10 -m venv ~/model_export_env
source ~/model_export_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-export.txt
```

Install `ultralytics` only when exporting current YOLO models, or
`open_clip_torch` only when experimentally exporting OpenCLIP.

## 2. Prepare the Hailo environment

Download/install the matching Hailo Software Suite (including Dataflow Compiler
and HailoRT) from the Hailo Developer Zone:
https://hailo.ai/developer-zone/

Install the matching Hailo Model Zoo v2.x checkout from the official releases:
https://github.com/hailo-ai/hailo_model_zoo/releases

Example (pin to v2.18.0):

```bash
git clone https://github.com/hailo-ai/hailo_model_zoo.git
cd hailo_model_zoo
git checkout v2.18.0
```

Use the exact Model Zoo tag that matches your installed Hailo Software Suite /
DFC release notes.

Check the machine:

```bash
source ~/hailo_env/bin/activate
./hmct doctor   --model-zoo-root ~/Video_streaming/hailo_model_zoo   --calib-dir ~/calibration_images   --test-gpu
```

List the classifier architectures available in the separate export environment:

```bash
source ~/model_export_env/bin/activate
./hmct export-list --pattern resnet
./hmct export-list --pattern vit
```

## 3. See what your installed Model Zoo can compile

```bash
./hmct zoo-list --root ~/Video_streaming/hailo_model_zoo
./hmct zoo-list --root ~/Video_streaming/hailo_model_zoo --pattern yolo
./hmct zoo-list --root ~/Video_streaming/hailo_model_zoo --pattern resnet
./hmct zoo-list --root ~/Video_streaming/hailo_model_zoo --pattern vit
```

See [`docs/MODEL_ZOO_MODELS.md`](docs/MODEL_ZOO_MODELS.md) for a Hailo-8 public
snapshot. The local command is more reliable than a static list.

## 4. Choose your conversion path

| Starting point | Recommended path |
|---|---|
| Stock model already in Model Zoo | `compile_zoo_model.sh` without `--onnx` |
| Fine-tuned ResNet/ViT/classifier matching a Zoo architecture | export ONNX, then `compile_zoo_model.sh --onnx ...` |
| Custom YOLOv8/10/11 | `export_yolo_onnx.sh`, then matching YOLO recipe |
| YOLOv5 with NMS node mismatch | `convert_yolov5_dynamic.sh` |
| Architecture absent from Zoo | `prepare_calibration.py` + `compile_direct_dfc.py` |

Model Zoo presence is not a universal whitelist. It means Hailo has provided a
known graph, parser boundaries, optimization script, preprocessing assumptions,
and often postprocessing. A model outside the Zoo can still be attempted, but
operator support, quantization, end-node selection, and hardware mapping become
your responsibility.

## 5. Fine-tuned ResNet-50 example

In the export environment:

```bash
./hmct export-classifier   --arch resnet50   --checkpoint /path/to/best.pt   --num-classes 2   --image-size 224   --output ~/models/resnet50_finetuned.onnx

./scripts/check_onnx.py ~/models/resnet50_finetuned.onnx
```

The exporter handles raw state dicts and common keys such as `state_dict`,
`model_state_dict`, `model`, and `ema`. Use `--checkpoint-key` for a custom
layout. Use `--trusted-checkpoint` only for a full pickled model from a trusted
source.

In the Hailo compiler environment:

```bash
./scripts/compile_zoo_model.sh   --model resnet_v1_50   --onnx ~/models/resnet50_finetuned.onnx   --classes 2   --model-zoo-root ~/Video_streaming/hailo_model_zoo   --calib-dir ~/resnet50_calibration_images   --hw-arch hailo8   --output-dir ~/hailo_outputs/resnet50
```

This works when the exported graph remains compatible with the Zoo's ResNet-50
recipe. If it does not, use the direct DFC route and supply parser boundaries or
an ALLS script.

## 6. ViT guidance

Hailo-8's public Zoo includes known-good variants such as `vit_tiny`,
`vit_small`, `vit_base`, DeiT, Swin, LeViT, and other efficient transformer
families. Compile them exactly like the stock ResNet example.

**ViT-L/14 is different:** it is not in the public Hailo-8 classification list.
The repository includes an experimental OpenCLIP image-encoder exporter, but an
ONNX export is not a promise that Hailo-8 can parse, quantize, or map it. Treat
it as a BYOM feasibility experiment.

## 7. Calibration

Calibration data estimates activation ranges for quantization. Use unlabeled,
representative deployment images. Preserve the same resize, crop, channel order,
and normalization used during training. Some ALLS recipes require a minimum
number of samples; 1024 is a safe starting point for the models that previously
rejected 500 images.

For direct DFC:

```bash
./scripts/prepare_calibration.py   --images ~/dataset/train   --output ~/calib/resnet50_224.npy   --height 224 --width 224 --count 1024 --center-crop
```

## 8. Direct DFC conversion

```bash
source ~/hailo_env/bin/activate
./scripts/compile_direct_dfc.py   --onnx ~/models/custom.onnx   --network-name custom_classifier   --calib-npy ~/calib/custom_224.npy   --input-name input   --input-shape 1 3 224 224   --hw-arch hailo8   --output-dir ~/hailo_outputs/custom
```

Add `--end-nodes` when automatic parser boundaries are wrong and
`--model-script custom.alls` when optimization/compiler instructions are
needed. Direct conversion cannot dynamically make unsupported operators
supported; inspect the parser/compiler error rather than blindly retrying.

## 9. Run on Raspberry Pi

```bash
scp ~/hailo_outputs/resnet50/*.hef pi@raspberrypi.local:~/models/
ssh pi@raspberrypi.local
hailortcli fw-control identify
hailortcli parse-hef ~/models/resnet_v1_50.hef
hailortcli benchmark ~/models/resnet_v1_50.hef
```

For a full inference example, see `scripts/run_classification.py`. Accelerator
FPS and end-to-end camera latency are different measurements.

## 10. Validate accuracy after quantization

A compiled model is not automatically a correct model. Compare at least:

1. PyTorch FP32 predictions
2. ONNX Runtime FP32 predictions
3. Hailo quantized/emulator predictions when available
4. Physical Hailo predictions

Use a held-out labeled set and report accuracy loss, not only FPS and HEF size.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/MODEL_ZOO_MODELS.md`](docs/MODEL_ZOO_MODELS.md)
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
- [`docs/RASPBERRY_PI.md`](docs/RASPBERRY_PI.md)
- [`docs/REFERENCES.md`](docs/REFERENCES.md)
- [`examples/commands.md`](examples/commands.md)

## Scope and honesty

The scripts automate repeatable parts of the workflow. They cannot guarantee
that every neural network is compatible with Hailo-8. Conversion depends on the
actual ONNX graph, supported operations, static shapes, quantization behavior,
memory/resource mapping, and postprocessing requirements.
