# Project history

This toolkit consolidates the scripts developed while compiling and benchmarking
YOLO models on an x86_64 WSL2 Hailo compiler host, then deploying the HEFs to a
Raspberry Pi 5 with a Hailo-8 M.2 accelerator.

The original work established these practical lessons:

- Model Zoo recipes compiled YOLOv8, YOLOv10, YOLOv11, and several other
  variants without retraining.
- YOLOv5 exports required dynamic repair of NMS layer references.
- Calibration requirements differed across models; YOLO26 requested 1024.
- Large detector compilation could take hours and use multi-context mapping.
- A visible NVIDIA GPU was not enough; TensorFlow cuDNN Conv2D had to work.
- HEFs were benchmarked on the Pi with `hailortcli benchmark`.

The repository intentionally keeps the dynamic YOLOv5 fix while replacing
one-off batch scripts with reusable, argument-driven tools.
