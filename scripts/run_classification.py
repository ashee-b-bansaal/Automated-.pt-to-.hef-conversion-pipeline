#!/usr/bin/env python3
"""Single-image classification with a compiled HEF using pyHailoRT.

Preprocessing must match training. This script defaults to ImageNet
resize/normalization and requests FLOAT32 virtual streams. Depending on how
normalization was embedded in the HEF, use --raw-255 or adjust mean/std.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

import numpy as np
from PIL import Image
from hailo_platform import (
    HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams,
    InputVStreamParams, OutputVStreamParams, VDevice,
)

p = argparse.ArgumentParser()
p.add_argument("--hef", required=True)
p.add_argument("--image", required=True)
p.add_argument("--labels")
p.add_argument("--size", type=int, default=224)
p.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
p.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
p.add_argument("--raw-255", action="store_true", help="Do not normalize; pass 0..255 RGB values")
p.add_argument("--topk", type=int, default=5)
args = p.parse_args()

image = Image.open(args.image).convert("RGB").resize((args.size, args.size))
x = np.asarray(image, dtype=np.float32)
if not args.raw_255:
    x = x / 255.0
    x = (x - np.asarray(args.mean, dtype=np.float32)) / np.asarray(args.std, dtype=np.float32)
x = x[None, ...]

labels = None
if args.labels:
    labels = [line.strip() for line in Path(args.labels).read_text().splitlines() if line.strip()]

hef = HEF(args.hef)
with VDevice() as device:
    params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    group = device.configure(hef, params)[0]
    activation = group.create_params()
    inputs = InputVStreamParams.make(group, quantized=False, format_type=FormatType.FLOAT32)
    outputs = OutputVStreamParams.make(group, quantized=False, format_type=FormatType.FLOAT32)
    input_name = hef.get_input_vstream_infos()[0].name
    with InferVStreams(group, inputs, outputs) as pipeline:
        with group.activate(activation):
            start = time.perf_counter()
            result = pipeline.infer({input_name: x})
            elapsed = (time.perf_counter() - start) * 1000

output_name, raw = next(iter(result.items()))
logits = np.asarray(raw).reshape(-1)
probs = np.exp(logits - logits.max())
probs /= probs.sum()
indices = np.argsort(probs)[::-1][:args.topk]
print(f"Output: {output_name}; end-to-end call: {elapsed:.2f} ms")
for i in indices:
    name = labels[i] if labels and i < len(labels) else str(i)
    print(f"{i:4d}  {probs[i]:.6f}  {name}")
