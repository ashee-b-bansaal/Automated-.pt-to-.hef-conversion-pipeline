#!/usr/bin/env python3
"""Create an NHWC float32 calibration array for direct DFC conversion."""
import argparse
from pathlib import Path
import random

import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--images", required=True)
p.add_argument("--output", required=True)
p.add_argument("--height", type=int, required=True)
p.add_argument("--width", type=int, required=True)
p.add_argument("--count", type=int, default=1024)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--center-crop", action="store_true")
args = p.parse_args()

root = Path(args.images)
files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
if not files:
    raise SystemExit(f"No images found under {root}")
random.Random(args.seed).shuffle(files)
files = files[: min(args.count, len(files))]

images = []
for path in files:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if args.center_crop:
            w, h = image.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            image = image.crop((left, top, left + side, top + side))
        image = image.resize((args.width, args.height), Image.Resampling.BILINEAR)
        images.append(np.asarray(image, dtype=np.float32))
array = np.stack(images, axis=0)
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, array)
print(f"Saved {out}: shape={array.shape}, dtype={array.dtype}, range=({array.min()}, {array.max()})")
