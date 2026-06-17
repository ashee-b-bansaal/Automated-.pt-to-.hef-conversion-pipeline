#!/usr/bin/env python3
"""Experimental OpenCLIP image-encoder export.

Export success does NOT imply Hailo-8 parse/compile support. ViT-L/14 is not in
the public Hailo-8 Model Zoo classification list; treat it as BYOM research.
"""
import argparse
from pathlib import Path

import torch
import open_clip

p = argparse.ArgumentParser()
p.add_argument("--model", default="ViT-L-14")
p.add_argument("--pretrained", required=True, help="OpenCLIP pretrained tag or checkpoint path")
p.add_argument("--output", required=True)
p.add_argument("--image-size", type=int, default=224)
p.add_argument("--opset", type=int, default=17)
args = p.parse_args()

model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
visual = model.visual.eval()
dummy = torch.randn(1, 3, args.image_size, args.image_size)
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    torch.onnx.export(
        visual,
        dummy,
        str(out),
        input_names=["image"],
        output_names=["embedding"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
print("Exported", out)
