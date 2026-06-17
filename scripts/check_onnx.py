#!/usr/bin/env python3
import argparse
from pathlib import Path

import onnx

p = argparse.ArgumentParser()
p.add_argument("model")
args = p.parse_args()
path = Path(args.model)
model = onnx.load(path)
onnx.checker.check_model(model)
print("ONNX CHECK: OK")
print("Inputs:")
for value in model.graph.input:
    dims = [d.dim_value or d.dim_param or "?" for d in value.type.tensor_type.shape.dim]
    print(f"  {value.name}: {dims}")
print("Outputs:")
for value in model.graph.output:
    dims = [d.dim_value or d.dim_param or "?" for d in value.type.tensor_type.shape.dim]
    print(f"  {value.name}: {dims}")
