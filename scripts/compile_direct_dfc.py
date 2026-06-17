#!/usr/bin/env python3
"""Direct ONNX -> HAR -> optimized HAR -> HEF using hailo_sdk_client.

This is the BYOM escape hatch for models without a Model Zoo recipe. It is
necessarily more manual: unsupported operators, parser boundary selection, and
resource mapping still require model-specific work.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--onnx", required=True)
p.add_argument("--network-name", required=True)
p.add_argument("--calib-npy", required=True, help="NHWC float32 array created by prepare_calibration.py")
p.add_argument("--output-dir", default="outputs")
p.add_argument("--hw-arch", default="hailo8")
p.add_argument("--start-nodes", nargs="*")
p.add_argument("--end-nodes", nargs="*")
p.add_argument("--input-name")
p.add_argument("--input-shape", nargs=4, type=int, metavar=("N", "C", "H", "W"))
p.add_argument("--model-script", help="Optional .alls file")
args = p.parse_args()

from hailo_sdk_client import ClientRunner

onnx = Path(args.onnx).resolve()
calib_path = Path(args.calib_npy).resolve()
out = Path(args.output_dir).resolve()
out.mkdir(parents=True, exist_ok=True)
if not onnx.is_file():
    raise SystemExit(f"ONNX not found: {onnx}")
if not calib_path.is_file():
    raise SystemExit(f"Calibration array not found: {calib_path}")

runner = ClientRunner(hw_arch=args.hw_arch)
kwargs = {}
if args.start_nodes:
    kwargs["start_node_names"] = args.start_nodes
if args.end_nodes:
    kwargs["end_node_names"] = args.end_nodes
if args.input_name and args.input_shape:
    kwargs["net_input_shapes"] = {args.input_name: list(args.input_shape)}

print("[1/3] Parsing ONNX")
runner.translate_onnx_model(str(onnx), args.network_name, **kwargs)
parsed_har = out / f"{args.network_name}.parsed.har"
runner.save_har(str(parsed_har))
print("Saved", parsed_har)

if args.model_script:
    print("Loading model script", args.model_script)
    runner.load_model_script(Path(args.model_script).read_text())

print("[2/3] Optimizing/quantizing")
calib = np.load(calib_path)
if calib.ndim != 4 or calib.shape[-1] not in (1, 3, 4):
    raise SystemExit(f"Expected NHWC calibration array; got {calib.shape}")
runner.optimize(calib)
optimized_har = out / f"{args.network_name}.optimized.har"
runner.save_har(str(optimized_har))
print("Saved", optimized_har)

print("[3/3] Compiling")
hef_bytes = runner.compile()
hef = out / f"{args.network_name}.{args.hw_arch}.hef"
hef.write_bytes(hef_bytes)
print("SUCCESS:", hef)
