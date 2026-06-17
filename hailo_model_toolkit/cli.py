from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .exporters import export_torchvision_classifier
from .zoo import scan_model_zoo


def _command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"unavailable ({exc})"
    text = (result.stdout + result.stderr).strip().splitlines()
    return text[0] if text else f"exit={result.returncode}"


def cmd_doctor(args):
    print("=== Hailo Model Toolkit environment check ===")
    print("Python:", sys.version.splitlines()[0])
    for command in ("hailomz", "hailo", "hailortcli", "nvidia-smi"):
        print(f"{command:12s}:", shutil.which(command) or "NOT FOUND")
    print("hailomz version:", _command_version(["hailomz", "--version"]))
    print("hailortcli version:", _command_version(["hailortcli", "--version"]))

    if args.model_zoo_root:
        root = Path(args.model_zoo_root).expanduser().resolve()
        print("Model Zoo root:", root)
        try:
            models = scan_model_zoo(root)
            print("Model Zoo YAML recipes:", len(models))
            git = _command_version(["git", "-C", str(root), "describe", "--always", "--dirty"])
            print("Model Zoo revision:", git)
        except Exception as exc:
            print("Model Zoo check: FAILED:", exc)

    if args.calib_dir:
        calib = Path(args.calib_dir).expanduser()
        files = [p for p in calib.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        print("Calibration images:", len(files), "in", calib)

    try:
        import numpy, onnx
        print("numpy:", numpy.__version__)
        print("onnx:", onnx.__version__)
    except Exception as exc:
        print("Python package check:", exc)

    try:
        import tensorflow as tf
        print("tensorflow:", tf.__version__)
        print("TensorFlow GPUs:", tf.config.list_physical_devices("GPU"))
        if args.test_gpu and tf.config.list_physical_devices("GPU"):
            with tf.device("/GPU:0"):
                x = tf.random.normal([1, 32, 32, 3])
                k = tf.random.normal([3, 3, 3, 8])
                y = tf.nn.conv2d(x, k, strides=1, padding="SAME")
                print("GPU Conv2D: OK", y.shape)
    except Exception as exc:
        print("TensorFlow/GPU check: FAILED:", exc)


def cmd_zoo_list(args):
    root = Path(args.root).expanduser().resolve()
    models = scan_model_zoo(root)
    if args.task:
        models = [m for m in models if m.task == args.task]
    if args.pattern:
        needle = args.pattern.lower()
        models = [m for m in models if needle in m.name.lower()]

    if args.json:
        print(json.dumps([
            {
                "name": m.name,
                "task": m.task,
                "yaml": str(m.yaml),
                "alls": str(m.alls) if m.alls else None,
                "nms": str(m.nms) if m.nms else None,
            }
            for m in models
        ], indent=2))
        return

    print(f"{'MODEL':35s} {'TASK':20s} ALLS NMS")
    print("-" * 70)
    for m in models:
        print(f"{m.name:35s} {m.task:20s} {'yes' if m.alls else 'no ':4s} {'yes' if m.nms else 'no'}")
    print(f"\n{len(models)} model recipe(s)")


def cmd_export_list(args):
    try:
        import torchvision.models as tvm
    except Exception as exc:
        raise RuntimeError("torchvision is required in the export environment") from exc
    names = sorted(tvm.list_models())
    if args.pattern:
        names = [name for name in names if args.pattern.lower() in name.lower()]
    for name in names:
        print(name)
    print(f"\n{len(names)} torchvision architecture(s)")


def cmd_export_classifier(args):
    metadata = export_torchvision_classifier(
        arch=args.arch,
        output=Path(args.output).expanduser(),
        checkpoint=Path(args.checkpoint).expanduser() if args.checkpoint else None,
        num_classes=args.num_classes,
        image_size=args.image_size,
        pretrained=args.pretrained,
        checkpoint_key=args.checkpoint_key,
        opset=args.opset,
        strict=not args.non_strict,
        trusted_checkpoint=args.trusted_checkpoint,
        mean=args.mean,
        std=args.std,
    )
    print("Exported:", Path(args.output).expanduser().resolve())
    print(json.dumps(metadata, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="hmct",
        description="Hailo model export, inventory, and conversion helpers",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check Hailo tools, versions, GPU, and calibration data")
    doctor.add_argument("--model-zoo-root")
    doctor.add_argument("--calib-dir")
    doctor.add_argument("--test-gpu", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    zoo = sub.add_parser("zoo-list", help="List recipes in the installed Hailo Model Zoo")
    zoo.add_argument("--root", required=True)
    zoo.add_argument("--task", choices=["classification", "object_detection", "segmentation", "pose", "other"])
    zoo.add_argument("--pattern")
    zoo.add_argument("--json", action="store_true")
    zoo.set_defaults(func=cmd_zoo_list)

    export_list = sub.add_parser("export-list", help="List torchvision architectures available in the export environment")
    export_list.add_argument("--pattern")
    export_list.set_defaults(func=cmd_export_list)

    export = sub.add_parser("export-classifier", help="Export a torchvision classifier to static ONNX")
    export.add_argument("--arch", required=True, help="torchvision architecture, e.g. resnet50 or vit_b_16")
    export.add_argument("--output", required=True)
    export.add_argument("--checkpoint", help="Fine-tuned .pt/.pth checkpoint. Omit for stock pretrained model.")
    export.add_argument("--checkpoint-key", help="Explicit nested checkpoint key")
    export.add_argument("--num-classes", type=int)
    export.add_argument("--image-size", type=int, default=224)
    export.add_argument("--opset", type=int, default=11)
    export.add_argument("--pretrained", action="store_true")
    export.add_argument("--non-strict", action="store_true")
    export.add_argument(
        "--trusted-checkpoint",
        action="store_true",
        help="Allow loading a full pickled PyTorch module. Only use for a checkpoint you trust.",
    )
    export.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    export.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    export.set_defaults(func=cmd_export_classifier)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
