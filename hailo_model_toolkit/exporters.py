from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .checkpoint import clean_state_dict, extract_state_dict


def _replace_classifier(model, num_classes: int):
    import torch.nn as nn

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return "fc"

    if hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return "heads.head"

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
            return "classifier"
        if isinstance(classifier, nn.Sequential):
            layers = list(classifier.children())
            for i in range(len(layers) - 1, -1, -1):
                if isinstance(layers[i], nn.Linear):
                    layers[i] = nn.Linear(layers[i].in_features, num_classes)
                    model.classifier = nn.Sequential(*layers)
                    return f"classifier.{i}"

    raise ValueError(
        "Could not automatically replace the classifier head. "
        "Add the architecture to _replace_classifier() or export the model yourself."
    )


def export_torchvision_classifier(
    *,
    arch: str,
    output: Path,
    checkpoint: Path | None,
    num_classes: int | None,
    image_size: int,
    pretrained: bool,
    checkpoint_key: str | None,
    opset: int,
    strict: bool,
    trusted_checkpoint: bool,
    mean: Iterable[float],
    std: Iterable[float],
):
    import torch
    import torchvision.models as tvm

    try:
        weights = "DEFAULT" if pretrained and checkpoint is None else None
        model = tvm.get_model(arch, weights=weights)
    except Exception as exc:
        available = sorted(tvm.list_models())
        raise ValueError(
            f"Unable to create torchvision model {arch!r}: {exc}\n"
            f"Examples available in this torchvision install: {', '.join(available[:30])}"
        ) from exc

    head = None
    if num_classes is not None:
        head = _replace_classifier(model, num_classes)

    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        # PyTorch 2.6+ defaults weights_only=True. Full-module checkpoints require
        # weights_only=False and must only be loaded from a trusted source.
        kwargs = {"map_location": "cpu"}
        if trusted_checkpoint:
            kwargs["weights_only"] = False
        try:
            ckpt = torch.load(checkpoint, **kwargs)
        except TypeError:
            kwargs.pop("weights_only", None)
            ckpt = torch.load(checkpoint, **kwargs)

        if hasattr(ckpt, "eval") and hasattr(ckpt, "state_dict") and not isinstance(ckpt, dict):
            model = ckpt
        else:
            state = clean_state_dict(extract_state_dict(ckpt, checkpoint_key))
            missing, unexpected = model.load_state_dict(state, strict=strict)
            if missing:
                print("WARNING: missing keys:", missing)
            if unexpected:
                print("WARNING: unexpected keys:", unexpected)

    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output),
            input_names=["input"],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )

    metadata = {
        "architecture": arch,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "pretrained": pretrained,
        "num_classes": num_classes,
        "image_size": image_size,
        "input_layout": "NCHW",
        "input_name": "input",
        "output_name": "logits",
        "normalization_mean": list(mean),
        "normalization_std": list(std),
        "normalization_embedded_in_onnx": False,
        "classifier_head_replaced": head,
        "opset": opset,
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
