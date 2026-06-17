from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any


def extract_state_dict(checkpoint: Any, explicit_key: str | None = None):
    """Extract a state_dict from common PyTorch checkpoint formats."""
    if explicit_key:
        if not isinstance(checkpoint, dict) or explicit_key not in checkpoint:
            raise KeyError(f"Checkpoint key {explicit_key!r} was not found")
        checkpoint = checkpoint[explicit_key]

    if hasattr(checkpoint, "state_dict") and not isinstance(checkpoint, dict):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint is neither a torch.nn.Module nor a dictionary. "
            "Use --checkpoint-key if weights are nested under a custom key."
        )

    for key in ("state_dict", "model_state_dict", "model", "ema", "weights"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
        if hasattr(value, "state_dict"):
            return value.state_dict()

    # A raw state_dict is a mapping whose values are tensor-like.
    if checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
        return checkpoint

    raise KeyError(
        "Could not identify a state_dict. Available top-level keys: "
        + ", ".join(map(str, checkpoint.keys()))
    )


def clean_state_dict(state_dict):
    """Remove common wrappers such as module., model., and _orig_mod."""
    cleaned = OrderedDict()
    prefixes = ("module.", "_orig_mod.")
    for key, value in state_dict.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        cleaned[key] = value
    return cleaned
