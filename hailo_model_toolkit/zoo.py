from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ZooModel:
    name: str
    yaml: Path
    alls: Path | None
    nms: Path | None
    task: str


def _guess_task(name: str, text: str) -> str:
    lowered = (name + "\n" + text).lower()
    explicit = re.search(r"(?:network_type|task|task_type)\s*:\s*([a-zA-Z0-9_-]+)", text)
    if explicit:
        return explicit.group(1).lower()
    if any(x in lowered for x in ("seg", "segmentation")):
        return "segmentation"
    if any(x in lowered for x in ("pose", "keypoint")):
        return "pose"
    if any(x in lowered for x in ("yolo", "ssd", "centernet", "nanodet", "detr", "efficientdet")):
        return "object_detection"
    if any(x in lowered for x in ("resnet", "mobilenet", "efficientnet", "vit", "deit", "swin", "classification")):
        return "classification"
    return "other"


def scan_model_zoo(root: Path) -> list[ZooModel]:
    cfg = root / "hailo_model_zoo" / "cfg"
    networks = cfg / "networks"
    if not networks.is_dir():
        raise FileNotFoundError(f"Model Zoo networks directory not found: {networks}")

    models: list[ZooModel] = []
    for yaml in sorted(networks.glob("*.yaml")):
        name = yaml.stem
        text = yaml.read_text(errors="ignore")
        alls_matches = list((cfg / "alls").rglob(f"{name}.alls")) if (cfg / "alls").exists() else []
        nms_matches = (
            list((cfg / "postprocess_config").rglob(f"{name}_nms_config.json"))
            if (cfg / "postprocess_config").exists()
            else []
        )
        models.append(
            ZooModel(
                name=name,
                yaml=yaml,
                alls=alls_matches[0] if alls_matches else None,
                nms=nms_matches[0] if nms_matches else None,
                task=_guess_task(name, text),
            )
        )
    return models
