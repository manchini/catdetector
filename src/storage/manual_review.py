"""
Persistencia de anotacoes manuais para revisao de eventos DVR.

Cada frame revisado gera uma imagem em detections/manual_review/images,
um label YOLO correspondente em detections/manual_review/labels e registros
no SQLite para manter as estatisticas do projeto coerentes.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from . import database

logger = logging.getLogger(__name__)

BASE_DIR = Path("detections/manual_review")
IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"
CLASSES_PATH = BASE_DIR / "classes.txt"
ANNOTATIONS_PATH = BASE_DIR / "annotations.json"

YOLO_CLASSES = ("cat", "my_dog", "other")
YOLO_CLASS_TO_ID = {name: idx for idx, name in enumerate(YOLO_CLASSES)}
VALID_STATUSES = {"annotated", "false_positive"}


@dataclass(frozen=True)
class ManualBox:
    label: str
    bbox: tuple[int, int, int, int]


def init_manual_review_dirs(base_dir: Path = BASE_DIR) -> tuple[Path, Path]:
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    classes_path = base_dir / "classes.txt"
    if not classes_path.exists():
        classes_path.write_text("\n".join(YOLO_CLASSES) + "\n", encoding="utf-8")
    return images_dir, labels_dir


def bbox_px_to_yolo(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = validate_bbox_px(bbox, width, height)
    box_w = x2 - x1
    box_h = y2 - y1
    cx = x1 + box_w / 2
    cy = y1 + box_h / 2
    return cx / width, cy / height, box_w / width, box_h / height


def validate_bbox_px(
    bbox: Iterable[int | float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    values = list(bbox)
    if len(values) != 4:
        raise ValueError("bbox deve ter quatro valores")

    x1, y1, x2, y2 = [int(round(float(v))) for v in values]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox precisa ter largura e altura positivas")
    return x1, y1, x2, y2


def normalize_boxes(
    boxes: list[dict],
    width: int,
    height: int,
    status: str,
) -> list[ManualBox]:
    if status not in VALID_STATUSES:
        raise ValueError(f"status invalido: {status}")
    if status == "false_positive":
        return []
    if not boxes:
        raise ValueError("anotacao precisa de ao menos uma bbox")

    normalized = []
    for box in boxes:
        label = str(box.get("label", "")).strip()
        if label not in YOLO_CLASS_TO_ID:
            raise ValueError(f"label invalido: {label}")
        bbox = box.get("bbox_px") or box.get("bbox")
        if bbox is None:
            raise ValueError("bbox ausente")
        normalized.append(
            ManualBox(
                label=label,
                bbox=validate_bbox_px(bbox, width, height),
            )
        )
    return normalized


def _safe_stem(value: str) -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value
    )
    return safe.strip("_") or "manual"


def _load_index(path: Path = ANNOTATIONS_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Nao foi possivel carregar indice manual: %s", e)
        return {}


def _save_index(index: dict, path: Path = ANNOTATIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def get_saved_annotation(source_id: str, base_dir: Path = BASE_DIR) -> dict | None:
    index = _load_index(base_dir / "annotations.json")
    return index.get(source_id)


def save_manual_annotation(
    *,
    source_image_path: Path,
    width: int,
    height: int,
    camera_id: str,
    camera_name: str,
    timestamp: datetime,
    status: str,
    boxes: list[dict],
    source_id: str,
    base_dir: Path = BASE_DIR,
) -> dict:
    images_dir, labels_dir = init_manual_review_dirs(base_dir)
    normalized_boxes = normalize_boxes(boxes, width, height, status)

    timestamp_stem = timestamp.strftime("%Y%m%d_%H%M%S")
    stem = _safe_stem(f"{timestamp_stem}_{camera_id}_{source_id[:12]}")
    image_path = images_dir / f"{stem}.jpg"
    label_path = labels_dir / f"{stem}.txt"

    shutil.copyfile(source_image_path, image_path)

    yolo_lines = []
    for box in normalized_boxes:
        class_id = YOLO_CLASS_TO_ID[box.label]
        cx, cy, bw, bh = bbox_px_to_yolo(box.bbox, width, height)
        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

    database.init_db()
    detection_ids = []
    if status == "false_positive":
        detection_ids.append(
            database.save_manual_detection(
                camera_id=camera_id,
                camera_name=camera_name,
                timestamp=timestamp,
                user_label="false_positive",
                image_path=str(image_path),
                bbox=None,
            )
        )
    else:
        for box in normalized_boxes:
            detection_ids.append(
                database.save_manual_detection(
                    camera_id=camera_id,
                    camera_name=camera_name,
                    timestamp=timestamp,
                    user_label=box.label,
                    image_path=str(image_path),
                    bbox=box.bbox,
                )
            )

    result = {
        "source_id": source_id,
        "status": status,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "classes_path": str(base_dir / "classes.txt"),
        "detection_ids": detection_ids,
        "boxes": [
            {"label": box.label, "bbox_px": list(box.bbox)}
            for box in normalized_boxes
        ],
        "timestamp": timestamp.isoformat(),
    }

    index_path = base_dir / "annotations.json"
    index = _load_index(index_path)
    index[source_id] = result
    _save_index(index, index_path)
    return result
