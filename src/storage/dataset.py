"""
Gerencia o dataset de imagens classificadas pelo Telegram.
Organiza as imagens em pastas por label para retreino futuro.
"""

import shutil
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path("detections")
LABELED_DIR = BASE_DIR / "labeled"
UNLABELED_DIR = BASE_DIR / "unlabeled"

# Labels possíveis
LABELS = ["cat", "my_dog", "other", "false_positive"]


def init_dirs():
    """Cria a estrutura de diretórios."""
    UNLABELED_DIR.mkdir(parents=True, exist_ok=True)
    for label in LABELS:
        (LABELED_DIR / label).mkdir(parents=True, exist_ok=True)
    logger.info("Diretórios do dataset inicializados")


def save_frame(
    frame: np.ndarray,
    camera_id: str,
    confidence: float,
    model_class: str,
) -> str:
    """
    Salva o frame completo na pasta unlabeled.
    Retorna o caminho do arquivo.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    filename = f"{timestamp}_{camera_id}_{model_class}_{confidence:.0%}.jpg"
    filepath = UNLABELED_DIR / filename

    cv2.imwrite(str(filepath), frame)
    logger.debug(f"Frame salvo: {filepath}")
    return str(filepath)


def save_crop(
    crop: np.ndarray,
    camera_id: str,
    confidence: float,
    model_class: str,
) -> str:
    """
    Salva o recorte do animal detectado.
    Retorna o caminho do arquivo.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    filename = f"crop_{timestamp}_{camera_id}_{model_class}_{confidence:.0%}.jpg"
    filepath = UNLABELED_DIR / filename

    cv2.imwrite(str(filepath), crop)
    logger.debug(f"Crop salvo: {filepath}")
    return str(filepath)


def label_image(image_path: str, label: str):
    """
    Move a imagem da pasta unlabeled para a pasta do label correto.
    Chamado quando o usuário classifica pelo Telegram.
    """
    if label not in LABELS:
        logger.warning(f"Label inválido: {label}")
        return

    src = Path(image_path)
    if not src.exists():
        logger.warning(f"Imagem não encontrada: {src}")
        return

    dst = LABELED_DIR / label / src.name
    shutil.move(str(src), str(dst))
    logger.info(f"Imagem movida: {src.name} → {label}/")


def get_dataset_stats() -> dict:
    """Retorna estatísticas do dataset."""
    stats = {"unlabeled": 0, "labeled": {}}

    # Contar unlabeled
    if UNLABELED_DIR.exists():
        stats["unlabeled"] = len(list(UNLABELED_DIR.glob("*.jpg")))

    # Contar por label
    for label in LABELS:
        label_dir = LABELED_DIR / label
        if label_dir.exists():
            stats["labeled"][label] = len(list(label_dir.glob("*.jpg")))

    stats["total_labeled"] = sum(stats["labeled"].values())
    return stats
