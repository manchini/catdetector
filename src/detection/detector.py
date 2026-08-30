"""
Detector de animais usando YOLOv8s + SAHI (sliced inference).
Estágio 2: roda apenas quando o detector de movimento encontra algo.

SAHI divide o frame em tiles sobrepostos antes da inferência, permitindo
detectar animais pequenos em câmeras de segurança com ângulo aberto.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Em câmeras IR, o modelo COCO frequentemente misclassifica objetos
# (caixa d'água → toilet, grade → person, etc). Filtramos tudo que
# nunca é animal e mantemos apenas classes que PODEM ser animais
# (cat, dog, bird, horse, sheep, cow, bear, zebra, giraffe, elephant).
# O usuário classifica via Telegram para coletar dados de fine-tuning.
ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}


class AnimalDetector:
    """
    Wrapper para YOLOv8s com SAHI para detecção de animais em câmeras de segurança.
    Usa sliced inference para encontrar animais pequenos em frames de alta resolução.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.15,
    ):
        self.confidence = confidence
        self._model = None
        self._sahi_model = None
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]):
        """Carrega o modelo YOLO e inicializa o wrapper SAHI."""
        try:
            from ultralytics import YOLO

            if model_path and Path(model_path).exists():
                logger.info(f"Carregando modelo custom: {model_path}")
                model_source = model_path
            else:
                logger.info("Carregando YOLOv8s pré-treinado (COCO)")
                model_source = "yolov8s.pt"

            self._model = YOLO(model_source)

            # Inicializa wrapper SAHI para sliced inference
            from sahi import AutoDetectionModel
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=model_source,
                confidence_threshold=self.confidence,
            )

            logger.info("Modelo YOLO + SAHI carregados com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detecta animais em um frame usando SAHI (sliced inference).

        Returns:
            Lista de detecções, cada uma com:
                - class_name: str ("cat", "dog", "bird", etc)
                - class_id: int
                - confidence: float (0-1)
                - bbox: tuple (x1, y1, x2, y2)
                - crop: np.ndarray (imagem recortada do animal)
        """
        start = time.time()

        # Sliced inference via SAHI
        from sahi.predict import get_sliced_prediction
        import PIL.Image

        pil_image = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            result = get_sliced_prediction(
                image=pil_image,
                detection_model=self._sahi_model,
                slice_height=480,
                slice_width=480,
                overlap_height_ratio=0.3,
                overlap_width_ratio=0.3,
                verbose=0,
            )
        finally:
            pil_image.close()

        detections = []
        for pred in result.object_prediction_list:
            class_name = pred.category.name
            if class_name not in ANIMAL_CLASSES:
                continue

            conf = pred.score.value
            x1 = int(pred.bbox.minx)
            y1 = int(pred.bbox.miny)
            x2 = int(pred.bbox.maxx)
            y2 = int(pred.bbox.maxy)

            crop = frame[y1:y2, x1:x2].copy()

            detections.append({
                "class_name": class_name,
                "class_id": pred.category.id,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "crop": crop,
            })

        elapsed = time.time() - start
        if detections:
            logger.info(
                f"YOLO+SAHI: {len(detections)} detecções em {elapsed:.2f}s "
                f"({', '.join(d['class_name'] for d in detections)})"
            )

        return detections

    def draw_detections(
        self, frame: np.ndarray, detections: list[dict]
    ) -> np.ndarray:
        """Desenha bounding boxes e labels no frame."""
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 255, 255)  # Amarelo padrão para todas as classes
            label = f"{det['class_name']} {det['confidence']:.0%}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

        return annotated
