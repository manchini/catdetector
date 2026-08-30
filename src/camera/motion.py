"""
Detector de movimento usando Background Subtraction do OpenCV.
Estágio 1 (leve): roda em todas as câmeras o tempo todo.
Só quando detecta movimento significativo é que o frame vai pro YOLO.
"""

import cv2
import time
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class MotionDetector:
    """
    Detecta movimento comparando frames consecutivos.
    Usa MOG2 Background Subtractor — leve e eficiente.
    Suporta zonas de detecção (detection_zone) para limitar análise a regiões específicas.
    """

    def __init__(
        self,
        threshold: int = 5000,
        min_area: int = 500,
        detection_zone: list | None = None,
    ) -> None:
        """
        Args:
            threshold: Quantidade mínima de pixels em movimento para considerar
                       que houve movimento significativo.
            min_area: Área mínima de um contorno para ser considerado (filtra ruído).
            detection_zone: Lista de zonas normalizadas (0-1): [[x1, y1, x2, y2], ...]
                           Se fornecido, análise é limitada a essas regiões.
        """
        self.threshold = threshold
        self.min_area = min_area
        self.detection_zone = detection_zone  # Zona(s) de interesse
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False,
        )
        self._last_motion_time = 0
        self._warmup_frames = 30  # frames iniciais para calibrar o fundo
        self._frame_count = 0

    def detect(self, frame: np.ndarray) -> dict:
        """
        Analisa um frame e retorna se houve movimento.

        Returns:
            dict com:
                - motion: bool (houve movimento significativo?)
                - score: int (quantidade de pixels em movimento)
                - contours: list (contornos do movimento)
                - bounding_boxes: list (retângulos ao redor do movimento)
        """
        self._frame_count += 1

        # Reduz resolução para análise (muito mais rápido)
        small = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Se há zona de detecção definida, mascara apenas essa região
        if self.detection_zone:
            mask = np.zeros_like(gray)
            h, w = small.shape[:2]
            for zone in self.detection_zone:
                x1, y1, x2, y2 = zone
                # Converte coordenadas normalizadas (0-1) para pixel
                x1, y1 = int(x1 * w), int(y1 * h)
                x2, y2 = int(x2 * w), int(y2 * h)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            gray = cv2.bitwise_and(gray, gray, mask=mask)

        # Aplica background subtraction
        fg_mask = self._bg_subtractor.apply(gray)

        # Remove ruído
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Binariza
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Calcula score (total de pixels em movimento)
        motion_score = cv2.countNonZero(thresh)

        # Ainda em warmup (calibrando o fundo)
        if self._frame_count < self._warmup_frames:
            return {
                "motion": False,
                "score": 0,
                "contours": [],
                "bounding_boxes": [],
            }

        # Encontra contornos
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filtra contornos pequenos (ruído)
        significant_contours = [
            c for c in contours if cv2.contourArea(c) > self.min_area
        ]

        # Calcula bounding boxes (escala de volta para resolução original)
        h_orig, w_orig = frame.shape[:2]
        scale_x = w_orig / 320
        scale_y = h_orig / 240
        bounding_boxes = []
        for c in significant_contours:
            x, y, w, h = cv2.boundingRect(c)
            bounding_boxes.append((
                int(x * scale_x),
                int(y * scale_y),
                int(w * scale_x),
                int(h * scale_y),
            ))

        has_motion = motion_score > self.threshold

        if has_motion:
            self._last_motion_time = time.time()

        return {
            "motion": has_motion,
            "score": motion_score,
            "contours": significant_contours,
            "bounding_boxes": bounding_boxes,
        }

    @property
    def last_motion_time(self) -> float:
        return self._last_motion_time

    def reset(self) -> None:
        """Reinicia o detector (útil ao trocar de cena)."""
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False,
        )
        self._frame_count = 0


def _zones_to_mask(zones, width: int, height: int) -> np.ndarray | None:
    """Converte zonas normalizadas em máscara binária do tamanho informado."""
    if not zones:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    for zone in zones:
        x1, y1, x2, y2 = zone
        px1, py1 = int(x1 * width), int(y1 * height)
        px2, py2 = int(x2 * width), int(y2 * height)
        cv2.rectangle(mask, (px1, py1), (px2, py2), 255, -1)
    return mask


def filter_frames_with_motion(
    frames,
    detection_zone=None,
    threshold: int = 30,
    min_area: int = 20,
    analysis_size: tuple = (320, 240),
    debug_dir: Path | None = None,
    debug_prefix: str = "zero_motion",
) -> list[tuple[float, np.ndarray, int]]:
    """
    Pré-filtro de movimento para uma sequência finita de frames extraídos
    offline de um .dav. Não tem warmup (ao contrário de MotionDetector),
    pois usa diferença contra a mediana dos primeiros frames como fundo.

    Args:
        frames: lista de tuplas (timestamp_sec, frame_ndarray).
        detection_zone: lista de zonas normalizadas [[x1,y1,x2,y2], ...].
        threshold: pixels de diferença (na zona) para considerar movimento.
        min_area: área mínima de um contorno para contar.
        analysis_size: (w, h) de redução para análise.

    Returns:
        Lista filtrada [(ts, frame, score)] apenas com frames em que
        houve movimento significativo dentro das zonas.
    """
    if not frames:
        return []

    aw, ah = analysis_size

    gray_frames = []
    for _, f in frames:
        small = cv2.resize(f, (aw, ah))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        gray_frames.append(gray)

    baseline_count = min(15, len(gray_frames))
    baseline = np.median(
        np.stack(gray_frames[:baseline_count], axis=0), axis=0
    ).astype(np.uint8)

    zone_mask = _zones_to_mask(detection_zone, aw, ah)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    kept = []
    max_score = 0
    max_area = 0.0
    max_frame = None
    max_ts = 0.0
    for (ts, frame), gray in zip(frames, gray_frames):
        diff = cv2.absdiff(gray, baseline)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        masked = (
            cv2.bitwise_and(thresh, thresh, mask=zone_mask)
            if zone_mask is not None
            else thresh
        )

        score = int(cv2.countNonZero(masked))
        contours, _ = cv2.findContours(
            masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        frame_max_area = max((cv2.contourArea(c) for c in contours), default=0.0)
        if score > max_score:
            max_score = score
            max_area = frame_max_area
            max_frame = frame
            max_ts = ts
        has_significant = any(cv2.contourArea(c) >= min_area for c in contours)

        if has_significant and score >= threshold:
            kept.append((ts, frame, score))

    logger.info(
        "Pre-filtro motion offline: %d/%d frame(s) com movimento na zona "
        "(threshold=%d, min_area=%d, max_score=%d, max_area=%.1f, zonas=%d)",
        len(kept),
        len(frames),
        threshold,
        min_area,
        max_score,
        max_area,
        len(detection_zone or []),
    )
    if not kept and debug_dir is not None and max_frame is not None:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            out = debug_dir / f"{debug_prefix}_max_{int(max_ts * 10):06d}.jpg"
            cv2.imwrite(str(out), max_frame)
            logger.info("Frame debug zero-motion salvo em %s", out)
        except OSError as e:
            logger.warning("Nao foi possivel salvar debug zero-motion: %s", e)
    return kept
