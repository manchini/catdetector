"""
Validacao end-to-end: segmento DVR 00:00:00-00:30:00 do cam_frente
- Extrai frames do .dav ja baixado
- Aplica pre-filtro de movimento nas zonas
- Roda YOLO+SAHI
- Envia alertas para o Telegram (bot real)

Uso:
    python -m scripts.validate_full_pipeline
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import yaml
from dotenv import load_dotenv
from telegram import Bot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.camera.motion import filter_frames_with_motion
from src.detection.detector import AnimalDetector
from src.dvr.offline_processor import frames_from_video
from src.storage import database, dataset
from src.bot.notifications import send_detection_alert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-20s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate")

SEGMENT = Path(
    "detections/analysis/20260414_000329_cam_frente/"
    "segment_00.00.00-00.30.00[R][0@0][0].dav"
)
CAM_ID = "cam_frente"
CAM_NAME = "Frente"
SEGMENT_START = datetime(2026, 4, 14, 0, 0, 0)
FRAME_INTERVAL = 2


def load_cam_config(cam_id: str) -> dict:
    with open("config/cameras.yaml") as f:
        cfg = yaml.safe_load(f)
    for cam in cfg.get("cameras", []):
        if cam.get("id") == cam_id:
            return cam
    raise ValueError(f"Camera {cam_id} nao encontrada no config")


async def main():
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID ausentes em .env")
        return

    if not SEGMENT.exists():
        logger.error(f"Segmento nao encontrado: {SEGMENT}")
        return

    database.init_db()
    dataset.init_dirs()

    cam_cfg = load_cam_config(CAM_ID)
    zones = cam_cfg.get("detection_zone")
    logger.info(f"Zonas de deteccao ({CAM_ID}): {len(zones) if zones else 0}")

    logger.info(f"Extraindo frames de {SEGMENT.name} (interval={FRAME_INTERVAL}s)")
    frames_list = list(frames_from_video(SEGMENT, frame_interval_sec=FRAME_INTERVAL))
    logger.info(f"Frames extraidos: {len(frames_list)}")

    if not frames_list:
        logger.error("Nenhum frame extraido; abortando")
        return

    logger.info("Aplicando pre-filtro de movimento nas zonas...")
    filtered = filter_frames_with_motion(frames_list, detection_zone=zones)
    logger.info(
        f"Pre-filtro: {len(filtered)}/{len(frames_list)} frames preservados "
        f"({100*len(filtered)/len(frames_list):.1f}%)"
    )

    if not filtered:
        logger.warning("Nenhum frame passou no pre-filtro; nada para enviar")
        return

    logger.info("Carregando YOLO+SAHI (pode demorar no primeiro load)...")
    detector = AnimalDetector(confidence=0.2)

    bot = Bot(token=token)
    total_sent = 0
    total_detections = 0

    for idx, (ts_sec, frame, motion_score) in enumerate(filtered, 1):
        frame_time = SEGMENT_START + timedelta(seconds=ts_sec)
        logger.info(
            f"[{idx}/{len(filtered)}] ts={frame_time.strftime('%H:%M:%S')} "
            f"score={motion_score} -> YOLO"
        )

        try:
            detections = detector.detect(frame)
        except Exception as e:
            logger.error(f"YOLO falhou: {e}")
            continue

        if not detections:
            logger.info(f"  nenhuma deteccao (sem animal)")
            continue

        total_detections += len(detections)
        for det in detections:
            class_name = det["class_name"]
            conf = det["confidence"]
            logger.info(f"  ACHOU: {class_name} ({conf:.0%})")

            image_path = dataset.save_frame(frame, CAM_ID, conf, class_name)
            crop_path = None
            if det.get("crop") is not None and det["crop"].size > 0:
                crop_path = dataset.save_crop(
                    det["crop"], CAM_ID, conf, class_name
                )

            annotated_path = image_path.replace(".jpg", "_annotated.jpg")
            annotated = detector.draw_detections(frame, [det])
            cv2.imwrite(annotated_path, annotated)

            detection_id = database.save_detection(
                camera_id=CAM_ID,
                camera_name=CAM_NAME,
                model_class=class_name,
                model_confidence=conf,
                image_path=image_path,
                crop_path=crop_path,
                bbox=det.get("bbox"),
            )

            try:
                msg_id = await send_detection_alert(
                    bot=bot,
                    chat_id=chat_id,
                    detection_id=detection_id,
                    image_path=annotated_path,
                    camera_name=CAM_NAME,
                    model_class=class_name,
                    confidence=conf,
                    timestamp=frame_time.strftime("%d/%m/%Y %H:%M:%S"),
                )
                database.update_telegram_msg_id(detection_id, msg_id)
                total_sent += 1
                logger.info(f"  Telegram OK (msg_id={msg_id})")
            except Exception as e:
                logger.error(f"  Telegram FALHOU: {e}")

            await asyncio.sleep(1)

    logger.info("=" * 60)
    logger.info(
        f"FIM: {len(filtered)} frames analisados | "
        f"{total_detections} deteccoes | {total_sent} enviadas ao Telegram"
    )


if __name__ == "__main__":
    asyncio.run(main())
