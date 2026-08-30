"""
Ferramenta visual para calibrar zonas de detecção em câmeras.
Permite desenhar retângulos nas áreas onde a detecção de movimento é desejada.

Uso:
    python -m src.tools.zone_calibrator --camera cam_frente

Controles:
    - Clique e arraste para desenhar zona
    - ENTER para confirmar e próxima câmera
    - ESC para cancelar e próxima câmera
    - 'c' para limpar desenho atual
    - 'q' para sair
"""

import cv2
import argparse
import logging
import yaml
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ZoneCalibrator:
    """Interface visual para calibrar zonas de detecção."""

    def __init__(self, config_path: str = "config/cameras.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.dvrs_config = self.config.get("dvrs", {})

    def _load_config(self) -> dict:
        """Carrega configuração de câmeras."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config não encontrada: {self.config_path}")
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _save_config(self):
        """Salva configuração atualizada."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Configuração salva: {self.config_path}")

    def calibrate_camera(self, camera_id: Optional[str] = None) -> bool:
        """
        Interface visual para calibrar uma câmera.

        Args:
            camera_id: ID da câmera (se None, lista todas)

        Returns:
            True se calibração foi salva, False se cancelada
        """
        # Procura câmera
        camera_config = None
        camera_idx = None
        for i, cam in enumerate(self.config.get("cameras", [])):
            if cam["id"] == camera_id or camera_id is None:
                camera_config = cam
                camera_idx = i
                break

        if not camera_config:
            logger.error(f"Câmera não encontrada: {camera_id}")
            return False

        cam_id = camera_config["id"]
        cam_name = camera_config["name"]
        dvr_id = camera_config.get("dvr")
        channel = camera_config.get("channel")

        if not dvr_id or dvr_id not in self.dvrs_config:
            logger.error(f"DVR não configurado para {cam_id}")
            return False

        dvr_cfg = self.dvrs_config[dvr_id]

        logger.info(f"\nCalibr ando: {cam_name} (ch{channel})")
        logger.info(f"Conectando ao stream RTSP...")

        # Constrói URL RTSP
        rtsp_url = (
            f"rtsp://{dvr_cfg['username']}:{dvr_cfg['password']}"
            f"@{dvr_cfg['host']}:{dvr_cfg['port']}"
            f"/cam/realmonitor?channel={channel}&subtype={dvr_cfg.get('subtype', 1)}"
        )

        # Abre stream
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.error(f"Falha ao conectar: {cam_name}")
            return False

        # Lê um frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.error(f"Falha ao capturar frame: {cam_name}")
            return False

        logger.info(f"Frame capturado: {frame.shape}")

        # Interface de desenho
        zones = []
        drawing = False
        start_pt = None

        def on_mouse(event, x, y, flags, param):
            nonlocal drawing, start_pt, zones

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_pt = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                # Preview do retângulo
                preview = frame.copy()
                h, w = frame.shape[:2]
                cv2.rectangle(preview, start_pt, (x, y), (0, 255, 0), 2)
                for zone in zones:
                    pt1 = (int(zone[0]*w), int(zone[1]*h))
                    pt2 = (int(zone[2]*w), int(zone[3]*h))
                    cv2.rectangle(preview, pt1, pt2, (0, 255, 0), 2)
                cv2.imshow(f"{cam_name} - Calibrador", preview)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                if start_pt and (x, y) != start_pt:
                    # Normaliza coordenadas (0-1)
                    h, w = frame.shape[:2]
                    x1, y1 = start_pt
                    x2, y2 = x, y
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1

                    zone = [x1/w, y1/h, x2/w, y2/h]
                    zones.append(zone)
                    logger.info(f"Zona {len(zones)} adicionada: {[f'{v:.2f}' for v in zone]}")

                    # Redesenha
                    preview = frame.copy()
                    for zone in zones:
                        pt1 = (int(zone[0]*w), int(zone[1]*h))
                        pt2 = (int(zone[2]*w), int(zone[3]*h))
                        cv2.rectangle(preview, pt1, pt2, (0, 255, 0), 2)
                    cv2.imshow(f"{cam_name} - Calibrador", preview)

        # Mostra frame e aguarda interação
        cv2.namedWindow(f"{cam_name} - Calibrador", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{cam_name} - Calibrador", 1024, 768)
        cv2.setMouseCallback(f"{cam_name} - Calibrador", on_mouse)

        display = frame.copy()
        cv2.imshow(f"{cam_name} - Calibrador", display)

        logger.info("\nControles:")
        logger.info("  Clique e arraste para desenhar zona")
        logger.info("  ENTER para confirmar")
        logger.info("  'c' para limpar")
        logger.info("  ESC para cancelar")
        logger.info("  'q' para sair\n")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Sai completamente
                cv2.destroyAllWindows()
                return False
            elif key == 27:  # ESC
                # Próxima câmera
                logger.info("Cancelado.")
                cv2.destroyAllWindows()
                return False
            elif key == ord('c'):
                # Limpa zonas
                zones = []
                cv2.imshow(f"{cam_name} - Calibrador", frame.copy())
                logger.info("Zonas limpas.")
            elif key == 13:  # ENTER
                # Salva zonas
                break

        cv2.destroyAllWindows()

        if zones:
            self.config["cameras"][camera_idx]["detection_zone"] = zones
            self._save_config()
            logger.info(f"✓ {len(zones)} zona(s) salva(s) para {cam_name}")
            return True
        else:
            logger.info(f"Nenhuma zona definida para {cam_name}")
            return False

    def calibrate_all(self):
        """Calibra todas as câmeras."""
        cameras = self.config.get("cameras", [])
        logger.info(f"\nCalibração de {len(cameras)} câmera(s)\n")

        for cam in cameras:
            if not cam.get("enabled", True):
                logger.info(f"⊘ {cam['name']} desabilitada, pulando")
                continue

            self.calibrate_camera(cam["id"])


def main():
    parser = argparse.ArgumentParser(description="Calibrador de zonas de detecção")
    parser.add_argument("--camera", "-c", help="ID da câmera (se omitido, calibra todas)")
    parser.add_argument("--config", default="config/cameras.yaml", help="Arquivo de config")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    calibrator = ZoneCalibrator(args.config)

    if args.camera:
        calibrator.calibrate_camera(args.camera)
    else:
        calibrator.calibrate_all()


if __name__ == "__main__":
    main()
