"""
Captura de frames via RTSP das câmeras Intelbras.
Cada câmera roda em sua própria thread para não bloquear o sistema.
"""

import cv2
import time
import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    id: str
    name: str
    channel: int
    enabled: bool = True
    motion_threshold: int = 5000
    detection_zone: list = None


@dataclass
class CameraState:
    """Estado atual de uma câmera."""
    connected: bool = False
    paused: bool = False
    last_frame_time: float = 0
    total_frames: int = 0
    errors: int = 0
    last_error: str = ""


class CameraStream:
    """Gerencia a conexão RTSP com uma câmera Intelbras."""

    def __init__(self, config: CameraConfig, dvr_config: dict):
        self.config = config
        self.dvr = dvr_config
        self.state = CameraState()
        self._cap = None
        self._lock = threading.Lock()
        self._running = False

    @property
    def rtsp_url(self) -> str:
        return (
            f"rtsp://{self.dvr['username']}:{self.dvr['password']}"
            f"@{self.dvr['host']}:{self.dvr['port']}"
            f"/cam/realmonitor?channel={self.config.channel}"
            f"&subtype={self.dvr.get('subtype', 1)}"
        )

    def connect(self) -> bool:
        """Conecta à câmera via RTSP."""
        try:
            self._cap = cv2.VideoCapture(self.rtsp_url)
            # Configurações para reduzir latência
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FPS, 5)

            if self._cap.isOpened():
                self.state.connected = True
                logger.info(f"[{self.config.name}] Conectado via RTSP")
                return True
            else:
                self.state.connected = False
                self.state.last_error = "Falha ao abrir stream"
                logger.error(f"[{self.config.name}] Falha na conexão RTSP")
                return False
        except Exception as e:
            self.state.connected = False
            self.state.last_error = str(e)
            self.state.errors += 1
            logger.error(f"[{self.config.name}] Erro: {e}")
            return False

    def read_frame(self):
        """Lê o frame mais recente da câmera."""
        if not self._cap or not self._cap.isOpened():
            return None

        try:
            ret, frame = self._cap.read()
            if ret:
                self.state.last_frame_time = time.time()
                self.state.total_frames += 1
                return frame
            else:
                self.state.errors += 1
                return None
        except Exception as e:
            self.state.last_error = str(e)
            self.state.errors += 1
            return None

    def start_capture(self, frame_callback):
        """Inicia captura contínua em thread separada."""
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            args=(frame_callback,),
            daemon=True,
            name=f"cam-{self.config.id}"
        )
        self._thread.start()
        logger.info(f"[{self.config.name}] Thread de captura iniciada")

    def _capture_loop(self, frame_callback):
        """Loop principal de captura."""
        reconnect_delay = 5

        while self._running:
            if self.state.paused:
                time.sleep(1)
                continue

            if not self.state.connected:
                if not self.connect():
                    logger.warning(
                        f"[{self.config.name}] Reconectando em {reconnect_delay}s..."
                    )
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 60)
                    continue
                reconnect_delay = 5
                self.state.errors = 0  # Reseta contador de erros após reconexão bem-sucedida

            frame = self.read_frame()
            if frame is not None:
                frame_callback(self.config, frame)
            else:
                # Frame falhou, tentar reconectar
                self.state.connected = False
                self.disconnect()
                time.sleep(1)

    def disconnect(self):
        """Desconecta da câmera."""
        if self._cap:
            self._cap.release()
            self._cap = None
        self.state.connected = False

    def stop(self):
        """Para a captura."""
        self._running = False
        self.disconnect()
        logger.info(f"[{self.config.name}] Captura parada")

    def pause(self):
        self.state.paused = True
        logger.info(f"[{self.config.name}] Pausada")

    def resume(self):
        self.state.paused = False
        logger.info(f"[{self.config.name}] Retomada")
