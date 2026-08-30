"""
Pipeline principal: orquestra câmeras → movimento → YOLO → notificação.
Usa fila (Queue) para desacoplar captura da inferência.
"""

import time
import logging
import asyncio
import threading
from queue import Empty, Full, Queue
from datetime import datetime, timedelta
from pathlib import Path

import tempfile
import cv2
import numpy as np
import yaml

from .camera.stream import CameraStream, CameraConfig
from .camera.motion import MotionDetector, filter_frames_with_motion
from .detection.detector import AnimalDetector
from .storage import database, dataset
from .bot.notifications import send_detection_alert
from .dvr.dahua_api import DahuaHTTPClient, RecordedFile, format_bytes
from .dvr.event_poller import DVREventPoller
from .dvr.offline_processor import frames_from_video
from .dvr.rtsp_opencv_extractor import extract_frames_from_rtsp_playback
from .dvr.state import DVRPollerState

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """
    Pipeline principal do sistema.

    Fluxo:
    1. Cada câmera captura frames em thread própria
    2. Detector de movimento filtra frames sem atividade
    3. Frames com movimento vão para fila de inferência
    4. Workers YOLO processam a fila
    5. Detecções geram notificações no Telegram
    """

    def __init__(self, config_path: str = "config/cameras.yaml"):
        self.config = self._load_config(config_path)
        self.cameras: dict[str, CameraStream] = {}
        self.motion_detectors: dict[str, MotionDetector] = {}
        self.inference_queue: Queue = Queue(maxsize=20)
        self.detector: AnimalDetector | None = None
        self._cooldowns: dict[str, float] = {}
        self._notification_cooldowns: dict[str, float] = {}
        self._last_frame_time: dict[str, float] = {}
        self._lock = threading.Lock()
        self._running = False

        # DVR polling (mode: motion_source = dvr)
        self._pollers: list[DVREventPoller] = []
        self._dvr_clients: dict[str, DahuaHTTPClient] = {}
        # Mapa (dvr_name, channel) -> camera_id  — preenchido no setup()
        self._dvr_channel_to_cam: dict[tuple[str, int], str] = {}
        # Mapa cam_id -> cam_cfg dict — evita lookups lineares em _on_dvr_event
        self._cam_configs: dict[str, dict] = {}
        self._dvr_state = DVRPollerState()

        # Telegram bot reference (set by main.py)
        self.bot = None
        self.chat_id = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def _load_config(self, path: str) -> dict:
        """Carrega configuração das câmeras."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Arquivo de configuração não encontrado: {path}\n"
                f"Copie cameras.example.yaml para cameras.yaml e edite."
            )

        with open(config_path) as f:
            return yaml.safe_load(f)

    def _evict_stale_entries(
        self,
        mapping: dict[str, float],
        now: float,
        max_age: float = 3600.0,
    ) -> None:
        """Remove entradas mais antigas que max_age (segundos) do dict.

        Evita crescimento ilimitado de _cooldowns / _last_frame_time quando
        câmeras aparecem e somem ao longo de runs longos.
        """
        stale = [k for k, t in mapping.items() if now - t > max_age]
        for k in stale:
            mapping.pop(k, None)

    def setup(self) -> None:
        """Inicializa todos os componentes."""
        logger.info("Inicializando pipeline...")

        # Banco de dados
        database.init_db()
        dataset.init_dirs()

        # Detector YOLO
        conf = self.config.get("detection", {}).get("confidence_threshold", 0.45)
        self.detector = AnimalDetector(confidence=conf)

        # Câmeras — suporta múltiplos DVRs
        dvrs = self.config.get("dvrs", {})
        # Retrocompatibilidade: campo "dvr" único do formato antigo
        legacy_dvr = self.config.get("dvr", {})

        for cam_cfg in self.config.get("cameras", []):
            if not cam_cfg.get("enabled", True):
                continue

            # Resolve qual DVR esta câmera usa
            dvr_id = cam_cfg.get("dvr")
            if dvr_id and dvr_id in dvrs:
                dvr_config = dvrs[dvr_id]
            else:
                dvr_config = legacy_dvr

            cam = CameraConfig(
                id=cam_cfg["id"],
                name=cam_cfg["name"],
                channel=cam_cfg["channel"],
                enabled=cam_cfg.get("enabled", True),
                motion_threshold=cam_cfg.get("motion_threshold", 5000),
                detection_zone=cam_cfg.get("detection_zone"),
            )
            stream = CameraStream(cam, dvr_config)
            self.cameras[cam.id] = stream

            # Um detector de movimento por câmera
            self.motion_detectors[cam.id] = MotionDetector(
                threshold=cam.motion_threshold,
                detection_zone=cam.detection_zone,
            )

        logger.info(f"{len(self.cameras)} câmeras configuradas")

        # Inicializa clientes HTTP do DVR se modo dvr
        motion_source = self.config.get("motion_source", "mog2")
        if motion_source == "dvr":
            self._setup_dvr_clients(dvrs)

    def _setup_dvr_clients(self, dvrs: dict) -> None:
        """
        Instancia DahuaHTTPClient para cada DVR e constrói o mapa
        (dvr_name, channel) → camera_id a partir das câmeras configuradas.
        """
        for dvr_id, dvr_cfg in dvrs.items():
            http_port = dvr_cfg.get("http_port", 80)
            http_user = dvr_cfg.get("http_username") or dvr_cfg.get("username", "admin")
            http_pass = dvr_cfg.get("http_password") or dvr_cfg.get("password", "")

            client = DahuaHTTPClient(
                dvr_ip=dvr_cfg["host"],
                username=http_user,
                password=http_pass,
                dvr_name=dvr_id,
                port=http_port,
                rpc_username=dvr_cfg.get("rpc_username"),
                rpc_password=dvr_cfg.get("rpc_password"),
            )
            self._dvr_clients[dvr_id] = client

        # Mapeamento (dvr_name, channel) → camera_id e cam_id → cam_cfg
        for cam_cfg in self.config.get("cameras", []):
            if not cam_cfg.get("enabled", True):
                continue
            dvr_id = cam_cfg.get("dvr")
            channel = cam_cfg.get("channel")
            cam_id = cam_cfg.get("id")
            if dvr_id and channel and cam_id:
                self._dvr_channel_to_cam[(dvr_id, int(channel))] = cam_id
                self._cam_configs[cam_id] = cam_cfg

        logger.info(
            f"Clientes HTTP DVR: {list(self._dvr_clients.keys())} | "
            f"Mapeamentos canal→câmera: {len(self._dvr_channel_to_cam)}"
        )

    def start(self) -> None:
        """Inicia captura de todas as câmeras e workers de inferência."""
        self._running = True

        self._frame_interval = self.config.get("detection", {}).get("frame_interval", 2)

        motion_source = self.config.get("motion_source", "mog2")

        if motion_source == "dvr":
            self._start_dvr_pollers()
        else:
            for cam_id, stream in self.cameras.items():
                stream.start_capture(
                    lambda config, frame, cid=cam_id: self._on_frame(cid, config, frame)
                )

        # Inicia workers de inferência (2 threads)
        for i in range(2):
            t = threading.Thread(
                target=self._inference_worker,
                daemon=True,
                name=f"inference-{i}",
            )
            t.start()

        logger.info("Pipeline iniciado!")

    def _start_dvr_pollers(self) -> None:
        """Cria e inicia um DVREventPoller por DVR configurado."""
        dvrs = self.config.get("dvrs", {})
        detection_cfg = self.config.get("detection", {})

        # Prefer the documented location under `detection`, but keep a
        # compatibility fallback for configs that placed it under `notification`.
        force_reprocess_str = detection_cfg.get("force_reprocess_since")
        if not force_reprocess_str:
            notification_cfg = self.config.get("notification", {})
            force_reprocess_str = notification_cfg.get("force_reprocess_since")
            if force_reprocess_str:
                logger.warning(
                    "force_reprocess_since found under 'notification'; "
                    "move it to 'detection'"
                )
        force_reprocess_since = None
        if force_reprocess_str:
            try:
                # Aceita formatos: "2026-04-14", "2026-04-14 10:30:00", "today", "yesterday"
                if force_reprocess_str.lower() == "today":
                    force_reprocess_since = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                elif force_reprocess_str.lower() == "yesterday":
                    force_reprocess_since = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    # Tenta parsear como data/datetime
                    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                        try:
                            force_reprocess_since = datetime.strptime(force_reprocess_str, fmt)
                            break
                        except ValueError:
                            continue
                if force_reprocess_since:
                    logger.info(f"Modo reprocessamento ativado: desde {force_reprocess_since}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Não foi possível parsear force_reprocess_since '{force_reprocess_str}': {e}")

        for dvr_id, dvr_cfg in dvrs.items():
            client = self._dvr_clients.get(dvr_id)
            if not client:
                continue

            # Coleta todos os canais que pertencem a este DVR
            channels = [
                ch for (did, ch) in self._dvr_channel_to_cam
                if did == dvr_id
            ]
            if not channels:
                logger.warning(f"DVR {dvr_id}: nenhum canal mapeado, pulando poller")
                continue

            poll_interval = dvr_cfg.get("poll_interval", 30)

            poller = DVREventPoller(
                client=client,
                channels=sorted(channels),
                dvr_name=dvr_id,
                on_event=self._on_dvr_event,
                poll_interval=poll_interval,
                state=self._dvr_state,
                force_reprocess_since=force_reprocess_since,
            )
            poller.start()
            self._pollers.append(poller)

    def _on_dvr_event(self, rec: RecordedFile) -> bool:
        """
        Callback chamado pelo DVREventPoller para cada segmento novo.
        Baixa o arquivo .dav e extrai frames para YOLO.

        Returns:
            True se o evento foi tratado ou descartado definitivamente;
            False se houve falha transitÃ³ria e o poller deve tentar de novo.
        """
        dvr_name = rec.dvr_name
        channel = rec.channel
        cam_id = self._dvr_channel_to_cam.get((dvr_name, channel))

        if not cam_id:
            logger.warning(
                f"DVR {dvr_name} ch{channel}: sem câmera mapeada, ignorando evento"
            )
            return True

        cam_cfg_entry = self._cam_configs.get(cam_id, {})
        cam_name = cam_cfg_entry.get("name", cam_id)

        # Verifica se o DVR está configurado e se o cliente HTTP existe
        dvrs = self.config.get("dvrs", {})
        if dvr_name not in dvrs:
            logger.warning(f"[{cam_id}] DVR {dvr_name} não configurado")
            return True
        client = self._dvr_clients.get(dvr_name)
        if not client:
            logger.warning(f"[{cam_id}] Cliente HTTP do DVR {dvr_name} não encontrado")
            return True

        logger.info(f"[{cam_id}] Novo evento DVR: {rec}")

        # Baixa o segmento .dav localmente e extrai frames do arquivo.
        frame_interval = self.config.get("detection", {}).get("frame_interval", 2)
        download_dir = Path("detections/dvr_downloads")
        download_dir.mkdir(parents=True, exist_ok=True)

        temp_handle = tempfile.NamedTemporaryFile(
            suffix=".dav",
            prefix=f"{cam_id}_",
            dir=download_dir,
            delete=False,
        )
        temp_file = Path(temp_handle.name)
        temp_handle.close()

        frames_list = []
        fallback_reason = None
        try:
            logger.info(f"[{cam_id}] Iniciando download do .dav: {temp_file.name}")
            download = client.download_file_detailed(
                rec.file_path,
                temp_file,
                expected_size=rec.length_bytes or None,
            )
            if download.ok:
                frames_list = list(
                    frames_from_video(temp_file, frame_interval_sec=frame_interval)
                )
                if not frames_list:
                    fallback_reason = "download_ok_no_frames"
            else:
                fallback_reason = f"download_failed:{download.reason}"
                logger.warning(
                    f"[{cam_id}] Download HTTP falhou "
                    f"({download.reason}, escrito={format_bytes(download.bytes_written)}, "
                    f"esperado={format_bytes(download.expected_bytes)}); "
                    "caindo para RTSP playback"
                )
        except Exception as e:
            fallback_reason = "video_decode_failed"
            logger.error(f"[{cam_id}] Erro ao processar arquivo DVR local: {e}")
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(
                        f"[{cam_id}] Não foi possível remover temporário "
                        f"{temp_file.name}: {e}"
                    )

        if not frames_list:
            dvr_cfg = dvrs[dvr_name]
            logger.info(
                f"[{cam_id}] Tentando RTSP playback ch{rec.channel} "
                f"{rec.start_time.strftime('%H:%M:%S')}-{rec.end_time.strftime('%H:%M:%S')} "
                f"(motivo={fallback_reason or 'no_local_frames'})"
            )
            try:
                frames_list = extract_frames_from_rtsp_playback(
                    host=dvr_cfg["host"],
                    port=dvr_cfg.get("port", 554),
                    username=dvr_cfg["username"],
                    password=dvr_cfg["password"],
                    channel=rec.channel,
                    start_time=rec.start_time,
                    end_time=rec.end_time,
                    duration_seconds=rec.duration_seconds,
                    frame_interval=frame_interval,
                    subtype=0,
                    timeout_seconds=int(rec.duration_seconds) + 30,
                )
            except Exception as e:
                logger.error(f"[{cam_id}] RTSP playback falhou: {e}")
                return False

        if not frames_list:
            logger.warning(f"[{cam_id}] Nenhum frame extraído do arquivo DVR")
            return False

        logger.info(f"[{cam_id}] {len(frames_list)} frame(s) extraído(s)")

        # Pré-filtro de movimento: só mantém frames com atividade dentro das
        # zonas calibradas — reduz drasticamente o trabalho do YOLO+SAHI.
        detection_zone = self._cam_configs.get(cam_id, {}).get("detection_zone")

        debug_zero_motion = self.config.get("detection", {}).get(
            "debug_zero_motion", False
        )
        filtered = filter_frames_with_motion(
            frames_list,
            detection_zone=detection_zone,
            debug_dir=Path("detections/analysis/zero_motion")
            if debug_zero_motion
            else None,
            debug_prefix=(
                f"{cam_id}_{rec.start_time.strftime('%Y%m%d_%H%M%S')}"
            ),
        )

        if not filtered:
            logger.info(f"[{cam_id}] Nenhum frame com movimento na zona; ignorando segmento")
            return True

        # Enfileira frames para YOLO
        frames_enqueued = 0
        for ts_sec, frame, motion_score in filtered:
            frame_time = rec.start_time + timedelta(seconds=ts_sec)
            try:
                self.inference_queue.put_nowait({
                    "frame": frame.copy(),
                    "camera_id": cam_id,
                    "camera_name": cam_name,
                    "timestamp": frame_time,
                    "motion_score": motion_score,
                    "bounding_boxes": [],
                })
                frames_enqueued += 1
            except Full:
                logger.warning(f"[{cam_id}] Fila de inferência cheia — frame DVR descartado")

        logger.info(f"[{cam_id}] {frames_enqueued} frame(s) enfileirado(s) do evento DVR")
        return True

    def _on_frame(self, cam_id: str, config: CameraConfig, frame: np.ndarray) -> None:
        """
        Callback chamado por cada câmera a cada frame.
        Aplica detecção de movimento e enfileira se necessário.
        """
        # Throttle: só processa 1 frame a cada N segundos por câmera
        now = time.time()
        with self._lock:
            if now - self._last_frame_time.get(cam_id, 0) < self._frame_interval:
                return
            self._last_frame_time[cam_id] = now
            self._evict_stale_entries(self._last_frame_time, now)

        motion_det = self.motion_detectors.get(cam_id)
        if not motion_det:
            return

        result = motion_det.detect(frame)

        if result["score"] > 0:
            logger.debug(
                f"[{cam_id}] Movimento: score={result['score']} "
                f"threshold={motion_det.threshold}"
            )

        if result["motion"]:
            logger.info(
                f"[{cam_id}] Movimento detectado! score={result['score']}"
            )
            # Checa cooldown (evita spam de notificações)
            cooldown = self.config.get("detection", {}).get("cooldown_seconds", 30)
            with self._lock:
                last_time = self._cooldowns.get(cam_id, 0)
                if now - last_time < cooldown:
                    logger.debug(f"[{cam_id}] Cooldown ativo, ignorando")
                    return
                self._cooldowns[cam_id] = now
                self._evict_stale_entries(self._cooldowns, now)

            # Enfileira para inferência YOLO
            try:
                self.inference_queue.put_nowait({
                    "frame": frame.copy(),
                    "camera_id": cam_id,
                    "camera_name": config.name,
                    "timestamp": datetime.now(),
                    "motion_score": result["score"],
                    "bounding_boxes": result["bounding_boxes"],
                })
            except Full:
                logger.warning(f"[{cam_id}] Fila de inferência cheia — frame descartado")

    def _inference_worker(self) -> None:
        """Worker que processa frames da fila com YOLO."""
        logger.info(f"Worker de inferência iniciado: {threading.current_thread().name}")

        while self._running:
            try:
                item = self.inference_queue.get(timeout=1)
            except Empty:
                continue

            frame = item["frame"]
            cam_id = item["camera_id"]
            cam_name = item["camera_name"]
            timestamp = item["timestamp"]

            # Roda YOLO com proteção contra falhas
            try:
                detections = self.detector.detect(frame)
            except Exception as e:
                logger.error(f"Erro na inferência YOLO (câmera {cam_id}): {e}")
                continue

            if not detections:
                continue

            for det in detections:
                self._handle_detection(
                    frame=frame,
                    detection=det,
                    camera_id=cam_id,
                    camera_name=cam_name,
                    timestamp=timestamp,
                )

    def _handle_detection(
        self,
        frame: np.ndarray,
        detection: dict,
        camera_id: str,
        camera_name: str,
        timestamp: datetime,
    ) -> None:
        """Processa uma detecção: salva, loga e notifica."""
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        # Salva frame completo e crop
        image_path = dataset.save_frame(
            frame, camera_id, confidence, class_name
        )
        crop_path = None
        if detection.get("crop") is not None and detection["crop"].size > 0:
            crop_path = dataset.save_crop(
                detection["crop"], camera_id, confidence, class_name
            )

        # Desenha detecção no frame para a notificação
        annotated_path = image_path.replace(".jpg", "_annotated.jpg")
        annotated = self.detector.draw_detections(frame, [detection])
        cv2.imwrite(annotated_path, annotated)

        # Salva no banco
        detection_id = database.save_detection(
            camera_id=camera_id,
            camera_name=camera_name,
            model_class=class_name,
            model_confidence=confidence,
            image_path=image_path,
            crop_path=crop_path,
            bbox=detection.get("bbox"),
        )

        logger.info(
            f"Detecção #{detection_id}: {class_name} ({confidence:.0%}) "
            f"em {camera_name}"
        )

        # Notifica via Telegram (async), com cooldown apenas depois de deteccao real.
        should_notify = True
        cooldown = self.config.get("detection", {}).get("cooldown_seconds", 30)
        with self._lock:
            now = time.time()
            last_time = self._notification_cooldowns.get(camera_id, 0)
            if now - last_time < cooldown:
                should_notify = False
            else:
                self._notification_cooldowns[camera_id] = now
                self._evict_stale_entries(self._notification_cooldowns, now)

        if should_notify and self.bot and self.chat_id and self._event_loop is not None:
            asyncio.run_coroutine_threadsafe(
                self._send_telegram_alert(
                    detection_id=detection_id,
                    image_path=annotated_path,
                    camera_name=camera_name,
                    model_class=class_name,
                    confidence=confidence,
                    timestamp=timestamp,
                ),
                self._event_loop,
            )
        elif not should_notify:
            logger.info(
                f"Notificacao em cooldown para {camera_name}; "
                f"deteccao #{detection_id} salva sem alerta"
            )

    async def _send_telegram_alert(
        self,
        detection_id: int,
        image_path: str,
        camera_name: str,
        model_class: str,
        confidence: float,
        timestamp: datetime,
    ) -> None:
        """Envia alerta no Telegram com retry."""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                msg_id = await send_detection_alert(
                    bot=self.bot,
                    chat_id=self.chat_id,
                    detection_id=detection_id,
                    image_path=image_path,
                    camera_name=camera_name,
                    model_class=model_class,
                    confidence=confidence,
                    timestamp=timestamp.strftime("%d/%m/%Y %H:%M:%S"),
                )
                database.update_telegram_msg_id(detection_id, msg_id)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Erro ao enviar alerta (tentativa {attempt+1}/{max_retries}): {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Falha ao enviar alerta Telegram após {max_retries} tentativas: {e}")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Define o event loop do asyncio (chamado pelo main)."""
        self._event_loop = loop

    def stop(self) -> None:
        """Para o pipeline."""
        self._running = False
        for stream in self.cameras.values():
            stream.stop()
        for poller in self._pollers:
            poller.stop()
        logger.info("Pipeline parado")
