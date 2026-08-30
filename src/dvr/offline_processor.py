"""
Processador offline de videos gravados nos DVRs Intelbras.
Consulta os periodos com movimento no DVR, baixa e processa com YOLO+SAHI.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import yaml

from src.dvr.dahua_api import DahuaHTTPClient, RecordedFile
from src.dvr.rtsp_opencv_extractor import extract_frames_from_rtsp_playback

logger = logging.getLogger(__name__)

DOWNLOAD_DIR = Path("detections/dvr_downloads")


def load_dvr_configs(config_path: str = "config/cameras.yaml") -> list[dict]:
    """Carrega DVRs do mesmo YAML usado pelo pipeline principal."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {config_path}")

    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    dvrs = config.get("dvrs", {})
    cameras = config.get("cameras", [])
    dvr_configs = []

    for dvr_id, dvr_cfg in dvrs.items():
        channels = sorted({
            int(cam["channel"])
            for cam in cameras
            if cam.get("enabled", True) and cam.get("dvr") == dvr_id and cam.get("channel") is not None
        })

        dvr_configs.append({
            "name": dvr_id,
            "ip": dvr_cfg["host"],
            "username": dvr_cfg.get("http_username") or dvr_cfg.get("username", "admin"),
            "password": dvr_cfg.get("http_password") or dvr_cfg.get("password", ""),
            "rpc_username": dvr_cfg.get("rpc_username"),
            "rpc_password": dvr_cfg.get("rpc_password"),
            "rtsp_port": dvr_cfg.get("port", 554),
            "rtsp_username": dvr_cfg.get("username", "admin"),
            "rtsp_password": dvr_cfg.get("password", ""),
            "rtsp_subtype": dvr_cfg.get("subtype", 0),
            "channels": channels or [1, 2, 3, 4, 5],
        })

    return dvr_configs


def frames_from_video(video_path: Path, frame_interval_sec: float = 2.0):
    """
    Gera frames de um arquivo de video local em intervalos regulares.

    Yields:
        (timestamp_relativo, frame_numpy)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Nao foi possivel abrir: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_skip = max(1, int(fps * frame_interval_sec))
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                timestamp_sec = frame_count / fps
                yield timestamp_sec, frame

            frame_count += 1
    finally:
        cap.release()


class DVROfflineProcessor:
    """Processa videos com movimento dos DVRs usando o detector YOLO existente."""

    def __init__(self, detector, notifier=None, db=None, frame_interval_sec: float = 2.0):
        self.detector = detector
        self.notifier = notifier
        self.db = db
        self.frame_interval_sec = frame_interval_sec
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def process_file(self, rec: RecordedFile, video_path: Path) -> int:
        detections_count = 0
        logger.info(f"Processando: {video_path.name} ({rec.duration_seconds:.0f}s)")

        for ts, frame in frames_from_video(video_path, self.frame_interval_sec):
            detections_count += self._process_frame_results(rec, ts, frame)

        return detections_count

    def process_frames_list(self, rec: RecordedFile, frames_list: list[tuple[float, object]]) -> int:
        detections_count = 0
        logger.info(f"Processando {len(frames_list)} frame(s) via RTSP playback ({rec.duration_seconds:.0f}s)")

        for ts, frame in frames_list:
            detections_count += self._process_frame_results(rec, ts, frame)

        return detections_count

    def _process_frame_results(self, rec: RecordedFile, ts: float, frame) -> int:
        detections_count = 0
        try:
            results = self.detector.detect(frame)
            if not results:
                return 0

            frame_time = rec.start_time + timedelta(seconds=ts)

            for det in results:
                detections_count += 1
                logger.info(
                    f"  [{frame_time.strftime('%H:%M:%S')}] "
                    f"Canal {rec.channel}: {det['class_name']} "
                    f"({det['confidence']:.2f}) @ DVR {rec.dvr_name}"
                )

                if self.notifier:
                    self.notifier.send_detection(
                        frame=frame,
                        detections=results,
                        camera_name=f"{rec.dvr_name} ch{rec.channel}",
                        timestamp=frame_time,
                    )

                if self.db:
                    self.db.save_detection(
                        camera=f"{rec.dvr_name}_ch{rec.channel}",
                        timestamp=frame_time,
                        detections=results,
                    )

        except Exception as e:
            logger.error(f"Erro ao processar frame em {ts:.1f}s: {e}")
            return 0

        return detections_count


    def process_dvr(
        self,
        client: DahuaHTTPClient,
        dvr_config: dict,
        channels: list[int],
        start: datetime,
        end: datetime,
        keep_downloads: bool = False,
    ) -> dict:
        stats = {
            "dvr": client.dvr_name,
            "period": f"{start} - {end}",
            "files_processed": 0,
            "detections": 0,
            "errors": 0,
        }
        rtsp_only = False

        for channel in channels:
            events = client.list_motion_events(channel=channel, start=start, end=end)
            for rec in events:
                filename = Path(rec.file_path).name
                video_path = DOWNLOAD_DIR / f"ch{channel}_{filename}"

                try:
                    processed = False

                    if not rtsp_only and client.download_file(rec.file_path, video_path):
                        n = self.process_file(rec, video_path)
                        processed = True
                        if not keep_downloads and video_path.exists():
                            video_path.unlink()
                            logger.debug(f"Removido: {video_path.name}")
                    else:
                        if not rtsp_only:
                            rtsp_only = True
                            logger.warning(
                                f"DVR {client.dvr_name}: download falhou; "
                                f"usando RTSP playback para os proximos eventos desta execucao"
                            )
                        timeout = max(25, int(rec.duration_seconds * 2))
                        frames_list = extract_frames_from_rtsp_playback(
                            host=dvr_config["ip"],
                            port=dvr_config.get("rtsp_port", 554),
                            username=dvr_config.get("rtsp_username", dvr_config["username"]),
                            password=dvr_config.get("rtsp_password", dvr_config["password"]),
                            channel=rec.channel,
                            start_time=rec.start_time,
                            duration_seconds=rec.duration_seconds,
                            frame_interval=self.frame_interval_sec,
                            subtype=dvr_config.get("rtsp_subtype", 0),
                            timeout_seconds=timeout,
                        )
                        if frames_list:
                            n = self.process_frames_list(rec, frames_list)
                            processed = True
                        else:
                            n = 0

                    if processed:
                        stats["files_processed"] += 1
                        stats["detections"] += n
                    else:
                        stats["errors"] += 1

                except Exception as e:
                    logger.error(f"Erro ao processar evento {rec.file_path}: {e}")
                    stats["errors"] += 1

        return stats


def run_offline(
    dvr_configs: list[dict],
    hours_back: int = 24,
    channels: list[int] | None = None,
    frame_interval: float = 2.0,
    keep_downloads: bool = False,
):
    """Executa o processamento offline dos DVRs."""
    from src.detection.detector import AnimalDetector

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    end = datetime.now()
    start = end - timedelta(hours=hours_back)

    logger.info(f"Iniciando processamento offline: {start} -> {end}")

    detector = AnimalDetector()
    processor = DVROfflineProcessor(detector, frame_interval_sec=frame_interval)

    total_stats = {"files": 0, "detections": 0}

    for config in dvr_configs:
        client = DahuaHTTPClient(
            dvr_ip=config["ip"],
            username=config["username"],
            password=config["password"],
            dvr_name=config.get("name", config["ip"]),
            rpc_username=config.get("rpc_username"),
            rpc_password=config.get("rpc_password"),
        )

        if not client.check_connection():
            logger.error(f"DVR {config.get('name', config['ip'])}: sem conexao, pulando")
            continue

        dvr_channels = channels or config.get("channels", [1, 2, 3, 4, 5])
        stats = processor.process_dvr(
            client=client,
            dvr_config=config,
            channels=dvr_channels,
            start=start,
            end=end,
            keep_downloads=keep_downloads,
        )

        logger.info(
            f"DVR {stats['dvr']}: "
            f"{stats['files_processed']} arquivo(s), "
            f"{stats['detections']} deteccao(oes)"
        )
        total_stats["files"] += stats["files_processed"]
        total_stats["detections"] += stats["detections"]

    logger.info(
        f"Processamento concluido: "
        f"{total_stats['files']} arquivo(s), "
        f"{total_stats['detections']} deteccao(oes) no total"
    )
    return total_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Processador offline de DVR Intelbras")
    parser.add_argument("--config", default="config/cameras.yaml", help="Arquivo YAML de configuracao")
    parser.add_argument("--hours", type=int, default=24, help="Horas para tras")
    parser.add_argument("--channels", type=int, nargs="+", help="Canais especificos")
    parser.add_argument("--frame-interval", type=float, default=2.0, help="Intervalo entre frames (s)")
    parser.add_argument("--keep-downloads", action="store_true", help="Mantem videos baixados")
    args = parser.parse_args()

    run_offline(
        dvr_configs=load_dvr_configs(args.config),
        hours_back=args.hours,
        channels=args.channels,
        frame_interval=args.frame_interval,
        keep_downloads=args.keep_downloads,
    )
