"""
Extrator de frames via RTSP usando OpenCV diretamente.
Suporta dois modos:
  - Stream ao vivo: rtsp://...?channel=N&subtype=1
  - Playback de gravação: rtsp://...?channel=N&subtype=0&starttime=YYYYMMDDTHHMMSSz
"""

import logging
import re
import threading
import cv2
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


def _redact(url: str) -> str:
    """Remove senha da URL RTSP para uso seguro em logs."""
    return re.sub(r"(rtsp://[^:]+):[^@]+@", r"\1:***@", url)


def get_rtsp_url(host: str, port: int, username: str, password: str, channel: int, subtype: int = 1) -> str:
    """Constrói URL RTSP para stream ao vivo."""
    return f"rtsp://{username}:{password}@{host}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"


def get_rtsp_playback_url(
    host: str, port: int, username: str, password: str,
    channel: int, start_time: datetime,
    end_time: datetime | None = None, subtype: int = 0
) -> str:
    """
    Constrói URL RTSP para playback de gravação por timestamp.

    Formato validado empiricamente no firmware MHDX 4.x:
        /cam/playback?channel=N&starttime=YYYY_MM_DD_HH_MM_SS&endtime=...
    Passar end_time torna o playback auto-encerrável no fim do segmento.
    """
    start_str = start_time.strftime("%Y_%m_%d_%H_%M_%S")
    url = (
        f"rtsp://{username}:{password}@{host}:{port}"
        f"/cam/playback?channel={channel}&starttime={start_str}"
    )
    if end_time is not None:
        end_str = end_time.strftime("%Y_%m_%d_%H_%M_%S")
        url += f"&endtime={end_str}"
    return url


def extract_frames_from_rtsp_playback(
    host: str,
    port: int,
    username: str,
    password: str,
    channel: int,
    start_time: datetime,
    duration_seconds: float,
    frame_interval: float = 2.0,
    subtype: int = 0,
    timeout_seconds: int = 60,
    end_time: datetime | None = None,
) -> list[tuple[float, np.ndarray]]:
    """
    Extrai frames de uma gravação do DVR via RTSP playback por timestamp.

    O DVR Dahua/Intelbras suporta reprodução de gravações com:
        rtsp://...?channel=N&subtype=0&starttime=YYYYMMDDTHHMMSSz

    Args:
        host, port, username, password, channel: credenciais DVR
        start_time: início da gravação
        duration_seconds: duração total da gravação
        frame_interval: intervalo entre frames a extrair (segundos)
        subtype: 0=main stream, 1=sub stream
        timeout_seconds: timeout total da operação

    Returns:
        Lista de (timestamp_relativo_segundos, frame_numpy)
    """
    rtsp_url = get_rtsp_playback_url(
        host, port, username, password, channel, start_time,
        end_time=end_time, subtype=subtype,
    )
    num_frames = max(1, int(duration_seconds / frame_interval))

    logger.info(
        f"RTSP playback {_redact(rtsp_url)} ch{channel} desde {start_time.strftime('%H:%M:%S')} "
        f"(dur={duration_seconds:.0f}s, ~{num_frames} frames, interval={frame_interval}s)"
    )

    frames: list[tuple[float, np.ndarray]] = []
    error_msg = None
    cap = None

    def _worker():
        nonlocal cap, frames, error_msg
        try:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            if not cap.isOpened():
                error_msg = "Nao foi possivel abrir stream RTSP playback"
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = 25.0
            logger.debug(f"Stream playback: FPS={fps}")

            frame_skip = max(1, int(fps * frame_interval))
            frame_count = 0
            extracted = 0

            while extracted < num_frames:
                try:
                    ret, frame = cap.read()
                except Exception:
                    # OpenCV lança exceção C++ ao atingir o fim da gravação — normal
                    logger.debug("Stream playback encerrado (fim da gravacao)")
                    break

                if not ret:
                    logger.debug(f"Stream playback encerrado no frame {frame_count}")
                    break

                if frame_count % frame_skip == 0:
                    ts_sec = frame_count / fps
                    frames.append((ts_sec, frame.copy()))
                    extracted += 1
                    logger.debug(f"Frame {extracted}/{num_frames} em t={ts_sec:.1f}s")

                frame_count += 1

        except Exception as e:
            error_msg = _redact(str(e))
            logger.error(f"Erro no worker RTSP playback: {error_msg}")
        finally:
            if cap:
                cap.release()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        logger.warning(f"Timeout RTSP playback (>{timeout_seconds}s), usando {len(frames)} frames obtidos")
        if cap:
            try:
                cap.release()
            except Exception:
                pass

    if error_msg:
        logger.error(f"Erro RTSP playback: {_redact(error_msg)}")

    logger.info(f"RTSP playback: {len(frames)} frame(s) extraido(s)")
    return frames
