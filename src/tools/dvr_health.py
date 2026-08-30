"""
Diagnostico rapido da integracao DVR.

Uso:
  python -m src.tools.dvr_health --dvr dvr_casa --channel 1 --minutes 30
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from src.dvr.dahua_api import (
    DEFAULT_EVENT_TYPES,
    DahuaHTTPClient,
    RecordedFile,
    format_bytes,
)
from src.dvr.offline_processor import frames_from_video

logger = logging.getLogger("dvr_health")


def _load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _make_client(dvr_name: str, dvr_cfg: dict) -> DahuaHTTPClient:
    return DahuaHTTPClient(
        dvr_ip=dvr_cfg["host"],
        username=dvr_cfg.get("http_username") or dvr_cfg.get("username", "admin"),
        password=dvr_cfg.get("http_password") or dvr_cfg.get("password", ""),
        dvr_name=dvr_name,
        port=dvr_cfg.get("http_port", 80),
        rpc_username=dvr_cfg.get("rpc_username"),
        rpc_password=dvr_cfg.get("rpc_password"),
    )


def _print_event(rec: RecordedFile) -> None:
    logger.info(
        "  %s file=%s flags=%s length=%s",
        rec,
        rec.file_path,
        rec.flags or [],
        format_bytes(rec.length_bytes),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostico de DVR Intelbras/Dahua")
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--dvr", required=True)
    parser.add_argument("--channel", required=True, type=int)
    parser.add_argument("--minutes", default=30, type=int)
    parser.add_argument("--frame-interval", default=2.0, type=float)
    parser.add_argument("--keep-download", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = _load_config(args.config)
    dvrs = cfg.get("dvrs", {})
    if args.dvr not in dvrs:
        logger.error("DVR %s nao encontrado. Disponiveis: %s", args.dvr, list(dvrs))
        return 2

    client = _make_client(args.dvr, dvrs[args.dvr])
    if not client.check_connection():
        logger.error("Sem conexao HTTP com %s", args.dvr)
        return 3

    end = datetime.now()
    start = end - timedelta(minutes=args.minutes)
    logger.info(
        "Listando eventos ch%d entre %s e %s",
        args.channel,
        start.strftime("%Y-%m-%d %H:%M:%S"),
        end.strftime("%Y-%m-%d %H:%M:%S"),
    )
    events = client.find_media_files(
        channel=args.channel,
        start=start,
        end=end,
        event_types=DEFAULT_EVENT_TYPES,
        flags=["Event"],
    )
    logger.info("Eventos encontrados: %d", len(events))
    for rec in events[-10:]:
        _print_event(rec)

    if not events:
        logger.warning("Nenhum evento recente para testar download")
        return 0

    rec = events[-1]
    logger.info("Testando evento mais recente: %s", rec)
    out_dir = Path("detections/dvr_health")
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_handle = tempfile.NamedTemporaryFile(
        suffix=".dav",
        prefix=f"{args.dvr}_ch{args.channel}_",
        dir=out_dir,
        delete=False,
    )
    video_path = Path(temp_handle.name)
    temp_handle.close()

    try:
        outcome = client.download_file_detailed(
            rec.file_path,
            video_path,
            expected_size=rec.length_bytes or None,
        )
        logger.info(
            "Download: ok=%s reason=%s status=%s bytes=%s/%s attempts=%d "
            "relogin=%d resumed=%s",
            outcome.ok,
            outcome.reason,
            outcome.http_status,
            format_bytes(outcome.bytes_written),
            format_bytes(outcome.expected_bytes),
            outcome.attempts,
            outcome.relogin_count,
            outcome.resumed,
        )
        if not outcome.ok:
            return 4

        frames = list(frames_from_video(video_path, args.frame_interval))
        logger.info("Frames extraidos: %d", len(frames))
        if not frames:
            return 5
        return 0
    finally:
        if not args.keep_download:
            video_path.unlink(missing_ok=True)
            video_path.with_name(video_path.name + ".part").unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
