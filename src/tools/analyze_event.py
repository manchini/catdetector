"""
Analise pontual de um evento conhecido do DVR.

Uso:
  python -m src.tools.analyze_event \\
      --dvr dvr_casa --channel 1 \\
      --at "2026-04-14 00:03:29" \\
      --window-min 5

Lista eventos do DVR na janela, baixa o segmento que contem o instante pedido,
extrai todos os frames a 1s e roda YOLO+SAHI com conf 0.1 em cada um.
Salva os frames (com e sem deteccao) anotados em detections/analysis/<stamp>_<cam>/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import yaml

from src.detection.detector import AnimalDetector
from src.dvr.dahua_api import (
    DEFAULT_EVENT_TYPES,
    DahuaHTTPClient,
    RecordedFile,
    format_bytes,
)
from src.dvr.offline_processor import frames_from_video

logger = logging.getLogger("analyze_event")

ANALYSIS_DIR = Path("detections/analysis")

def list_events_all_types(
    client: DahuaHTTPClient,
    channel: int,
    start: datetime,
    end: datetime,
) -> tuple[list[RecordedFile], dict[str, int]]:
    """Busca por cada tipo conhecido + uma busca sem filtro de evento, deduplica por file_path."""
    seen: dict[str, RecordedFile] = {}
    counts: dict[str, int] = {}

    for evt in DEFAULT_EVENT_TYPES:
        files = client.find_media_files(
            channel=channel,
            start=start,
            end=end,
            event_types=[evt],
            flags=["Event"],
        )
        counts[evt] = len(files)
        logger.info(f"  {evt}: {len(files)} arquivo(s)")
        for f in files:
            seen.setdefault(f.file_path, f)

    files_nofilter = client.find_media_files(
        channel=channel,
        start=start,
        end=end,
    )
    counts["<sem filtro Event>"] = len(files_nofilter)
    logger.info(f"  <sem filtro Event>: {len(files_nofilter)} arquivo(s)")
    for f in files_nofilter:
        seen.setdefault(f.file_path, f)

    return list(seen.values()), counts


def load_dvr_config(dvr_name: str, config_path: str = "config/cameras.yaml") -> tuple[dict, dict]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(config_path)

    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    dvrs = config.get("dvrs", {})
    if dvr_name not in dvrs:
        raise KeyError(f"DVR '{dvr_name}' nao encontrado em {config_path}. Disponiveis: {list(dvrs)}")

    cameras = config.get("cameras", [])
    return dvrs[dvr_name], {c["id"]: c for c in cameras}


def make_client(dvr_name: str, dvr_cfg: dict) -> DahuaHTTPClient:
    return DahuaHTTPClient(
        dvr_ip=dvr_cfg["host"],
        username=dvr_cfg.get("http_username") or dvr_cfg.get("username", "admin"),
        password=dvr_cfg.get("http_password") or dvr_cfg.get("password", ""),
        dvr_name=dvr_name,
        port=dvr_cfg.get("http_port", 80),
        rpc_username=dvr_cfg.get("rpc_username"),
        rpc_password=dvr_cfg.get("rpc_password"),
    )


def find_event_covering(events: list[RecordedFile], target: datetime) -> RecordedFile | None:
    for rec in events:
        if rec.start_time <= target <= rec.end_time:
            return rec
    if not events:
        return None
    return min(events, key=lambda r: min(abs((r.start_time - target).total_seconds()),
                                         abs((r.end_time - target).total_seconds())))


def _zone_bboxes_px(zones: list[list[float]] | None, frame_shape) -> list[tuple[int, int, int, int]]:
    if not zones:
        return []
    h, w = frame_shape[:2]
    out = []
    for z in zones:
        x1, y1, x2, y2 = z
        out.append((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
    return out


def _bbox_overlaps_any_zone(bbox, zones_px) -> bool:
    """True se o centro do bbox cai em alguma zona OU se ha area de interseccao."""
    if not zones_px:
        return True  # sem zonas configuradas, aceita tudo
    bx1, by1, bx2, by2 = bbox
    cx = (bx1 + bx2) // 2
    cy = (by1 + by2) // 2
    for zx1, zy1, zx2, zy2 in zones_px:
        if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
            return True
        ix1 = max(bx1, zx1); iy1 = max(by1, zy1)
        ix2 = min(bx2, zx2); iy2 = min(by2, zy2)
        if ix2 > ix1 and iy2 > iy1:
            return True
    return False


def annotate_frame(frame, detections: list[dict], timestamp_txt: str,
                   zones_px: list[tuple[int, int, int, int]] | None = None,
                   detections_in_zone: set[int] | None = None):
    annotated = frame.copy()
    if zones_px:
        for zx1, zy1, zx2, zy2 in zones_px:
            cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (255, 128, 0), 1)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        in_zone = detections_in_zone is None or i in detections_in_zone
        color = (0, 255, 255) if in_zone else (128, 128, 128)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        if not in_zone:
            label += " [fora]"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(annotated, timestamp_txt, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return annotated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dvr", required=True, help="nome do DVR no cameras.yaml (ex: dvr_casa)")
    parser.add_argument("--channel", type=int, required=True)
    parser.add_argument("--at", required=True, help="instante alvo YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--window-min", type=int, default=5, help="minutos antes/depois do alvo para listar eventos")
    parser.add_argument("--frame-interval", type=float, default=1.0)
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--save-all-frames", action="store_true",
                        help="Salva todos os frames, nao so os com deteccao")
    parser.add_argument("--skip-download", action="store_true",
                        help="Reusa o .dav existente no diretorio de saida (pula download)")
    parser.add_argument("--ignore-zones", action="store_true",
                        help="Nao aplica filtro de zonas calibradas (detection_zone)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    target = datetime.strptime(args.at, "%Y-%m-%d %H:%M:%S")
    start = target - timedelta(minutes=args.window_min)
    end = target + timedelta(minutes=args.window_min)

    dvr_cfg, cameras_by_id = load_dvr_config(args.dvr)
    cam_id = next(
        (cid for cid, c in cameras_by_id.items()
         if c.get("dvr") == args.dvr and int(c.get("channel", -1)) == args.channel),
        f"{args.dvr}_ch{args.channel}",
    )

    stamp = target.strftime("%Y%m%d_%H%M%S")
    out_dir = ANALYSIS_DIR / f"{stamp}_{cam_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saida: {out_dir}")

    client = make_client(args.dvr, dvr_cfg)
    if not client.check_connection():
        logger.error(f"Sem conexao com DVR {args.dvr}")
        return 2

    logger.info(f"Listando eventos do canal {args.channel} entre {start} e {end} (multi-tipo)")
    events, counts = list_events_all_types(client, args.channel, start, end)
    logger.info(f"Eventos encontrados (unicos): {len(events)}")
    logger.info(f"Contagem por tipo: {counts}")
    for rec in sorted(events, key=lambda r: r.start_time):
        logger.info(f"  - {rec}  events={rec.events}")

    rec = find_event_covering(events, target)
    if rec is None:
        logger.error("Nenhum evento na janela. Aumente --window-min ou verifique o DVR.")
        return 3

    if not (rec.start_time <= target <= rec.end_time):
        logger.warning(
            f"Nenhum evento contem exatamente {target}. Usando o mais proximo: {rec}"
        )
    else:
        logger.info(f"Evento que contem {target}: {rec}")

    video_path = out_dir / f"segment_{Path(rec.file_path).name}"
    if args.skip_download and video_path.exists() and video_path.stat().st_size > 0:
        size_mb = video_path.stat().st_size / 1024 / 1024
        logger.info(f"Skip download: reusando {video_path.name} ({size_mb:.1f} MB)")
    else:
        logger.info(f"Baixando {rec.file_path} -> {video_path}")
        outcome = client.download_file_detailed(
            rec.file_path,
            video_path,
            expected_size=rec.length_bytes or None,
        )
        if not outcome.ok:
            logger.error(
                "Download falhou: %s (escrito=%s, esperado=%s)",
                outcome.reason,
                format_bytes(outcome.bytes_written),
                format_bytes(outcome.expected_bytes),
            )
            return 4
        logger.info(f"Download ok: {format_bytes(video_path.stat().st_size)}")

    cam_cfg = cameras_by_id.get(cam_id, {})
    zones_cfg = None if args.ignore_zones else cam_cfg.get("detection_zone")
    if zones_cfg:
        logger.info(f"Zonas calibradas para {cam_id}: {len(zones_cfg)} retangulo(s)")
    elif args.ignore_zones:
        logger.info("Filtro de zonas desativado por --ignore-zones")
    else:
        logger.info(f"Camera {cam_id} sem detection_zone; aceitando deteccoes em qualquer posicao")

    logger.info(f"Carregando detector (conf={args.confidence})")
    detector = AnimalDetector(confidence=args.confidence)

    logger.info(f"Extraindo e analisando frames a cada {args.frame_interval}s")
    rows: list[tuple[str, int, str, bool]] = []
    total_frames = 0
    detected_frames = 0
    in_zone_frames = 0
    zones_px: list[tuple[int, int, int, int]] = []

    for ts_rel, frame in frames_from_video(video_path, args.frame_interval):
        total_frames += 1
        frame_time = rec.start_time + timedelta(seconds=ts_rel)
        ts_txt = frame_time.strftime("%H:%M:%S")

        if not zones_px and zones_cfg:
            zones_px = _zone_bboxes_px(zones_cfg, frame.shape)

        detections = detector.detect(frame)
        if detections:
            detected_frames += 1

        in_zone_idx: set[int] = set()
        has_in_zone = False
        for i, d in enumerate(detections):
            is_in = _bbox_overlaps_any_zone(d["bbox"], zones_px) if zones_cfg else True
            if is_in:
                in_zone_idx.add(i)
                has_in_zone = True
            rows.append((ts_txt, int(d["confidence"] * 100), d["class_name"], is_in))

        if has_in_zone:
            in_zone_frames += 1

        is_target_window = abs((frame_time - target).total_seconds()) < 1.5
        should_save = (
            has_in_zone
            or args.save_all_frames
            or is_target_window
            or (bool(detections) and not zones_cfg)
        )
        if should_save:
            annotated = annotate_frame(
                frame, detections,
                f"{ts_txt} rel={ts_rel:.1f}s",
                zones_px=zones_px,
                detections_in_zone=in_zone_idx if zones_cfg else None,
            )
            fname = f"f{int(ts_rel*10):05d}_{ts_txt.replace(':','')}.jpg"
            cv2.imwrite(str(out_dir / fname), annotated)

    logger.info(
        f"Frames analisados: {total_frames}, com deteccao: {detected_frames}, "
        f"com deteccao na zona: {in_zone_frames}"
    )
    logger.info("Resumo (hora -> conf -> classe -> zona):")
    for t, c, cls, iz in rows:
        marker = "ZONE" if iz else "fora"
        logger.info(f"  {t}  {c:3d}%  {cls:8s}  [{marker}]")

    rows_in_zone = [r for r in rows if r[3]]
    if not rows:
        logger.warning("Nenhuma deteccao em NENHUM frame do segmento.")
        logger.warning("Reduza --confidence ou use --ignore-zones para diagnosticar.")
    elif not rows_in_zone and zones_cfg:
        logger.warning("Houve deteccoes, mas TODAS fora das zonas calibradas.")

    summary_path = out_dir / "SUMMARY.txt"
    with open(summary_path, "w", encoding="utf-8") as h:
        h.write(f"Event analysis\n")
        h.write(f"dvr={args.dvr} channel={args.channel} cam_id={cam_id} target={target}\n")
        h.write(f"segment={rec}\n")
        h.write(f"zones_configured={bool(zones_cfg)} n_zones={len(zones_cfg) if zones_cfg else 0}\n")
        h.write(
            f"frames_total={total_frames} frames_with_detections={detected_frames} "
            f"frames_with_detection_in_zone={in_zone_frames}\n\n"
        )
        for t, c, cls, iz in rows:
            marker = "ZONE" if iz else "fora"
            h.write(f"{t}  {c:3d}%  {cls:8s}  [{marker}]\n")
    logger.info(f"Sumario: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
