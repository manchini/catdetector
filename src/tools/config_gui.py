"""
GUI web local para configuracao e revisao manual de eventos DVR.

Uso:
    python -m src.tools.config_gui --config config/cameras.yaml --port 8765
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import webbrowser
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import yaml

from src.dvr.dahua_api import DahuaHTTPClient, RecordedFile
from src.dvr.offline_processor import frames_from_video
from src.dvr.rtsp_opencv_extractor import extract_frames_from_rtsp_playback
from src.storage.manual_review import get_saved_annotation, save_manual_annotation

try:
    from src.dvr.dahua_api import DEFAULT_EVENT_TYPES
except ImportError:  # pragma: no cover - compat com versoes antigas do modulo
    DEFAULT_EVENT_TYPES = ["VideoMotion"]


logger = logging.getLogger("config_gui")


def _parse_datetime(value: str) -> datetime:
    value = value.strip()
    if not value:
        raise ValueError("datetime vazio")
    return datetime.fromisoformat(value.replace("Z", "+00:00").replace("T", " "))


def _event_id(rec: RecordedFile, camera_id: str) -> str:
    raw = "|".join(
        [
            rec.dvr_name,
            camera_id,
            str(rec.channel),
            rec.start_time.isoformat(),
            rec.end_time.isoformat(),
            rec.file_path,
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _camera_rtsp_config(dvr_cfg: dict) -> dict:
    return {
        "host": dvr_cfg["host"],
        "port": int(dvr_cfg.get("port", 554)),
        "username": dvr_cfg.get("username", "admin"),
        "password": dvr_cfg.get("password", ""),
        "subtype": int(dvr_cfg.get("subtype", 1)),
    }


class ConfigGuiState:
    def __init__(self, config_path: Path, config: dict | None = None):
        self.config_path = config_path
        self.config = config if config is not None else self.load_config()
        self.cache_dir = Path("detections/manual_review/cache")
        self.events: dict[str, dict] = {}
        self.frames: dict[str, dict] = {}

    def load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(self.config_path)
        with open(self.config_path, encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def refresh_config(self) -> None:
        self.config = self.load_config()

    def sanitized_config(self) -> dict:
        cfg = json.loads(json.dumps(self.config))
        for dvr in (cfg.get("dvrs") or {}).values():
            for key in ("password", "http_password", "rpc_password"):
                if key in dvr:
                    dvr[f"{key}_set"] = bool(dvr.get(key))
                    dvr[key] = ""
        return cfg

    def _camera(self, camera_id: str) -> dict:
        for cam in self.config.get("cameras", []):
            if cam.get("id") == camera_id:
                return cam
        raise KeyError(f"camera nao encontrada: {camera_id}")

    def _dvr(self, dvr_name: str) -> dict:
        dvrs = self.config.get("dvrs", {})
        if dvr_name not in dvrs:
            raise KeyError(f"DVR nao encontrado: {dvr_name}")
        return dvrs[dvr_name]

    def _client(self, dvr_name: str, dvr_cfg: dict) -> DahuaHTTPClient:
        return DahuaHTTPClient(
            dvr_ip=dvr_cfg["host"],
            username=dvr_cfg.get("http_username") or dvr_cfg.get("username", "admin"),
            password=dvr_cfg.get("http_password") or dvr_cfg.get("password", ""),
            dvr_name=dvr_name,
            port=int(dvr_cfg.get("http_port", 80)),
            rpc_username=dvr_cfg.get("rpc_username"),
            rpc_password=dvr_cfg.get("rpc_password"),
        )

    def list_events(self, camera_id: str, start: datetime, end: datetime) -> list[dict]:
        cam = self._camera(camera_id)
        dvr_name = cam.get("dvr")
        channel = int(cam.get("channel"))
        dvr_cfg = self._dvr(dvr_name)
        client = self._client(dvr_name, dvr_cfg)

        records = client.find_media_files(
            channel=channel,
            start=start,
            end=end,
            event_types=DEFAULT_EVENT_TYPES,
            flags=["Event"],
        )

        out = []
        for rec in records:
            eid = _event_id(rec, camera_id)
            self.events[eid] = {
                "record": rec,
                "camera": cam,
                "dvr_name": dvr_name,
                "dvr_cfg": dvr_cfg,
            }
            out.append(self._event_payload(eid, rec, cam))
        return out

    def _event_payload(self, event_id: str, rec: RecordedFile, cam: dict) -> dict:
        return {
            "id": event_id,
            "camera_id": cam.get("id"),
            "camera_name": cam.get("name", cam.get("id")),
            "channel": rec.channel,
            "start_time": rec.start_time.isoformat(),
            "end_time": rec.end_time.isoformat(),
            "duration_seconds": rec.duration_seconds,
            "length_bytes": rec.length_bytes,
            "events": rec.events,
            "video_stream": rec.video_stream,
            "file_path": rec.file_path,
        }

    def open_event(self, event_id: str, frame_interval: float) -> dict:
        entry = self.events.get(event_id)
        if not entry:
            raise KeyError("evento expirou; busque novamente")

        rec: RecordedFile = entry["record"]
        cam = entry["camera"]
        dvr_name = entry["dvr_name"]
        dvr_cfg = entry["dvr_cfg"]
        client = self._client(dvr_name, dvr_cfg)
        event_dir = self.cache_dir / event_id
        frames_dir = event_dir / "frames"
        event_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        video_path = event_dir / f"segment_{Path(rec.file_path).name}"

        frames_list = []
        if video_path.exists() and video_path.stat().st_size > 0:
            frames_list = list(frames_from_video(video_path, frame_interval))
        if not frames_list:
            self._download_event(client, rec, video_path)
            if video_path.exists() and video_path.stat().st_size > 0:
                frames_list = list(frames_from_video(video_path, frame_interval))

        if not frames_list:
            rtsp_cfg = _camera_rtsp_config(dvr_cfg)
            frames_list = extract_frames_from_rtsp_playback(
                host=rtsp_cfg["host"],
                port=rtsp_cfg["port"],
                username=rtsp_cfg["username"],
                password=rtsp_cfg["password"],
                channel=rec.channel,
                start_time=rec.start_time,
                end_time=rec.end_time,
                duration_seconds=rec.duration_seconds,
                frame_interval=frame_interval,
                subtype=0,
                timeout_seconds=max(30, int(rec.duration_seconds) + 30),
            )

        if not frames_list:
            raise RuntimeError("nenhum frame extraido do evento")

        # Limpa frames deste evento da memória antes de recarregar
        for old_id in [fid for fid, m in self.frames.items() if m.get("event_id") == event_id]:
            del self.frames[old_id]

        _MAX_FRAMES_PER_EVENT = 300
        if len(frames_list) > _MAX_FRAMES_PER_EVENT:
            logger.warning(
                f"Evento {event_id}: {len(frames_list)} frames, "
                f"limitando a {_MAX_FRAMES_PER_EVENT}"
            )
            frames_list = frames_list[:_MAX_FRAMES_PER_EVENT]

        frame_payloads = []
        for idx, (ts_sec, frame) in enumerate(frames_list):
            frame_time = rec.start_time + timedelta(seconds=float(ts_sec))
            frame_id = hashlib.sha1(
                f"{event_id}:{idx}:{ts_sec:.3f}".encode("utf-8")
            ).hexdigest()
            frame_path = frames_dir / f"f{idx:05d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"falha ao salvar frame cache: {frame_path}")

            height, width = frame.shape[:2]
            source_id = hashlib.sha1(
                f"{event_id}:{idx}:{frame_time.isoformat()}".encode("utf-8")
            ).hexdigest()
            meta = {
                "id": frame_id,
                "source_id": source_id,
                "event_id": event_id,
                "path": str(frame_path),
                "camera_id": cam.get("id"),
                "camera_name": cam.get("name", cam.get("id")),
                "timestamp": frame_time.isoformat(),
                "ts_sec": float(ts_sec),
                "width": width,
                "height": height,
            }
            self.frames[frame_id] = meta
            frame_payloads.append(
                {
                    "id": frame_id,
                    "timestamp": meta["timestamp"],
                    "ts_sec": meta["ts_sec"],
                    "width": width,
                    "height": height,
                    "annotated": get_saved_annotation(source_id) is not None,
                }
            )

        return {
            "event": self._event_payload(event_id, rec, cam),
            "frames": frame_payloads,
        }

    def _download_event(
        self,
        client: DahuaHTTPClient,
        rec: RecordedFile,
        video_path: Path,
    ) -> None:
        if hasattr(client, "download_file_detailed"):
            outcome = client.download_file_detailed(
                rec.file_path,
                video_path,
                expected_size=rec.length_bytes or None,
            )
            if not outcome.ok:
                logger.warning("download evento falhou: %s", outcome.reason)
            return

        if not client.download_file(rec.file_path, video_path):
            logger.warning("download evento falhou")

    def get_zone_frame(self, camera_id: str) -> bytes:
        """Captura um frame ao vivo da camera via RTSP e retorna como JPEG bytes."""
        cam = self._camera(camera_id)
        dvr_name = cam.get("dvr")
        dvr_cfg = self._dvr(dvr_name)
        channel = int(cam.get("channel"))

        url = (
            f"rtsp://{dvr_cfg.get('username', 'admin')}:{dvr_cfg.get('password', '')}"
            f"@{dvr_cfg['host']}:{int(dvr_cfg.get('port', 554))}"
            f"/cam/realmonitor?channel={channel}&subtype={int(dvr_cfg.get('subtype', 1))}"
        )
        cap = cv2.VideoCapture(url)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            frame = None
            for _ in range(10):
                ok, f = cap.read()
                if ok:
                    frame = f
                    break
            if frame is None:
                raise RuntimeError("nenhum frame RTSP recebido")
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return bytes(buf)
        finally:
            cap.release()

    def zones_for_camera(self, camera_id: str) -> list[list[float]]:
        cam = self._camera(camera_id)
        return list(cam.get("detection_zone") or [])

    def save_zones(self, camera_id: str, zones: list[list[float]]) -> None:
        for cam in self.config.get("cameras", []):
            if cam.get("id") == camera_id:
                cam["detection_zone"] = zones
                break
        else:
            raise KeyError(f"camera nao encontrada: {camera_id}")
        with open(self.config_path, "w", encoding="utf-8") as fh:
            yaml.dump(self.config, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)

    def frame_path(self, frame_id: str) -> Path:
        meta = self.frames.get(frame_id)
        if not meta:
            raise KeyError("frame nao encontrado")
        return Path(meta["path"])

    def annotation_for_frame(self, frame_id: str) -> dict:
        meta = self.frames.get(frame_id)
        if not meta:
            raise KeyError("frame nao encontrado")
        return get_saved_annotation(meta["source_id"]) or {
            "source_id": meta["source_id"],
            "status": "new",
            "boxes": [],
        }

    def save_frame_annotation(
        self,
        frame_id: str,
        status: str,
        boxes: list[dict],
    ) -> dict:
        meta = self.frames.get(frame_id)
        if not meta:
            raise KeyError("frame nao encontrado")
        return save_manual_annotation(
            source_image_path=Path(meta["path"]),
            width=int(meta["width"]),
            height=int(meta["height"]),
            camera_id=meta["camera_id"],
            camera_name=meta["camera_name"],
            timestamp=datetime.fromisoformat(meta["timestamp"]),
            status=status,
            boxes=boxes,
            source_id=meta["source_id"],
        )


_TEMPLATE_PATH = Path(__file__).with_name("config_gui_template.html")


def _load_html_template() -> str:
    """Carrega o template HTML do disco a cada request (sem cache para hot-reload em dev)."""
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


class ConfigGuiHandler(BaseHTTPRequestHandler):
    state: ConfigGuiState

    def log_message(self, fmt, *args):
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status=status)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        if length > 1_000_000:
            raise ValueError(f"Content-Length muito grande: {length}")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = _load_html_template().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/config":
                self.state.refresh_config()
                self._send_json(self.state.sanitized_config())
                return
            if parsed.path == "/api/review/events":
                qs = parse_qs(parsed.query)
                camera_id = (qs.get("camera_id") or [""])[0]
                start = _parse_datetime((qs.get("start") or [""])[0])
                end = _parse_datetime((qs.get("end") or [""])[0])
                events = self.state.list_events(camera_id, start, end)
                self._send_json({"events": events})
                return
            if parsed.path == "/api/review/frame":
                qs = parse_qs(parsed.query)
                frame_id = (qs.get("id") or [""])[0]
                frame_path = self.state.frame_path(frame_id)
                data = frame_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/api/review/annotations":
                qs = parse_qs(parsed.query)
                frame_id = (qs.get("frame_id") or [""])[0]
                self._send_json(self.state.annotation_for_frame(frame_id))
                return
            if parsed.path == "/api/zones/frame":
                qs = parse_qs(parsed.query)
                camera_id = (qs.get("camera_id") or [""])[0]
                data = self.state.get_zone_frame(camera_id)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/api/zones":
                qs = parse_qs(parsed.query)
                camera_id = (qs.get("camera_id") or [""])[0]
                zones = self.state.zones_for_camera(camera_id)
                self._send_json({"zones": zones})
                return
            self._send_error("rota nao encontrada", status=404)
        except (KeyError, ValueError) as e:
            self._send_error(str(e), status=400)
        except Exception as e:
            logger.exception("GET falhou")
            self._send_error(str(e), status=500)

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            payload = self._read_json()
            if parsed.path == "/api/review/open":
                result = self.state.open_event(
                    str(payload.get("event_id", "")),
                    float(payload.get("frame_interval") or 1.0),
                )
                self._send_json(result)
                return
            if parsed.path == "/api/review/save":
                result = self.state.save_frame_annotation(
                    frame_id=str(payload.get("frame_id", "")),
                    status=str(payload.get("status", "")),
                    boxes=list(payload.get("boxes") or []),
                )
                self._send_json(result)
                return
            if parsed.path == "/api/zones/save":
                camera_id = str(payload.get("camera_id", ""))
                zones = list(payload.get("zones") or [])
                self.state.save_zones(camera_id, zones)
                self.state.refresh_config()
                self._send_json({"ok": True, "count": len(zones)})
                return
            self._send_error("rota nao encontrada", status=404)
        except (KeyError, ValueError) as e:
            self._send_error(str(e), status=400)
        except Exception as e:
            logger.exception("POST falhou")
            self._send_error(str(e), status=500)


def run_server(config_path: str, host: str, port: int, open_browser: bool) -> None:
    state = ConfigGuiState(Path(config_path))

    class Handler(ConfigGuiHandler):
        pass

    Handler.state = state
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    logger.info("Config GUI em %s", url)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Encerrando config GUI")
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description="GUI local do Cat Detector")
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.host not in ("127.0.0.1", "localhost"):
        logger.warning("Servidor exposto fora de localhost; use apenas em rede confiavel")

    run_server(args.config, args.host, args.port, args.open_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
