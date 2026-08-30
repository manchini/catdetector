"""
Cliente HTTP para DVRs Intelbras/Dahua.
Consulta eventos de movimento e baixa videos para processamento offline.
"""

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator

import requests
from requests.auth import HTTPDigestAuth

logger = logging.getLogger(__name__)


def _interruptible_sleep(seconds: float, stop_event: threading.Event | None) -> None:
    """Dorme por `seconds` mas acorda a cada 1s para checar stop_event."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            return
        time.sleep(min(1.0, deadline - time.monotonic()))


MOTION_GRID_COLS = 22
MOTION_GRID_ROWS = 18
DEFAULT_EVENT_TYPES = [
    "VideoMotion",
    "SmartMotionHuman",
    "SmartMotionVehicle",
    "CrossLineDetection",
    "CrossRegionDetection",
]


def _safe_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def format_bytes(size: int | None) -> str:
    if size is None:
        return "unknown"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    if size < 1024 * 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MiB"
    return f"{size / 1024 / 1024 / 1024:.2f} GiB"


@dataclass
class RecordedFile:
    channel: int
    start_time: datetime
    end_time: datetime
    file_path: str
    events: list[str]
    length_bytes: int
    video_stream: str
    dvr_ip: str
    dvr_name: str
    flags: list[str] | None = None
    duration: int | None = None
    disk: int | None = None
    partition: int | None = None
    cluster: int | None = None
    cut_length: int | None = None
    raw_fields: dict[str, str] | None = None

    @property
    def has_motion(self) -> bool:
        return any("motion" in e.lower() for e in self.events)

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    def __str__(self) -> str:
        events_str = ",".join(self.events) if self.events else "Timing"
        return (
            f"Ch{self.channel} [{self.start_time.strftime('%H:%M:%S')}-"
            f"{self.end_time.strftime('%H:%M:%S')}] "
            f"{events_str} {self.video_stream} ({format_bytes(self.length_bytes)})"
        )


@dataclass
class DownloadOutcome:
    ok: bool
    file_path: str
    dest: Path
    url: str
    attempts: int = 0
    expected_bytes: int | None = None
    content_length: int | None = None
    bytes_written: int = 0
    http_status: int | None = None
    error: str | None = None
    relogin_count: int = 0
    resumed: bool = False

    @property
    def reason(self) -> str:
        if self.ok:
            return "ok"
        return self.error or "unknown"


@dataclass
class MotionZone:
    """Zona de deteccao de movimento configurada no DVR."""

    channel: int
    window_id: int
    name: str
    grid: list[list[int]]
    sensitivity: int

    def cells_active(self) -> list[tuple[int, int]]:
        """Retorna coordenadas (linha, coluna) das células ativas na grade."""
        return [
            (r, c)
            for r, row in enumerate(self.grid)
            for c, val in enumerate(row)
            if val
        ]


def _parse_files_response(text: str, dvr_ip: str, dvr_name: str) -> list[RecordedFile]:
    raw: dict[int, dict] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        m = re.match(r"items\[(\d+)\]\.(.+)", key.strip())
        if not m:
            continue
        idx = int(m.group(1))
        raw.setdefault(idx, {})[m.group(2)] = value.strip()

    result = []
    for idx in sorted(raw.keys()):
        data = raw[idx]
        try:
            start = datetime.strptime(data.get("StartTime", ""), "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(data.get("EndTime", ""), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        events = [v for k, v in data.items() if k.startswith("Events[")]
        flags = [v for k, v in data.items() if k.startswith("Flags[")]
        result.append(
            RecordedFile(
                channel=_safe_int(data.get("Channel"), 0) + 1,
                start_time=start,
                end_time=end,
                file_path=data.get("FilePath", ""),
                events=events,
                length_bytes=_safe_int(data.get("Length"), 0),
                video_stream=data.get("VideoStream", "Main"),
                dvr_ip=dvr_ip,
                dvr_name=dvr_name,
                flags=flags,
                duration=_safe_int(data.get("Duration")) if "Duration" in data else None,
                disk=_safe_int(data.get("Disk")) if "Disk" in data else None,
                partition=_safe_int(data.get("Partition")) if "Partition" in data else None,
                cluster=_safe_int(data.get("Cluster")) if "Cluster" in data else None,
                cut_length=_safe_int(data.get("CutLength")) if "CutLength" in data else None,
                raw_fields=dict(data),
            )
        )

    return result


def _decode_region_grid(region_values: list[int]) -> list[list[int]]:
    grid = []
    for val in region_values[:MOTION_GRID_ROWS]:
        row = [(val >> (MOTION_GRID_COLS - 1 - col)) & 1 for col in range(MOTION_GRID_COLS)]
        grid.append(row)
    return grid


def _parse_motion_zones(text: str) -> list[MotionZone]:
    raw: dict[int, dict[int, dict]] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        m = re.match(
            r"table\.MotionDetect\[(\d+)\]\.MotionDetectWindow\[(\d+)\]\.(.+)",
            key.strip(),
        )
        if not m:
            continue

        channel = int(m.group(1))
        window_id = int(m.group(2))
        field = m.group(3)
        raw.setdefault(channel, {}).setdefault(window_id, {})[field] = value.strip()

    zones = []
    for channel, windows in sorted(raw.items()):
        for window_id, fields in sorted(windows.items()):
            region_vals = []
            for i in range(MOTION_GRID_ROWS):
                key = f"Region[{i}]"
                if key in fields:
                    region_vals.append(_safe_int(fields[key]))

            if not region_vals:
                continue

            zones.append(
                MotionZone(
                    channel=channel + 1,
                    window_id=window_id,
                    name=fields.get("Name", f"Regiao {window_id + 1}"),
                    grid=_decode_region_grid(region_vals),
                    sensitivity=_safe_int(fields.get("Sensitive", "0")),
                )
            )

    return zones


def _encode_motion_zones(zones: list[MotionZone]) -> dict:
    """
    Codifica lista de MotionZone de volta para formato CGI.

    Returns:
        dict com parâmetros para POST em /cgi-bin/configManager.cgi
        Exemplo: {'action': 'setConfig', 'name': 'MotionDetect',
                  'table.MotionDetect[0].MotionDetectWindow[0].Sensitive': '40', ...}
    """
    params = {
        "action": "setConfig",
        "name": "MotionDetect",
    }

    for zone in zones:
        ch = zone.channel - 1
        win = zone.window_id

        params[
            f"table.MotionDetect[{ch}].MotionDetectWindow[{win}].Sensitive"
        ] = str(zone.sensitivity)

        for row_idx, row in enumerate(zone.grid):
            region_val = 0
            for col_idx, val in enumerate(row):
                if val:
                    region_val |= 1 << (MOTION_GRID_COLS - 1 - col_idx)
            params[
                f"table.MotionDetect[{ch}].MotionDetectWindow[{win}].Region[{row_idx}]"
            ] = str(region_val)

    return params


class DahuaHTTPClient:
    """
    Cliente HTTP para DVRs Intelbras baseados em Dahua.
    Usa Digest Auth e a API CGI para listar e baixar gravacoes.
    """

    def __init__(self, dvr_ip: str, username: str, password: str,
                 dvr_name: str = "", port: int = 80,
                 rpc_username: str | None = None,
                 rpc_password: str | None = None) -> None:
        self.dvr_ip = dvr_ip
        self.dvr_name = dvr_name or dvr_ip
        self.username = username
        self.password = password
        self.rpc_username = rpc_username or username
        self.rpc_password = rpc_password or password
        self.base_url = f"http://{dvr_ip}:{port}"
        self._session = requests.Session()
        self._session.auth = HTTPDigestAuth(username, password)
        self._rpc_session: requests.Session | None = None
        self._rpc_session_id: str | None = None
        # Sessão RPC pode ser invalidada de qualquer thread (poller ou pipeline);
        # serializa mutações para evitar dois logins simultâneos / cookie corrompido.
        self._rpc_lock = threading.RLock()

    def _rpc_login(self) -> requests.Session | None:
        """
        Autentica via /RPC2_Login (JSON-RPC) e devolve uma Session com o
        cookie WebClientHttpSessionID necessário para baixar arquivos pelo
        endpoint /RPC_Loadfile. O digest auth tradicional é aceito pela CGI
        de listagem mas NÃO libera streaming do payload pelo Loadfile neste
        firmware (MHDX 4.x); por isso precisamos do login RPC.
        """
        session = requests.Session()
        user = self.rpc_username
        pwd = self.rpc_password

        try:
            r1 = session.post(
                f"{self.base_url}/RPC2_Login",
                json={
                    "method": "global.login",
                    "params": {
                        "userName": user,
                        "password": "",
                        "clientType": "Web3.0",
                        "loginType": "Direct",
                        "authorityType": "Default",
                    },
                    "id": 1,
                    "session": 0,
                },
                timeout=10,
            )
            d1 = r1.json()
            params = d1.get("params") or {}
            random_str = params.get("random")
            realm = params.get("realm")
            sess_id = d1.get("session")
            if not (random_str and realm and sess_id):
                logger.error(f"RPC login challenge inválido: {d1}")
                return None

            pwd_hash = hashlib.md5(f"{user}:{realm}:{pwd}".encode()).hexdigest().upper()
            final_hash = (
                hashlib.md5(f"{user}:{random_str}:{pwd_hash}".encode())
                .hexdigest()
                .upper()
            )

            r2 = session.post(
                f"{self.base_url}/RPC2_Login",
                json={
                    "method": "global.login",
                    "params": {
                        "userName": user,
                        "password": final_hash,
                        "clientType": "Web3.0",
                        "loginType": "Direct",
                        "authorityType": "Default",
                    },
                    "id": 2,
                    "session": sess_id,
                },
                timeout=10,
            )
            d2 = r2.json()
            if not d2.get("result"):
                err = (d2.get("error") or {}).get("message", "desconhecido")
                logger.error(f"RPC login falhou para {user}@{self.dvr_ip}: {err}")
                return None

            with self._rpc_lock:
                self._rpc_session = session
                self._rpc_session_id = str(d2.get("session", sess_id))
            logger.info(f"DVR {self.dvr_name}: RPC login OK")
            return session
        except (requests.RequestException, ValueError, KeyError) as e:
            logger.exception(f"Erro durante RPC login: {e}")
            return None

    def _get_rpc_session(self) -> requests.Session | None:
        with self._rpc_lock:
            if self._rpc_session is None:
                self._rpc_login()
            return self._rpc_session

    def _invalidate_rpc_session(self) -> None:
        with self._rpc_lock:
            self._rpc_session = None
            self._rpc_session_id = None

    def _get(self, path: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", 30)
        return self._session.get(f"{self.base_url}{path}", **kwargs)

    def check_connection(self) -> bool:
        try:
            response = self._get("/cgi-bin/magicBox.cgi?action=getDeviceType")
            if response.status_code == 200:
                logger.info(f"DVR {self.dvr_name}: conectado ({response.text.strip()})")
                return True
        except requests.RequestException as e:
            logger.error(f"DVR {self.dvr_name}: erro de conexao: {e}")
        return False

    def get_motion_zones(self, channel: int | None = None) -> list[MotionZone]:
        response = self._get("/cgi-bin/configManager.cgi?action=getConfig&name=MotionDetect")
        if response.status_code != 200:
            logger.error(f"DVR {self.dvr_name}: nao foi possivel obter zonas de motion")
            return []

        zones = _parse_motion_zones(response.text)
        if channel is not None:
            zones = [zone for zone in zones if zone.channel == channel]
        return zones

    def set_motion_zones(self, zones: list[MotionZone]) -> bool:
        """
        Escreve configuração de zonas de motion no DVR via POST.

        Args:
            zones: Lista de MotionZone com sensibilidade atualizada.

        Returns:
            True se sucesso (HTTP 200), False caso contrário.
        """
        params = _encode_motion_zones(zones)

        try:
            response = self._session.post(
                f"{self.base_url}/cgi-bin/configManager.cgi",
                data=params,
                timeout=30,
            )

            if response.status_code == 200:
                logger.info(f"DVR {self.dvr_name}: zonas de motion atualizadas")
                return True

            logger.error(
                f"DVR {self.dvr_name}: erro ao atualizar zonas ({response.status_code})"
            )
            return False
        except requests.RequestException as e:
            logger.error(f"DVR {self.dvr_name}: erro ao escrever zonas: {e}")
            return False

    def _list_media_once(
        self,
        channel: int,
        start: datetime,
        end: datetime,
        event_type: str | None = None,
        flags: list[str] | None = None,
        types: tuple[str, ...] = ("dav",),
        max_files: int = 500,
    ) -> list[RecordedFile]:
        response = self._get("/cgi-bin/mediaFileFind.cgi?action=factory.create")
        if response.status_code != 200:
            return []

        obj_id = response.text.strip().split("=")[-1].strip()
        start_str = start.strftime("%Y-%m-%d%%20%H:%M:%S")
        end_str = end.strftime("%Y-%m-%d%%20%H:%M:%S")

        parts = [
            "/cgi-bin/mediaFileFind.cgi",
            "?action=findFile",
            f"&object={obj_id}",
            f"&condition.Channel={channel}",
            f"&condition.StartTime={start_str}",
            f"&condition.EndTime={end_str}",
        ]
        for idx, media_type in enumerate(types):
            parts.append(f"&condition.Types[{idx}]={media_type}")
        if event_type is not None:
            parts.append(f"&condition.Events[0]={event_type}")
        if flags:
            for idx, flag in enumerate(flags):
                parts.append(f"&condition.Flags[{idx}]={flag}")

        try:
            response_find = self._get("".join(parts))
            if response_find.status_code != 200:
                return []

            response_next = self._get(
                f"/cgi-bin/mediaFileFind.cgi"
                f"?action=findNextFile&object={obj_id}&count={max_files}"
            )
            if response_next.status_code != 200:
                return []

            return _parse_files_response(
                response_next.text, self.dvr_ip, self.dvr_name
            )
        finally:
            self._get(f"/cgi-bin/mediaFileFind.cgi?action=destroy&object={obj_id}")

    def find_media_files(
        self,
        channel: int,
        start: datetime,
        end: datetime,
        event_types: list[str | None] | tuple[str | None, ...] | None = None,
        flags: list[str] | None = None,
        types: tuple[str, ...] = ("dav",),
        max_files: int = 500,
        since: datetime | None = None,
        main_stream_only: bool = True,
    ) -> list[RecordedFile]:
        """
        Busca gravacoes do DVR via mediaFileFind.

        event_types=None faz uma busca sem filtro de evento. Para consultar
        varios tipos, passe uma lista e os resultados serao deduplicados por
        file_path.
        """
        event_filters = [None] if event_types is None else list(event_types)
        seen: dict[str, RecordedFile] = {}

        for event_type in event_filters:
            files = self._list_media_once(
                channel=channel,
                start=start,
                end=end,
                event_type=event_type,
                flags=flags,
                types=types,
                max_files=max_files,
            )
            for file in files:
                key = file.file_path or (
                    f"{file.channel}:{file.start_time.isoformat()}:"
                    f"{file.end_time.isoformat()}"
                )
                seen.setdefault(key, file)

        files = sorted(seen.values(), key=lambda item: item.start_time)
        if main_stream_only:
            files = [file for file in files if file.video_stream == "Main"]
        if since is not None:
            files = [file for file in files if file.start_time > since]
        return files

    def list_motion_events(
        self,
        channel: int,
        start: datetime,
        end: datetime,
        max_files: int = 500,
        since: datetime | None = None,
    ) -> list[RecordedFile]:
        files = self.find_media_files(
            channel=channel,
            start=start,
            end=end,
            event_types=["VideoMotion"],
            flags=["Event"],
            max_files=max_files,
            since=since,
        )
        logger.info(
            f"DVR {self.dvr_name} ch{channel}: "
            f"{len(files)} evento(s) de motion ({start.date()}..{end.date()})"
        )
        return files

    def list_all_recordings(
        self,
        channel: int,
        start: datetime,
        end: datetime,
        max_files: int = 500,
    ) -> list[RecordedFile]:
        return self.find_media_files(
            channel=channel,
            start=start,
            end=end,
            max_files=max_files,
        )

    def download_file_detailed(
        self,
        file_path: str,
        dest: Path,
        expected_size: int | None = None,
        chunk_size: int = 256 * 1024,
        max_retries: int = 3,
        stop_event: threading.Event | None = None,
    ) -> DownloadOutcome:
        """
        Baixa um arquivo de video do DVR para disco.

        O firmware MHDX 4.x aceita digest auth para GET em /RPC_Loadfile
        (responde 200 com Content-Length correto) mas fecha a conexão sem
        transmitir payload. A forma que funciona: login via /RPC2_Login e
        reutilizar o cookie WebClientHttpSessionID no GET.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.unlink(missing_ok=True)
        part_path = dest.with_name(dest.name + ".part")
        file_url = f"{self.base_url}/RPC_Loadfile{file_path}"
        outcome = DownloadOutcome(
            ok=False,
            file_path=file_path,
            dest=dest,
            url=file_url,
            expected_bytes=expected_size,
        )

        for attempt in range(max_retries):
            outcome.attempts = attempt + 1
            session = self._get_rpc_session()
            if session is None:
                outcome.error = "rpc_login_failed"
                if attempt < max_retries - 1:
                    _interruptible_sleep(2.0, stop_event)
                    continue
                break

            headers = {}
            resume_from = part_path.stat().st_size if part_path.exists() else 0
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
                outcome.resumed = True

            try:
                with session.get(
                    file_url,
                    stream=True,
                    timeout=(10, 60),
                    headers=headers,
                ) as response:
                    outcome.http_status = response.status_code
                    if response.status_code == 401:
                        logger.info("RPC session expirou; refazendo login")
                        self._invalidate_rpc_session()
                        outcome.relogin_count += 1
                        outcome.error = "unauthorized"
                        continue
                    if response.status_code not in (200, 206):
                        logger.error(
                            f"Download falhou ({response.status_code}), "
                            f"tentativa {attempt + 1}/{max_retries}"
                        )
                        outcome.error = f"http_{response.status_code}"
                        if attempt < max_retries - 1:
                            _interruptible_sleep(2.0 + attempt, stop_event)
                        continue

                    if response.status_code == 200 and resume_from > 0:
                        part_path.unlink(missing_ok=True)
                        resume_from = 0
                        outcome.resumed = False

                    content_length = _safe_int(
                        response.headers.get("CONTENT-LENGTH")
                        or response.headers.get("Content-Length"),
                        0,
                    )
                    if content_length:
                        outcome.content_length = content_length
                        if response.status_code == 206:
                            outcome.expected_bytes = expected_size or (
                                resume_from + content_length
                            )
                        elif outcome.expected_bytes is None:
                            outcome.expected_bytes = content_length

                    if (
                        expected_size
                        and content_length
                        and response.status_code == 200
                        and content_length != expected_size
                    ):
                        logger.warning(
                            "Tamanho anunciado diverge do RecordedFile: "
                            f"Content-Length={format_bytes(content_length)} "
                            f"RecordedFile={format_bytes(expected_size)}"
                        )

                    logger.info(
                        f"Iniciando download: {Path(file_path).name} "
                        f"(esperado={format_bytes(outcome.expected_bytes)}, "
                        f"retomado_de={format_bytes(resume_from)})"
                    )

                    mode = (
                        "ab"
                        if resume_from > 0 and response.status_code == 206
                        else "wb"
                    )
                    with open(part_path, mode) as file_handle:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                file_handle.write(chunk)

                final_size = part_path.stat().st_size if part_path.exists() else 0
                outcome.bytes_written = final_size

                if final_size == 0:
                    logger.warning(
                        f"DVR devolveu 0 bytes para {dest.name} "
                        f"(tentativa {attempt + 1}/{max_retries})"
                    )
                    self._invalidate_rpc_session()
                    outcome.relogin_count += 1
                    outcome.error = "zero_bytes"
                    if attempt < max_retries - 1:
                        _interruptible_sleep(2.0 + attempt, stop_event)
                    continue

                if outcome.expected_bytes is None or final_size >= outcome.expected_bytes:
                    part_path.replace(dest)
                    outcome.ok = True
                    outcome.error = None
                    logger.info(
                        f"Download completo: {dest.name} "
                        f"({format_bytes(final_size)})"
                    )
                    return outcome

                logger.warning(
                    f"Download incompleto "
                    f"({format_bytes(final_size)}/"
                    f"{format_bytes(outcome.expected_bytes)}), "
                    f"tentativa {attempt + 1}/{max_retries}"
                )
                outcome.error = "incomplete"
                if attempt < max_retries - 1:
                    _interruptible_sleep(2.0 + attempt, stop_event)
            except Exception as e:
                # Download falha por muitas razões transitórias (RST, timeout,
                # disk error, parser do urllib3). Capturar amplamente é
                # intencional para que o loop de retry continue.
                partial_size = part_path.stat().st_size if part_path.exists() else 0
                outcome.bytes_written = partial_size
                outcome.error = f"{type(e).__name__}: {e}"
                logger.error(
                    f"Erro tentativa {attempt + 1}: {e} "
                    f"(parcial={format_bytes(partial_size)})"
                )
                self._invalidate_rpc_session()
                outcome.relogin_count += 1
                if attempt < max_retries - 1:
                    _interruptible_sleep(2.0 + attempt, stop_event)

        logger.error(
            f"Download falhou apos {max_retries} tentativas "
            f"({outcome.reason}, escrito={format_bytes(outcome.bytes_written)})"
        )
        dest.unlink(missing_ok=True)
        part_path.unlink(missing_ok=True)
        return outcome

    def download_file(
        self,
        file_path: str,
        dest: Path,
        chunk_size: int = 256 * 1024,
        max_retries: int = 3,
        expected_size: int | None = None,
    ) -> bool:
        outcome = self.download_file_detailed(
            file_path=file_path,
            dest=dest,
            expected_size=expected_size,
            chunk_size=chunk_size,
            max_retries=max_retries,
        )
        return outcome.ok

    def iter_motion_videos(
        self,
        channels: list[int],
        start: datetime,
        end: datetime,
        download_dir: Path,
        skip_existing: bool = True,
    ) -> Generator[tuple[RecordedFile, Path], None, None]:
        download_dir.mkdir(parents=True, exist_ok=True)

        for channel in channels:
            events = self.list_motion_events(channel, start, end)
            for rec in events:
                filename = Path(rec.file_path).name
                local_path = download_dir / f"ch{channel}_{filename}"

                if skip_existing and local_path.exists():
                    size = local_path.stat().st_size
                    if size > 0:
                        logger.debug(f"Ja existe: {local_path.name}")
                        yield rec, local_path
                        continue

                    logger.warning(f"Arquivo vazio encontrado, baixando novamente: {local_path.name}")
                    local_path.unlink(missing_ok=True)

                logger.info(f"Baixando: {rec}")
                if self.download_file(
                    rec.file_path,
                    local_path,
                    expected_size=rec.length_bytes or None,
                ):
                    yield rec, local_path
                else:
                    logger.warning(f"Falha no download: {rec.file_path}")
