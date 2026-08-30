"""
Diagnostico: descobrir qual credencial autoriza /RPC_Loadfile em cada DVR.

Hipotese: dvr_casa2 fecha a conexao durante o download porque o RPC login
esta acontecendo como `vlc` (username), que nao tem permissao de Loadfile
no firmware; enquanto `admin` (rpc_username) teria.

Testa para cada DVR, com cada par de credenciais:
  1. Lista eventos de motion recente (digest auth)
  2. Faz RPC login com o par
  3. Tenta baixar 64KB do primeiro evento via /RPC_Loadfile
  4. Reporta: status RPC, bytes baixados, headers, erro

Uso:
    python scripts/diagnose_dvr_downloads.py
"""

import hashlib
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests
import yaml
from requests.auth import HTTPDigestAuth

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dvr.dahua_api import DahuaHTTPClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("diag")


def load_config() -> dict:
    with open("config/cameras.yaml") as f:
        return yaml.safe_load(f)


def rpc_login(base_url: str, user: str, pwd: str) -> tuple[requests.Session | None, str]:
    """Retorna (session, status_msg). session=None em falha."""
    session = requests.Session()
    try:
        r1 = session.post(
            f"{base_url}/RPC2_Login",
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
            return None, f"challenge invalido: {d1}"

        pwd_hash = hashlib.md5(f"{user}:{realm}:{pwd}".encode()).hexdigest().upper()
        final_hash = hashlib.md5(f"{user}:{random_str}:{pwd_hash}".encode()).hexdigest().upper()

        r2 = session.post(
            f"{base_url}/RPC2_Login",
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
            return None, f"login rejeitado: {err}"
        return session, f"OK (session={str(d2.get('session', sess_id))[:8]}...)"
    except Exception as e:
        return None, f"excecao: {type(e).__name__}: {e}"


def attempt_partial_download(
    session: requests.Session,
    base_url: str,
    file_path: str,
    max_bytes: int = 65536,
) -> tuple[int, dict, str]:
    """Tenta baixar os primeiros max_bytes. Retorna (bytes_recv, headers, err)."""
    url = f"{base_url}/RPC_Loadfile{file_path}"
    try:
        with session.get(url, stream=True, timeout=15) as r:
            headers = dict(r.headers)
            total = 0
            try:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        total += len(chunk)
                        if total >= max_bytes:
                            break
                return total, headers, ""
            except Exception as e:
                return total, headers, f"{type(e).__name__}: {e}"
    except Exception as e:
        return 0, {}, f"{type(e).__name__}: {e}"


def list_first_event(dvr_cfg: dict, channel: int = 1) -> str | None:
    """Lista o primeiro evento das ultimas 2h. Usa cliente original."""
    client = DahuaHTTPClient(
        dvr_ip=dvr_cfg["host"],
        username=dvr_cfg["username"],
        password=dvr_cfg["password"],
        dvr_name=dvr_cfg.get("name", dvr_cfg["host"]),
        port=dvr_cfg.get("http_port", 80),
        rpc_username=dvr_cfg.get("rpc_username"),
        rpc_password=dvr_cfg.get("rpc_password"),
    )
    end = datetime.now()
    start = end - timedelta(hours=2)
    try:
        events = client.list_motion_events(channel=channel, start=start, end=end)
        if events:
            return events[0].file_path
    except Exception as e:
        log.error(f"list_motion_events falhou: {e}")
    return None


def diagnose_dvr(dvr_name: str, dvr_cfg: dict):
    log.info("=" * 72)
    log.info(f"DVR: {dvr_name} ({dvr_cfg['host']})")
    log.info("=" * 72)

    base_url = f"http://{dvr_cfg['host']}:{dvr_cfg.get('http_port', 80)}"

    http_user = dvr_cfg["username"]
    http_pwd = dvr_cfg["password"]
    rpc_user = dvr_cfg.get("rpc_username", http_user)
    rpc_pwd = dvr_cfg.get("rpc_password", http_pwd)

    log.info(f"http_user={http_user!r}  rpc_user={rpc_user!r}")

    # Canal 1 em dvr_casa, canal 2 em dvr_casa2 (unica camera enabled la)
    channel = 2 if dvr_name == "dvr_casa2" else 1
    log.info(f"Listando eventos recentes no canal {channel}...")
    file_path = list_first_event(dvr_cfg, channel=channel)
    if not file_path:
        log.warning(f"Nenhum evento recente em {dvr_name} canal {channel}; pulando download.")
        return
    log.info(f"Evento alvo: {file_path}")

    for label, user, pwd in [
        ("HTTP_USER", http_user, http_pwd),
        ("RPC_USER ", rpc_user, rpc_pwd),
    ]:
        log.info("-" * 60)
        log.info(f"[{label}] Tentando RPC login como {user!r}...")
        session, status = rpc_login(base_url, user, pwd)
        log.info(f"[{label}] RPC login: {status}")
        if session is None:
            continue

        log.info(f"[{label}] Baixando ate 64KB de {file_path}...")
        bytes_recv, headers, err = attempt_partial_download(session, base_url, file_path)
        log.info(f"[{label}] bytes recebidos: {bytes_recv}")
        interesting = {
            k: v
            for k, v in headers.items()
            if k.lower() in ("content-length", "content-type", "connection", "server")
        }
        log.info(f"[{label}] headers: {interesting}")
        if err:
            log.info(f"[{label}] erro durante stream: {err}")
        if bytes_recv > 0 and not err:
            log.info(f"[{label}] ==> DOWNLOAD FUNCIONA COM ESTA CREDENCIAL <==")
        elif bytes_recv == 0:
            log.info(f"[{label}] ==> download retornou 0 bytes (auth ok mas sem payload)")


def main():
    cfg = load_config()
    dvrs = cfg.get("dvrs", {})
    if not dvrs:
        log.error("Nenhum DVR no config")
        return

    for dvr_name, dvr_cfg in dvrs.items():
        try:
            diagnose_dvr(dvr_name, dvr_cfg)
        except Exception as e:
            log.error(f"Erro diagnosticando {dvr_name}: {e}")


if __name__ == "__main__":
    main()
