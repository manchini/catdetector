"""
Persistência do estado de polling do DVR.
Guarda o último timestamp processado por (dvr_name, channel) para que, após
um restart, o poller não reprocesse eventos já vistos nem perca os do intervalo.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

STATE_FILE = Path("detections/dvr_state.json")
DEFAULT_LOOKBACK_MINUTES = 5


class DVRPollerState:
    """
    Persiste o último start_time de evento visto por canal de cada DVR.

    Formato do JSON:
        {
            "dvr_name": {
                "1": "2026-04-14T10:23:00",
                "2": "2026-04-14T10:21:45"
            }
        }
    """

    def __init__(self, state_file: Path = STATE_FILE):
        self._file = state_file
        self._lock = Lock()
        self._data: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self):
        try:
            if self._file.exists():
                with open(self._file, encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.debug(f"Estado DVR carregado de {self._file}")
        except Exception as e:
            logger.warning(f"Não foi possível carregar estado DVR ({self._file}): {e}")
            self._data = {}

    def _save(self):
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self._file)
        except Exception as e:
            logger.warning(f"Não foi possível salvar estado DVR: {e}")

    def get_last_seen(self, dvr_name: str, channel: int) -> datetime:
        """
        Retorna o último timestamp visto para o canal.
        Na primeira vez, retorna `now - DEFAULT_LOOKBACK_MINUTES`.
        """
        with self._lock:
            ch_str = str(channel)
            iso = self._data.get(dvr_name, {}).get(ch_str)
            if iso:
                try:
                    return datetime.fromisoformat(iso)
                except ValueError:
                    pass
            return datetime.now() - timedelta(minutes=DEFAULT_LOOKBACK_MINUTES)

    def update_last_seen(self, dvr_name: str, channel: int, ts: datetime):
        """Atualiza e persiste o timestamp mais recente para o canal."""
        with self._lock:
            ch_str = str(channel)
            self._data.setdefault(dvr_name, {})[ch_str] = ts.isoformat(
                timespec="seconds"
            )
            self._save()
