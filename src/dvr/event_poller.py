"""
Thread de polling de eventos de movimento do DVR Intelbras/Dahua.

A cada `poll_interval` segundos, consulta a API HTTP do DVR por novos
segmentos de motion em cada canal configurado e dispara um callback
com cada RecordedFile novo encontrado.

Integração com o pipeline:
    poller = DVREventPoller(
        client=DahuaHTTPClient(...),
        channels=[1, 2, 3],
        dvr_name="MHDX",
        on_event=pipeline._on_dvr_event,
    )
    poller.start()
    ...
    poller.stop()
"""

import logging
import threading
import time
from datetime import datetime
from typing import Callable

from src.dvr.dahua_api import DahuaHTTPClient, RecordedFile
from src.dvr.state import DVRPollerState

logger = logging.getLogger(__name__)

class DVREventPoller:
    """
    Consulta periodicamente o DVR por novos segmentos de movimento e chama
    `on_event(rec)` para cada RecordedFile novo.

    Thread-safe; pode haver um poller por DVR.
    """

    def __init__(
        self,
        client: DahuaHTTPClient,
        channels: list[int],
        dvr_name: str,
        on_event: Callable[[RecordedFile], bool],
        poll_interval: int = 30,
        state: DVRPollerState | None = None,
        force_reprocess_since: datetime | None = None,
    ) -> None:
        """
        Args:
            client: Cliente HTTP já configurado para o DVR.
            channels: Lista de canais a monitorar (1-based).
            dvr_name: Nome do DVR (usado como chave no estado persistido).
            on_event: Callback chamado para cada RecordedFile novo.
                      Retorna True para avancar estado; False para retry.
                      Executado na thread do poller — deve ser thread-safe.
            poll_interval: Segundos entre consultas ao DVR.
            state: Instância de DVRPollerState compartilhada (ou nova se None).
            force_reprocess_since: Se fornecido, força reprocessamento desde esta data
                                   (ignora o estado persistido).
        """
        self.client = client
        self.channels = channels
        self.dvr_name = dvr_name
        self.on_event = on_event
        self.poll_interval = poll_interval
        self._state = state or DVRPollerState()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._force_reprocess_since = force_reprocess_since
        # chave: (channel, start_time_iso) → nº de falhas consecutivas
        self._fail_counts: dict[tuple, int] = {}
        self._max_event_retries = 3

    def start(self) -> None:
        """Inicia a thread de polling em background."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"dvr-poller-{self.dvr_name}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"DVREventPoller iniciado para {self.dvr_name} "
            f"(canais: {self.channels}, intervalo: {self.poll_interval}s)"
        )

    def stop(self) -> None:
        """Sinaliza parada e aguarda a thread encerrar."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning(
                    f"DVREventPoller {self.dvr_name}: thread nao encerrou em 5s "
                    f"(provavelmente bloqueada em download)"
                )

    def _run(self) -> None:
        """Loop principal da thread."""
        # Testa conectividade na primeira iteração antes de entrar no loop
        if not self.client.check_connection():
            logger.error(
                f"DVREventPoller {self.dvr_name}: sem conexão com o DVR. "
                f"Tentará novamente a cada ciclo."
            )

        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception:
                # Loop principal não pode morrer; log com traceback completo.
                logger.exception(f"DVREventPoller {self.dvr_name}: erro no poll")

            # Aguarda o intervalo mas acorda a cada segundo para checar stop
            for _ in range(self.poll_interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def _poll_once(self) -> None:
        """Executa uma rodada de consulta para todos os canais."""
        now = datetime.now()

        for channel in self.channels:
            if self._stop_event.is_set():
                break

            # Se força reprocessamento, ignora estado persistido
            if self._force_reprocess_since:
                last_seen = self._force_reprocess_since
            else:
                last_seen = self._state.get_last_seen(self.dvr_name, channel)

            # Consulta desde last_seen para retry real de eventos falhos.
            query_start = last_seen

            try:
                events = self.client.list_motion_events(
                    channel=channel,
                    start=query_start,
                    end=now,
                    since=last_seen,
                )
            except Exception as e:
                logger.warning(
                    f"DVR {self.dvr_name} ch{channel}: erro ao listar eventos: {e}"
                )
                continue

            if not events:
                logger.debug(
                    f"DVR {self.dvr_name} ch{channel}: nenhum evento novo"
                )
                continue

            logger.info(
                f"DVR {self.dvr_name} ch{channel}: "
                f"{len(events)} novo(s) evento(s) de motion"
            )

            newest_ts = last_seen
            for rec in sorted(events, key=lambda item: item.start_time):
                fail_key = (channel, rec.start_time.isoformat())
                fail_count = self._fail_counts.get(fail_key, 0)

                if fail_count >= self._max_event_retries:
                    logger.warning(
                        f"DVR {self.dvr_name} ch{channel}: evento "
                        f"{rec.start_time.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"ignorado apos {fail_count} falhas consecutivas"
                    )
                    # Avança o estado para não bloquear eventos futuros
                    if rec.start_time > newest_ts:
                        newest_ts = rec.start_time
                    continue

                processed = False
                try:
                    processed = bool(self.on_event(rec))
                except Exception:
                    # Callback falhou: log com traceback e trata como falha
                    # transitória (será re-tentado até _max_event_retries).
                    logger.exception(
                        f"DVR {self.dvr_name} ch{channel}: "
                        f"erro no callback on_event"
                    )

                if not processed:
                    self._fail_counts[fail_key] = fail_count + 1
                    logger.warning(
                        f"DVR {self.dvr_name} ch{channel}: evento "
                        f"{rec.start_time.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"nao concluido (falha {fail_count + 1}/{self._max_event_retries})"
                    )
                    break

                self._fail_counts.pop(fail_key, None)
                if rec.start_time > newest_ts:
                    newest_ts = rec.start_time

            # Atualiza estado apenas se houve eventos novos
            if newest_ts > last_seen:
                self._state.update_last_seen(self.dvr_name, channel, newest_ts)
