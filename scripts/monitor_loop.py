"""
Loop autônomo de monitoramento e auto-correção do pipeline DVR.

Fluxo:
1. Inicia `python -m src.main` em subprocesso.
2. Lê stdout/stderr do pipeline em streaming.
3. Classifica cada linha de log em categorias (OK, WARN, ERROR, FATAL).
4. Agrega estatísticas em janela deslizante.
5. Se a taxa de erro ultrapassa threshold, aplica correção conhecida.
6. Reinicia pipeline se travar ou errar demais.

Correções automáticas implementadas:
- IncompleteRead repetido → throttle (aumenta intervalo de poll)
- force_reprocess_since causando avalanche → desabilita reprocessamento
- UnicodeEncodeError nos logs → define PYTHONIOENCODING=utf-8
- Pipeline sem progresso por >5min → restart

Uso:
    python scripts/monitor_loop.py [--max-iterations N] [--dry-run]
"""

import argparse
import logging
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Thread
from queue import Queue, Empty

LOG_DIR = Path("detections/loop_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

MONITOR_LOG = LOG_DIR / "monitor.log"
PIPELINE_LOG = LOG_DIR / "pipeline.log"
CORRECTIONS_LOG = LOG_DIR / "corrections.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    handlers=[
        logging.FileHandler(MONITOR_LOG, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("monitor")

ERROR_PATTERNS = {
    "incomplete_read": re.compile(r"IncompleteRead"),
    "connection_refused": re.compile(r"WinError 10061|Connection refused"),
    "connection_reset": re.compile(r"WinError 10054|Connection reset"),
    "unicode_encode": re.compile(r"UnicodeEncodeError"),
    "auth_failure": re.compile(r"401|Unauthorized|autentica"),
    "no_space": re.compile(r"No space left|OSError.*28"),
    "yolo_error": re.compile(r"Erro na infer.ncia YOLO"),
}

PROGRESS_PATTERNS = {
    "event_detected": re.compile(r"Novo evento DVR"),
    "frames_extracted": re.compile(r"frame\(s\) extra"),
    "detection_made": re.compile(r"Detec.*#\d+:"),
    "pipeline_started": re.compile(r"Pipeline iniciado"),
}

WINDOW_SIZE = 60
ERROR_RATE_THRESHOLD = 0.4
STALL_TIMEOUT_SEC = 300
MAX_RESTARTS = 5


class PipelineMonitor:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.proc: subprocess.Popen | None = None
        self.log_queue: Queue = Queue()
        self.line_window: deque[tuple[datetime, str, str]] = deque(maxlen=WINDOW_SIZE)
        self.last_progress_at: datetime = datetime.now()
        self.restart_count = 0
        self.corrections_applied: list[str] = []

    def classify(self, line: str) -> str:
        for name, pat in ERROR_PATTERNS.items():
            if pat.search(line):
                return f"ERROR:{name}"
        for name, pat in PROGRESS_PATTERNS.items():
            if pat.search(line):
                return f"PROGRESS:{name}"
        if "ERROR" in line:
            return "ERROR:generic"
        if "WARNING" in line:
            return "WARN:generic"
        return "INFO"

    def start_pipeline(self):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [sys.executable, "-u", "-m", "src.main"]
        logger.info(f"Iniciando pipeline (tentativa #{self.restart_count + 1}): {' '.join(cmd)}")

        if self.dry_run:
            logger.info("[DRY-RUN] pipeline não foi de fato iniciado")
            return

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(Path(__file__).parent.parent),
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.last_progress_at = datetime.now()

        Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        if not self.proc or not self.proc.stdout:
            return
        with open(PIPELINE_LOG, "a", encoding="utf-8") as f:
            for line in self.proc.stdout:
                line = line.rstrip()
                f.write(line + "\n")
                f.flush()
                self.log_queue.put(line)

    def drain_logs(self):
        while True:
            try:
                line = self.log_queue.get_nowait()
            except Empty:
                break
            category = self.classify(line)
            self.line_window.append((datetime.now(), category, line))
            if category.startswith("PROGRESS"):
                self.last_progress_at = datetime.now()

    def compute_stats(self) -> dict:
        if not self.line_window:
            return {"total": 0, "errors": 0, "error_rate": 0.0, "by_error": {}}

        total = len(self.line_window)
        errors = sum(1 for _, cat, _ in self.line_window if cat.startswith("ERROR"))
        by_error: dict[str, int] = {}
        for _, cat, _ in self.line_window:
            if cat.startswith("ERROR:"):
                key = cat.split(":", 1)[1]
                by_error[key] = by_error.get(key, 0) + 1

        return {
            "total": total,
            "errors": errors,
            "error_rate": errors / total if total else 0.0,
            "by_error": by_error,
        }

    def log_correction(self, name: str, details: str):
        self.corrections_applied.append(name)
        msg = f"CORRECAO APLICADA: {name} -- {details}"
        logger.warning(msg)
        with open(CORRECTIONS_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} | {name} | {details}\n")

    def apply_correction_force_reprocess(self) -> bool:
        cfg_path = Path("config/cameras.yaml")
        text = cfg_path.read_text(encoding="utf-8")
        if "force_reprocess_since" not in text:
            return False
        new_text = re.sub(
            r"^\s*force_reprocess_since:.*$",
            "  # force_reprocess_since: desabilitado pelo monitor",
            text,
            flags=re.MULTILINE,
        )
        if new_text == text:
            return False
        if not self.dry_run:
            cfg_path.write_text(new_text, encoding="utf-8")
        self.log_correction(
            "desabilita_force_reprocess",
            "force_reprocess_since comentado -- evita avalanche de downloads",
        )
        return True

    def apply_correction_increase_poll_interval(self) -> bool:
        cfg_path = Path("config/cameras.yaml")
        text = cfg_path.read_text(encoding="utf-8")
        m = re.search(r"poll_interval:\s*(\d+)", text)
        if not m:
            return False
        current = int(m.group(1))
        if current >= 120:
            return False
        new_val = min(120, current * 2)
        new_text = re.sub(r"poll_interval:\s*\d+", f"poll_interval: {new_val}", text)
        if not self.dry_run:
            cfg_path.write_text(new_text, encoding="utf-8")
        self.log_correction(
            "aumenta_poll_interval",
            f"poll_interval {current}s -> {new_val}s (reduz pressao no DVR)",
        )
        return True

    def decide_action(self, stats: dict) -> str | None:
        by_error = stats.get("by_error", {})

        if stats["error_rate"] >= ERROR_RATE_THRESHOLD and stats["total"] >= 20:
            if by_error.get("incomplete_read", 0) >= 5:
                if "desabilita_force_reprocess" not in self.corrections_applied:
                    if self.apply_correction_force_reprocess():
                        return "restart"
                if "aumenta_poll_interval" not in self.corrections_applied:
                    if self.apply_correction_increase_poll_interval():
                        return "restart"
            logger.error(
                f"Taxa de erro alta ({stats['error_rate']:.0%}) sem correcao "
                f"conhecida. Erros: {by_error}"
            )
            return "restart"

        elapsed = (datetime.now() - self.last_progress_at).total_seconds()
        if elapsed > STALL_TIMEOUT_SEC:
            logger.warning(
                f"Pipeline sem progresso ha {elapsed:.0f}s -- restart"
            )
            return "restart"

        return None

    def stop_pipeline(self):
        if not self.proc:
            return
        logger.info("Parando pipeline atual")
        if self.dry_run:
            return
        try:
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def run(self, max_iterations: int | None = None):
        logger.info("=" * 60)
        logger.info("MONITOR LOOP INICIADO")
        logger.info(f"Logs: {LOG_DIR.absolute()}")
        logger.info(f"Dry-run: {self.dry_run}")
        logger.info("=" * 60)

        self.start_pipeline()
        iteration = 0

        try:
            while True:
                iteration += 1
                if max_iterations and iteration > max_iterations:
                    logger.info(f"max_iterations ({max_iterations}) atingido -- encerrando")
                    break

                time.sleep(30)
                self.drain_logs()

                if not self.dry_run and not self.is_alive():
                    logger.error("Pipeline morreu inesperadamente!")
                    if self.restart_count >= MAX_RESTARTS:
                        logger.critical(f"MAX_RESTARTS ({MAX_RESTARTS}) atingido -- aborta")
                        break
                    self.restart_count += 1
                    self.start_pipeline()
                    continue

                stats = self.compute_stats()
                logger.info(
                    f"[iter {iteration}] linhas={stats['total']} "
                    f"erros={stats['errors']} rate={stats['error_rate']:.0%} "
                    f"detalhe={stats['by_error']}"
                )

                action = self.decide_action(stats)
                if action == "restart":
                    if self.restart_count >= MAX_RESTARTS:
                        logger.critical(f"MAX_RESTARTS ({MAX_RESTARTS}) atingido -- aborta")
                        break
                    self.stop_pipeline()
                    self.restart_count += 1
                    self.line_window.clear()
                    self.start_pipeline()

        except KeyboardInterrupt:
            logger.info("Interrompido pelo usuario")
        finally:
            self.stop_pipeline()
            logger.info(
                f"MONITOR LOOP FINALIZADO -- restarts={self.restart_count} "
                f"correcoes={self.corrections_applied}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    monitor = PipelineMonitor(dry_run=args.dry_run)
    monitor.run(max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
