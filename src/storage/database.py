"""
Banco de dados SQLite para armazenar detecções e classificações.
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("detections/catdetector.db")

# Serializa escritas: SQLite WAL aceita leituras concorrentes mas só
# uma transação de escrita por vez. Threads de inferência + callbacks do
# Telegram podem entrar em conflito sem este lock.
_WRITE_LOCK = threading.Lock()


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def db_connection() -> Iterator[sqlite3.Connection]:
    """Context manager que garante fechamento da conexão em qualquer caminho."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def db_write() -> Iterator[sqlite3.Connection]:
    """Conexão serializada para operações de escrita."""
    with _WRITE_LOCK, db_connection() as conn:
        yield conn


def init_db() -> None:
    """Cria as tabelas se não existirem."""
    with db_write() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS detections (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
                camera_id       TEXT NOT NULL,
                camera_name     TEXT NOT NULL,
                model_class     TEXT,
                model_confidence REAL,
                user_label      TEXT,
                labeled_at      DATETIME,
                image_path      TEXT NOT NULL,
                crop_path       TEXT,
                bbox            TEXT,
                telegram_msg_id INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_detections_camera
                ON detections(camera_id);
            CREATE INDEX IF NOT EXISTS idx_detections_timestamp
                ON detections(timestamp);
            CREATE INDEX IF NOT EXISTS idx_detections_user_label
                ON detections(user_label);
        """)
        conn.commit()

        try:
            conn.execute("SELECT bbox FROM detections LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE detections ADD COLUMN bbox TEXT")
            conn.commit()
            logger.info("Migração: coluna 'bbox' adicionada")

    logger.info("Banco de dados inicializado")


def save_detection(
    camera_id: str,
    camera_name: str,
    model_class: str,
    model_confidence: float,
    image_path: str,
    crop_path: Optional[str] = None,
    bbox: Optional[tuple] = None,
    telegram_msg_id: Optional[int] = None,
) -> int:
    """Salva uma detecção e retorna o ID."""
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}" if bbox else None
    with db_write() as conn:
        cursor = conn.execute(
            """
            INSERT INTO detections
                (camera_id, camera_name, model_class, model_confidence,
                 image_path, crop_path, bbox, telegram_msg_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (camera_id, camera_name, model_class, model_confidence,
             image_path, crop_path, bbox_str, telegram_msg_id),
        )
        detection_id = cursor.lastrowid
        conn.commit()
    return detection_id


def save_manual_detection(
    camera_id: str,
    camera_name: str,
    timestamp: datetime,
    user_label: str,
    image_path: str,
    bbox: Optional[tuple] = None,
    model_class: str = "manual",
    model_confidence: float = 1.0,
    crop_path: Optional[str] = None,
) -> int:
    """Salva uma anotação manual com timestamp real do frame."""
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}" if bbox else None
    with db_write() as conn:
        cursor = conn.execute(
            """
            INSERT INTO detections
                (timestamp, camera_id, camera_name, model_class, model_confidence,
                 user_label, labeled_at, image_path, crop_path, bbox)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
            """,
            (
                timestamp.isoformat(sep=" ", timespec="seconds"),
                camera_id,
                camera_name,
                model_class,
                model_confidence,
                user_label,
                image_path,
                crop_path,
                bbox_str,
            ),
        )
        detection_id = cursor.lastrowid
        conn.commit()
    return detection_id


def update_label(detection_id: int, user_label: str) -> None:
    """Atualiza a classificação feita pelo usuário via Telegram."""
    with db_write() as conn:
        conn.execute(
            """
            UPDATE detections
            SET user_label = ?, labeled_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (user_label, detection_id),
        )
        conn.commit()
    logger.info(f"Detecção #{detection_id} classificada como: {user_label}")


def update_telegram_msg_id(detection_id: int, msg_id: int) -> None:
    """Atualiza o ID da mensagem do Telegram."""
    with db_write() as conn:
        conn.execute(
            "UPDATE detections SET telegram_msg_id = ? WHERE id = ?",
            (msg_id, detection_id),
        )
        conn.commit()


def get_detection(detection_id: int) -> Optional[dict]:
    """Busca uma detecção pelo ID."""
    with db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM detections WHERE id = ?", (detection_id,)
        ).fetchone()
    return dict(row) if row else None


def get_stats(days: int = 7) -> dict:
    """Estatísticas das últimas N dias."""
    since = datetime.now() - timedelta(days=days)
    with db_connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE timestamp > ?",
            (since.isoformat(),),
        ).fetchone()[0]

        by_label = conn.execute(
            """
            SELECT user_label, COUNT(*) as count
            FROM detections
            WHERE timestamp > ? AND user_label IS NOT NULL
            GROUP BY user_label
            """,
            (since.isoformat(),),
        ).fetchall()

        by_camera = conn.execute(
            """
            SELECT camera_name, COUNT(*) as count
            FROM detections
            WHERE timestamp > ?
            GROUP BY camera_name
            ORDER BY count DESC
            """,
            (since.isoformat(),),
        ).fetchall()

        unlabeled = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE user_label IS NULL",
        ).fetchone()[0]

        accuracy_row = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE
                    WHEN model_class = user_label THEN 1
                    WHEN model_class = 'dog' AND user_label = 'my_dog' THEN 1
                    WHEN user_label IN ('false_positive', 'other') THEN 0
                    ELSE 0
                END) as correct
            FROM detections
            WHERE user_label IS NOT NULL AND timestamp > ?
            """,
            (since.isoformat(),),
        ).fetchone()

    accuracy = None
    if accuracy_row and accuracy_row[0] > 0:
        accuracy = accuracy_row[1] / accuracy_row[0]

    return {
        "total": total,
        "by_label": {row[0]: row[1] for row in by_label},
        "by_camera": {row[0]: row[1] for row in by_camera},
        "unlabeled": unlabeled,
        "model_accuracy": accuracy,
        "days": days,
    }


def get_labeled_counts() -> dict:
    """Contagem total de imagens classificadas por label."""
    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT user_label, COUNT(*) as count
            FROM detections
            WHERE user_label IS NOT NULL
            GROUP BY user_label
            """
        ).fetchall()
    return {row[0]: row[1] for row in rows}
