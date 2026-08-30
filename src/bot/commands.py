"""
Comandos do Telegram Bot.
/start, /stats, /cameras, /pause, /resume, /snapshot
"""

import os
import logging

import cv2
from telegram import Update
from telegram.ext import ContextTypes

from ..storage import database, dataset

logger = logging.getLogger(__name__)

# Referência global ao pipeline (setada pelo main.py)
_pipeline = None


def set_pipeline(pipeline):
    global _pipeline
    _pipeline = pipeline


async def _check_auth(update: Update) -> bool:
    """Verifica se o usuário está autorizado."""
    expected_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not expected_chat_id:
        return True  # Sem configuração, permite

    actual_chat_id = str(update.effective_chat.id)
    if actual_chat_id != expected_chat_id:
        await update.message.reply_text("❌ Acesso negado. Você não está autorizado.")
        logger.warning(f"Acesso não autorizado de chat_id {actual_chat_id}")
        return False
    return True


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start — mostra status do sistema."""
    if not await _check_auth(update):
        return

    cam_status = ""
    if _pipeline:
        for cam_id, stream in _pipeline.cameras.items():
            state = stream.state
            icon = "🟢" if state.connected else "🔴"
            paused = " (pausada)" if state.paused else ""
            cam_status += f"  {icon} {stream.config.name}{paused}\n"

    ds_stats = dataset.get_dataset_stats()

    text = (
        "🐱 *Cat Detector — Online*\n\n"
        f"📷 *Câmeras:*\n{cam_status}\n"
        f"📊 *Dataset:*\n"
        f"  Classificadas: {ds_stats['total_labeled']}\n"
        f"  Não classificadas: {ds_stats['unlabeled']}\n\n"
        "💡 *Comandos:*\n"
        "  /stats — Estatísticas\n"
        "  /cameras — Status das câmeras\n"
        "  /snapshot cam\\_id — Foto atual da câmera\n"
        "  /pause cam\\_id — Pausar câmera\n"
        "  /resume cam\\_id — Retomar câmera\n"
        "  /dataset — Status do dataset para treino"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /stats — estatísticas de detecção."""
    if not await _check_auth(update):
        return

    days = 7
    if context.args:
        try:
            days = int(context.args[0])
        except ValueError:
            pass

    stats = database.get_stats(days=days)
    labels = stats.get("by_label", {})
    cameras = stats.get("by_camera", {})

    text = f"📊 *Estatísticas — últimos {days} dias*\n\n"
    text += f"Total: *{stats['total']}* detecções\n"

    if labels:
        text += "\n*Por classificação:*\n"
        emoji_map = {
            "cat": "🐱", "my_dog": "🐶",
            "other": "🦝", "false_positive": "❌",
        }
        for label, count in labels.items():
            emoji = emoji_map.get(label, "•")
            text += f"  {emoji} {label}: {count}\n"

    if cameras:
        text += "\n*Por câmera:*\n"
        for cam, count in cameras.items():
            text += f"  📷 {cam}: {count}\n"

    if stats["model_accuracy"] is not None:
        text += f"\n🎯 Precisão do modelo: {stats['model_accuracy']:.0%}"

    if stats["unlabeled"] > 0:
        text += f"\n⚠️ Pendentes de classificação: {stats['unlabeled']}"

    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_cameras(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /cameras — status detalhado das câmeras."""
    if not await _check_auth(update):
        return

    if not _pipeline:
        await update.message.reply_text("Sistema não inicializado.")
        return

    text = "📷 *Status das Câmeras*\n\n"
    for cam_id, stream in _pipeline.cameras.items():
        state = stream.state
        icon = "🟢" if state.connected else "🔴"
        paused = " ⏸️" if state.paused else ""

        text += (
            f"{icon} *{stream.config.name}*{paused}\n"
            f"  ID: `{cam_id}`\n"
            f"  Frames: {state.total_frames}\n"
            f"  Erros: {state.errors}\n\n"
        )

    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /pause cam_id — pausa uma câmera."""
    if not await _check_auth(update):
        return

    if not context.args:
        await update.message.reply_text(
            "Uso: /pause cam\\_id\nExemplo: /pause cam\\_frente"
        )
        return

    cam_id = context.args[0]
    if _pipeline and cam_id in _pipeline.cameras:
        _pipeline.cameras[cam_id].pause()
        await update.message.reply_text(f"⏸️ Câmera `{cam_id}` pausada.")
    else:
        await update.message.reply_text(f"❌ Câmera `{cam_id}` não encontrada.")


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /resume cam_id — retoma uma câmera."""
    if not await _check_auth(update):
        return

    if not context.args:
        await update.message.reply_text(
            "Uso: /resume cam\\_id\nExemplo: /resume cam\\_frente"
        )
        return

    cam_id = context.args[0]
    if _pipeline and cam_id in _pipeline.cameras:
        _pipeline.cameras[cam_id].resume()
        await update.message.reply_text(f"▶️ Câmera `{cam_id}` retomada.")
    else:
        await update.message.reply_text(f"❌ Câmera `{cam_id}` não encontrada.")


async def cmd_dataset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /dataset — status da coleta de dados para fine-tuning."""
    if not await _check_auth(update):
        return

    ds_stats = dataset.get_dataset_stats()
    labeled = ds_stats.get("labeled", {})

    # Conta detecções com bbox no banco (necessário para treino)
    from ..storage import database as db
    conn = db.get_connection()
    try:
        bbox_counts = conn.execute(
            """
            SELECT user_label, COUNT(*) as count
            FROM detections
            WHERE user_label IN ('cat', 'my_dog')
              AND bbox IS NOT NULL
            GROUP BY user_label
            """
        ).fetchall()
    finally:
        conn.close()
    bbox_map = {row[0]: row[1] for row in bbox_counts}

    min_samples = 50
    cat_count = bbox_map.get("cat", 0)
    dog_count = bbox_map.get("my_dog", 0)

    text = "🎯 *Dataset para Fine\\-tuning*\n\n"
    text += "*Imagens classificadas \\(com bbox\\):*\n"
    text += f"  🐱 Gato: {cat_count}"
    if cat_count >= min_samples:
        text += " ✅\n"
    else:
        text += f" \\(faltam {min_samples - cat_count}\\)\n"

    text += f"  🐶 Cachorro: {dog_count}"
    if dog_count >= min_samples:
        text += " ✅\n"
    else:
        text += f" \\(faltam {min_samples - dog_count}\\)\n"

    text += f"\n*Mínimo recomendado:* {min_samples} por classe\n"

    total = cat_count + dog_count
    if total >= min_samples * 2:
        text += "\n✅ *Pronto para treinar\\!*\n"
        text += "`python \\-m src\\.export\\_dataset`\n"
        text += "`python \\-m src\\.train`"
    else:
        text += "\n⏳ Continue classificando detecções pelo Telegram\\."

    await update.message.reply_text(text, parse_mode="MarkdownV2")


async def cmd_snapshot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /snapshot cam_id — envia frame atual de uma câmera."""
    if not await _check_auth(update):
        return

    if not _pipeline:
        await update.message.reply_text("Sistema não inicializado.")
        return

    # Sem argumento: lista câmeras disponíveis
    if not context.args:
        cam_list = "\n".join(
            f"  `{cid}` — {s.config.name}"
            for cid, s in _pipeline.cameras.items()
        )
        await update.message.reply_text(
            f"Uso: /snapshot cam\\_id\n\n📷 *Câmeras:*\n{cam_list}",
            parse_mode="Markdown",
        )
        return

    cam_id = context.args[0]
    if cam_id not in _pipeline.cameras:
        await update.message.reply_text(f"❌ Câmera `{cam_id}` não encontrada.")
        return

    stream = _pipeline.cameras[cam_id]
    if not stream.state.connected:
        await update.message.reply_text(f"🔴 Câmera `{cam_id}` desconectada.")
        return

    frame = stream.read_frame()
    if frame is None:
        await update.message.reply_text(f"⚠️ Não foi possível capturar frame de `{cam_id}`.")
        return

    # Codifica frame como JPEG em memória
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    from io import BytesIO
    photo = BytesIO(buf.tobytes())
    photo.name = f"{cam_id}.jpg"

    await update.message.reply_photo(
        photo=photo,
        caption=f"📷 {stream.config.name} (`{cam_id}`)",
        parse_mode="Markdown",
    )
