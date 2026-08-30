"""
Handlers para os callbacks dos botões inline do Telegram.
Quando o usuário toca em 🐱/🐶/❌, processa a classificação.
"""

import os
import logging

from telegram import Update
from telegram.ext import ContextTypes

from ..storage import database, dataset

logger = logging.getLogger(__name__)


async def _check_auth(update: Update) -> bool:
    """Verifica se o usuário está autorizado."""
    expected_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not expected_chat_id:
        return True  # Sem configuração, permite

    actual_chat_id = str(update.effective_chat.id)
    if actual_chat_id != expected_chat_id:
        logger.warning(f"Acesso não autorizado de chat_id {actual_chat_id}")
        return False
    return True

# Mapa de labels para texto amigável
LABEL_NAMES = {
    "cat": "🐱 Gato",
    "my_dog": "🐶 Meu Cachorro",
    "other": "🦝 Outro Animal",
    "false_positive": "❌ Falso Alarme",
}


async def handle_classification(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    """
    Processa o callback quando o usuário toca um botão de classificação.
    Formato do callback_data: "label:{detection_id}:{label}"
    """
    query = update.callback_query
    await query.answer()

    if not await _check_auth(update):
        return

    data = query.data
    if not data.startswith("label:"):
        return

    parts = data.split(":")
    if len(parts) != 3:
        return

    _, detection_id_str, label = parts

    try:
        detection_id = int(detection_id_str)
    except ValueError:
        return

    if label not in LABEL_NAMES:
        return

    # Busca a detecção no banco
    detection = database.get_detection(detection_id)
    if not detection:
        await query.edit_message_caption(
            caption="⚠️ Detecção não encontrada no banco de dados."
        )
        return

    # Atualiza o label no banco
    database.update_label(detection_id, label)

    # Move a imagem para a pasta correta do dataset
    if detection.get("image_path"):
        dataset.label_image(detection["image_path"], label)
    if detection.get("crop_path"):
        dataset.label_image(detection["crop_path"], label)

    # Contagens atualizadas
    counts = database.get_labeled_counts()
    counts_text = " | ".join(
        f"{LABEL_NAMES.get(k, k)}: {v}" for k, v in counts.items()
    )

    # Atualiza a mensagem original com a classificação
    label_name = LABEL_NAMES[label]
    original_caption = query.message.caption or ""

    new_caption = (
        f"{original_caption}\n\n"
        f"✅ Classificado como: *{label_name}*\n"
        f"📊 {counts_text}"
    )

    try:
        await query.edit_message_caption(
            caption=new_caption,
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.warning(f"Erro ao editar mensagem: {e}")

    logger.info(
        f"Detecção #{detection_id} classificada como {label} "
        f"pelo usuário via Telegram"
    )
