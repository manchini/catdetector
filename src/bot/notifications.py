"""
Envia notificações no Telegram com foto do animal detectado
e botões inline para classificação (gato / meu cachorro / nada).
"""

import logging
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

logger = logging.getLogger(__name__)


def build_detection_message(
    camera_name: str,
    model_class: str,
    confidence: float,
    timestamp: str,
) -> str:
    """Monta o texto da notificação."""
    emoji = "🐱" if model_class == "cat" else "🐶" if model_class == "dog" else "❓"

    return (
        f"{emoji} *Animal detectado!*\n"
        f"📷 Câmera: *{camera_name}*\n"
        f"🕐 {timestamp}\n"
        f"🤖 Modelo: {model_class} ({confidence:.0%})"
    )


def build_classification_keyboard(detection_id: int) -> InlineKeyboardMarkup:
    """Monta os botões inline para classificação."""
    keyboard = [
        [
            InlineKeyboardButton("🐱 Gato", callback_data=f"label:{detection_id}:cat"),
            InlineKeyboardButton(
                "🐶 Meu Cachorro", callback_data=f"label:{detection_id}:my_dog"
            ),
        ],
        [
            InlineKeyboardButton("🦝 Outro", callback_data=f"label:{detection_id}:other"),
            InlineKeyboardButton(
                "❌ Falso Alarme", callback_data=f"label:{detection_id}:false_positive"
            ),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


async def send_detection_alert(
    bot,
    chat_id: str,
    detection_id: int,
    image_path: str,
    camera_name: str,
    model_class: str,
    confidence: float,
    timestamp: str,
) -> int:
    """
    Envia alerta com foto e botões de classificação.
    Retorna o message_id do Telegram.
    """
    text = build_detection_message(camera_name, model_class, confidence, timestamp)
    keyboard = build_classification_keyboard(detection_id)

    photo_path = Path(image_path)
    if not photo_path.exists():
        logger.error(f"Imagem não encontrada: {image_path}")
        msg = await bot.send_message(
            chat_id=chat_id, text=text + "\n\n⚠️ Imagem não disponível",
            parse_mode="Markdown", reply_markup=keyboard,
        )
        return msg.message_id

    with open(photo_path, "rb") as photo:
        msg = await bot.send_photo(
            chat_id=chat_id, photo=photo, caption=text,
            parse_mode="Markdown", reply_markup=keyboard,
            read_timeout=30, connect_timeout=10,
        )

    logger.info(f"Alerta enviado: detecção #{detection_id} ({camera_name})")
    return msg.message_id


async def send_daily_report(bot, chat_id: str, stats: dict):
    """Envia relatório diário de detecções."""
    labels = stats.get("by_label", {})
    cameras = stats.get("by_camera", {})

    text = "📊 *Relatório Diário*\n\n"
    text += f"Total de detecções: *{stats['total']}*\n"

    if labels:
        text += "\nPor classificação:\n"
        emoji_map = {
            "cat": "🐱", "my_dog": "🐶",
            "other": "🦝", "false_positive": "❌",
        }
        for label, count in labels.items():
            emoji = emoji_map.get(label, "•")
            text += f"  {emoji} {label}: {count}\n"

    if cameras:
        text += "\nPor câmera:\n"
        for cam, count in cameras.items():
            text += f"  📷 {cam}: {count}\n"

    if stats.get("unlabeled", 0) > 0:
        text += f"\n⚠️ Não classificadas: {stats['unlabeled']}"

    if stats.get("model_accuracy") is not None:
        text += f"\n🎯 Precisão do modelo: {stats['model_accuracy']:.0%}"

    await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
    logger.info("Relatório diário enviado")
