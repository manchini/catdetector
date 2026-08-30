"""
Cat Detector — Entry Point
Inicia o pipeline de detecção e o bot do Telegram.
"""

import os
import sys
import asyncio
import logging
import signal
from pathlib import Path

from dotenv import load_dotenv
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
)

from .pipeline import DetectionPipeline
from .bot.handler import handle_classification
from .bot.commands import (
    cmd_start, cmd_stats, cmd_cameras,
    cmd_pause, cmd_resume, cmd_snapshot, cmd_dataset, set_pipeline,
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-20s] %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("cat_detector.log"),
    ],
)
logger = logging.getLogger("cat-detector")

for noisy_logger in ("httpx", "httpcore", "telegram", "telegram.ext", "telegram.request"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def main():
    # Carrega variáveis de ambiente
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        error_msg = (
            "⚠️  Configuração do Telegram incompleta\n"
            "=" * 40 + "\n\n"
            "As variáveis de ambiente TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID "
            "não foram encontradas.\n\n"
            "📝 Para configurar:\n"
            "1. Copie o arquivo .env.example para .env:\n"
            f'   cp .env.example .env\n\n'
            "2. Edite o arquivo .env com seus valores:\n"
            "   TELEGRAM_BOT_TOKEN=seu_token_aqui\n"
            "   TELEGRAM_CHAT_ID=sua_chat_id_aqui\n\n"
            "🔍 Como obter o token do bot:\n"
            "- Chame @BotFather no Telegram para criar um novo bot\n"
            "- Ele retornará um token como: 1234567890:AAH...XY\n\n"
            "📱 Como obter a TELEGRAM_CHAT_ID:\n"
            "- Envie uma mensagem de /start para o seu bot\n"
            "- A resposta mostrará sua Chat ID, OU\n"
            "- Use um serviço como https://telegram.org/chatid/\n\n"
        )
        logger.error(error_msg)
        sys.exit(1)

    # Cria instâncias do pipeline e da aplicação
    pipeline = DetectionPipeline()
    app = ApplicationBuilder().token(token).build()

    # Passa referência do pipeline para os comandos
    set_pipeline(pipeline)

    # Registra comandos
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("cameras", cmd_cameras))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("snapshot", cmd_snapshot))
    app.add_handler(CommandHandler("dataset", cmd_dataset))

    # Registra handler de classificação (botões inline)
    app.add_handler(CallbackQueryHandler(handle_classification))

    # Conecta o bot ao pipeline
    pipeline.bot = app.bot
    pipeline.chat_id = chat_id

    # ---- Inicia tudo ----
    logger.info("=" * 60)
    logger.info("  🐱 Cat Detector — Iniciando...")
    logger.info("=" * 60)

    # Pega o event loop do asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    pipeline.set_event_loop(loop)

    # Inicializa componentes (câmeras, YOLO, DVR clients, etc.)
    pipeline.setup()

    # Inicia o pipeline de detecção (threads)
    pipeline.start()

    # Inicia o bot do Telegram (bloqueia aqui)
    logger.info("Bot do Telegram iniciado. Ctrl+C para parar.")

    try:
        app.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        logger.info("Encerrando...")
    finally:
        pipeline.stop()
        logger.info("Cat Detector encerrado.")


if __name__ == "__main__":
    main()
