from __future__ import annotations

import logging
import shutil

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession

from sticker_bot.bot.handlers import create_router
from sticker_bot.config import load_settings
from sticker_bot.services.masking import MaskingService
from sticker_bot.services.processing import ProcessingService
from sticker_bot.services.scheduler import TaskScheduler
from sticker_bot.services.uploading import UploadService
from sticker_bot.storage.rate_limits import RateLimitStore


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


async def main() -> None:
    configure_logging()
    settings = load_settings()
    settings.ensure_runtime_dirs()

    if not settings.bot_token:
        raise RuntimeError("BOT_TOKEN not set. Create .env with BOT_TOKEN=...")
    if not settings.sam2_checkpoint.exists():
        raise RuntimeError(f"SAM2 checkpoint not found: {settings.sam2_checkpoint}")
    if not settings.sam2_config_dir.exists():
        raise RuntimeError(f"SAM2 config directory not found: {settings.sam2_config_dir}")
    if shutil.which(settings.ffmpeg_cmd) is None:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg and restart the shell.")

    rate_limits = RateLimitStore(
        settings.limits_file,
        window_seconds=settings.rate_limit_window,
        max_requests=settings.max_files_per_hour,
    )
    masking = MaskingService(settings)
    masking.initialize()
    processor = ProcessingService(settings, masking)
    uploader = UploadService(settings)
    scheduler = TaskScheduler(settings, processor, uploader)

    dispatcher = Dispatcher()
    dispatcher.include_router(create_router(settings, rate_limits, scheduler))

    session = AiohttpSession()
    bot = Bot(
        token=settings.bot_token,
        session=session,
        default=DefaultBotProperties(parse_mode="HTML"),
    )

    workers = await scheduler.start_workers(bot)
    try:
        await dispatcher.start_polling(bot)
    finally:
        await scheduler.stop_workers(workers)
        await bot.session.close()

