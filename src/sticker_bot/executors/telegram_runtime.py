from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession


logger = logging.getLogger("sticker_bot.executor.telegram")


async def create_executor_bot(token: str, role: str) -> Bot:
    session = AiohttpSession()
    bot = Bot(
        token=token,
        session=session,
        default=DefaultBotProperties(parse_mode="HTML"),
    )
    me = await bot.get_me()
    startup_message = f"Executor startup: role={role} bot=@{me.username} token_id={token.split(':', 1)[0]}"
    print(startup_message, flush=True)
    logger.info(
        startup_message,
    )
    return bot


@asynccontextmanager
async def executor_bot_lifespan(token: str | None, role: str) -> AsyncIterator[Bot | None]:
    if not token:
        logger.warning("Executor startup: role=%s missing Telegram bot token", role)
        yield None
        return

    bot = await create_executor_bot(token, role)
    try:
        yield bot
    finally:
        await bot.session.close()
