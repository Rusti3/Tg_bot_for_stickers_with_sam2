from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sticker_bot.domain.tasks import AddCommandOptions, StickerTask
from sticker_bot.services.masking import parse_back_param
from sticker_bot.storage.rate_limits import RateLimitStore

if TYPE_CHECKING:
    from aiogram import Router
else:
    Router = Any


def parse_add_command_args(args_text: str | None, max_grid_width: int) -> AddCommandOptions:
    options = AddCommandOptions()
    args = args_text.split() if args_text else []

    for arg in args:
        if arg.isdigit():
            options.w_count = min(max(int(arg), 1), max_grid_width)
            continue

        if not arg.startswith("back="):
            continue

        raw_value = arg.replace("back=", "", 1).lower()
        if raw_value in {"auto", "none"}:
            options.back_mode = raw_value
            continue

        color_target, tolerance = parse_back_param(raw_value)
        if color_target:
            options.back_mode = raw_value
            options.tolerance = tolerance

    return options


def create_router(settings, rate_limits: RateLimitStore, scheduler) -> Router:
    from aiogram import F, Router
    from aiogram.filters import Command, CommandObject
    from aiogram.types import Message

    router = Router()

    @router.message(Command("stats"))
    async def handle_stats(message: Message) -> None:
        total_users, total_requests = rate_limits.stats()
        await message.answer(f"Total users: {total_users}\nCreated packs: {total_requests}")

    @router.message(Command("add"), F.photo | F.video | F.animation | F.document | F.reply_to_message)
    async def handle_add(message: Message, command: CommandObject) -> None:
        user_id = message.from_user.id
        if not rate_limits.record_request(user_id):
            await message.answer("Rate limit reached. Try again later.")
            return

        target = message.reply_to_message or message
        file_obj = target.photo[-1] if target.photo else (target.animation or target.video or target.document)
        if not file_obj:
            await message.answer("File not found.")
            return

        options = parse_add_command_args(command.args, settings.max_grid_width)
        document_mime = (target.document.mime_type or "").lower() if target.document else ""
        document_name = (target.document.file_name or "").lower() if target.document else ""
        is_gif = bool(target.document and (document_mime == "image/gif" or document_name.endswith(".gif")))
        is_video = bool(target.video or target.animation or (target.document and ("video" in document_mime or is_gif)))

        status_message = await message.answer("Queue: ...")
        task = StickerTask(
            user_id=user_id,
            user_name=message.from_user.username or str(user_id),
            file_id=file_obj.file_id,
            w_count=options.w_count,
            back_mode=options.back_mode,
            tolerance=options.tolerance,
            is_video=is_video,
            is_gif=is_gif,
            status_message=status_message,
            source_message=message,
        )
        queue_size = await scheduler.enqueue(task)
        await status_message.edit_text(f"Queue: {queue_size}")

    return router
