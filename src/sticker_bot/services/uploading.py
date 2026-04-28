from __future__ import annotations

import asyncio
import io
import logging
import secrets
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter
    from aiogram.types import BufferedInputFile, InlineKeyboardButton, InputSticker
    from aiogram.utils.keyboard import InlineKeyboardBuilder
except ModuleNotFoundError:
    class TelegramRetryAfter(Exception):
        def __init__(self, retry_after: int = 0):
            super().__init__("retry later")
            self.retry_after = retry_after

    class TelegramNetworkError(Exception):
        pass

    class BufferedInputFile:
        def __init__(self, data: bytes, filename: str):
            self.data = data
            self.filename = filename

    class InputSticker:
        def __init__(self, *, sticker, emoji_list, format):
            self.sticker = sticker
            self.emoji_list = emoji_list
            self.format = format

    class InlineKeyboardButton:
        def __init__(self, *, text: str, url: str):
            self.text = text
            self.url = url

    class _Markup:
        def __init__(self, rows):
            self.rows = rows

    class InlineKeyboardBuilder:
        def __init__(self) -> None:
            self.rows = []

        def row(self, *buttons):
            self.rows.append(buttons)
            return self

        def as_markup(self):
            return _Markup(self.rows)

from sticker_bot.config import Settings
from sticker_bot.domain.tasks import StickerTask


logger = logging.getLogger("StickerBot")
PUZZLE_EMOJI = ["🧩"]


async def safe_api_call(func, *args, **kwargs):
    while True:
        try:
            return await func(*args, **kwargs)
        except TelegramRetryAfter as exc:
            await asyncio.sleep(exc.retry_after)
        except (TelegramNetworkError, asyncio.TimeoutError):
            await asyncio.sleep(5)


class UploadService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def compress_sticker(
        self,
        segment_data: bytes,
        fmt: str,
        ext: str,
        compression_level: int = 1,
    ) -> bytes:
        try:
            if fmt == "static":
                image = Image.open(io.BytesIO(segment_data))
                quality = max(20, 95 - compression_level * 15)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG", optimize=True, quality=quality)
                return buffer.getvalue()

            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp_in:
                tmp_in.write(segment_data)
                input_path = Path(tmp_in.name)
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp_out:
                output_path = Path(tmp_out.name)

            crf = 28 + compression_level * 2
            bitrate = max(50, 200 - compression_level * 10)
            command = [
                self.settings.ffmpeg_cmd,
                "-y",
                "-i",
                str(input_path),
                "-c:v",
                "libvpx-vp9",
                "-pix_fmt",
                "yuva420p",
                "-crf",
                str(crf),
                "-b:v",
                f"{bitrate}k",
                "-deadline",
                "realtime",
                "-an",
                str(output_path),
            ]
            subprocess.run(command, stderr=subprocess.DEVNULL, check=True)
            result = output_path.read_bytes()
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            return result
        except Exception as exc:
            logger.error("Compression error (%s): %s", fmt, exc)
            return segment_data

    async def create_sticker_set(
        self,
        bot,
        *,
        user_id: int,
        pack_name: str,
        title: str,
        first_segment: bytes,
        emoji_list: list[str],
        fmt: str,
        ext: str,
        max_retries: int = 5,
    ) -> bytes:
        segment = first_segment
        for attempt in range(max_retries):
            try:
                await safe_api_call(
                    bot.create_new_sticker_set,
                    user_id=user_id,
                    name=pack_name,
                    title=title,
                    stickers=[
                        InputSticker(
                            sticker=BufferedInputFile(segment, filename=f"0.{ext}"),
                            emoji_list=emoji_list,
                            format=fmt,
                        )
                    ],
                    sticker_type="custom_emoji",
                )
                return segment
            except TelegramRetryAfter as exc:
                logger.warning("Rate limit, waiting %ss", exc.retry_after)
                await asyncio.sleep(exc.retry_after)
            except Exception as exc:
                error_msg = str(exc).lower()
                if "file is too big" in error_msg or "too big" in error_msg:
                    segment = await self.compress_sticker(segment, fmt, ext, attempt + 1)
                    await asyncio.sleep(0.5)
                    continue
                logger.error("Sticker pack creation error: %s", exc)
                raise
        raise RuntimeError(f"Failed to create sticker pack after {max_retries} attempts")

    async def upload_single_sticker(
        self,
        bot,
        user_id: int,
        pack_name: str,
        segment: bytes,
        emoji_list: list[str],
        fmt: str,
        ext: str,
        *,
        max_retries: int = 5,
    ) -> bool:
        for attempt in range(max_retries):
            try:
                await safe_api_call(
                    bot.add_sticker_to_set,
                    user_id=user_id,
                    name=pack_name,
                    sticker=InputSticker(
                        sticker=BufferedInputFile(segment, filename=f"sticker.{ext}"),
                        emoji_list=emoji_list,
                        format=fmt,
                    ),
                )
                return True
            except TelegramRetryAfter as exc:
                logger.warning("Rate limit, waiting %ss", exc.retry_after)
                await asyncio.sleep(exc.retry_after)
            except Exception as exc:
                error_msg = str(exc).lower()
                if "file is too big" in error_msg or "too big" in error_msg:
                    segment = await self.compress_sticker(segment, fmt, ext, attempt + 1)
                    await asyncio.sleep(0.5)
                    continue
                logger.error("Sticker upload error: %s", exc)
                raise
        logger.error("Failed to upload sticker after %s attempts", max_retries)
        return False

    @staticmethod
    def _render_grid_html(custom_emoji_ids: list[str], cols: int, rows: int) -> str:
        return "\n".join(
            "".join(
                f'<tg-emoji emoji-id="{custom_emoji_ids[row * cols + col]}">🧩</tg-emoji>'
                for col in range(cols)
            )
            for row in range(rows)
        )

    async def publish_pack_from_delivery(
        self,
        bot,
        *,
        user_id: int,
        user_name: str,
        chat_id: int,
        reply_to_message_id: int | None,
        segments: list[bytes],
        cols: int,
        rows: int,
        fmt: str,
        ext: str,
        start_time: float | None = None,
    ) -> dict[str, Any]:
        if not segments:
            raise RuntimeError("Puzzle segments are missing.")

        bot_username = (await bot.get_me()).username
        pack_name = f"puzzle_{secrets.token_hex(4)}_by_{bot_username}"
        await self.create_sticker_set(
            bot,
            user_id=user_id,
            pack_name=pack_name,
            title=f"Puzzle {cols}x{rows}",
            first_segment=segments[0],
            emoji_list=PUZZLE_EMOJI,
            fmt=fmt,
            ext=ext,
        )

        failed_count = 0
        for index in range(1, len(segments)):
            uploaded = await self.upload_single_sticker(
                bot,
                user_id,
                pack_name,
                segments[index],
                PUZZLE_EMOJI,
                fmt,
                ext,
            )
            if not uploaded:
                failed_count += 1
                logger.warning("Skipped sticker %s/%s", index, len(segments))

        sticker_set = await bot.get_sticker_set(pack_name)
        custom_emoji_ids = [sticker.custom_emoji_id for sticker in sticker_set.stickers]
        if len(custom_emoji_ids) < cols * rows or any(not emoji_id for emoji_id in custom_emoji_ids):
            raise RuntimeError("Telegram returned incomplete custom emoji ids for the created puzzle pack.")

        keyboard = InlineKeyboardBuilder().row(
            InlineKeyboardButton(
                text="Install",
                url=f"https://t.me/addstickers/{pack_name}",
            )
        )
        await safe_api_call(
            bot.send_message,
            chat_id=chat_id,
            text=self._render_grid_html(custom_emoji_ids, cols, rows),
            reply_markup=keyboard.as_markup(),
            reply_to_message_id=reply_to_message_id,
        )
        logger.info(
            "Task completed: user=%s, grid=%sx%s, time=%.2fs, skipped=%s",
            user_name,
            cols,
            rows,
            (time.perf_counter() - start_time) if start_time is not None else 0.0,
            failed_count,
        )
        return {
            "packName": pack_name,
            "customEmojiIds": [emoji_id for emoji_id in custom_emoji_ids if emoji_id],
            "failedCount": failed_count,
        }

    async def publish_pack(
        self,
        bot,
        task: StickerTask,
        user_dir: Path,
        segments: list[bytes],
        cols: int,
        rows: int,
        fmt: str,
        ext: str,
    ) -> None:
        try:
            await safe_api_call(task.status_message.edit_text, text="Creating sticker pack...")
            await self.publish_pack_from_delivery(
                bot,
                user_id=task.user_id,
                user_name=task.user_name,
                chat_id=task.source_message.chat.id,
                reply_to_message_id=task.source_message.message_id,
                segments=segments,
                cols=cols,
                rows=rows,
                fmt=fmt,
                ext=ext,
                start_time=task.start_time,
            )
            await safe_api_call(task.status_message.delete)
        except Exception:
            logger.error(traceback.format_exc())
            await safe_api_call(
                task.status_message.edit_text,
                text="Failed to create sticker pack. Please try again later.",
            )
        finally:
            shutil.rmtree(user_dir, ignore_errors=True)
