from __future__ import annotations

import asyncio

from sticker_bot.config import load_settings
from sticker_bot.services.uploading import UploadService


class FakeBot:
    def __init__(self) -> None:
        self.add_calls = 0
        self.create_calls = 0

    async def add_sticker_to_set(self, **kwargs):
        self.add_calls += 1
        if self.add_calls == 1:
            raise RuntimeError("file is too big")

    async def create_new_sticker_set(self, **kwargs):
        self.create_calls += 1
        if self.create_calls == 1:
            raise RuntimeError("file is too big")


def test_upload_single_sticker_retries_with_compression(tmp_path):
    settings = load_settings({"BOT_TOKEN": "token"}, project_root=tmp_path, load_env_file=False)
    service = UploadService(settings)
    bot = FakeBot()
    compression_calls = []

    async def fake_compress(segment, fmt, ext, compression_level):
        compression_calls.append((fmt, ext, compression_level))
        return b"smaller"

    service.compress_sticker = fake_compress

    result = asyncio.run(
        service.upload_single_sticker(
            bot,
            1,
            "pack_name",
            b"large-segment",
            ["🧩"],
            "static",
            "png",
        )
    )

    assert result is True
    assert bot.add_calls == 2
    assert compression_calls == [("static", "png", 1)]


def test_create_sticker_set_retries_with_compression(tmp_path):
    settings = load_settings({"BOT_TOKEN": "token"}, project_root=tmp_path, load_env_file=False)
    service = UploadService(settings)
    bot = FakeBot()
    compression_calls = []

    async def fake_compress(segment, fmt, ext, compression_level):
        compression_calls.append((fmt, ext, compression_level))
        return b"smaller"

    service.compress_sticker = fake_compress

    result = asyncio.run(
        service.create_sticker_set(
            bot,
            user_id=1,
            pack_name="pack_name",
            title="Puzzle 1x1",
            first_segment=b"large-segment",
            emoji_list=["🧩"],
            fmt="static",
            ext="png",
        )
    )

    assert result == b"smaller"
    assert bot.create_calls == 2
    assert compression_calls == [("static", "png", 1)]
