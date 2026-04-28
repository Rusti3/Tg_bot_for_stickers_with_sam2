from __future__ import annotations

import asyncio
import io
from pathlib import Path

import numpy as np
from PIL import Image

from sticker_bot.config import load_settings
from sticker_bot.services.processing import ProcessingService


class DummyMasking:
    predictor = None

    def get_smart_mask(self, image):
        raise AssertionError("Smart mask should not be used in this test")


class RecordingProcessingService(ProcessingService):
    def __init__(self, settings, masking):
        super().__init__(settings, masking)
        self.recorded_frames: list[bytes] = []

    def _encode_tile_segment(self, output_path: Path, frames: list[bytes], fps: int) -> bytes:
        self.recorded_frames.extend(frames)
        output_path.write_bytes(b"segment")
        return b"segment"


class EncodingRecordingProcessingService(ProcessingService):
    def __init__(self, settings, masking):
        super().__init__(settings, masking)
        self.last_encode_args = None

    def _encode_raw_video(
        self,
        output_path: Path,
        frames: list[bytes],
        fps: int,
        width: int,
        height: int,
        *,
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> bytes:
        self.last_encode_args = {
            "fps": fps,
            "width": width,
            "height": height,
            "output_width": output_width,
            "output_height": output_height,
            "frame_count": len(frames),
        }
        output_path.write_bytes(b"segment")
        return b"segment"


def make_settings(tmp_path):
    return load_settings(
        {"BOT_TOKEN": "token", "FFMPEG_CMD": "ffmpeg"},
        project_root=tmp_path,
        load_env_file=False,
    )


def test_process_photo_sync_splits_into_tiles(tmp_path):
    settings = make_settings(tmp_path)
    service = ProcessingService(settings, DummyMasking())

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :100] = (0, 0, 255)
    image[:, 100:] = (0, 255, 0)
    input_path = tmp_path / "sample.png"
    Image.fromarray(image[:, :, ::-1]).save(input_path)

    segments, cols, rows = service.process_photo_sync(input_path, 2, "none", 10)

    assert cols == 2
    assert rows == 1
    assert len(segments) == 2
    decoded = Image.open(io.BytesIO(segments[0]))
    assert decoded.size == (100, 100)


def test_process_gif_with_alpha_preserves_transparency(tmp_path):
    settings = make_settings(tmp_path)
    service = RecordingProcessingService(settings, DummyMasking())

    frame_one = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    frame_two = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    for frame in (frame_one, frame_two):
        for x in range(10, 40):
            for y in range(10, 40):
                frame.putpixel((x, y), (255, 0, 0, 255))

    gif_path = tmp_path / "animated.gif"
    frame_one.save(
        gif_path,
        save_all=True,
        append_images=[frame_two],
        duration=80,
        loop=0,
        transparency=0,
        disposal=2,
    )

    segments, cols, rows = asyncio.run(service.process_gif_with_alpha(gif_path, tmp_path, 1))

    assert segments == [b"segment"]
    assert cols == 1
    assert rows == 1
    alpha_channel = service.recorded_frames[0][3::4]
    assert min(alpha_channel) == 0
    assert max(alpha_channel) == 255


def test_encode_tile_segment_keeps_legacy_custom_emoji_dimensions(tmp_path):
    settings = make_settings(tmp_path)
    service = EncodingRecordingProcessingService(settings, DummyMasking())

    result = service._encode_tile_segment(tmp_path / "segment.webm", [b"\x00" * (100 * 100 * 4)], 24)

    assert result == b"segment"
    assert service.last_encode_args == {
        "fps": 24,
        "width": 100,
        "height": 100,
        "output_width": None,
        "output_height": None,
        "frame_count": 1,
    }
