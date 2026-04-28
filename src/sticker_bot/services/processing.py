from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sticker_bot.config import Settings
from sticker_bot.services.masking import MaskingService, get_color_mask, parse_back_param


logger = logging.getLogger("StickerBot")


class ProcessingService:
    # `/add` legacy parity relies on 100x100 custom emoji tiles.
    STICKER_SIZE = 100
    WORK_TILE_SIZE = 100

    def __init__(self, settings: Settings, masking: MaskingService) -> None:
        self.settings = settings
        self.masking = masking

    def process_photo_sync(
        self,
        input_path: str | Path,
        w_count: int,
        back_mode: str,
        tolerance: int,
    ) -> tuple[list[bytes], int, int]:
        logger.debug(
            "Processing photo: %s, w_count=%s, back_mode=%s, tolerance=%s",
            input_path,
            w_count,
            back_mode,
            tolerance,
        )
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {input_path}")

        height, width = image.shape[:2]
        color_target, _ = parse_back_param(back_mode)
        if back_mode == "auto":
            mask = (self.masking.get_smart_mask(image) * 255).astype(np.uint8)
        elif color_target:
            mask = get_color_mask(image, color_target, tolerance)
        else:
            mask = np.full((height, width), 255, dtype=np.uint8)

        image_rgba = cv2.merge([*cv2.split(image), mask])
        cell = self.WORK_TILE_SIZE
        cols = w_count
        rows = max(1, int(round(cols * (height / max(width, 1)))))
        image_resized = cv2.resize(
            image_rgba,
            (cols * cell, rows * cell),
            interpolation=cv2.INTER_LANCZOS4,
        )

        segments: list[bytes] = []
        for row in range(rows):
            for col in range(cols):
                tile = image_resized[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell]
                _, buffer = cv2.imencode(".png", tile)
                segments.append(buffer.tobytes())
        return segments, cols, rows

    async def process_video(
        self,
        input_path: str | Path,
        user_dir: str | Path,
        w_count: int,
        back_mode: str,
        tolerance: int,
        *,
        is_gif: bool = False,
    ) -> tuple[list[bytes], int, int]:
        base_dir = Path(user_dir)
        if is_gif:
            return await self.process_gif_with_alpha(input_path, base_dir, w_count)

        frames_dir = base_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        cell = self.WORK_TILE_SIZE
        fps = 24
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ratio = (frame_height / frame_width) if frame_width else 1.0
        rows = max(1, int(round(w_count * ratio)))
        target_w = w_count * cell
        target_h = rows * cell

        frames: list[np.ndarray] = []
        count = 0
        while count < 70:
            ok, frame = cap.read()
            if not ok:
                break
            resized = cv2.resize(frame, (target_w, target_h))
            frame_path = frames_dir / f"{count:05d}.jpg"
            cv2.imwrite(str(frame_path), resized)
            frames.append(resized)
            count += 1
        cap.release()

        if not frames:
            raise RuntimeError("Video does not contain readable frames.")

        masks = self._build_video_masks(frames_dir, frames, back_mode, tolerance, target_h, target_w)

        segments: list[bytes] = []
        for row in range(rows):
            for col in range(w_count):
                tile_frames = []
                for frame, mask in zip(frames, masks, strict=True):
                    y1, y2 = row * cell, (row + 1) * cell
                    x1, x2 = col * cell, (col + 1) * cell
                    tile = cv2.merge(
                        [
                            *cv2.split(frame[y1:y2, x1:x2]),
                            (mask[y1:y2, x1:x2] * 255).astype(np.uint8),
                        ]
                    )
                    tile_frames.append(tile.tobytes())
                segments.append(
                    await asyncio.to_thread(
                        self._encode_tile_segment,
                        base_dir / f"s_{row}_{col}.webm",
                        tile_frames,
                        fps,
                    )
                )

        shutil.rmtree(frames_dir, ignore_errors=True)
        return segments, w_count, rows

    def remove_background_photo_sync(self, input_path: str | Path) -> bytes:
        logger.debug("Removing background from photo: %s", input_path)
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {input_path}")

        mask = (self.masking.get_smart_mask(image) * 255).astype(np.uint8)
        rgba = cv2.merge([*cv2.split(image), mask])
        ok, buffer = cv2.imencode(".png", rgba)
        if not ok:
            raise RuntimeError("Unable to encode PNG with alpha")
        return buffer.tobytes()

    async def remove_background_video(
        self,
        input_path: str | Path,
        user_dir: str | Path,
        *,
        is_gif: bool = False,
    ) -> bytes:
        base_dir = Path(user_dir)
        if is_gif:
            return await self._render_full_gif_webm(input_path, base_dir, use_smart_mask=True)

        frames_dir = base_dir / "removebg_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        fps = 24
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        target_w, target_h = self._fit_video_size(frame_width, frame_height)

        frames: list[np.ndarray] = []
        count = 0
        while count < 70:
            ok, frame = cap.read()
            if not ok:
                break
            resized = cv2.resize(frame, (target_w, target_h))
            frame_path = frames_dir / f"{count:05d}.jpg"
            cv2.imwrite(str(frame_path), resized)
            frames.append(resized)
            count += 1
        cap.release()

        if not frames:
            raise RuntimeError("Video does not contain readable frames.")

        masks = self._build_video_masks(frames_dir, frames, "auto", 10, target_h, target_w)
        rgba_frames = [
            cv2.merge([*cv2.split(frame), (mask * 255).astype(np.uint8)]).tobytes()
            for frame, mask in zip(frames, masks, strict=True)
        ]
        output_path = base_dir / "removebg_output.webm"
        result = await asyncio.to_thread(
            self._encode_raw_video,
            output_path,
            rgba_frames,
            fps,
            target_w,
            target_h,
        )
        shutil.rmtree(frames_dir, ignore_errors=True)
        return result

    def create_circle_video_sync(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> bytes:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.settings.ffmpeg_cmd,
            "-y",
            "-i",
            str(input_path),
            "-vf",
            "crop='min(iw,ih)':'min(iw,ih)',scale=240:240,fps=24",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0 or not output.exists():
            details = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed while creating circle video: {details}")
        return output.read_bytes()

    async def process_gif_with_alpha(
        self,
        input_path: str | Path,
        user_dir: Path,
        w_count: int,
    ) -> tuple[list[bytes], int, int]:
        logger.info("Processing GIF with alpha: %s", input_path)
        gif = Image.open(input_path)
        frames: list[Image.Image] = []
        alphas: list[Image.Image] = []

        frame_count = 0
        while frame_count < 70:
            try:
                frame_rgba = gif.convert("RGBA")
                frames.append(frame_rgba.convert("RGB"))
                alphas.append(frame_rgba.split()[3])
                frame_count += 1
                gif.seek(gif.tell() + 1)
            except EOFError:
                break

        if not frames:
            raise RuntimeError("GIF has no readable frames.")

        orig_w, orig_h = gif.size
        cell = self.WORK_TILE_SIZE
        fps = 24
        rows = max(1, int(round(w_count * (orig_h / max(orig_w, 1)))))
        target_w = w_count * cell
        target_h = rows * cell

        segments: list[bytes] = []
        for row in range(rows):
            for col in range(w_count):
                tile_frames = []
                for frame_rgb, alpha in zip(frames, alphas, strict=True):
                    frame_resized = frame_rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    alpha_resized = alpha.resize((target_w, target_h), Image.Resampling.LANCZOS)

                    x1, y1 = col * cell, row * cell
                    x2, y2 = x1 + cell, y1 + cell
                    tile_rgb = frame_resized.crop((x1, y1, x2, y2))
                    tile_alpha = alpha_resized.crop((x1, y1, x2, y2))
                    tile_rgba = Image.merge("RGBA", (*tile_rgb.split(), tile_alpha))
                    r_chan, g_chan, b_chan, a_chan = tile_rgba.split()
                    tile_bgra = Image.merge("RGBA", (b_chan, g_chan, r_chan, a_chan))
                    tile_frames.append(tile_bgra.tobytes())

                segments.append(
                    await asyncio.to_thread(
                        self._encode_tile_segment,
                        user_dir / f"s_{row}_{col}.webm",
                        tile_frames,
                        fps,
                    )
                )

        return segments, w_count, rows

    async def _render_full_gif_webm(
        self,
        input_path: str | Path,
        user_dir: Path,
        *,
        use_smart_mask: bool,
    ) -> bytes:
        gif = Image.open(input_path)
        frames: list[Image.Image] = []
        masks: list[Image.Image] = []

        frame_count = 0
        while frame_count < 70:
            try:
                frame_rgba = gif.convert("RGBA")
                frame_rgb = frame_rgba.convert("RGB")
                frames.append(frame_rgb)
                if use_smart_mask:
                    frame_bgr = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
                    alpha = Image.fromarray((self.masking.get_smart_mask(frame_bgr) * 255).astype(np.uint8))
                else:
                    alpha = frame_rgba.split()[3]
                masks.append(alpha)
                frame_count += 1
                gif.seek(gif.tell() + 1)
            except EOFError:
                break

        if not frames:
            raise RuntimeError("GIF has no readable frames.")

        orig_w, orig_h = gif.size
        target_w, target_h = self._fit_video_size(orig_w, orig_h)
        rgba_frames = []
        for frame_rgb, alpha in zip(frames, masks, strict=True):
            frame_resized = frame_rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)
            alpha_resized = alpha.resize((target_w, target_h), Image.Resampling.LANCZOS)
            rgba = Image.merge("RGBA", (*frame_resized.split(), alpha_resized))
            r_chan, g_chan, b_chan, a_chan = rgba.split()
            bgra = Image.merge("RGBA", (b_chan, g_chan, r_chan, a_chan))
            rgba_frames.append(bgra.tobytes())

        return await asyncio.to_thread(
            self._encode_raw_video,
            user_dir / "removebg_output.webm",
            rgba_frames,
            24,
            target_w,
            target_h,
        )

    def _build_video_masks(
        self,
        frames_dir: Path,
        frames: list[np.ndarray],
        back_mode: str,
        tolerance: int,
        target_h: int,
        target_w: int,
    ) -> list[np.ndarray]:
        color_target, _ = parse_back_param(back_mode)

        if back_mode == "auto":
            if self.masking.predictor is None:
                raise RuntimeError("MaskingService must be initialized before video processing.")
            state = self.masking.predictor.init_state(video_path=str(frames_dir))
            first_mask = self.masking.get_smart_mask(frames_dir / "00000.jpg")
            self.masking.predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=first_mask)
            masks = []
            for _, _, out_logits in self.masking.predictor.propagate_in_video(state):
                masks.append((out_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0])
            self.masking.predictor.reset_state(state)
            return masks

        if color_target:
            return [
                (get_color_mask(frame, color_target, tolerance) / 255).astype(np.uint8)
                for frame in frames
            ]

        return [np.ones((target_h, target_w), dtype=np.uint8) for _ in frames]

    @staticmethod
    def _fit_video_size(width: int, height: int, max_dim: int = 512) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            return 256, 256
        scale = min(1.0, max_dim / max(width, height))
        target_w = max(2, int(round(width * scale)))
        target_h = max(2, int(round(height * scale)))
        if target_w % 2:
            target_w += 1
        if target_h % 2:
            target_h += 1
        return target_w, target_h

    def _encode_tile_segment(self, output_path: Path, frames: list[bytes], fps: int) -> bytes:
        return self._encode_raw_video(
            output_path,
            frames,
            fps,
            self.WORK_TILE_SIZE,
            self.WORK_TILE_SIZE,
        )

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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.settings.ffmpeg_cmd,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "bgra",
            "-r",
            str(fps),
            "-i",
            "-",
        ]
        if output_width and output_height and (output_width != width or output_height != height):
            command.extend(
                [
                    "-vf",
                    f"scale={output_width}:{output_height}:flags=lanczos",
                ]
            )
        command.extend(
            [
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuva420p",
            "-crf",
            "30",
            "-b:v",
            "200k",
            "-deadline",
            "realtime",
            "-an",
            str(output_path),
            ]
        )
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        try:
            for frame in frames:
                proc.stdin.write(frame)
        except BrokenPipeError:
            logger.error("ffmpeg closed stdin early while creating %s", output_path)
        finally:
            proc.stdin.close()
        stderr = b""
        if proc.stderr is not None:
            stderr = proc.stderr.read()
        proc.wait()
        if proc.returncode != 0 or not output_path.exists():
            details = stderr.decode("utf-8", errors="replace").strip()
            if not details:
                details = "ffmpeg exited without stderr output."
            raise RuntimeError(f"ffmpeg failed while creating segment: {output_path}; {details}")
        return output_path.read_bytes()
