import asyncio
import os
import shutil
import cv2
import subprocess
import logging
import numpy as np
import torch
import traceback
import secrets
import time
import json
import re
import io
import tempfile
from dataclasses import dataclass
from typing import Optional, List, Dict, Set, Tuple
from PIL import Image
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import Message, BufferedInputFile, InputSticker, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramRetryAfter, TelegramNetworkError
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import hydra
from hydra.core.global_hydra import GlobalHydra

# ... РІРЅСѓС‚СЂРё init_models() ...
if not GlobalHydra.instance().is_initialized():
    # Р“РѕРІРѕСЂРёРј Hydra РёСЃРєР°С‚СЊ РєРѕРЅС„РёРіРё РїСЂСЏРјРѕ РІ РЅР°С€РµР№ РїР°РїРєРµ
    hydra.initialize_config_dir(config_dir=os.path.abspath("sam2_configs"), version_base="1.2")

# ====== РќРђРЎРўР РћР™РљР ======
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
TEMP_DIR = "bot_temp"
LIMITS_FILE = "user_limits.json"
FFMPEG_CMD = 'ffmpeg'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_GRID_WIDTH = 10
CONCURRENT_GPU_TASKS = 2

# Р›РёРјРёС‚С‹
MAX_FILES_PER_HOUR = 30
RATE_LIMIT_WINDOW = 3600

# Р¦РІРµС‚РѕРІР°СЏ РєР°СЂС‚Р°
COLOR_MAP = {
    "white": (255, 255, 255), "Р±РµР»С‹Р№": (255, 255, 255),
    "black": (0, 0, 0), "С‡РµСЂРЅС‹Р№": (0, 0, 0),
    "green": (0, 255, 0), "Р·РµР»РµРЅС‹Р№": (0, 255, 0),
    "blue": (255, 0, 0), "СЃРёРЅРёР№": (255, 0, 0), # OpenCV РёСЃРїРѕР»СЊР·СѓРµС‚ BGR
    "red": (0, 0, 255), "РєСЂР°СЃРЅС‹Р№": (0, 0, 255)
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "sam2.1_hiera_base_plus.pt")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StickerBot")

dp = Dispatcher()

@dataclass
class StickerTask:
    user_id: int
    user_name: str
    file_id: str
    w_count: int
    back_mode: str  # "auto", "none", РёР»Рё HEX/ColorName
    tolerance: int  # Р”РћР‘РђР’Р›Р•РќРћ
    is_video: bool
    is_gif: bool
    status_msg: Message
    msg_ref: Message
    start_time: float = 0.0

# === Р“Р›РћР‘РђР›Р¬РќР«Р• РџР•Р Р•РњР•РќРќР«Р• ===
user_tasks: Dict[int, List[StickerTask]] = {}
user_queue_order: List[int] = []
active_gpu_users: Set[int] = set()
scheduler_cond: Optional[asyncio.Condition] = None

# ====== РЈРўРР›РРўР« Р¦Р’Р•РўРђ ======
def parse_back_param(param: str) -> Tuple[Optional[Tuple[int, int, int]], int]:
    param = param.lower()
    # РС‰РµРј С†РёС„СЂС‹ РІ РєРѕРЅС†Рµ СЃС‚СЂРѕРєРё (РЅР°РїСЂРёРјРµСЂ, 'black120' -> 120)
    match_tol = re.search(r'(\d+)$', param)
    tolerance = 10  # РЎС‚Р°РЅРґР°СЂС‚РЅРѕРµ Р·РЅР°С‡РµРЅРёРµ
    color_part = param

    if match_tol:
        tolerance = int(match_tol.group(1))
        color_part = param[:match_tol.start()]

    color_bgr = None
    if color_part in COLOR_MAP:
        color_bgr = COLOR_MAP[color_part]
    else:
        match_hex = re.search(r'(?:#)?([0-9a-fA-F]{6})', color_part)
        if match_hex:
            h = match_hex.group(1)
            color_bgr = (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

    return color_bgr, tolerance

def get_color_mask(img_bgr, target_bgr, tolerance=50):
    lower = np.array([max(c - tolerance, 0) for c in target_bgr])
    upper = np.array([min(c + tolerance, 255) for c in target_bgr])
    mask = cv2.inRange(img_bgr, lower, upper)

    kernel = np.ones((3,3), np.uint8)
    # Р•СЃР»Рё tolerance РІС‹СЃРѕРєРёР№, СЃСЂРµР·Р°РµРј РєРѕРЅС‚СѓСЂ Р°РіСЂРµСЃСЃРёРІРЅРµРµ
    iters = 1 if tolerance < 80 else 2
    mask = cv2.dilate(mask, kernel, iterations=iters)

    return cv2.bitwise_not(mask)

# ====== РЎРРЎРўР•РњРђ Р›РРњРРўРћР’ ======
def load_limits() -> Dict[int, List[float]]:
    if os.path.exists(LIMITS_FILE):
        try:
            with open(LIMITS_FILE, "r") as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"РћС€РёР±РєР° Р·Р°РіСЂСѓР·РєРё Р»РёРјРёС‚РѕРІ: {e}")
    return {}

def save_limits(history):
    try:
        with open(LIMITS_FILE, "w") as f:
            json.dump(history, f)
    except Exception as e:
        logger.error(f"РћС€РёР±РєР° СЃРѕС…СЂР°РЅРµРЅРёСЏ Р»РёРјРёС‚РѕРІ: {e}")

user_request_history = load_limits()

# ====== РРќРР¦РРђР›РР—РђР¦РРЇ РњРћР”Р•Р›Р•Р™ ======
predictor = None
birefnet = None
birefnet_transform = None

def init_models():
    global predictor, birefnet, birefnet_transform
    logger.info(f"рџљЂ РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ РјРѕРґРµР»РµР№ РЅР° {DEVICE}...")
    try:
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor("sam2.1_hiera_b+.yaml", SAM2_CHECKPOINT, device=DEVICE)
        birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True, torch_dtype=torch.float32)
        birefnet.to(DEVICE).eval()
        birefnet_transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
        return True
    except Exception as e:
        logger.error(f"вќЊ РћС€РёР±РєР° Р·Р°РіСЂСѓР·РєРё РјРѕРґРµР»РµР№: {e}")
        return False

async def safe_api_call(func, *args, **kwargs):
    while True:
        try:
            return await func(*args, **kwargs)
        except TelegramRetryAfter as e:
            await asyncio.sleep(e.retry_after)
        except (TelegramNetworkError, asyncio.TimeoutError):
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"API Error: {e}")
            raise

# ====== РЇР”Р Рћ РћР‘Р РђР‘РћРўРљР ======
def get_smart_mask(image_path_or_np):
    if isinstance(image_path_or_np, str):
        image = Image.open(image_path_or_np).convert("RGB")
        logger.debug(f"рџ¤– BiRefNet: Р·Р°РіСЂСѓР·РєР° РёР·РѕР±СЂР°Р¶РµРЅРёСЏ {image_path_or_np}, СЂР°Р·РјРµСЂ={image.size}")
    else:
        image = Image.fromarray(cv2.cvtColor(image_path_or_np, cv2.COLOR_BGR2RGB))
        logger.debug(f"рџ¤– BiRefNet: РѕР±СЂР°Р±РѕС‚РєР° РёР· numpy array, СЂР°Р·РјРµСЂ={image.size}")
    orig_w, orig_h = image.size
    input_tensor = birefnet_transform(image).unsqueeze(0).to(DEVICE)
    logger.debug(f"рџ“Љ BiRefNet: input_tensor shape={input_tensor.shape}, device={DEVICE}")
    with torch.no_grad():
        preds = birefnet(input_tensor)[-1]
        logger.debug(f"рџ“Љ BiRefNet: preds shape={preds.shape}, min={preds.min():.4f}, max={preds.max():.4f}")
        mask = (preds.sigmoid().cpu()[0][0].numpy() > 0.4).astype(np.uint8)
        logger.debug(f"вњ… BiRefNet: РјР°СЃРєР° СЃРіРµРЅРµСЂРёСЂРѕРІР°РЅР°, nonzero={np.count_nonzero(mask)}/{mask.size} ({100*np.count_nonzero(mask)/mask.size:.1f}%)")
    return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

def process_photo_sync(input_path, w_count, back_mode, tolerance):
    logger.debug(f"рџ“Ґ РћР±СЂР°Р±РѕС‚РєР° С„РѕС‚Рѕ: {input_path}, w_count={w_count}, back_mode={back_mode}, tolerance={tolerance}")
    img = cv2.imread(input_path)
    h_orig, w_orig = img.shape[:2]
    logger.debug(f"рџ“ђ Р Р°Р·РјРµСЂ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ: {w_orig}x{h_orig}")

    # Р’С‹Р±РѕСЂ РјР°СЃРєРё
    color_target, _ = parse_back_param(back_mode)
    if back_mode == "auto":
        logger.debug("рџ¤– Р РµР¶РёРј РјР°СЃРєРё: AUTO (BiRefNet)")
        mask = (get_smart_mask(img) * 255).astype(np.uint8)
    elif color_target:
        logger.debug(f"рџЋЁ Р РµР¶РёРј РјР°СЃРєРё: COLOR (target={color_target}, tolerance={tolerance})")
        mask = get_color_mask(img, color_target, tolerance)
    else:
        logger.debug("вљЄ Р РµР¶РёРј РјР°СЃРєРё: NONE (РїРѕР»РЅС‹Р№)")
        mask = np.full((h_orig, w_orig), 255, dtype=np.uint8)

    logger.debug(f"рџ“Љ РЎС‚Р°С‚РёСЃС‚РёРєР° РјР°СЃРєРё: min={mask.min()}, max={mask.max()}, nonzero={np.count_nonzero(mask)}")
    img_rgba = cv2.merge([*cv2.split(img), mask])
    cell, cols = 100, w_count
    rows = max(1, int(round(cols * (h_orig / w_orig))))
    logger.debug(f"рџ“¦ РЎРµС‚РєР°: {cols}x{rows} СЏС‡РµРµРє РїРѕ {cell}x{cell}px")
    img_resized = cv2.resize(img_rgba, (cols * cell, rows * cell), interpolation=cv2.INTER_LANCZOS4)

    segments = []
    for r in range(rows):
        for c in range(cols):
            tile = img_resized[r*cell:(r+1)*cell, c*cell:(c+1)*cell]
            _, buf = cv2.imencode(".png", tile)
            segments.append(buf.tobytes())
    logger.debug(f"вњ… РЎС„РѕСЂРјРёСЂРѕРІР°РЅРѕ СЃРµРіРјРµРЅС‚РѕРІ: {len(segments)}")
    return segments, cols, rows

async def process_video_sync(input_path, user_dir, w_count, back_mode, tolerance, is_gif=False):
    logger.debug(
        f"Processing video: {input_path}, w_count={w_count}, back_mode={back_mode}, "
        f"tolerance={tolerance}, is_gif={is_gif}"
    )
    frames_dir = os.path.join(user_dir, "frames")
    alpha_dir = os.path.join(user_dir, "alpha")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(alpha_dir, exist_ok=True)

    if is_gif:
        logger.info("GIF file detected: processing with alpha preservation")
        return await process_gif_with_alpha(input_path, user_dir, frames_dir, alpha_dir, w_count)

    cap = cv2.VideoCapture(input_path)
    cell, fps = 100, 24
    target_w = w_count * cell
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ratio = (frame_h / frame_w) if frame_w else 1.0
    rows = max(1, int(round(w_count * ratio)))
    target_h = rows * cell
    logger.debug(f"Frame size: {target_w}x{target_h}, rows={rows}, fps={fps}")

    count = 0
    while count < 70:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{frames_dir}/{count:05d}.jpg", cv2.resize(frame, (target_w, target_h)))
        count += 1
    cap.release()
    logger.debug(f"Extracted frames: {count}")

    masks = []
    color_target, _ = parse_back_param(back_mode)

    if back_mode == "auto":
        logger.debug("Mask mode: AUTO (SAM2)")
        state = predictor.init_state(video_path=frames_dir)
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=get_smart_mask(f"{frames_dir}/00000.jpg"))
        for frame_idx, _, out_logits in predictor.propagate_in_video(state):
            masks.append((out_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0])
            if frame_idx % 10 == 0:
                logger.debug(f"Processed frame {frame_idx}/{count}")
        predictor.reset_state(state)
    elif color_target:
        logger.debug(f"Mask mode: COLOR target={color_target}, tolerance={tolerance}")
        for idx in range(count):
            frame = cv2.imread(f"{frames_dir}/{idx:05d}.jpg")
            m = get_color_mask(frame, color_target, tolerance)
            masks.append((m / 255).astype(np.uint8))
    else:
        logger.debug("Mask mode: NONE")
        masks = [np.ones((target_h, target_w), dtype=np.uint8) for _ in range(count)]

    sticker_segments = []
    logger.debug(f"Segmenting grid: {w_count}x{rows}")
    for r in range(rows):
        for c in range(w_count):
            seg_out = os.path.join(user_dir, f"s_{r}_{c}.webm")
            cmd = [
                FFMPEG_CMD, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '100x100',
                '-pix_fmt', 'bgra', '-r', str(fps), '-i', '-', '-c:v', 'libvpx-vp9',
                '-pix_fmt', 'yuva420p', '-crf', '30', '-b:v', '200k', '-deadline', 'realtime',
                '-an', seg_out
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            for idx in range(len(masks)):
                frame = cv2.imread(f"{frames_dir}/{idx:05d}.jpg")
                y1, y2, x1, x2 = r * cell, (r + 1) * cell, c * cell, (c + 1) * cell
                tile = cv2.merge([
                    *cv2.split(frame[y1:y2, x1:x2]),
                    (masks[idx][y1:y2, x1:x2] * 255).astype(np.uint8)
                ])
                proc.stdin.write(tile.tobytes())
            proc.stdin.close()
            proc.wait()
            if os.path.exists(seg_out):
                with open(seg_out, 'rb') as f:
                    sticker_segments.append(f.read())

    logger.debug(f"Video segments created: {len(sticker_segments)}")
    return sticker_segments, w_count, rows


async def process_gif_with_alpha(input_path, user_dir, frames_dir, alpha_dir, w_count):
    logger.info("Processing GIF with alpha: %s", input_path)
    gif = Image.open(input_path)

    frame_paths = []
    alpha_paths = []
    frame_count = 0
    max_frames = 70

    while frame_count < max_frames:
        try:
            frame_rgba = gif.convert("RGBA")
            frame_rgb = frame_rgba.convert("RGB")

            frame_path = os.path.join(frames_dir, f"{frame_count:05d}.png")
            alpha_path = os.path.join(alpha_dir, f"{frame_count:05d}.png")
            frame_rgb.save(frame_path)
            frame_rgba.split()[3].save(alpha_path)

            frame_paths.append(frame_path)
            alpha_paths.append(alpha_path)
            frame_count += 1
            gif.seek(gif.tell() + 1)
        except EOFError:
            break

    if frame_count == 0:
        logger.error("GIF has no frames: %s", input_path)
        return [], w_count, 1

    orig_w, orig_h = gif.size
    cell, fps = 100, 24
    target_w = w_count * cell
    rows = max(1, int(round(w_count * (orig_h / max(orig_w, 1)))))
    target_h = rows * cell

    sticker_segments = []
    for r in range(rows):
        for c in range(w_count):
            seg_out = os.path.join(user_dir, f"s_{r}_{c}.webm")
            cmd = [
                FFMPEG_CMD, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '100x100',
                '-pix_fmt', 'bgra', '-r', str(fps), '-i', '-', '-c:v', 'libvpx-vp9',
                '-pix_fmt', 'yuva420p', '-crf', '30', '-b:v', '200k', '-deadline', 'realtime',
                '-an', seg_out
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

            for idx in range(frame_count):
                frame_rgb = Image.open(frame_paths[idx]).convert("RGB")
                alpha = Image.open(alpha_paths[idx]).convert("L")

                frame_resized = frame_rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)
                alpha_resized = alpha.resize((target_w, target_h), Image.Resampling.LANCZOS)

                x1, y1 = c * cell, r * cell
                x2, y2 = x1 + cell, y1 + cell
                tile_rgb = frame_resized.crop((x1, y1, x2, y2))
                tile_alpha = alpha_resized.crop((x1, y1, x2, y2))
                tile_rgba = Image.merge("RGBA", (*tile_rgb.split(), tile_alpha))
                r_chan, g_chan, b_chan, a_chan = tile_rgba.split()
                tile_bgra = Image.merge("RGBA", (b_chan, g_chan, r_chan, a_chan))
                proc.stdin.write(tile_bgra.tobytes())

            proc.stdin.close()
            proc.wait()
            if os.path.exists(seg_out):
                with open(seg_out, 'rb') as f:
                    sticker_segments.append(f.read())

    shutil.rmtree(frames_dir, ignore_errors=True)
    shutil.rmtree(alpha_dir, ignore_errors=True)
    logger.debug(f"GIF segments created: {len(sticker_segments)}")
    return sticker_segments, w_count, rows

# ====== Р¤РћРќРћР’РђРЇ Р—РђР“Р РЈР—РљРђ Р Р’РћР РљР•Р  ======
async def compress_sticker(segment_data, fmt, ext, compression_level=1):
    """РЎР¶РёРјР°РµС‚ СЃС‚РёРєРµСЂ, РµСЃР»Рё РѕРЅ СЃР»РёС€РєРѕРј Р±РѕР»СЊС€РѕР№"""
    try:
        if fmt == "static":
            img = Image.open(io.BytesIO(segment_data))
            # РЈРјРµРЅСЊС€Р°РµРј РєР°С‡РµСЃС‚РІРѕ PNG
            quality = max(20, 95 - compression_level * 15)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True, quality=quality)
            result = buf.getvalue()
            logger.debug(f"рџ—њпёЏ РЎР¶Р°С‚РёРµ PNG: {len(segment_data)} -> {len(result)} bytes (quality={quality})")
            return result
        else:
            # Р”Р»СЏ РІРёРґРµРѕ СѓРјРµРЅСЊС€Р°РµРј Р±РёС‚СЂРµР№С‚ С‡РµСЂРµР· ffmpeg
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp_in:
                tmp_in.write(segment_data)
                tmp_in_path = tmp_in.name
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp_out:
                tmp_out_path = tmp_out.name
            # РЈРІРµР»РёС‡РёРІР°РµРј CRF Рё СѓРјРµРЅСЊС€Р°РµРј Р±РёС‚СЂРµР№С‚ СЃ РєР°Р¶РґРѕР№ РїРѕРїС‹С‚РєРѕР№
            crf = 28 + compression_level * 2
            bitrate = max(50, 200 - compression_level * 10)
            cmd = [FFMPEG_CMD, '-y', '-i', tmp_in_path, '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p', '-crf', str(crf), '-b:v', f'{bitrate}k', '-deadline', 'realtime', '-an', tmp_out_path]
            subprocess.run(cmd, stderr=subprocess.DEVNULL, check=True)
            with open(tmp_out_path, 'rb') as f:
                result = f.read()
            os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)
            logger.debug(f"рџ—њпёЏ РЎР¶Р°С‚РёРµ WebM: {len(segment_data)} -> {len(result)} bytes (crf={crf}, bitrate={bitrate}k)")
            return result
    except Exception as e:
        logger.error(f"вќЊ РћС€РёР±РєР° СЃР¶Р°С‚РёСЏ ({fmt}): {e}")
        return segment_data

async def upload_single_sticker(bot, task, pack_name, segment, emoji_list, fmt, ext, max_retries=5):
    """РџС‹С‚Р°РµС‚СЃСЏ Р·Р°РіСЂСѓР·РёС‚СЊ СЃС‚РёРєРµСЂ, РїСЂРё РѕС€РёР±РєРµ 'too big' СЃР¶РёРјР°РµС‚ Рё РїСЂРѕР±СѓРµС‚ СЃРЅРѕРІР°"""
    for attempt in range(max_retries):
        try:
            await safe_api_call(bot.add_sticker_to_set, user_id=task.user_id, name=pack_name, sticker=InputSticker(sticker=BufferedInputFile(segment, filename=f"sticker.{ext}"), emoji_list=emoji_list, format=fmt))
            return True
        except TelegramRetryAfter as e:
            logger.warning(f"вЏі Rate limit, Р¶РґС‘Рј {e.retry_after}s")
            await asyncio.sleep(e.retry_after)
        except Exception as e:
            error_msg = str(e)
            if "file is too big" in error_msg.lower() or "too big" in error_msg.lower():
                logger.warning(f"вљ пёЏ РЎС‚РёРєРµСЂС‚ СЃР»РёС€РєРѕРј Р±РѕР»СЊС€РѕР№ ({len(segment)} bytes), РїРѕРїС‹С‚РєР° {attempt + 1}/{max_retries}, СЃР¶Р°С‚РёРµ...")
                segment = await compress_sticker(segment, fmt, ext, attempt + 1)
                await asyncio.sleep(0.5)
            else:
                logger.error(f"вќЊ РћС€РёР±РєР° Р·Р°РіСЂСѓР·РєРё СЃС‚РёРєРµСЂР°: {e}")
                raise
    logger.error(f"вќЊ РќРµ СѓРґР°Р»РѕСЃСЊ Р·Р°РіСЂСѓР·РёС‚СЊ СЃС‚РёРєРµСЂ РїРѕСЃР»Рµ {max_retries} РїРѕРїС‹С‚РѕРє")
    return False

async def background_uploader(bot, task, user_dir, segments, cols, rows, fmt, ext):
    bot_username = (await bot.get_me()).username
    logger.debug(f"рџ“¤ Р—Р°РіСЂСѓР·РєР° СЃС‚РёРєРµСЂРѕРІ: user={task.user_name}, fmt={fmt}, segments={len(segments)}, grid={cols}x{rows}")
    try:
        await safe_api_call(task.status_msg.edit_text, text="рџ“¤ РЎРѕР·РґР°РЅРёРµ РЅР°Р±РѕСЂР°...")
        pack_name = f"puzzle_{secrets.token_hex(4)}_by_{bot_username}"
        logger.debug(f"рџ“¦ РЎРѕР·РґР°РЅРёРµ СЃС‚РёРєРµСЂРїР°РєР°: {pack_name}")
        await safe_api_call(bot.create_new_sticker_set, user_id=task.user_id, name=pack_name, title=f"Puzzle {cols}x{rows}", stickers=[InputSticker(sticker=BufferedInputFile(segments[0], filename=f"0.{ext}"), emoji_list=["рџ§©"], format=fmt)], sticker_type="custom_emoji")
        logger.debug(f"вњ… РџРµСЂРІС‹Р№ СЃС‚РёРєРµСЂ Р·Р°РіСЂСѓР¶РµРЅ, Р·Р°РіСЂСѓР·РєР° РѕСЃС‚Р°Р»СЊРЅС‹С…...")
        failed_count = 0
        for i in range(1, len(segments)):
            success = await upload_single_sticker(bot, task, pack_name, segments[i], ["рџ§©"], fmt, ext)
            if not success:
                failed_count += 1
                logger.warning(f"вљ пёЏ РџСЂРѕРїСѓС‰РµРЅ СЃС‚РёРєРµСЂ {i}/{len(segments)}")
            if (i + 1) % 10 == 0:
                logger.debug(f"  рџ“Ќ Р—Р°РіСЂСѓР¶РµРЅРѕ СЃС‚РёРєРµСЂРѕРІ: {i + 1 - failed_count}/{len(segments)} (РїСЂРѕРїСѓС‰РµРЅРѕ: {failed_count})")

        logger.debug(f"рџ“¦ РџРѕР»СѓС‡РµРЅРёРµ РёРЅС„РѕСЂРјР°С†РёРё Рѕ СЃС‚РёРєРµСЂРїР°РєРµ...")
        sticker_set = await bot.get_sticker_set(pack_name)
        ids = [s.custom_emoji_id for s in sticker_set.stickers]
        logger.debug(f"вњ… РџРѕР»СѓС‡РµРЅРѕ custom_emoji_id: {len(ids)} С€С‚.")
        grid_html = ["".join([f'<tg-emoji emoji-id="{ids[r*cols+c]}">рџ§©</tg-emoji>' for c in range(cols)]) for r in range(rows)]
        kb = InlineKeyboardBuilder().row(InlineKeyboardButton(text="РЈСЃС‚Р°РЅРѕРІРёС‚СЊ", url=f"https://t.me/addstickers/{pack_name}"))
        logger.debug(f"рџ“© РћС‚РїСЂР°РІРєР° СЃРѕРѕР±С‰РµРЅРёСЏ СЃ grid {rows} СЃС‚СЂРѕРє")
        await safe_api_call(task.msg_ref.answer, text="\n".join(grid_html), reply_markup=kb.as_markup())
        logger.debug(f"вњ… РЎРѕРѕР±С‰РµРЅРёРµ РѕС‚РїСЂР°РІР»РµРЅРѕ, СѓРґР°Р»РµРЅРёРµ status_msg")
        await safe_api_call(task.status_msg.delete)
        logger.info(f"вњ… Р—Р°РґР°С‡Р° Р·Р°РІРµСЂС€РµРЅР°: user={task.user_name}, grid={cols}x{rows}, РІСЂРµРјСЏ={time.perf_counter() - task.start_time:.2f}s, РїСЂРѕРїСѓС‰РµРЅРѕ: {failed_count}")
    except: logger.error(traceback.format_exc())
    finally: shutil.rmtree(user_dir, ignore_errors=True)

async def gpu_worker(worker_id, bot):
    logger.info(f"рџ”§ GPU Worker {worker_id} Р·Р°РїСѓС‰РµРЅ")
    while True:
        async with scheduler_cond:
            while True:
                selected_user = next((uid for uid in user_queue_order if uid not in active_gpu_users), None)
                if selected_user is not None:
                    task = user_tasks[selected_user].pop(0)
                    active_gpu_users.add(selected_user)
                    user_queue_order.remove(selected_user)
                    if user_tasks[selected_user]: user_queue_order.append(selected_user)
                    else: del user_tasks[selected_user]
                    logger.debug(f"рџЋЇ Worker {worker_id} РІР·СЏР» Р·Р°РґР°С‡Сѓ: user={task.user_name}, is_video={task.is_video}, w_count={task.w_count}")
                    break
                await scheduler_cond.wait()
        user_dir = os.path.join(TEMP_DIR, f"run_{worker_id}_{secrets.token_hex(3)}")
        try:
            os.makedirs(user_dir, exist_ok=True)
            file_info = await bot.get_file(task.file_id)
            logger.debug(f"рџ“Ґ РЎРєР°С‡РёРІР°РЅРёРµ С„Р°Р№Р»Р°: file_path={file_info.file_path}, size={file_info.file_size} bytes")
            input_p = os.path.join(user_dir, "input_source")
            await bot.download_file(file_info.file_path, input_p)
            logger.debug(f"рџ’ѕ Р¤Р°Р№Р» СЃРѕС…СЂР°РЅС‘РЅ: {input_p}")
            if task.is_video:
                logger.debug("рџЋ¬ Р—Р°РїСѓСЃРє РѕР±СЂР°Р±РѕС‚РєРё РІРёРґРµРѕ...")
                segments, cols, rows = await process_video_sync(input_p, user_dir, task.w_count, task.back_mode, task.tolerance, task.is_gif)
            else:
                logger.debug("рџ“ё Р—Р°РїСѓСЃРє РѕР±СЂР°Р±РѕС‚РєРё С„РѕС‚Рѕ...")
                segments, cols, rows = await asyncio.to_thread(process_photo_sync, input_p, task.w_count, task.back_mode, task.tolerance)
            fmt, ext = ("video", "webm") if task.is_video else ("static", "png")
            logger.debug(f"рџ“¦ РћР±СЂР°Р±РѕС‚РєР° Р·Р°РІРµСЂС€РµРЅР°: {len(segments)} СЃРµРіРјРµРЅС‚РѕРІ, С„РѕСЂРјР°С‚={fmt}")
            asyncio.create_task(background_uploader(bot, task, user_dir, segments, cols, rows, fmt, ext))
        except: shutil.rmtree(user_dir, ignore_errors=True)
        finally:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            async with scheduler_cond:
                active_gpu_users.discard(task.user_id)
                scheduler_cond.notify_all()

# ====== РҐР•РќР”Р›Р•Р Р« ======
@dp.message(Command("stats"))
async def handle_stats(message: Message):
    total_users = len(user_request_history)
    total_requests = sum(len(h) for h in user_request_history.values())
    await message.answer(f"рџ“Љ Р’СЃРµРіРѕ РїРѕР»СЊР·РѕРІР°С‚РµР»РµР№: {total_users}\nрџ§© РЎРѕР·РґР°РЅРѕ: {total_requests}")

@dp.message(Command("add"), F.photo | F.video | F.animation | F.document | F.reply_to_message)
async def handle_add(message: Message, command: CommandObject):
    uid = message.from_user.id
    current_time = time.time()

    if uid not in user_request_history: user_request_history[uid] = []
    user_request_history[uid] = [t for t in user_request_history[uid] if current_time - t < RATE_LIMIT_WINDOW]
    if len(user_request_history[uid]) >= MAX_FILES_PER_HOUR:
        return await message.answer("рџ›‘ Р›РёРјРёС‚! РџРѕРїСЂРѕР±СѓР№С‚Рµ РїРѕР·Р¶Рµ.")

    user_request_history[uid].append(current_time)
    save_limits(user_request_history)

    target = message.reply_to_message or message
    args = command.args.split() if command.args else []

    w_count = 1
    back_mode = "none"
    tolerance = 10

    for arg in args:
        if arg.isdigit():
            w_count = min(max(int(arg), 1), MAX_GRID_WIDTH)
        if arg.startswith("back="):
            raw_val = arg.replace("back=", "")
            if raw_val == "auto":
                back_mode = "auto"
            else:
                color_target, tol = parse_back_param(raw_val)
                if color_target:
                    back_mode = raw_val
                    tolerance = tol

    file_obj = target.photo[-1] if target.photo else (target.animation or target.video or target.document)
    if not file_obj: return await message.answer("вќЊ Р¤Р°Р№Р» РЅРµ РЅР°Р№РґРµРЅ.")

    document_mime = (target.document.mime_type or "").lower() if target.document else ""
    document_name = (target.document.file_name or "").lower() if target.document else ""
    is_gif = bool(target.document and (document_mime == "image/gif" or document_name.endswith(".gif")))
    is_video = bool(target.video or target.animation or (target.document and ("video" in document_mime or is_gif)))

    async with scheduler_cond:
        if uid not in user_tasks:
            user_tasks[uid] = []
            user_queue_order.append(uid)
        qsize = sum(len(tasks) for tasks in user_tasks.values())
        status = await message.answer(f"вЏі РћС‡РµСЂРµРґСЊ: {qsize + 1}")
        user_tasks[uid].append(StickerTask(user_id=uid, user_name=message.from_user.username or str(uid), file_id=file_obj.file_id, w_count=w_count, back_mode=back_mode, tolerance=tolerance, is_video=is_video, is_gif=is_gif, status_msg=status, msg_ref=message, start_time=time.perf_counter()))
        scheduler_cond.notify()

async def main():
    global scheduler_cond
    scheduler_cond = asyncio.Condition()
    if not TOKEN:
        logger.error("BOT_TOKEN not set. Create .env with BOT_TOKEN=...")
        return
    if not os.path.exists(SAM2_CHECKPOINT):
        logger.error(f"SAM2 checkpoint not found: {SAM2_CHECKPOINT}")
        return
    if shutil.which(FFMPEG_CMD) is None:
        logger.error("ffmpeg not found in PATH. Install ffmpeg and restart the shell.")
        return
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if init_models():
        session = AiohttpSession()
        bot = Bot(token=TOKEN, session=session, default=DefaultBotProperties(parse_mode="HTML"))
        for i in range(CONCURRENT_GPU_TASKS): asyncio.create_task(gpu_worker(i, bot))
        await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

