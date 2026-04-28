from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from sticker_bot.config import Settings


logger = logging.getLogger("StickerBot")

COLOR_MAP = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
}


def parse_back_param(param: str) -> tuple[Optional[tuple[int, int, int]], int]:
    normalized = param.lower()
    match_tol = re.search(r"(\d+)$", normalized)
    tolerance = 10
    color_part = normalized

    if match_tol:
        tolerance = int(match_tol.group(1))
        color_part = normalized[: match_tol.start()]

    color_bgr = None
    if color_part in COLOR_MAP:
        color_bgr = COLOR_MAP[color_part]
    else:
        match_hex = re.search(r"(?:#)?([0-9a-fA-F]{6})", color_part)
        if match_hex:
            value = match_hex.group(1)
            color_bgr = (int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16))

    return color_bgr, tolerance


def get_color_mask(
    image_bgr: np.ndarray,
    target_bgr: tuple[int, int, int],
    tolerance: int = 50,
) -> np.ndarray:
    lower = np.array([max(component - tolerance, 0) for component in target_bgr])
    upper = np.array([min(component + tolerance, 255) for component in target_bgr])
    mask = cv2.inRange(image_bgr, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    iterations = 1 if tolerance < 80 else 2
    mask = cv2.dilate(mask, kernel, iterations=iterations)
    return cv2.bitwise_not(mask)


class MaskingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        import torch

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = None
        self.birefnet = None
        self.birefnet_transform = None

    def initialize(self) -> None:
        import hydra
        from hydra.core.global_hydra import GlobalHydra
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        logger.info("Initializing models on %s...", self.device)
        self._initialize_hydra(hydra, GlobalHydra)

        from sam2.build_sam import build_sam2_video_predictor

        self.predictor = build_sam2_video_predictor(
            self.settings.sam2_config_name,
            str(self.settings.sam2_checkpoint),
            device=self.device,
        )
        self.birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet",
            trust_remote_code=True,
            dtype=self._torch.float32,
        )
        self.birefnet.to(self.device).eval()
        self.birefnet_transform = transforms.Compose(
            [transforms.Resize((1024, 1024)), transforms.ToTensor()]
        )

    def _initialize_hydra(self, hydra_module, hydra_state) -> None:
        config_dir = self.settings.sam2_config_dir
        if not config_dir.exists():
            raise FileNotFoundError(f"SAM2 config directory not found: {config_dir}")
        if not hydra_state.instance().is_initialized():
            hydra_module.initialize_config_dir(
                config_dir=str(config_dir),
                version_base="1.2",
            )

    def get_smart_mask(self, image_path_or_np: str | Path | np.ndarray) -> np.ndarray:
        if self.birefnet is None or self.birefnet_transform is None:
            raise RuntimeError("MaskingService must be initialized before use.")

        if isinstance(image_path_or_np, (str, Path)):
            image = Image.open(image_path_or_np).convert("RGB")
            logger.debug("BiRefNet: loading image %s, size=%s", image_path_or_np, image.size)
        else:
            image = Image.fromarray(cv2.cvtColor(image_path_or_np, cv2.COLOR_BGR2RGB))
            logger.debug("BiRefNet: processing numpy image, size=%s", image.size)

        orig_w, orig_h = image.size
        input_tensor = self.birefnet_transform(image).unsqueeze(0).to(self.device)
        logger.debug("BiRefNet: input_tensor shape=%s, device=%s", input_tensor.shape, self.device)
        with self._torch.no_grad():
            predictions = self.birefnet(input_tensor)[-1]
            mask = (predictions.sigmoid().cpu()[0][0].numpy() > 0.4).astype(np.uint8)
        return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
