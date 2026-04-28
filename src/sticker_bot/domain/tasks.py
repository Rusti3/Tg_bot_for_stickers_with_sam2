from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiogram.types import Message
else:
    Message = Any


@dataclass(slots=True)
class AddCommandOptions:
    w_count: int = 1
    back_mode: str = "none"
    tolerance: int = 10


@dataclass(slots=True)
class StickerTask:
    user_id: int
    user_name: str
    file_id: str
    w_count: int
    back_mode: str
    tolerance: int
    is_video: bool
    is_gif: bool
    status_message: Message
    source_message: Message
    start_time: float = field(default_factory=time.perf_counter)
