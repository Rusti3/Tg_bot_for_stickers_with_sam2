from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable


logger = logging.getLogger("StickerBot")


class RateLimitStore:
    def __init__(
        self,
        path: Path,
        *,
        window_seconds: int,
        max_requests: int,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.path = path
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.clock = clock or time.time
        self.history = self._load()

    def _load(self) -> dict[int, list[float]]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            logger.error("Failed to load rate limits: %s", exc)
            return {}
        return {int(user_id): timestamps for user_id, timestamps in data.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("w", encoding="utf-8") as handle:
                json.dump(self.history, handle)
        except Exception as exc:
            logger.error("Failed to save rate limits: %s", exc)

    def prune(self, user_id: int, now: float | None = None) -> list[float]:
        current_time = self.clock() if now is None else now
        timestamps = self.history.get(user_id, [])
        fresh = [timestamp for timestamp in timestamps if current_time - timestamp < self.window_seconds]
        if fresh:
            self.history[user_id] = fresh
        elif user_id in self.history:
            del self.history[user_id]
        return fresh

    def allow_request(self, user_id: int, now: float | None = None) -> bool:
        return len(self.prune(user_id, now)) < self.max_requests

    def record_request(self, user_id: int, now: float | None = None) -> bool:
        current_time = self.clock() if now is None else now
        timestamps = self.prune(user_id, current_time)
        if len(timestamps) >= self.max_requests:
            return False
        timestamps.append(current_time)
        self.history[user_id] = timestamps
        self.save()
        return True

    def stats(self) -> tuple[int, int]:
        return len(self.history), sum(len(timestamps) for timestamps in self.history.values())
