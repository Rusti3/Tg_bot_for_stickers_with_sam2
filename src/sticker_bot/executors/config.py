from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

from sticker_bot.config import Settings, load_settings


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    return int(value)


@dataclass(frozen=True)
class ExecutorSettings:
    runtime: Settings
    object_storage_endpoint: str
    object_storage_access_key: str
    object_storage_secret_key: str
    object_storage_bucket: str
    object_storage_secure: bool
    port: int


def load_executor_settings(
    env: Mapping[str, str] | None = None,
    *,
    project_root: Path | None = None,
) -> ExecutorSettings:
    runtime = load_settings(env=env, project_root=project_root)
    root = runtime.project_root
    source: dict[str, str] = {
        key: value
        for key, value in dotenv_values(root / ".env").items()
        if value is not None
    }
    source.update(dict(env or os.environ))

    return ExecutorSettings(
        runtime=runtime,
        object_storage_endpoint=source.get("OBJECT_STORAGE_ENDPOINT", "minio:9000"),
        object_storage_access_key=source.get("OBJECT_STORAGE_ACCESS_KEY", "minioadmin"),
        object_storage_secret_key=source.get("OBJECT_STORAGE_SECRET_KEY", "minioadmin"),
        object_storage_bucket=source.get("OBJECT_STORAGE_BUCKET", "sticker-bot"),
        object_storage_secure=_parse_bool(source.get("OBJECT_STORAGE_USE_SSL"), False),
        port=_parse_int(source.get("PORT"), 8001),
    )
