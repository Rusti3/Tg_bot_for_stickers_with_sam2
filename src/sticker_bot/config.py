from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


DEFAULT_SAM2_CHECKPOINT_URL = (
    "https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/"
    "sam2.1_hiera_base_plus.pt?download=true"
)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    bot_token: str | None
    project_root: Path
    data_dir: Path
    temp_dir: Path
    limits_file: Path
    sam2_checkpoint: Path
    sam2_config_name: str
    ffmpeg_cmd: str
    concurrent_gpu_tasks: int
    max_grid_width: int
    max_files_per_hour: int
    rate_limit_window: int
    auto_download_sam2: bool
    sam2_checkpoint_url: str

    @property
    def src_root(self) -> Path:
        return self.project_root / "src"

    @property
    def sam2_config_dir(self) -> Path:
        return self.project_root / "sam2_configs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.sam2_checkpoint.parent

    @property
    def state_dir(self) -> Path:
        return self.limits_file.parent

    def ensure_runtime_dirs(self) -> None:
        for path in (self.data_dir, self.temp_dir, self.checkpoints_dir, self.state_dir):
            path.mkdir(parents=True, exist_ok=True)


def _resolve_path(raw_value: str | None, default: Path, project_root: Path) -> Path:
    candidate = Path(raw_value) if raw_value else default
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def load_settings(
    env: Mapping[str, str] | None = None,
    *,
    project_root: Path | None = None,
    load_env_file: bool = True,
) -> Settings:
    root = project_root or Path(__file__).resolve().parents[2]
    source: dict[str, str] = {}
    if load_env_file:
        source.update(
            {
                key: value
                for key, value in dotenv_values(root / ".env").items()
                if value is not None
            }
        )
    source.update(dict(env or os.environ))

    data_dir = _resolve_path(source.get("DATA_DIR"), root / "data", root)
    temp_dir = _resolve_path(source.get("TEMP_DIR"), data_dir / "tmp", root)
    limits_file = _resolve_path(
        source.get("LIMITS_FILE"),
        data_dir / "state" / "user_limits.json",
        root,
    )
    sam2_checkpoint = _resolve_path(
        source.get("SAM2_CHECKPOINT"),
        data_dir / "checkpoints" / "sam2.1_hiera_base_plus.pt",
        root,
    )

    return Settings(
        bot_token=source.get("BOT_TOKEN") or source.get("TELEGRAM_BOT_TOKEN"),
        project_root=root.resolve(),
        data_dir=data_dir,
        temp_dir=temp_dir,
        limits_file=limits_file,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config_name=source.get("SAM2_CONFIG_NAME", "sam2.1_hiera_b+.yaml"),
        ffmpeg_cmd=source.get("FFMPEG_CMD", "ffmpeg"),
        concurrent_gpu_tasks=_parse_int(source.get("CONCURRENT_GPU_TASKS"), 2),
        max_grid_width=_parse_int(source.get("MAX_GRID_WIDTH"), 10),
        max_files_per_hour=_parse_int(source.get("MAX_FILES_PER_HOUR"), 30),
        rate_limit_window=_parse_int(source.get("RATE_LIMIT_WINDOW"), 3600),
        auto_download_sam2=_parse_bool(source.get("AUTO_DOWNLOAD_SAM2"), True),
        sam2_checkpoint_url=source.get(
            "SAM2_CHECKPOINT_URL",
            DEFAULT_SAM2_CHECKPOINT_URL,
        ),
    )
