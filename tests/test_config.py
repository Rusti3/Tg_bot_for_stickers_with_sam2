from __future__ import annotations

from sticker_bot.config import load_settings


def test_load_settings_resolves_paths_and_defaults(tmp_path):
    settings = load_settings(
        {
            "BOT_TOKEN": "token",
            "DATA_DIR": "runtime-data",
            "TEMP_DIR": "runtime-data/tmp-files",
            "LIMITS_FILE": "runtime-data/state/limits.json",
            "SAM2_CHECKPOINT": "runtime-data/checkpoints/model.pt",
            "CONCURRENT_GPU_TASKS": "4",
            "AUTO_DOWNLOAD_SAM2": "false",
        },
        project_root=tmp_path,
        load_env_file=False,
    )

    assert settings.bot_token == "token"
    assert settings.data_dir == (tmp_path / "runtime-data").resolve()
    assert settings.temp_dir == (tmp_path / "runtime-data" / "tmp-files").resolve()
    assert settings.limits_file == (tmp_path / "runtime-data" / "state" / "limits.json").resolve()
    assert settings.sam2_checkpoint == (tmp_path / "runtime-data" / "checkpoints" / "model.pt").resolve()
    assert settings.concurrent_gpu_tasks == 4
    assert settings.auto_download_sam2 is False


def test_ensure_runtime_dirs_creates_expected_tree(tmp_path):
    settings = load_settings(
        {"BOT_TOKEN": "token"},
        project_root=tmp_path,
        load_env_file=False,
    )

    settings.ensure_runtime_dirs()

    assert settings.data_dir.is_dir()
    assert settings.temp_dir.is_dir()
    assert settings.checkpoints_dir.is_dir()
    assert settings.state_dir.is_dir()


def test_load_settings_accepts_telegram_bot_token_alias(tmp_path):
    settings = load_settings(
        {"TELEGRAM_BOT_TOKEN": "token-from-v2"},
        project_root=tmp_path,
        load_env_file=False,
    )

    assert settings.bot_token == "token-from-v2"
