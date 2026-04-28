from __future__ import annotations

import asyncio

from sticker_bot.config import load_settings
from sticker_bot.domain.tasks import StickerTask
from sticker_bot.services.scheduler import TaskScheduler


def make_task(user_id: int, file_id: str) -> StickerTask:
    return StickerTask(
        user_id=user_id,
        user_name=str(user_id),
        file_id=file_id,
        w_count=1,
        back_mode="none",
        tolerance=10,
        is_video=False,
        is_gif=False,
        status_message=None,
        source_message=None,
    )


def test_scheduler_fairness_between_users(tmp_path):
    settings = load_settings({"BOT_TOKEN": "token"}, project_root=tmp_path, load_env_file=False)
    scheduler = TaskScheduler(settings, processor=object(), uploader=object())

    async def scenario():
        await scheduler.enqueue(make_task(1, "u1-a"))
        await scheduler.enqueue(make_task(1, "u1-b"))
        await scheduler.enqueue(make_task(2, "u2-a"))

        first = await scheduler.acquire_next_task()
        second = await scheduler.acquire_next_task()
        await scheduler.mark_complete(1)
        third = await scheduler.acquire_next_task()

        return first.file_id, second.file_id, third.file_id

    assert asyncio.run(scenario()) == ("u1-a", "u2-a", "u1-b")


def test_scheduler_builds_unique_task_dirs(tmp_path):
    settings = load_settings({"BOT_TOKEN": "token"}, project_root=tmp_path, load_env_file=False)
    scheduler = TaskScheduler(settings, processor=object(), uploader=object())
    task = make_task(1, "BAACAgQAAxkBAAIB")

    first = scheduler._build_task_dir(0, task)
    second = scheduler._build_task_dir(0, task)

    assert first != second
    assert first.parent == settings.temp_dir
    assert second.parent == settings.temp_dir
