from __future__ import annotations

import asyncio
import logging
import secrets
import traceback
from collections import deque
from pathlib import Path

from sticker_bot.config import Settings
from sticker_bot.domain.tasks import StickerTask
from sticker_bot.services.processing import ProcessingService
from sticker_bot.services.uploading import UploadService, safe_api_call


logger = logging.getLogger("StickerBot")


class TaskScheduler:
    def __init__(
        self,
        settings: Settings,
        processor: ProcessingService,
        uploader: UploadService,
    ) -> None:
        self.settings = settings
        self.processor = processor
        self.uploader = uploader
        self.condition = asyncio.Condition()
        self.user_tasks: dict[int, deque[StickerTask]] = {}
        self.user_queue_order: deque[int] = deque()
        self.active_users: set[int] = set()
        self.background_tasks: set[asyncio.Task] = set()

    async def enqueue(self, task: StickerTask) -> int:
        async with self.condition:
            if task.user_id not in self.user_tasks:
                self.user_tasks[task.user_id] = deque()
            if task.user_id not in self.user_queue_order:
                self.user_queue_order.append(task.user_id)

            self.user_tasks[task.user_id].append(task)
            queue_size = sum(len(queue) for queue in self.user_tasks.values())
            self.condition.notify_all()
            return queue_size

    async def acquire_next_task(self) -> StickerTask:
        async with self.condition:
            while True:
                selected_user = next(
                    (
                        user_id
                        for user_id in self.user_queue_order
                        if user_id not in self.active_users and self.user_tasks.get(user_id)
                    ),
                    None,
                )
                if selected_user is not None:
                    task = self.user_tasks[selected_user].popleft()
                    self.active_users.add(selected_user)
                    self.user_queue_order.remove(selected_user)
                    if self.user_tasks[selected_user]:
                        self.user_queue_order.append(selected_user)
                    else:
                        del self.user_tasks[selected_user]
                    return task
                await self.condition.wait()

    async def mark_complete(self, user_id: int) -> None:
        async with self.condition:
            self.active_users.discard(user_id)
            self.condition.notify_all()

    async def start_workers(self, bot) -> list[asyncio.Task]:
        workers = []
        for worker_id in range(self.settings.concurrent_gpu_tasks):
            workers.append(asyncio.create_task(self.worker_loop(worker_id, bot)))
        return workers

    async def stop_workers(self, workers: list[asyncio.Task]) -> None:
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

    async def worker_loop(self, worker_id: int, bot) -> None:
        logger.info("GPU Worker %s started", worker_id)
        while True:
            task = await self.acquire_next_task()
            user_dir = self._build_task_dir(worker_id, task)
            try:
                user_dir.mkdir(parents=True, exist_ok=True)
                file_info = await bot.get_file(task.file_id)
                input_path = user_dir / "input_source"
                await bot.download_file(file_info.file_path, destination=str(input_path))

                if task.is_video:
                    segments, cols, rows = await self.processor.process_video(
                        input_path,
                        user_dir,
                        task.w_count,
                        task.back_mode,
                        task.tolerance,
                        is_gif=task.is_gif,
                    )
                    fmt, ext = "video", "webm"
                else:
                    segments, cols, rows = await asyncio.to_thread(
                        self.processor.process_photo_sync,
                        input_path,
                        task.w_count,
                        task.back_mode,
                        task.tolerance,
                    )
                    fmt, ext = "static", "png"

                publish_task = asyncio.create_task(
                    self.uploader.publish_pack(bot, task, user_dir, segments, cols, rows, fmt, ext)
                )
                self.background_tasks.add(publish_task)
                publish_task.add_done_callback(self.background_tasks.discard)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error(traceback.format_exc())
                await safe_api_call(
                    task.status_message.edit_text,
                    text="Processing failed. Please try again later.",
                )
            finally:
                self._empty_cuda_cache()
                await self.mark_complete(task.user_id)

    def _build_task_dir(self, worker_id: int, task: StickerTask) -> Path:
        # Use a per-task random suffix so async uploader cleanup cannot collide
        # with a later task from the same user/file-id prefix.
        return (
            self.settings.temp_dir
            / f"run_{worker_id}_{task.user_id}_{task.file_id[:6]}_{secrets.token_hex(3)}"
        )

    @staticmethod
    def _empty_cuda_cache() -> None:
        try:
            import torch
        except ModuleNotFoundError:
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
