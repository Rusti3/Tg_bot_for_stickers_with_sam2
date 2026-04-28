from __future__ import annotations

import asyncio
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from sticker_bot.executors.config import ExecutorSettings, load_executor_settings
from sticker_bot.executors.models import ExecutorRequestModel, ExecutorResponseModel
from sticker_bot.executors.storage import ObjectStorage
from sticker_bot.executors.telegram_runtime import executor_bot_lifespan
from sticker_bot.services.masking import MaskingService
from sticker_bot.services.processing import ProcessingService
from sticker_bot.services.uploading import UploadService


logger = logging.getLogger("sticker_bot.cpu_executor")


def create_app() -> FastAPI:
    settings = load_executor_settings()
    settings.runtime.ensure_runtime_dirs()
    storage = ObjectStorage(settings)
    masking = MaskingService(settings.runtime)
    processor = ProcessingService(settings.runtime, masking)
    uploader = UploadService(settings.runtime)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with executor_bot_lifespan(settings.runtime.bot_token, "cpu-executor") as bot:
            app.state.executor_bot = bot
            yield

    app = FastAPI(title="Sticker Bot CPU Executor", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"ok": "true", "role": "cpu"}

    @app.post("/execute", response_model=ExecutorResponseModel)
    async def execute(request: ExecutorRequestModel) -> ExecutorResponseModel:
        return await _execute_cpu_job(
            settings,
            storage,
            processor,
            uploader,
            app.state.executor_bot,
            request,
        )

    return app


async def _execute_cpu_job(
    settings: ExecutorSettings,
    storage: ObjectStorage,
    processor: ProcessingService,
    uploader: UploadService,
    bot,
    request: ExecutorRequestModel,
) -> ExecutorResponseModel:
    work_dir = settings.runtime.temp_dir / f"cpu_executor_{request.jobId}"
    work_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path(request.sourceObjectKey).name or "source.bin"
    source_path = storage.download_file(request.sourceObjectKey, work_dir / source_name)

    try:
        if request.jobType in {"puzzle", "stickers"}:
            command = request.payload.command or {}
            delivery = request.payload.delivery
            if bot is None:
                raise RuntimeError("CPU executor is missing Telegram bot credentials.")
            if delivery is None:
                raise RuntimeError("Puzzle delivery payload is missing.")

            if request.stage == "finalize":
                puzzle = request.payload.artifacts and request.payload.artifacts.puzzle
                if puzzle is None or not puzzle.segments:
                    raise RuntimeError("Puzzle finalize stage is missing rendered segments.")
                cols = puzzle.cols
                rows = puzzle.rows
                fmt = puzzle.format
                ext = puzzle.ext
                segments = [storage.download_bytes(segment.objectKey) for segment in puzzle.segments]
                segment_refs = [
                    {
                        "objectKey": segment.objectKey,
                        "contentType": segment.contentType,
                        "fileName": segment.fileName,
                    }
                    for segment in puzzle.segments
                ]
            else:
                w_count = getattr(command, "wCount", 1)
                back_mode = getattr(command, "backMode", "none")
                tolerance = getattr(command, "tolerance", 10)

                if request.sourceKind == "photo":
                    segments, cols, rows = await asyncio.to_thread(
                        processor.process_photo_sync,
                        source_path,
                        w_count,
                        back_mode,
                        tolerance,
                    )
                    fmt, ext = "static", "png"
                else:
                    segments, cols, rows = await processor.process_video(
                        source_path,
                        work_dir,
                        w_count,
                        back_mode,
                        tolerance,
                        is_gif=bool(request.payload.source and request.payload.source.isGif),
                    )
                    fmt, ext = "video", "webm"
                segment_refs = []

            pack_result = await uploader.publish_pack_from_delivery(
                bot,
                user_id=delivery.userId,
                user_name=delivery.username or str(delivery.userId),
                chat_id=delivery.chatId,
                reply_to_message_id=delivery.replyToMessageId,
                segments=segments,
                cols=cols,
                rows=rows,
                fmt=fmt,
                ext=ext,
            )

            return ExecutorResponseModel(
                stage="deliver",
                deliveryHandled=True,
                payloadPatch={
                    "artifacts": {
                        "puzzle": {
                            "cols": cols,
                            "rows": rows,
                            "format": fmt,
                            "ext": ext,
                            "segments": segment_refs,
                            "packName": pack_result["packName"],
                            "customEmojiIds": pack_result["customEmojiIds"],
                        }
                    }
                },
            )

        if request.jobType == "circle_video":
            output_path = work_dir / "circle.mp4"
            data = await asyncio.to_thread(processor.create_circle_video_sync, source_path, output_path)
            object_key = f"{request.resultPrefix}/circle.mp4"
            storage.upload_bytes(object_key, data, "video/mp4")
            return ExecutorResponseModel(
                stage="deliver",
                resultObjectKey=object_key,
                payloadPatch={
                    "artifacts": {
                        "outputFile": {
                            "objectKey": object_key,
                            "contentType": "video/mp4",
                            "fileName": "circle.mp4",
                        }
                    }
                },
            )

        raise ValueError(f"Unsupported CPU executor job type: {request.jobType}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


app = create_app()
