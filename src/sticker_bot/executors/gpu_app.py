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


logger = logging.getLogger("sticker_bot.gpu_executor")


def create_app() -> FastAPI:
    settings = load_executor_settings()
    settings.runtime.ensure_runtime_dirs()
    storage = ObjectStorage(settings)
    masking = MaskingService(settings.runtime)
    masking.initialize()
    processor = ProcessingService(settings.runtime, masking)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with executor_bot_lifespan(settings.runtime.bot_token, "gpu-executor") as bot:
            app.state.executor_bot = bot
            yield

    app = FastAPI(title="Sticker Bot GPU Executor", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"ok": "true", "role": "gpu"}

    @app.post("/execute", response_model=ExecutorResponseModel)
    async def execute(request: ExecutorRequestModel) -> ExecutorResponseModel:
        return await _execute_gpu_job(settings, storage, processor, request)

    return app


async def _execute_gpu_job(
    settings: ExecutorSettings,
    storage: ObjectStorage,
    processor: ProcessingService,
    request: ExecutorRequestModel,
) -> ExecutorResponseModel:
    work_dir = settings.runtime.temp_dir / f"gpu_executor_{request.jobId}"
    work_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path(request.sourceObjectKey).name or "source.bin"
    source_path = storage.download_file(request.sourceObjectKey, work_dir / source_name)

    try:
        if request.jobType == "remove_bg":
            if request.sourceKind == "photo":
                data = await asyncio.to_thread(processor.remove_background_photo_sync, source_path)
                object_key = f"{request.resultPrefix}/remove_bg.png"
                storage.upload_bytes(object_key, data, "image/png")
                return ExecutorResponseModel(
                    stage="deliver",
                    resultObjectKey=object_key,
                    payloadPatch={
                        "artifacts": {
                            "outputFile": {
                                "objectKey": object_key,
                                "contentType": "image/png",
                                "fileName": "remove_bg.png",
                            }
                        }
                    },
                )

            data = await processor.remove_background_video(
                source_path,
                work_dir,
                is_gif=bool(request.payload.source and request.payload.source.isGif),
            )
            object_key = f"{request.resultPrefix}/remove_bg.webm"
            storage.upload_bytes(object_key, data, "video/webm")
            return ExecutorResponseModel(
                stage="deliver",
                resultObjectKey=object_key,
                payloadPatch={
                    "artifacts": {
                        "outputFile": {
                            "objectKey": object_key,
                            "contentType": "video/webm",
                            "fileName": "remove_bg.webm",
                        }
                    }
                },
            )

        if request.jobType in {"puzzle", "stickers"}:
            command = request.payload.command or {}
            w_count = getattr(command, "wCount", 1)
            tolerance = getattr(command, "tolerance", 10)

            if request.sourceKind == "photo":
                segments, cols, rows = await asyncio.to_thread(
                    processor.process_photo_sync,
                    source_path,
                    w_count,
                    "auto",
                    tolerance,
                )
                fmt, ext, content_type = "static", "png", "image/png"
            else:
                segments, cols, rows = await processor.process_video(
                    source_path,
                    work_dir,
                    w_count,
                    "auto",
                    tolerance,
                    is_gif=bool(request.payload.source and request.payload.source.isGif),
                )
                fmt, ext, content_type = "video", "webm", "video/webm"

            segment_refs = []
            for index, segment in enumerate(segments):
                object_key = f"{request.resultPrefix}/segments/{index:03d}.{ext}"
                storage.upload_bytes(object_key, segment, content_type)
                segment_refs.append(
                    {
                        "objectKey": object_key,
                        "contentType": content_type,
                        "fileName": f"segment_{index:03d}.{ext}",
                    }
                )

            return ExecutorResponseModel(
                stage="finalize",
                payloadPatch={
                    "artifacts": {
                        "puzzle": {
                            "cols": cols,
                            "rows": rows,
                            "format": fmt,
                            "ext": ext,
                            "segments": segment_refs,
                        }
                    }
                },
            )

        raise ValueError(f"Unsupported GPU executor job type: {request.jobType}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


app = create_app()
