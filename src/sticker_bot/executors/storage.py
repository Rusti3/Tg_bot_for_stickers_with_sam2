from __future__ import annotations

import io
from pathlib import Path

from minio import Minio

from sticker_bot.executors.config import ExecutorSettings


class ObjectStorage:
    def __init__(self, settings: ExecutorSettings) -> None:
        self.settings = settings
        self.client = Minio(
            settings.object_storage_endpoint,
            access_key=settings.object_storage_access_key,
            secret_key=settings.object_storage_secret_key,
            secure=settings.object_storage_secure,
        )

    def download_file(self, object_key: str, destination: str | Path) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.client.fget_object(self.settings.object_storage_bucket, object_key, str(target))
        return target

    def upload_bytes(self, object_key: str, data: bytes, content_type: str) -> str:
        self.client.put_object(
            self.settings.object_storage_bucket,
            object_key,
            io.BytesIO(data),
            len(data),
            content_type=content_type,
        )
        return object_key

    def download_bytes(self, object_key: str) -> bytes:
        response = self.client.get_object(self.settings.object_storage_bucket, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
