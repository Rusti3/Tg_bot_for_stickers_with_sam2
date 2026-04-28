from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ArtifactRefModel(BaseModel):
    objectKey: str
    contentType: str | None = None
    fileName: str | None = None


class PuzzleArtifactsModel(BaseModel):
    cols: int
    rows: int
    format: Literal["static", "video"]
    ext: Literal["png", "webm"]
    segments: list[ArtifactRefModel] = Field(default_factory=list)
    packName: str | None = None
    customEmojiIds: list[str] = Field(default_factory=list)


class CommandModel(BaseModel):
    wCount: int = 1
    backMode: str = "none"
    tolerance: int = 10


class SourceModel(BaseModel):
    fileId: str
    fileName: str | None = None
    mimeType: str | None = None
    isGif: bool = False


class DeliveryModel(BaseModel):
    chatId: int
    replyToMessageId: int | None = None
    userId: int
    username: str | None = None


class ArtifactsModel(BaseModel):
    puzzle: PuzzleArtifactsModel | None = None
    outputFile: ArtifactRefModel | None = None


class PayloadModel(BaseModel):
    command: CommandModel | None = None
    source: SourceModel | None = None
    delivery: DeliveryModel | None = None
    artifacts: ArtifactsModel | None = None
    planCode: str | None = None


class ExecutorRequestModel(BaseModel):
    jobId: str
    jobType: Literal["puzzle", "stickers", "circle_video", "remove_bg"]
    stage: Literal["prepare", "gpu", "finalize", "deliver"]
    sourceKind: Literal["photo", "video"]
    sourceObjectKey: str
    resultPrefix: str
    payload: PayloadModel = Field(default_factory=PayloadModel)


class ExecutorResponseModel(BaseModel):
    stage: Literal["deliver", "finalize"]
    resultObjectKey: str | None = None
    deliveryHandled: bool = False
    payloadPatch: dict = Field(default_factory=dict)
