export const MOSCOW_TZ = "Europe/Moscow";
export const FREE_DAILY_LIMIT = 10;

export type JobType = "puzzle" | "stickers" | "circle_video" | "remove_bg";
export type SourceKind = "photo" | "video";
export type JobStatus =
  | "queued"
  | "processing"
  | "waiting_gpu"
  | "completed"
  | "failed"
  | "delivered";
export type JobStage = "prepare" | "gpu" | "finalize" | "deliver";
export type UsageEventType = "accepted" | "rejected_limit" | "completed" | "failed";
export type SendTaskKind = "message" | "invoice" | "job-result";
export type ExecutorStageResult = "deliver" | "finalize";

export interface BackOptions {
  backMode: string;
  tolerance: number;
  wCount: number;
}

export interface SourceDescriptor {
  fileId: string;
  fileName?: string | null;
  mimeType?: string | null;
  isGif: boolean;
}

export interface ArtifactRef {
  objectKey: string;
  contentType?: string;
  fileName?: string;
}

export interface PuzzleArtifacts {
  cols: number;
  rows: number;
  format: "static" | "video";
  ext: "png" | "webm";
  segments?: ArtifactRef[];
  packName?: string;
  customEmojiIds?: string[];
}

export interface JobPayload {
  command?: BackOptions;
  source?: SourceDescriptor;
  delivery?: {
    chatId: number;
    replyToMessageId?: number;
    userId: number;
    username?: string | null;
  };
  artifacts?: {
    puzzle?: PuzzleArtifacts;
    outputFile?: ArtifactRef;
  };
  planCode?: string;
}

export interface SendMessageTask {
  kind: "message";
  chatId: number;
  text: string;
  replyMarkup?: Record<string, unknown>;
}

export interface SendInvoiceTask {
  kind: "invoice";
  chatId: number;
  title: string;
  description: string;
  payload: string;
  amount: number;
  buttonText?: string;
}

export interface SendJobResultTask {
  kind: "job-result";
  jobId: string;
}

export type SendTask = SendMessageTask | SendInvoiceTask | SendJobResultTask;

export interface ExecutorRequest {
  jobId: string;
  jobType: JobType;
  stage: JobStage;
  sourceKind: SourceKind;
  sourceObjectKey: string;
  resultPrefix: string;
  payload: JobPayload;
}

export interface ExecutorResponse {
  stage: ExecutorStageResult;
  resultObjectKey?: string | null;
  deliveryHandled?: boolean;
  payloadPatch?: Partial<JobPayload>;
}

export interface EffectivePlan {
  planCode: string;
  dailyLimit: number;
  isPaid: boolean;
}

export interface TelegramUserShape {
  id: number;
  username?: string;
  first_name?: string;
  last_name?: string;
  language_code?: string;
}

export interface ParsedCommand {
  name: "add" | "removebg" | "circle" | "plans";
  args: string[];
}
