import { Worker } from "bullmq";

import { createSendQueue, SEND_QUEUE_NAME } from "../queues";
import { Repository } from "../repository";
import { ObjectStorageService } from "../services/object-storage";
import { TelegramApiError, TelegramClient } from "../telegram/client";
import { SendTask } from "../types";

export function createSenderWorker(args: {
  repository: Repository;
  connection: any;
  objectStorage: ObjectStorageService;
  telegram: TelegramClient;
}) {
  createSendQueue(args.connection);

  return new Worker<SendTask>(
    SEND_QUEUE_NAME,
    async (bullJob) => {
      if (bullJob.data.kind === "message") {
        await args.telegram.sendMessage(
          bullJob.data.chatId,
          bullJob.data.text,
          bullJob.data.replyMarkup,
        );
        return;
      }

      if (bullJob.data.kind === "invoice") {
        await args.telegram.sendInvoice(bullJob.data);
        return;
      }

      const storedJob = await args.repository.getJob(bullJob.data.jobId);
      if (!storedJob) {
        return;
      }

      try {
        if (storedJob.job_type === "puzzle" || storedJob.job_type === "stickers") {
          const errorText =
            "Puzzle delivery reached sender-worker unexpectedly. Legacy /add delivery must be handled by the Python CPU executor.";
          console.error(`[sender-worker] ${errorText} job=${storedJob.id}`);
          await args.repository.patchJob(storedJob.id, {
            status: "failed",
            stage: "deliver",
            errorText,
            markFinished: true,
          });
          return;
        } else if (storedJob.job_type === "remove_bg") {
          const outputFile = storedJob.payload.artifacts?.outputFile;
          if (!outputFile) {
            throw new Error("Background removal output is missing");
          }
          const data = await args.objectStorage.downloadBuffer(outputFile.objectKey);
          const message = await args.telegram.sendDocument(
            storedJob.payload.delivery?.chatId ?? storedJob.tg_user_id,
            outputFile,
            data,
            "Background removed.",
          );
          const resultFileId = message?.document?.file_id ?? message?.video?.file_id ?? null;
          await args.repository.patchJob(storedJob.id, {
            resultFileId,
          });
        } else if (storedJob.job_type === "circle_video") {
          const outputFile = storedJob.payload.artifacts?.outputFile;
          if (!outputFile) {
            throw new Error("Circle video output is missing");
          }
          const data = await args.objectStorage.downloadBuffer(outputFile.objectKey);
          await args.telegram.sendVideoNote(
            storedJob.payload.delivery?.chatId ?? storedJob.tg_user_id,
            outputFile.fileName ?? "circle.mp4",
            data,
          );
        }

        await args.repository.patchJob(storedJob.id, {
          status: "delivered",
          stage: "deliver",
          errorText: null,
        });
      } catch (error) {
        if (error instanceof TelegramApiError && error.errorCode === 403) {
          await args.repository.markUserBlocked(storedJob.tg_user_id);
          return;
        }
        throw error;
      }
    },
    {
      connection: args.connection,
      concurrency: 1,
    },
  );
}
