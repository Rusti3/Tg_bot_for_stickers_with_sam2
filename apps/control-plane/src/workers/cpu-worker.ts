import { DelayedError, Worker } from "bullmq";

import { AppConfig } from "../config";
import { getQuotaWindow } from "../lib/time";
import { createCpuQueue, createGpuQueue, createSendQueue, CPU_QUEUE_NAME } from "../queues";
import { Repository } from "../repository";
import { CpuLeaseService } from "../services/cpu-lease";
import { ObjectStorageService } from "../services/object-storage";
import { TelegramClient } from "../telegram/client";
import { ExecutorRequest } from "../types";
import { callExecutor } from "../executors";
import { buildResultPrefix, buildSourceObjectKey, inferContentType } from "./shared";

export function createCpuWorker(args: {
  config: AppConfig;
  repository: Repository;
  connection: any;
  leaseService: CpuLeaseService;
  objectStorage: ObjectStorageService;
  telegram: TelegramClient;
}) {
  const cpuQueue = createCpuQueue(args.connection);
  const gpuQueue = createGpuQueue(args.connection);
  const sendQueue = createSendQueue(args.connection);

  return new Worker(
    CPU_QUEUE_NAME,
    async (bullJob, token) => {
      const storedJob = await args.repository.getJob(bullJob.data.jobId);
      if (!storedJob) {
        return;
      }

      const leaseToken = `${storedJob.id}:${Date.now()}`;
      const leaseAcquired = await args.leaseService.acquire(storedJob.tg_user_id, leaseToken);
      if (!leaseAcquired) {
        const delayMs = 1500 + Math.floor(Math.random() * 1500);
        await bullJob.moveToDelayed(Date.now() + delayMs, token);
        throw new DelayedError();
      }

      const heartbeat = setInterval(() => {
        void args.leaseService.renew(storedJob.tg_user_id, leaseToken);
      }, Math.max(1000, Math.floor((args.config.cpuUserLeaseTtlSeconds * 1000) / 3)));

      try {
        await args.repository.patchJob(storedJob.id, {
          status: "processing",
          progress: 10,
          markStarted: true,
          errorText: null,
        });

        let sourceObjectKey = storedJob.source_object_key;
        if (!sourceObjectKey) {
          const file = await args.telegram.downloadFile(storedJob.source_file_id ?? "");
          sourceObjectKey = buildSourceObjectKey(storedJob.id, file.filePath);
          await args.objectStorage.uploadBuffer(
            sourceObjectKey,
            file.buffer,
            inferContentType(file.filePath, storedJob.source_kind),
          );
          await args.repository.patchJob(storedJob.id, {
            sourceObjectKey,
            progress: 25,
          });
        }

        if (storedJob.stage === "prepare" && storedJob.requires_gpu) {
          await args.repository.patchJob(storedJob.id, {
            status: "waiting_gpu",
            stage: "gpu",
            sourceObjectKey,
            progress: 35,
          });
          await gpuQueue.add(`gpu:${storedJob.id}`, { jobId: storedJob.id });
          return;
        }

        const request: ExecutorRequest = {
          jobId: storedJob.id,
          jobType: storedJob.job_type,
          stage: storedJob.stage,
          sourceKind: storedJob.source_kind,
          sourceObjectKey,
          resultPrefix: buildResultPrefix(storedJob.id),
          payload: storedJob.payload,
        };
        const result = await callExecutor(args.config.cpuExecutorUrl, request);
        const quotaDay = getQuotaWindow().quotaDate;
        const deliveryHandled = Boolean(result.deliveryHandled);

        await args.repository.patchJob(storedJob.id, {
          status: deliveryHandled ? "delivered" : "completed",
          stage: "deliver",
          progress: 100,
          resultObjectKey: result.resultObjectKey ?? null,
          payloadPatch: result.payloadPatch,
          markFinished: true,
          errorText: null,
        });
        await args.repository.appendUsageEvent(storedJob.user_id, quotaDay, "completed", storedJob.id);
        if (!deliveryHandled) {
          await sendQueue.add(`send:${storedJob.id}`, {
            kind: "job-result",
            jobId: storedJob.id,
          });
        }
      } catch (error) {
        const quotaDay = getQuotaWindow().quotaDate;
        await args.repository.patchJob(storedJob.id, {
          status: "failed",
          stage: "deliver",
          errorText: error instanceof Error ? error.message : String(error),
          markFinished: true,
        });
        await args.repository.appendUsageEvent(storedJob.user_id, quotaDay, "failed", storedJob.id);
        await sendQueue.add(`send-failed:${storedJob.id}`, {
          kind: "message",
          chatId: storedJob.payload.delivery?.chatId ?? storedJob.tg_user_id,
          text: "Job failed while processing. Please try again later.",
        });
        throw error;
      } finally {
        clearInterval(heartbeat);
        await args.leaseService.release(storedJob.tg_user_id, leaseToken);
      }
    },
    {
      connection: args.connection,
      concurrency: 1,
    },
  );
}
