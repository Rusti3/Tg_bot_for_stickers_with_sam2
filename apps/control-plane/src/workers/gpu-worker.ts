import { Worker } from "bullmq";

import { AppConfig } from "../config";
import { callExecutor } from "../executors";
import { getQuotaWindow } from "../lib/time";
import { createCpuQueue, createSendQueue, GPU_QUEUE_NAME } from "../queues";
import { Repository } from "../repository";
import { ExecutorRequest } from "../types";
import { buildResultPrefix } from "./shared";

export function createGpuWorker(args: {
  config: AppConfig;
  repository: Repository;
  connection: any;
}) {
  const cpuQueue = createCpuQueue(args.connection);
  const sendQueue = createSendQueue(args.connection);

  return new Worker(
    GPU_QUEUE_NAME,
    async (bullJob) => {
      const storedJob = await args.repository.getJob(bullJob.data.jobId);
      if (!storedJob) {
        return;
      }

      try {
        await args.repository.patchJob(storedJob.id, {
          status: "processing",
          stage: "gpu",
          progress: 60,
          markStarted: true,
          errorText: null,
        });

        const request: ExecutorRequest = {
          jobId: storedJob.id,
          jobType: storedJob.job_type,
          stage: "gpu",
          sourceKind: storedJob.source_kind,
          sourceObjectKey: storedJob.source_object_key ?? "",
          resultPrefix: buildResultPrefix(storedJob.id),
          payload: storedJob.payload,
        };
        const result = await callExecutor(args.config.gpuExecutorUrl, request);
        const quotaDay = getQuotaWindow().quotaDate;

        if (result.stage === "finalize") {
          await args.repository.patchJob(storedJob.id, {
            status: "processing",
            stage: "finalize",
            progress: 85,
            resultObjectKey: result.resultObjectKey ?? null,
            payloadPatch: result.payloadPatch,
            errorText: null,
          });
          await cpuQueue.add(`cpu-finalize:${storedJob.id}`, { jobId: storedJob.id });
          return;
        }

        await args.repository.patchJob(storedJob.id, {
          status: "completed",
          stage: "deliver",
          progress: 100,
          resultObjectKey: result.resultObjectKey ?? null,
          payloadPatch: result.payloadPatch,
          markFinished: true,
          errorText: null,
        });
        await args.repository.appendUsageEvent(storedJob.user_id, quotaDay, "completed", storedJob.id);
        await sendQueue.add(`send:${storedJob.id}`, {
          kind: "job-result",
          jobId: storedJob.id,
        });
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
          text: "GPU stage failed. Please try again later.",
        });
        throw error;
      }
    },
    {
      connection: args.connection,
      concurrency: 1,
    },
  );
}
