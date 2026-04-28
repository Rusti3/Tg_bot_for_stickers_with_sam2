import { Queue } from "bullmq";
import IORedis from "ioredis";

import { SendTask } from "./types";

export const CPU_QUEUE_NAME = "media.cpu";
export const GPU_QUEUE_NAME = "media.gpu";
export const SEND_QUEUE_NAME = "telegram.send";

export interface JobQueueData {
  jobId: string;
}

export function createCpuQueue(connection: IORedis): Queue<JobQueueData> {
  return new Queue<JobQueueData>(CPU_QUEUE_NAME, {
    connection,
    defaultJobOptions: {
      attempts: 5,
      backoff: {
        type: "exponential",
        delay: 2000,
      },
      removeOnComplete: 100,
      removeOnFail: 100,
    },
  });
}

export function createGpuQueue(connection: IORedis): Queue<JobQueueData> {
  return new Queue<JobQueueData>(GPU_QUEUE_NAME, {
    connection,
    defaultJobOptions: {
      attempts: 4,
      backoff: {
        type: "exponential",
        delay: 4000,
      },
      removeOnComplete: 100,
      removeOnFail: 100,
    },
  });
}

export function createSendQueue(connection: IORedis): Queue<SendTask> {
  return new Queue<SendTask>(SEND_QUEUE_NAME, {
    connection,
    defaultJobOptions: {
      attempts: 6,
      backoff: {
        type: "exponential",
        delay: 2000,
      },
      removeOnComplete: 200,
      removeOnFail: 200,
    },
  });
}
