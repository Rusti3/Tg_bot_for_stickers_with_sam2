import IORedis from "ioredis";

import { AppConfig } from "./config";

export function createRedis(config: AppConfig): IORedis {
  return new IORedis(config.redisUrl, {
    maxRetriesPerRequest: null,
    enableReadyCheck: false,
  });
}
