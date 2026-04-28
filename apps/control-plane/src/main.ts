import { loadConfig } from "./config";
import { createPool } from "./db";
import { runMigrations } from "./migrations";
import { createRedis } from "./redis";
import { Repository } from "./repository";
import { createServer } from "./server";
import { CpuLeaseService } from "./services/cpu-lease";
import { ObjectStorageService } from "./services/object-storage";
import { QuotaService } from "./services/quota-service";
import { TelegramClient } from "./telegram/client";
import { createCpuWorker } from "./workers/cpu-worker";
import { createGpuWorker } from "./workers/gpu-worker";
import { createSenderWorker } from "./workers/sender-worker";

async function main() {
  const config = loadConfig();
  const pool = createPool(config);
  await runMigrations(pool);

  const redis = createRedis(config);
  const repository = new Repository(pool, config);
  const telegram = new TelegramClient(config.telegramBotToken);
  const objectStorage = new ObjectStorageService(config);
  const quotaService = new QuotaService(redis);
  const cpuLeaseService = new CpuLeaseService(redis, config.cpuUserLeaseTtlSeconds);
  await logTelegramIdentity(config.serviceRole, config.telegramBotToken, telegram);

  if (config.serviceRole === "api") {
    if (config.telegramWebhookUrl) {
      validateWebhookUrl(config.telegramWebhookUrl);
    }
    const server = createServer({
      config,
      repository,
      quotaService,
      connection: redis,
      telegram,
    });
    if (config.telegramWebhookUrl) {
      await telegram.setWebhook(config.telegramWebhookUrl, config.telegramWebhookSecret);
    }
    await server.listen({
      host: "0.0.0.0",
      port: config.port,
    });
    return;
  }

  if (config.serviceRole === "cpu-worker") {
    const worker = createCpuWorker({
      config,
      repository,
      connection: redis,
      leaseService: cpuLeaseService,
      objectStorage,
      telegram,
    });
    worker.on("completed", (job) => console.log(`[cpu-worker] completed ${job.id}`));
    worker.on("failed", (job, error) => console.error(`[cpu-worker] failed ${job?.id}:`, error));
    return;
  }

  if (config.serviceRole === "gpu-worker") {
    const worker = createGpuWorker({
      config,
      repository,
      connection: redis,
    });
    worker.on("completed", (job) => console.log(`[gpu-worker] completed ${job.id}`));
    worker.on("failed", (job, error) => console.error(`[gpu-worker] failed ${job?.id}:`, error));
    return;
  }

  if (config.serviceRole === "sender-worker") {
    const worker = createSenderWorker({
      repository,
      connection: redis,
      objectStorage,
      telegram,
    });
    worker.on("completed", (job) => console.log(`[sender-worker] completed ${job.id}`));
    worker.on("failed", (job, error) => console.error(`[sender-worker] failed ${job?.id}:`, error));
  }
}

void main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

async function logTelegramIdentity(
  role: string,
  token: string,
  telegram: TelegramClient,
): Promise<void> {
  try {
    const me = await telegram.getMe();
    console.log(`[startup] role=${role} bot=@${me.username} token_id=${token.split(":")[0]}`);
  } catch (error) {
    console.warn(
      `[startup] role=${role} failed to resolve bot identity: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

function validateWebhookUrl(value: string): void {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(
      `TELEGRAM_WEBHOOK_URL must be a valid absolute URL, got: ${value}`,
    );
  }

  if (parsed.protocol !== "https:") {
    throw new Error(
      `TELEGRAM_WEBHOOK_URL must start with https:// and include /telegram/webhook, got: ${value}`,
    );
  }

  if (parsed.pathname !== "/telegram/webhook") {
    throw new Error(
      `TELEGRAM_WEBHOOK_URL must point to /telegram/webhook, got: ${value}`,
    );
  }
}
