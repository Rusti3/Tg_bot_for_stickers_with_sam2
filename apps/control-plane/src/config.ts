import path from "node:path";

function required(env: NodeJS.ProcessEnv, name: string): string {
  const value = env[name];
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

function optional(env: NodeJS.ProcessEnv, name: string, fallback: string): string {
  return env[name] ?? fallback;
}

function integer(env: NodeJS.ProcessEnv, name: string, fallback: number): number {
  const raw = env[name];
  if (!raw) {
    return fallback;
  }
  const parsed = Number.parseInt(raw, 10);
  if (Number.isNaN(parsed)) {
    throw new Error(`Environment variable ${name} must be an integer`);
  }
  return parsed;
}

function booleanFlag(env: NodeJS.ProcessEnv, name: string, fallback: boolean): boolean {
  const raw = env[name];
  if (!raw) {
    return fallback;
  }
  return ["1", "true", "yes", "on"].includes(raw.toLowerCase());
}

export interface AppConfig {
  serviceRole: "api" | "cpu-worker" | "gpu-worker" | "sender-worker";
  port: number;
  maxGridWidth: number;
  databaseUrl: string;
  redisUrl: string;
  telegramBotToken: string;
  telegramWebhookSecret: string;
  telegramWebhookUrl?: string;
  cpuWorkerReplicas: number;
  cpuPerUserConcurrency: number;
  cpuUserLeaseTtlSeconds: number;
  gpuProvider: "local" | "runpod";
  paidPlanCode: string;
  paidPlanDailyLimit: number;
  paidPlanDurationDays: number;
  paidPlanStarsAmount: number;
  objectStorageEndpoint: string;
  objectStorageRegion: string;
  objectStorageAccessKey: string;
  objectStorageSecretKey: string;
  objectStorageBucket: string;
  objectStorageUseSsl: boolean;
  cpuExecutorUrl: string;
  gpuExecutorUrl: string;
  repositoryRoot: string;
}

export function loadConfig(
  roleArg: string | undefined = process.argv[2],
  env: NodeJS.ProcessEnv = process.env,
): AppConfig {
  const role = (roleArg ?? env.PROCESS_ROLE ?? "api") as AppConfig["serviceRole"];
  const repositoryRoot = path.resolve(__dirname, "..", "..", "..");
  return {
    serviceRole: role,
    port: integer(env, "PORT", 3000),
    maxGridWidth: integer(env, "MAX_GRID_WIDTH", 10),
    databaseUrl: required(env, "DATABASE_URL"),
    redisUrl: required(env, "REDIS_URL"),
    telegramBotToken: required(env, "TELEGRAM_BOT_TOKEN"),
    telegramWebhookSecret: required(env, "TELEGRAM_WEBHOOK_SECRET"),
    telegramWebhookUrl: env.TELEGRAM_WEBHOOK_URL,
    cpuWorkerReplicas: integer(env, "CPU_WORKER_REPLICAS", 3),
    cpuPerUserConcurrency: integer(env, "CPU_PER_USER_CONCURRENCY", 1),
    cpuUserLeaseTtlSeconds: integer(env, "CPU_USER_LEASE_TTL_SECONDS", 900),
    gpuProvider: (optional(env, "GPU_PROVIDER", "local") as AppConfig["gpuProvider"]),
    paidPlanCode: optional(env, "PAID_PLAN_CODE", "pro_100_daily_30d"),
    paidPlanDailyLimit: integer(env, "PAID_PLAN_DAILY_LIMIT", 100),
    paidPlanDurationDays: integer(env, "PAID_PLAN_DURATION_DAYS", 30),
    paidPlanStarsAmount: integer(env, "PAID_PLAN_STARS_AMOUNT", 1),
    objectStorageEndpoint: required(env, "OBJECT_STORAGE_ENDPOINT"),
    objectStorageRegion: optional(env, "OBJECT_STORAGE_REGION", "us-east-1"),
    objectStorageAccessKey: required(env, "OBJECT_STORAGE_ACCESS_KEY"),
    objectStorageSecretKey: required(env, "OBJECT_STORAGE_SECRET_KEY"),
    objectStorageBucket: optional(env, "OBJECT_STORAGE_BUCKET", "sticker-bot"),
    objectStorageUseSsl: booleanFlag(env, "OBJECT_STORAGE_USE_SSL", false),
    cpuExecutorUrl: optional(env, "CPU_EXECUTOR_URL", "http://media-cpu-executor:8001"),
    gpuExecutorUrl: optional(env, "GPU_EXECUTOR_URL", "http://media-gpu-executor:8002"),
    repositoryRoot,
  };
}
