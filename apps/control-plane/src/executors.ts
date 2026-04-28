import { ExecutorRequest, ExecutorResponse } from "./types";

const TRANSIENT_ATTEMPTS = 90;
const TRANSIENT_DELAY_MS = 2000;

export async function callExecutor(
  baseUrl: string,
  request: ExecutorRequest,
): Promise<ExecutorResponse> {
  let lastError: unknown;

  for (let attempt = 1; attempt <= TRANSIENT_ATTEMPTS; attempt += 1) {
    try {
      const response = await fetch(`${baseUrl}/execute`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Executor ${baseUrl} failed: ${response.status} ${text}`);
      }

      return (await response.json()) as ExecutorResponse;
    } catch (error) {
      lastError = error;
      if (!isTransientExecutorError(error) || attempt === TRANSIENT_ATTEMPTS) {
        throw error;
      }

      await sleep(TRANSIENT_DELAY_MS);
    }
  }

  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

function isTransientExecutorError(error: unknown): boolean {
  if (!(error instanceof TypeError)) {
    return false;
  }

  return error.message.toLowerCase().includes("fetch failed");
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
