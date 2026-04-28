import assert from "node:assert/strict";

import { loadConfig } from "../src/config";
import {
  getCommandValidationError,
  inferSourceKind,
  parseBackOptions,
  parseCommand,
  requiresGpu,
} from "../src/lib/commands";
import { getQuotaWindow } from "../src/lib/time";
import { renderPuzzleHtml } from "../src/workers/shared";

function run(): void {
  assert.deepEqual(parseCommand("/add@TestBot 4 back=auto"), {
    name: "add",
    args: ["4", "back=auto"],
  });
  assert.equal(parseCommand("/puzzle 4 back=auto"), null);

  assert.deepEqual(parseBackOptions(["12", "back=black30"], 10), {
    wCount: 10,
    backMode: "black30",
    tolerance: 30,
  });
  assert.deepEqual(parseBackOptions(["2", "back=#00ff00"], 10), {
    wCount: 2,
    backMode: "#00ff00",
    tolerance: 10,
  });

  assert.equal(requiresGpu("puzzle", parseBackOptions(["3"], 10)), false);
  assert.equal(requiresGpu("puzzle", parseBackOptions(["3", "back=auto"], 10)), true);
  assert.equal(requiresGpu("remove_bg", parseBackOptions([], 10)), true);

  assert.deepEqual(
    inferSourceKind({
      document: {
        mime_type: "image/gif",
        file_name: "demo.gif",
      },
    }),
    { kind: "video", isGif: true },
  );
  assert.deepEqual(
    inferSourceKind({
      document: {
        mime_type: "image/png",
        file_name: "demo.png",
      },
    }),
    { kind: "photo", isGif: false },
  );
  assert.equal(
    inferSourceKind({
      document: {
        mime_type: "application/pdf",
        file_name: "demo.pdf",
      },
    }),
    null,
  );
  assert.equal(
    getCommandValidationError("removebg", { kind: "video", isGif: true }),
    "The /removebg command expects a photo or image document.",
  );
  assert.equal(
    getCommandValidationError("circle", { kind: "photo", isGif: false }),
    "The /circle command currently expects video, animation, or GIF input.",
  );
  assert.equal(getCommandValidationError("add", { kind: "photo", isGif: false }), null);

  const window = getQuotaWindow({
    setZone: () => ({
      startOf: () => ({
        toISODate: () => "2026-04-22",
        toFormat: () => "20260422",
        plus: () => ({
          toSeconds: () => 1776891600,
        }),
      }),
    }),
  } as any);
  assert.equal(window.quotaDate, "2026-04-22");
  assert.equal(window.quotaKeySuffix, "20260422");
  assert.equal(window.expireAtEpochSeconds, 1776891600);

  const config = loadConfig("api", {
    DATABASE_URL: "postgresql://test",
    REDIS_URL: "redis://test",
    TELEGRAM_BOT_TOKEN: "token",
    TELEGRAM_WEBHOOK_SECRET: "secret",
    OBJECT_STORAGE_ENDPOINT: "minio:9000",
    OBJECT_STORAGE_ACCESS_KEY: "minio",
    OBJECT_STORAGE_SECRET_KEY: "secret",
  } as NodeJS.ProcessEnv);
  assert.equal(config.paidPlanStarsAmount, 1);

  assert.equal(
    renderPuzzleHtml(["id1", "id2", "id3", "id4"], 2, 2),
    '<tg-emoji emoji-id="id1">🧩</tg-emoji><tg-emoji emoji-id="id2">🧩</tg-emoji>\n' +
      '<tg-emoji emoji-id="id3">🧩</tg-emoji><tg-emoji emoji-id="id4">🧩</tg-emoji>',
  );
  assert.equal(renderPuzzleHtml(["id1", null], 2, 1), null);

  console.log("control-plane manual tests passed");
}

run();
