import test from "node:test";
import assert from "node:assert/strict";

import {
  getCommandValidationError,
  inferSourceKind,
  parseBackOptions,
  parseCommand,
  requiresGpu,
} from "../src/lib/commands";

test("parseCommand supports bot mentions", () => {
  assert.deepEqual(parseCommand("/add@TestBot 4 back=auto"), {
    name: "add",
    args: ["4", "back=auto"],
  });
  assert.equal(parseCommand("/puzzle 4 back=auto"), null);
});

test("parseBackOptions keeps valid named colors and tolerance", () => {
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
});

test("requiresGpu only for auto background or removebg", () => {
  assert.equal(requiresGpu("puzzle", parseBackOptions(["3"], 10)), false);
  assert.equal(requiresGpu("puzzle", parseBackOptions(["3", "back=auto"], 10)), true);
  assert.equal(requiresGpu("remove_bg", parseBackOptions([], 10)), true);
});

test("inferSourceKind recognises GIF documents as video source", () => {
  assert.deepEqual(
    inferSourceKind({
      document: {
        mime_type: "image/gif",
        file_name: "demo.gif",
      },
    }),
    { kind: "video", isGif: true },
  );
});

test("inferSourceKind recognises image documents as photo source", () => {
  assert.deepEqual(
    inferSourceKind({
      document: {
        mime_type: "image/png",
        file_name: "demo.png",
      },
    }),
    { kind: "photo", isGif: false },
  );
});

test("inferSourceKind rejects unsupported documents", () => {
  assert.equal(
    inferSourceKind({
      document: {
        mime_type: "application/pdf",
        file_name: "demo.pdf",
      },
    }),
    null,
  );
});

test("command validation matches removebg and circle restrictions", () => {
  assert.equal(
    getCommandValidationError("removebg", { kind: "video", isGif: true }),
    "The /removebg command expects a photo or image document.",
  );
  assert.equal(
    getCommandValidationError("circle", { kind: "photo", isGif: false }),
    "The /circle command currently expects video, animation, or GIF input.",
  );
  assert.equal(getCommandValidationError("add", { kind: "photo", isGif: false }), null);
});
