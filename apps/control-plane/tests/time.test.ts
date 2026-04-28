import test from "node:test";
import assert from "node:assert/strict";

import { DateTime } from "luxon";

import { getQuotaWindow } from "../src/lib/time";

test("quota window uses Moscow day boundary", () => {
  const now = DateTime.fromISO("2026-04-21T23:30:00Z");
  const window = getQuotaWindow(now);
  assert.equal(window.quotaDate, "2026-04-22");
  assert.equal(window.quotaKeySuffix, "20260422");
  assert.equal(window.expireAtEpochSeconds, 1776891600);
});
