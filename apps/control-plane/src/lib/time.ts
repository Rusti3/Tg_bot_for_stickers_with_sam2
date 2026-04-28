import { DateTime } from "luxon";

import { MOSCOW_TZ } from "../types";

export interface QuotaWindow {
  quotaDate: string;
  quotaKeySuffix: string;
  expireAtEpochSeconds: number;
}

export function getQuotaWindow(now = DateTime.now().setZone(MOSCOW_TZ)): QuotaWindow {
  const start = now.setZone(MOSCOW_TZ).startOf("day");
  const nextDay = start.plus({ days: 1 });
  return {
    quotaDate: start.toISODate() ?? "",
    quotaKeySuffix: start.toFormat("yyyyLLdd"),
    expireAtEpochSeconds: Math.floor(nextDay.toSeconds()),
  };
}

export function toMoscowDayIso(input: Date | string): string {
  const value =
    input instanceof Date
      ? DateTime.fromJSDate(input, { zone: "utc" })
      : DateTime.fromISO(input, { zone: "utc" });
  return value.setZone(MOSCOW_TZ).toISODate() ?? "";
}
