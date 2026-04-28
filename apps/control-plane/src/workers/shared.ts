import path from "node:path";

import { JobPayload, JobType, SourceKind } from "../types";

export function inferContentType(filePath: string, sourceKind: SourceKind): string {
  const extension = path.extname(filePath).toLowerCase();
  if (extension === ".png") {
    return "image/png";
  }
  if (extension === ".gif") {
    return "image/gif";
  }
  if (extension === ".jpg" || extension === ".jpeg") {
    return "image/jpeg";
  }
  if (extension === ".webm") {
    return "video/webm";
  }
  if (extension === ".mp4") {
    return "video/mp4";
  }
  return sourceKind === "photo" ? "image/jpeg" : "application/octet-stream";
}

export function buildSourceObjectKey(jobId: string, filePath: string): string {
  return `jobs/${jobId}/source/${path.basename(filePath) || "source.bin"}`;
}

export function buildResultPrefix(jobId: string): string {
  return `jobs/${jobId}/results`;
}

export function buildPackName(jobId: string, botUsername: string): string {
  const compact = jobId.replace(/-/g, "").slice(0, 12);
  return `puzzle_${compact}_by_${botUsername}`.slice(0, 64);
}

export function renderPuzzleHtml(customEmojiIds: Array<string | null | undefined>, cols: number, rows: number): string | null {
  if (customEmojiIds.length < cols * rows || customEmojiIds.some((id) => !id)) {
    return null;
  }

  return Array.from({ length: rows }, (_, row) =>
    Array.from({ length: cols }, (_, col) => {
      const id = customEmojiIds[row * cols + col];
      return `<tg-emoji emoji-id="${id}">\uD83E\uDDE9</tg-emoji>`;
    }).join(""),
  ).join("\n");
}

export function renderPuzzleText(cols: number, rows: number): string {
  const piece = "\uD83E\uDDE9";
  return Array.from({ length: rows }, () => piece.repeat(cols)).join("\n");
}

export function shouldRouteThroughGpu(jobType: JobType, payload: JobPayload): boolean {
  return jobType === "remove_bg" || payload.command?.backMode === "auto";
}
