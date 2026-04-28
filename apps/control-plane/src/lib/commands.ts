import { BackOptions, JobType, ParsedCommand, SourceKind } from "../types";

const COLOR_NAMES = new Set(["white", "black", "green", "blue", "red"]);
const IMAGE_DOCUMENT_EXTENSIONS = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".bmp",
  ".tif",
  ".tiff",
  ".heic",
  ".heif",
]);
const VIDEO_DOCUMENT_EXTENSIONS = new Set([
  ".mp4",
  ".m4v",
  ".mov",
  ".webm",
  ".avi",
  ".mkv",
  ".mpeg",
  ".mpg",
]);

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function parseCommand(text?: string | null): ParsedCommand | null {
  if (!text || !text.startsWith("/")) {
    return null;
  }

  const [commandToken, ...args] = text.trim().split(/\s+/);
  const bareCommand = commandToken.slice(1).split("@")[0]?.toLowerCase();
  if (
    bareCommand !== "add" &&
    bareCommand !== "removebg" &&
    bareCommand !== "circle" &&
    bareCommand !== "plans"
  ) {
    return null;
  }

  return {
    name: bareCommand,
    args,
  };
}

export function parseBackOptions(args: string[], maxGridWidth: number): BackOptions {
  const options: BackOptions = {
    backMode: "none",
    tolerance: 10,
    wCount: 1,
  };

  for (const arg of args) {
    if (/^\d+$/.test(arg)) {
      options.wCount = clamp(Number.parseInt(arg, 10), 1, maxGridWidth);
      continue;
    }

    if (!arg.toLowerCase().startsWith("back=")) {
      continue;
    }

    const rawValue = arg.slice(5).toLowerCase();
    if (rawValue === "auto" || rawValue === "none") {
      options.backMode = rawValue;
      continue;
    }

    const parsed = parseBackParam(rawValue);
    if (parsed.valid) {
      options.backMode = rawValue;
      options.tolerance = parsed.tolerance;
    }
  }

  return options;
}

export function parseBackParam(rawValue: string): {
  valid: boolean;
  tolerance: number;
} {
  const normalized = rawValue.toLowerCase();
  if (COLOR_NAMES.has(normalized) || /^(?:#)?([0-9a-f]{6})$/i.test(normalized)) {
    return { valid: true, tolerance: 10 };
  }

  const toleranceMatch = normalized.match(/(\d+)$/);
  if (!toleranceMatch) {
    return { valid: false, tolerance: 10 };
  }

  const tolerance = Number.parseInt(toleranceMatch[1], 10);
  const colorPart = normalized.slice(0, toleranceMatch.index);
  if (COLOR_NAMES.has(colorPart)) {
    return { valid: true, tolerance };
  }

  return { valid: false, tolerance };
}

export function requiresGpu(jobType: JobType, options: BackOptions): boolean {
  return jobType === "remove_bg" || options.backMode === "auto";
}

export function inferSourceKind(
  message: Record<string, any>,
): { kind: SourceKind; isGif: boolean } | null {
  if (message.photo) {
    return { kind: "photo", isGif: false };
  }

  if (message.video || message.animation) {
    return { kind: "video", isGif: false };
  }

  if (!message.document) {
    return null;
  }

  const mimeType = String(message.document?.mime_type ?? "").toLowerCase();
  const fileName = String(message.document?.file_name ?? "").toLowerCase();
  const isGif = mimeType === "image/gif" || fileName.endsWith(".gif");
  if (isGif || mimeType.startsWith("video/") || hasKnownExtension(fileName, VIDEO_DOCUMENT_EXTENSIONS)) {
    return { kind: "video", isGif };
  }

  if (mimeType.startsWith("image/") || hasKnownExtension(fileName, IMAGE_DOCUMENT_EXTENSIONS)) {
    return { kind: "photo", isGif: false };
  }

  return null;
}

export function getCommandValidationError(
  commandName: ParsedCommand["name"],
  source: { kind: SourceKind; isGif: boolean },
): string | null {
  if (commandName === "removebg" && source.kind !== "photo") {
    return "The /removebg command expects a photo or image document.";
  }

  if (commandName === "circle" && source.kind !== "video") {
    return "The /circle command currently expects video, animation, or GIF input.";
  }

  return null;
}

function hasKnownExtension(fileName: string, extensions: Set<string>): boolean {
  const lastDot = fileName.lastIndexOf(".");
  if (lastDot === -1) {
    return false;
  }
  return extensions.has(fileName.slice(lastDot));
}
