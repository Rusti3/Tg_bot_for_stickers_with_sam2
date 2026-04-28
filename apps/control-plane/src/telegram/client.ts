import { ArtifactRef, SendInvoiceTask } from "../types";

export class TelegramApiError extends Error {
  constructor(
    message: string,
    readonly errorCode?: number,
    readonly description?: string,
  ) {
    super(message);
  }
}

interface FileResponse {
  file_id: string;
  file_unique_id: string;
  file_path: string;
}

interface StickerInfo {
  file_id: string;
  custom_emoji_id?: string | null;
}

interface StickerSetResponse {
  name: string;
  stickers: StickerInfo[];
}

export class TelegramClient {
  private readonly apiBaseUrl: string;
  private readonly fileBaseUrl: string;

  constructor(private readonly token: string) {
    this.apiBaseUrl = `https://api.telegram.org/bot${token}`;
    this.fileBaseUrl = `https://api.telegram.org/file/bot${token}`;
  }

  async getMe(): Promise<{ username: string }> {
    return this.apiCall("getMe");
  }

  async setWebhook(url: string, secretToken: string): Promise<void> {
    await this.apiCall("setWebhook", {
      url,
      secret_token: secretToken,
      allowed_updates: ["message", "callback_query", "pre_checkout_query"],
    });
  }

  async answerCallbackQuery(callbackQueryId: string): Promise<void> {
    await this.apiCall("answerCallbackQuery", {
      callback_query_id: callbackQueryId,
    });
  }

  async answerPreCheckoutQuery(preCheckoutQueryId: string, ok: boolean, errorMessage?: string) {
    await this.apiCall("answerPreCheckoutQuery", {
      pre_checkout_query_id: preCheckoutQueryId,
      ok,
      ...(errorMessage ? { error_message: errorMessage } : {}),
    });
  }

  async sendMessage(
    chatId: number,
    text: string,
    replyMarkup?: Record<string, unknown>,
    options?: {
      parseMode?: "HTML" | "MarkdownV2";
      replyToMessageId?: number;
    },
  ): Promise<void> {
    await this.apiCall("sendMessage", {
      chat_id: chatId,
      text,
      ...(replyMarkup ? { reply_markup: replyMarkup } : {}),
      ...(options?.parseMode ? { parse_mode: options.parseMode } : {}),
      ...(options?.replyToMessageId ? { reply_to_message_id: options.replyToMessageId } : {}),
    });
  }

  async sendInvoice(task: SendInvoiceTask): Promise<void> {
    await this.apiCall("sendInvoice", {
      chat_id: task.chatId,
      title: task.title,
      description: task.description,
      payload: task.payload,
      provider_token: "",
      currency: "XTR",
      prices: [{ label: task.buttonText ?? "Upgrade", amount: task.amount }],
    });
  }

  async getFile(fileId: string): Promise<FileResponse> {
    return this.apiCall("getFile", { file_id: fileId });
  }

  async downloadFile(fileId: string): Promise<{ buffer: Buffer; filePath: string }> {
    const file = await this.getFile(fileId);
    const response = await fetch(`${this.fileBaseUrl}/${file.file_path}`);
    if (!response.ok) {
      throw new TelegramApiError(`Failed to download Telegram file ${fileId}`);
    }
    const arrayBuffer = await response.arrayBuffer();
    return {
      buffer: Buffer.from(arrayBuffer),
      filePath: file.file_path,
    };
  }

  async sendDocument(chatId: number, artifact: ArtifactRef, data: Buffer, caption?: string) {
    const form = new FormData();
    form.set("chat_id", String(chatId));
    if (caption) {
      form.set("caption", caption);
    }
    form.set(
      "document",
      new Blob([bufferToBlobPart(data)], { type: artifact.contentType ?? "application/octet-stream" }),
      artifact.fileName ?? "result.bin",
    );
    const message = await this.apiCall("sendDocument", form, true);
    return message;
  }

  async sendVideoNote(chatId: number, fileName: string, data: Buffer): Promise<void> {
    const form = new FormData();
    form.set("chat_id", String(chatId));
    form.set("video_note", new Blob([bufferToBlobPart(data)], { type: "video/mp4" }), fileName);
    await this.apiCall("sendVideoNote", form, true);
  }

  async createNewStickerSet(args: {
    userId: number;
    name: string;
    title: string;
    stickerFormat: "static" | "video";
    stickerType?: "regular" | "custom_emoji";
    sticker: Buffer;
    ext: "png" | "webm";
  }): Promise<void> {
    const form = new FormData();
    const attachName = "sticker0";
    form.set("user_id", String(args.userId));
    form.set("name", args.name);
    form.set("title", args.title);
    form.set("sticker_format", args.stickerFormat);
    form.set("sticker_type", args.stickerType ?? "regular");
    form.set(
      attachName,
      new Blob([bufferToBlobPart(args.sticker)], {
        type: args.stickerFormat === "static" ? "image/png" : "video/webm",
      }),
      `sticker.${args.ext}`,
    );
    form.set(
      "stickers",
      JSON.stringify([
        {
          sticker: `attach://${attachName}`,
          format: args.stickerFormat,
          emoji_list: ["\uD83E\uDDE9"],
        },
      ]),
    );
    await this.apiCall("createNewStickerSet", form, true);
  }

  async addStickerToSet(args: {
    userId: number;
    name: string;
    stickerFormat: "static" | "video";
    sticker: Buffer;
    ext: "png" | "webm";
  }): Promise<void> {
    const form = new FormData();
    const attachName = "sticker0";
    form.set("user_id", String(args.userId));
    form.set("name", args.name);
    form.set(
      attachName,
      new Blob([bufferToBlobPart(args.sticker)], {
        type: args.stickerFormat === "static" ? "image/png" : "video/webm",
      }),
      `sticker.${args.ext}`,
    );
    form.set(
      "sticker",
      JSON.stringify({
        sticker: `attach://${attachName}`,
        format: args.stickerFormat,
        emoji_list: ["\uD83E\uDDE9"],
      }),
    );
    await this.apiCall("addStickerToSet", form, true);
  }

  async getStickerSet(name: string): Promise<StickerSetResponse> {
    return this.apiCall("getStickerSet", { name });
  }

  async apiCall<T = any>(
    method: string,
    payload?: Record<string, unknown> | FormData,
    isFormData = false,
  ): Promise<T> {
    const response = await fetch(`${this.apiBaseUrl}/${method}`, {
      method: "POST",
      headers: isFormData
        ? undefined
        : {
            "content-type": "application/json",
          },
      body: payload ? (isFormData ? (payload as FormData) : JSON.stringify(payload)) : undefined,
    });
    const data = (await response.json()) as { ok: boolean; result: T; error_code?: number; description?: string };
    if (!response.ok || !data.ok) {
      throw new TelegramApiError(
        `Telegram API call ${method} failed`,
        data.error_code,
        data.description,
      );
    }
    return data.result;
  }
}

function bufferToBlobPart(buffer: Buffer): BlobPart {
  return buffer as unknown as BlobPart;
}
