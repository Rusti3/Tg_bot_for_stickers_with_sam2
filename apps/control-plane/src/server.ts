import Fastify from "fastify";

import { AppConfig } from "./config";
import {
  parseBackOptions,
  parseCommand,
  requiresGpu,
  inferSourceKind,
  getCommandValidationError,
} from "./lib/commands";
import { getQuotaWindow } from "./lib/time";
import { createCpuQueue, createSendQueue } from "./queues";
import { Repository } from "./repository";
import { QuotaService } from "./services/quota-service";
import { TelegramClient } from "./telegram/client";
import { JobPayload, ParsedCommand, TelegramUserShape } from "./types";

function updateTypeOf(update: any): string {
  if (update.pre_checkout_query) {
    return "pre_checkout_query";
  }
  if (update.callback_query) {
    return "callback_query";
  }
  if (update.message?.successful_payment) {
    return "successful_payment";
  }
  if (update.message) {
    return "message";
  }
  return "unknown";
}

function userFromUpdate(update: any): TelegramUserShape | null {
  return update.message?.from ?? update.callback_query?.from ?? update.pre_checkout_query?.from ?? null;
}

function buildUpgradeMarkup() {
  return {
    inline_keyboard: [[{ text: "Upgrade with Stars", callback_data: "plans:buy" }]],
  };
}

function extractTargetMessage(message: any) {
  return message.reply_to_message ?? message;
}

function pickFileId(message: any): string | null {
  if (Array.isArray(message.photo) && message.photo.length > 0) {
    return message.photo[message.photo.length - 1]?.file_id ?? null;
  }
  return message.video?.file_id ?? message.animation?.file_id ?? message.document?.file_id ?? null;
}

function buildJobPayload(
  message: any,
  user: TelegramUserShape,
  command: ParsedCommand,
  maxGridWidth: number,
): JobPayload {
  const target = extractTargetMessage(message);
  const options = parseBackOptions(command.args, maxGridWidth);
  return {
    command: options,
    source: {
      fileId: pickFileId(target) ?? "",
      fileName: target.document?.file_name ?? null,
      mimeType: target.document?.mime_type ?? null,
      isGif: Boolean(target.document && inferSourceKind(target)?.isGif),
    },
    delivery: {
      chatId: message.chat.id,
      replyToMessageId: message.message_id,
      userId: user.id,
      username: user.username ?? null,
    },
  };
}

export function createServer(args: {
  config: AppConfig;
  repository: Repository;
  quotaService: QuotaService;
  connection: any;
  telegram: TelegramClient;
}) {
  const fastify = Fastify({
    logger: true,
  });
  const cpuQueue = createCpuQueue(args.connection);
  const sendQueue = createSendQueue(args.connection);

  fastify.get("/health", async () => ({
    ok: true,
    role: args.config.serviceRole,
  }));

  fastify.post("/telegram/webhook", async (request, reply) => {
    const secret = request.headers["x-telegram-bot-api-secret-token"];
    if (secret !== args.config.telegramWebhookSecret) {
      return reply.code(401).send({ ok: false });
    }

    const update = request.body as any;
    const updateId = Number(update.update_id);
    const updateType = updateTypeOf(update);
    const updateUser = userFromUpdate(update);
    const recorded = await args.repository.recordTelegramUpdate(
      updateId,
      updateUser?.id ?? null,
      updateType,
    );
    if (!recorded) {
      return reply.code(200).send({ ok: true, duplicate: true });
    }

    if (!updateUser) {
      return reply.code(200).send({ ok: true });
    }

    const userRow = await args.repository.upsertUser(updateUser);

    if (update.pre_checkout_query) {
      const query = update.pre_checkout_query;
      const payload = String(query.invoice_payload ?? "");
      const valid =
        payload.startsWith(`stars:${args.config.paidPlanCode}:`) &&
        query.currency === "XTR" &&
        query.total_amount === args.config.paidPlanStarsAmount;
      await args.telegram.answerPreCheckoutQuery(
        query.id,
        valid,
        valid ? undefined : "This payment payload is no longer valid.",
      );
      return reply.code(200).send({ ok: true });
    }

    if (update.callback_query) {
      const callback = update.callback_query;
      await args.telegram.answerCallbackQuery(callback.id);
      if (callback.data === "plans:buy") {
        await sendQueue.add(`invoice:${callback.from.id}:${Date.now()}`, {
          kind: "invoice",
          chatId: callback.message?.chat?.id ?? callback.from.id,
          title: "Sticker Bot Pro",
          description: `100 accepted jobs per day for ${args.config.paidPlanDurationDays} days.`,
          payload: `stars:${args.config.paidPlanCode}:${callback.from.id}`,
          amount: args.config.paidPlanStarsAmount,
          buttonText: "Upgrade",
        });
      }
      return reply.code(200).send({ ok: true });
    }

    if (update.message?.successful_payment) {
      const payment = update.message.successful_payment;
      await args.repository.createOrExtendEntitlement({
        userId: userRow.id,
        telegramPaymentChargeId: payment.telegram_payment_charge_id,
        providerPaymentChargeId: payment.provider_payment_charge_id ?? null,
        invoicePayload: payment.invoice_payload,
        amount: payment.total_amount,
        currency: payment.currency,
      });
      await sendQueue.add(`payment-ok:${updateUser.id}:${Date.now()}`, {
        kind: "message",
        chatId: update.message.chat.id,
        text: `Pro activated. Your daily limit is now ${args.config.paidPlanDailyLimit} jobs for ${args.config.paidPlanDurationDays} days.`,
      });
      return reply.code(200).send({ ok: true });
    }

    if (!update.message) {
      return reply.code(200).send({ ok: true });
    }

    const message = update.message;
    const command = parseCommand(message.text ?? message.caption);
    if (!command) {
      return reply.code(200).send({ ok: true });
    }

    if (command.name === "plans") {
      const quotaWindow = getQuotaWindow();
      const plan = await args.repository.getEffectivePlan(userRow.id);
      const used = await args.quotaService.getUsage(updateUser.id, quotaWindow.quotaKeySuffix);
      const statusLine = plan.isPaid
        ? `Current plan: ${plan.planCode} (${used}/${plan.dailyLimit} used today).`
        : `Current plan: free (${used}/${plan.dailyLimit} used today).`;
      await sendQueue.add(`plans:${updateUser.id}:${Date.now()}`, {
        kind: "message",
        chatId: message.chat.id,
        text:
          `${statusLine}\n` +
          `Free tier: 10 accepted jobs/day.\n` +
          `Pro tier: ${args.config.paidPlanDailyLimit} accepted jobs/day for ${args.config.paidPlanDurationDays} days.\n` +
          `Upgrade price: ${args.config.paidPlanStarsAmount} Stars.`,
        replyMarkup: buildUpgradeMarkup(),
      });
      return reply.code(200).send({ ok: true });
    }

    const targetMessage = extractTargetMessage(message);
    const source = inferSourceKind(targetMessage);
    const fileId = pickFileId(targetMessage);
    if (!source || !fileId) {
      const unsupportedDocument = Boolean(targetMessage.document && !source);
      await sendQueue.add(`${unsupportedDocument ? "unsupported-media" : "missing-media"}:${updateUser.id}:${Date.now()}`, {
        kind: "message",
        chatId: message.chat.id,
        text: unsupportedDocument
          ? "This document type is not supported. Use a photo, image document, video, animation, or GIF."
          : "Send or reply to a photo, image document, video, animation, or GIF with the command so I can create a job.",
      });
      return reply.code(200).send({ ok: true });
    }

    const validationError = getCommandValidationError(command.name, source);
    if (validationError) {
      await sendQueue.add(`command-invalid:${command.name}:${updateUser.id}:${Date.now()}`, {
        kind: "message",
        chatId: message.chat.id,
        text: validationError,
      });
      return reply.code(200).send({ ok: true });
    }

    const jobType =
      command.name === "removebg"
        ? "remove_bg"
        : command.name === "circle"
          ? "circle_video"
          : "puzzle";
    const payload = buildJobPayload(message, updateUser, command, args.config.maxGridWidth);
    const quotaWindow = getQuotaWindow();
    const plan = await args.repository.getEffectivePlan(userRow.id);
    const reservation = await args.quotaService.reserve(
      updateUser.id,
      quotaWindow.quotaKeySuffix,
      plan.dailyLimit,
      quotaWindow.expireAtEpochSeconds,
    );

    if (!reservation.accepted) {
      await args.repository.withTransaction(async (client) => {
        await args.repository.appendUsageEvent(userRow.id, quotaWindow.quotaDate, "rejected_limit", null, client);
        await args.repository.incrementDailyUsage(userRow.id, quotaWindow.quotaDate, "rejected", client);
      });
      await sendQueue.add(`quota-hit:${updateUser.id}:${Date.now()}`, {
        kind: "message",
        chatId: message.chat.id,
        text: `Daily limit reached (${plan.dailyLimit}/${plan.dailyLimit}). Try again tomorrow or upgrade to Pro.`,
        replyMarkup: buildUpgradeMarkup(),
      });
      return reply.code(200).send({ ok: true });
    }

    const options = payload.command ?? parseBackOptions(command.args, args.config.maxGridWidth);
    const requiresGpuFlag = requiresGpu(jobType, options);
    let createdJobId: string | null = null;
    try {
      const createdJob = await args.repository.withTransaction(async (client) => {
        const job = await args.repository.createQueuedJob(
          {
            userId: userRow.id,
            telegramUpdateId: updateId,
            jobType,
            sourceKind: source.kind,
            sourceFileId: fileId,
            requiresGpu: requiresGpuFlag,
            payload,
          },
          client,
        );
        await args.repository.appendUsageEvent(userRow.id, quotaWindow.quotaDate, "accepted", job.id, client);
        await args.repository.incrementDailyUsage(userRow.id, quotaWindow.quotaDate, "accepted", client);
        return job;
      });
      createdJobId = createdJob.id;
      await cpuQueue.add(`cpu:${createdJob.id}`, { jobId: createdJob.id });
    } catch (error) {
      await args.quotaService.rollback(updateUser.id, quotaWindow.quotaKeySuffix);
      if (createdJobId) {
        await args.repository.patchJob(createdJobId, {
          status: "failed",
          errorText: error instanceof Error ? error.message : String(error),
          markFinished: true,
        });
      }
      throw error;
    }

    await sendQueue.add(`queued:${createdJobId}`, {
      kind: "message",
      chatId: message.chat.id,
      text: "Job accepted and queued. I will send the result here when it is ready.",
    });

    return reply.code(200).send({ ok: true });
  });

  return fastify;
}
