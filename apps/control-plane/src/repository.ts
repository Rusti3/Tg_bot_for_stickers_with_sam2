import { randomUUID } from "node:crypto";

import { DateTime } from "luxon";

import { AppConfig } from "./config";
import {
  EffectivePlan,
  FREE_DAILY_LIMIT,
  JobPayload,
  JobStage,
  JobStatus,
  JobType,
  SourceKind,
  TelegramUserShape,
  UsageEventType,
} from "./types";

interface Queryable {
  query<T = any>(text: string, params?: any[]): Promise<{ rows: T[]; rowCount: number | null }>;
}

interface PoolLike extends Queryable {
  connect(): Promise<Queryable & { release(): void }>;
}

export interface StoredUser {
  id: number;
  tg_user_id: number;
  username: string | null;
  is_blocked: boolean;
}

export interface StoredJob {
  id: string;
  user_id: number;
  tg_user_id: number;
  username: string | null;
  telegram_update_id: number | null;
  job_type: JobType;
  source_kind: SourceKind;
  source_file_id: string | null;
  status: JobStatus;
  requires_gpu: boolean;
  progress: number;
  stage: JobStage;
  payload: JobPayload;
  source_object_key: string | null;
  result_object_key: string | null;
  provider_job_id: string | null;
  result_file_id: string | null;
  error_text: string | null;
}

export class Repository {
  constructor(
    private readonly pool: PoolLike,
    private readonly config: AppConfig,
  ) {}

  async withTransaction<T>(callback: (client: Queryable & { release(): void }) => Promise<T>): Promise<T> {
    const client = await this.pool.connect();
    try {
      await client.query("begin");
      const result = await callback(client);
      await client.query("commit");
      return result;
    } catch (error) {
      await client.query("rollback");
      throw error;
    } finally {
      client.release();
    }
  }

  async recordTelegramUpdate(
    updateId: number,
    tgUserId: number | null,
    updateType: string,
  ): Promise<boolean> {
    const result = await this.pool.query(
      `
        insert into telegram_updates(update_id, tg_user_id, update_type)
        values ($1, $2, $3)
        on conflict (update_id) do nothing
        returning update_id
      `,
      [updateId, tgUserId, updateType],
    );
    return Boolean(result.rowCount);
  }

  async upsertUser(user: TelegramUserShape, db: Queryable = this.pool): Promise<StoredUser> {
    const result = await db.query<StoredUser>(
      `
        insert into users (tg_user_id, username, first_name, last_name, language_code, last_seen_at)
        values ($1, $2, $3, $4, $5, now())
        on conflict (tg_user_id) do update
          set username = excluded.username,
              first_name = excluded.first_name,
              last_name = excluded.last_name,
              language_code = excluded.language_code,
              last_seen_at = now()
        returning id, tg_user_id, username, is_blocked
      `,
      [user.id, user.username ?? null, user.first_name ?? null, user.last_name ?? null, user.language_code ?? null],
    );
    return result.rows[0];
  }

  async getEffectivePlan(userId: number, now = DateTime.now()): Promise<EffectivePlan> {
    const asOf = now.toUTC().toISO();
    const result = await this.pool.query<{ plan_code: string; daily_limit: number }>(
      `
        select plan_code, daily_limit
        from user_entitlements
        where user_id = $1
          and is_active = true
          and starts_at <= $2::timestamptz
          and ends_at > $2::timestamptz
        order by ends_at desc
        limit 1
      `,
      [userId, asOf],
    );

    if (!result.rowCount) {
      return {
        planCode: "free",
        dailyLimit: FREE_DAILY_LIMIT,
        isPaid: false,
      };
    }

    return {
      planCode: result.rows[0].plan_code,
      dailyLimit: result.rows[0].daily_limit,
      isPaid: true,
    };
  }

  async createQueuedJob(
    args: {
      userId: number;
      telegramUpdateId: number;
      jobType: JobType;
      sourceKind: SourceKind;
      sourceFileId: string;
      requiresGpu: boolean;
      payload: JobPayload;
    },
    db: Queryable = this.pool,
  ): Promise<StoredJob> {
    const jobId = randomUUID();
    const result = await db.query<StoredJob>(
      `
        insert into jobs (
          id,
          user_id,
          telegram_update_id,
          job_type,
          source_kind,
          source_file_id,
          status,
          requires_gpu,
          stage,
          payload
        )
        values ($1, $2, $3, $4, $5, $6, 'queued', $7, 'prepare', $8::jsonb)
        returning
          id,
          user_id,
          0::bigint as tg_user_id,
          null::text as username,
          telegram_update_id,
          job_type,
          source_kind,
          source_file_id,
          status,
          requires_gpu,
          progress,
          stage,
          payload,
          source_object_key,
          result_object_key,
          provider_job_id,
          result_file_id,
          error_text
      `,
      [
        jobId,
        args.userId,
        args.telegramUpdateId,
        args.jobType,
        args.sourceKind,
        args.sourceFileId,
        args.requiresGpu,
        JSON.stringify(args.payload),
      ],
    );
    return result.rows[0];
  }

  async appendUsageEvent(
    userId: number,
    quotaDay: string,
    eventType: UsageEventType,
    jobId?: string | null,
    db: Queryable = this.pool,
  ): Promise<void> {
    await db.query(
      `
        insert into usage_events (user_id, job_id, event_type, quota_day)
        values ($1, $2, $3, $4::date)
      `,
      [userId, jobId ?? null, eventType, quotaDay],
    );
  }

  async incrementDailyUsage(
    userId: number,
    quotaDay: string,
    kind: "accepted" | "rejected",
    db: Queryable = this.pool,
  ): Promise<void> {
    const column = kind === "accepted" ? "accepted_count" : "rejected_count";
    await db.query(
      `
        insert into daily_usage(user_id, quota_day, accepted_count, rejected_count)
        values ($1, $2::date, $3, $4)
        on conflict (user_id, quota_day) do update
          set ${column} = daily_usage.${column} + 1
      `,
      [userId, quotaDay, kind === "accepted" ? 1 : 0, kind === "rejected" ? 1 : 0],
    );
  }

  async getJob(jobId: string): Promise<StoredJob | null> {
    const result = await this.pool.query<StoredJob>(
      `
        select
          j.id,
          j.user_id,
          u.tg_user_id,
          u.username,
          j.telegram_update_id,
          j.job_type,
          j.source_kind,
          j.source_file_id,
          j.status,
          j.requires_gpu,
          j.progress,
          j.stage,
          j.payload,
          j.source_object_key,
          j.result_object_key,
          j.provider_job_id,
          j.result_file_id,
          j.error_text
        from jobs j
        join users u on u.id = j.user_id
        where j.id = $1
      `,
      [jobId],
    );
    return result.rows[0] ?? null;
  }

  async patchJob(
    jobId: string,
    patch: {
      status?: JobStatus;
      stage?: JobStage;
      progress?: number;
      sourceObjectKey?: string | null;
      resultObjectKey?: string | null;
      providerJobId?: string | null;
      errorText?: string | null;
      resultFileId?: string | null;
      payloadPatch?: Partial<JobPayload>;
      markStarted?: boolean;
      markFinished?: boolean;
    },
  ): Promise<void> {
    await this.pool.query(
      `
        update jobs
        set
          status = coalesce($2, status),
          stage = coalesce($3, stage),
          progress = coalesce($4, progress),
          source_object_key = coalesce($5, source_object_key),
          result_object_key = coalesce($6, result_object_key),
          provider_job_id = coalesce($7, provider_job_id),
          error_text = $8,
          result_file_id = coalesce($9, result_file_id),
          payload = case
            when $10::jsonb is null then payload
            else payload || $10::jsonb
          end,
          started_at = case
            when $11 then coalesce(started_at, now())
            else started_at
          end,
          finished_at = case
            when $12 then now()
            else finished_at
          end
        where id = $1
      `,
      [
        jobId,
        patch.status ?? null,
        patch.stage ?? null,
        patch.progress ?? null,
        patch.sourceObjectKey ?? null,
        patch.resultObjectKey ?? null,
        patch.providerJobId ?? null,
        patch.errorText ?? null,
        patch.resultFileId ?? null,
        patch.payloadPatch ? JSON.stringify(patch.payloadPatch) : null,
        Boolean(patch.markStarted),
        Boolean(patch.markFinished),
      ],
    );
  }

  async markUserBlocked(tgUserId: number): Promise<void> {
    await this.pool.query("update users set is_blocked = true where tg_user_id = $1", [tgUserId]);
  }

  async getActiveEntitlementForUser(userId: number): Promise<{
    plan_code: string;
    daily_limit: number;
    ends_at: Date;
  } | null> {
    const result = await this.pool.query<{
      plan_code: string;
      daily_limit: number;
      ends_at: Date;
    }>(
      `
        select plan_code, daily_limit, ends_at
        from user_entitlements
        where user_id = $1
          and is_active = true
          and ends_at > now()
        order by ends_at desc
        limit 1
      `,
      [userId],
    );
    return result.rows[0] ?? null;
  }

  async createOrExtendEntitlement(args: {
    userId: number;
    telegramPaymentChargeId: string;
    providerPaymentChargeId?: string | null;
    invoicePayload: string;
    amount: number;
    currency: string;
  }): Promise<void> {
    await this.withTransaction(async (client) => {
      const paymentInsert = await client.query<{ id: number }>(
        `
          insert into stars_payments (
            user_id,
            telegram_payment_charge_id,
            provider_payment_charge_id,
            invoice_payload,
            amount,
            currency,
            status
          )
          values ($1, $2, $3, $4, $5, $6, 'paid')
          on conflict (telegram_payment_charge_id) do nothing
          returning id
        `,
        [
          args.userId,
          args.telegramPaymentChargeId,
          args.providerPaymentChargeId ?? null,
          args.invoicePayload,
          args.amount,
          args.currency,
        ],
      );

      if (!paymentInsert.rowCount) {
        return;
      }

      const activeEntitlement = await client.query<{ ends_at: Date }>(
        `
          select ends_at
          from user_entitlements
          where user_id = $1
            and plan_code = $2
            and is_active = true
            and ends_at > now()
          order by ends_at desc
          limit 1
        `,
        [args.userId, this.config.paidPlanCode],
      );

      const startAt =
        activeEntitlement.rows[0]?.ends_at instanceof Date
          ? DateTime.fromJSDate(activeEntitlement.rows[0].ends_at)
          : DateTime.now();
      const endsAt = startAt.plus({ days: this.config.paidPlanDurationDays });

      await client.query(
        `
          insert into user_entitlements (
            user_id,
            plan_code,
            daily_limit,
            starts_at,
            ends_at,
            is_active,
            source_payment_id
          )
          values ($1, $2, $3, $4, $5, true, $6)
        `,
        [
          args.userId,
          this.config.paidPlanCode,
          this.config.paidPlanDailyLimit,
          startAt.toUTC().toISO(),
          endsAt.toUTC().toISO(),
          paymentInsert.rows[0].id,
        ],
      );
    });
  }
}
