import IORedis from "ioredis";

export interface QuotaReservation {
  accepted: boolean;
  used: number;
}

const RESERVE_QUOTA_LUA = `
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
local limit = tonumber(ARGV[1])
local expire_at = tonumber(ARGV[2])
if current >= limit then
  return {0, current}
end
current = redis.call('INCR', KEYS[1])
if current == 1 then
  redis.call('EXPIREAT', KEYS[1], expire_at)
end
return {1, current}
`;

const ROLLBACK_QUOTA_LUA = `
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
if current <= 1 then
  redis.call('DEL', KEYS[1])
  return 0
end
return redis.call('DECR', KEYS[1])
`;

export class QuotaService {
  constructor(private readonly redis: IORedis) {}

  buildQuotaKey(tgUserId: number, suffix: string): string {
    return `quota:${tgUserId}:${suffix}`;
  }

  async reserve(
    tgUserId: number,
    suffix: string,
    limit: number,
    expireAt: number,
  ): Promise<QuotaReservation> {
    const key = this.buildQuotaKey(tgUserId, suffix);
    const result = (await this.redis.eval(RESERVE_QUOTA_LUA, 1, key, limit, expireAt)) as [
      number,
      number,
    ];
    return {
      accepted: result[0] === 1,
      used: result[1],
    };
  }

  async rollback(tgUserId: number, suffix: string): Promise<void> {
    const key = this.buildQuotaKey(tgUserId, suffix);
    await this.redis.eval(ROLLBACK_QUOTA_LUA, 1, key);
  }

  async getUsage(tgUserId: number, suffix: string): Promise<number> {
    const value = await this.redis.get(this.buildQuotaKey(tgUserId, suffix));
    return Number.parseInt(value ?? "0", 10) || 0;
  }
}
