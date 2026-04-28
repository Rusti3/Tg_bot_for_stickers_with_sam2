import IORedis from "ioredis";

const RENEW_LEASE_LUA = `
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return 0
`;

const RELEASE_LEASE_LUA = `
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
`;

export class CpuLeaseService {
  constructor(
    private readonly redis: IORedis,
    private readonly ttlSeconds: number,
  ) {}

  key(tgUserId: number): string {
    return `cpu:user:${tgUserId}`;
  }

  async acquire(tgUserId: number, token: string): Promise<boolean> {
    const result = await this.redis.set(this.key(tgUserId), token, "EX", this.ttlSeconds, "NX");
    return result === "OK";
  }

  async renew(tgUserId: number, token: string): Promise<boolean> {
    const result = (await this.redis.eval(
      RENEW_LEASE_LUA,
      1,
      this.key(tgUserId),
      token,
      this.ttlSeconds,
    )) as number;
    return result === 1;
  }

  async release(tgUserId: number, token: string): Promise<void> {
    await this.redis.eval(RELEASE_LEASE_LUA, 1, this.key(tgUserId), token);
  }
}
