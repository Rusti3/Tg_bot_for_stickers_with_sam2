# Telegram Sticker Bot for SAM2/BiRefNet

Docker-first Telegram bot for puzzle stickers, background removal, and circle videos.

The production runtime is webhook-only and is designed for a Linux Docker GPU environment:

- Node/TypeScript control plane
- PostgreSQL for users, jobs, usage history, payments, and entitlements
- Redis for atomic daily quotas and BullMQ queues
- Python CPU/GPU executors that reuse the existing SAM2/BiRefNet media pipeline
- MinIO for source/result artifacts
- Cloudflare Tunnel for public ingress without a public IP

## Requirements

- Docker Desktop or Docker Engine with Linux containers
- NVIDIA GPU driver
- NVIDIA Container Toolkit configured for Docker GPU access
- A Telegram bot token from BotFather
- A domain managed by Cloudflare DNS
- A Cloudflare remotely managed tunnel token

GPU mode is the supported deployment mode for this version.

## Architecture

The runtime is split into these services:

- `api`: receives Telegram webhooks, validates updates, upserts users, checks quota, creates jobs, and enqueues BullMQ tasks
- `cloudflared`: exposes the internal `api` service to the Internet through a remotely managed Cloudflare Tunnel
- `cpu-worker-1..3`: BullMQ workers with global CPU concurrency `3` and per-user CPU exclusivity enforced through Redis leases
- `gpu-worker`: executes GPU jobs through the GPU executor
- `sender-worker`: sends invoices, status messages, `/removebg`, and `/circle` results back to Telegram
- `media-cpu-executor`: Python HTTP service for CPU-side media processing
- `media-gpu-executor`: Python HTTP service for SAM2/BiRefNet and auto background removal
- `postgres`, `redis`, `minio`

`/add` puzzle-pack upload is handled inside the Python CPU executor to preserve the legacy custom-emoji puzzle behavior: `100x100` tiles, legacy video/GIF encoding, `custom_emoji` sticker sets, HTML `<tg-emoji>` grid, and install button.

## Cloudflare Tunnel

Production ingress is expected to go through Cloudflare Tunnel.

- no public IP is required
- no port forwarding is required
- no `A` record to the local machine is required
- the `api` service is not published to the host in the main compose file

Use a dedicated hostname such as `bot.yourdomain.com` or `bottg.example.com`.

In Cloudflare dashboard:

1. Move the domain to Cloudflare DNS if it is not already there.
2. Create a remotely managed tunnel.
3. Add a public hostname `bot.yourdomain.com`.
4. Route that hostname to `http://api:3000`.
5. Copy the tunnel token and place it into `.env` as `CLOUDFLARE_TUNNEL_TOKEN`.

Cloudflare will create the required DNS record automatically when you add the public hostname in the dashboard.

Do not create an `A` record to your home PC. The default compose file does not publish `api:3000` to the host; Cloudflare reaches it over the internal Docker network through `cloudflared`.

## CPU Fairness

The CPU lane implements the rule:

- up to `3` CPU jobs can run at the same time globally
- one Telegram user can hold only `1` active CPU lease at a time
- extra jobs from the same user stay queued and are retried later
- other users can occupy the remaining CPU workers immediately

This is enforced with Redis keys shaped like `cpu:user:{tg_user_id}` and lease TTL renewal while a worker is alive.

## Bot Commands

- `/add [w_count] [back=...]`
- `/removebg`
- `/circle`
- `/plans`

Current job mapping:

- `/add` -> `puzzle`
- `/removebg` -> `remove_bg`
- `/circle` -> `circle_video`

`/add` accepts media on the same message or as a reply. Examples:

```text
/add 3 back=auto
/add 4 back=black
/add 4 back=black120
/add 2 back=#00ff00
```

`/removebg` is image-only and returns a transparent PNG document.

`/circle` expects video, animation, or GIF media.

## Telegram Stars

The paid tier is configured by env vars and enabled through Telegram Stars invoices:

- free tier: `10` accepted jobs/day
- paid tier: `100` accepted jobs/day for `30` days

Successful payments are stored in Postgres and create or extend `user_entitlements`.

## Quick Start

1. Clone the repository:

```powershell
git clone https://github.com/Rusti3/Tg_bot_for_stickers_with_sam2.git
Set-Location Tg_bot_for_stickers_with_sam2
```

2. Copy the example env file:

```powershell
Copy-Item .env.example .env
```

3. Fill in at least:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_WEBHOOK_SECRET`
- `TELEGRAM_WEBHOOK_URL`
- `CLOUDFLARE_TUNNEL_TOKEN`

`TELEGRAM_WEBHOOK_URL` must be the public Cloudflare hostname plus `/telegram/webhook`, for example:

```env
TELEGRAM_WEBHOOK_URL=https://bot.yourdomain.com/telegram/webhook
```

4. Start the full stack:

```powershell
docker compose up --build -d
```

The first build is slow because it installs CUDA/PyTorch/SAM2/BiRefNet dependencies. Later Node-only changes rebuild much faster.

5. Check status and logs:

```powershell
docker compose ps
docker compose logs -f cloudflared
docker compose logs -f api
```

6. Check health through Cloudflare:

```powershell
Invoke-WebRequest https://bot.yourdomain.com/health -UseBasicParsing
```

7. Check Telegram webhook state:

```powershell
docker compose exec api node -e "const t=process.env.TELEGRAM_BOT_TOKEN; fetch('https://api.telegram.org/bot'+t+'/getWebhookInfo').then(r=>r.json()).then(j=>console.log(JSON.stringify(j.result,null,2)))"
```

The app calls `setWebhook` automatically on `api` startup when `TELEGRAM_WEBHOOK_URL` is set.

## Webhook Notes

- Telegram webhook URL must be exactly `https://bot.yourdomain.com/telegram/webhook`
- `TELEGRAM_WEBHOOK_SECRET` is the Telegram header secret
- `CLOUDFLARE_TUNNEL_TOKEN` is a separate Cloudflare credential
- the webhook path is `/telegram/webhook`
- the health endpoint is `/health`
- changing `.env` values requires container recreate, not rebuild:

```powershell
docker compose up -d --force-recreate api cpu-worker-1 cpu-worker-2 cpu-worker-3 gpu-worker sender-worker media-cpu-executor media-gpu-executor cloudflared
```

If logs still show an old bot, check the active token id:

```powershell
docker compose logs api cpu-worker-1 cpu-worker-2 cpu-worker-3 gpu-worker sender-worker | Select-String startup
```

## Environment

The main runtime variables are:

```env
POSTGRES_DB=sticker_bot
POSTGRES_USER=sticker_bot
POSTGRES_PASSWORD=sticker_bot
DATABASE_URL=postgresql://sticker_bot:sticker_bot@postgres:5432/sticker_bot
REDIS_URL=redis://redis:6379/0
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_WEBHOOK_SECRET=replace_me_with_random_secret
TELEGRAM_WEBHOOK_URL=https://bot.yourdomain.com/telegram/webhook
CLOUDFLARE_TUNNEL_TOKEN=your_cloudflare_tunnel_token_here
DATA_DIR=/app/data
TEMP_DIR=/app/data/tmp
SAM2_CHECKPOINT=/app/data/checkpoints/sam2.1_hiera_base_plus.pt
SAM2_CONFIG_NAME=sam2.1_hiera_b+.yaml
FFMPEG_CMD=ffmpeg
MAX_GRID_WIDTH=10
OBJECT_STORAGE_ENDPOINT=minio:9000
OBJECT_STORAGE_ACCESS_KEY=minioadmin
OBJECT_STORAGE_SECRET_KEY=minioadmin
OBJECT_STORAGE_BUCKET=sticker-bot
CPU_EXECUTOR_URL=http://media-cpu-executor:8001
GPU_EXECUTOR_URL=http://media-gpu-executor:8002
CPU_WORKER_REPLICAS=3
CPU_PER_USER_CONCURRENCY=1
CPU_USER_LEASE_TTL_SECONDS=900
GPU_PROVIDER=local
PAID_PLAN_CODE=pro_100_daily_30d
PAID_PLAN_DAILY_LIMIT=100
PAID_PLAN_DURATION_DAYS=30
PAID_PLAN_STARS_AMOUNT=1
```

## Data Model

The stack creates these core tables through migrations:

- `users`
- `jobs`
- `usage_events`
- `daily_usage`
- `telegram_updates`
- `stars_payments`
- `user_entitlements`

## Local Checks

Python checks:

```powershell
python -m pytest -q
python -m compileall src tests
```

Node control plane checks after installing dependencies in `apps/control-plane`:

```powershell
npm install
npm run build
npm test
```

Compose checks:

```powershell
docker compose --env-file .env.example config
```

## Troubleshooting

- `cloudflared` shows registered tunnel connections but the Cloudflare dashboard says waiting: verify the public hostname is attached to the same tunnel token and routes to `http://api:3000`.
- `https://your-host/health` returns `404`: the Cloudflare hostname path/routing is wrong, or the hostname is attached to another tunnel/application.
- Telegram says webhook must be HTTPS: `TELEGRAM_WEBHOOK_URL` must start with `https://` and end with `/telegram/webhook`.
- The bot still listens to the old bot: update `TELEGRAM_BOT_TOKEN` in the root `.env` and run the recreate command above. No rebuild is needed for token changes.
- First GPU startup can take time while models load/download. Keep `data/` mounted so checkpoints and caches survive restarts.

## Legacy Note

The old polling entrypoint is no longer the production path. Runtime starts through Docker Compose, Node webhook/control-plane services, and Python executors.
