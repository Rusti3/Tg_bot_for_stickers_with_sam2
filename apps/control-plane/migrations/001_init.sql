create extension if not exists pgcrypto;

create table if not exists schema_migrations (
  name text primary key,
  applied_at timestamptz not null default now()
);

create table if not exists users (
  id bigserial primary key,
  tg_user_id bigint not null unique,
  username text,
  first_name text,
  last_name text,
  language_code text,
  is_blocked boolean not null default false,
  created_at timestamptz not null default now(),
  last_seen_at timestamptz not null default now()
);

create table if not exists jobs (
  id uuid primary key default gen_random_uuid(),
  user_id bigint not null references users(id),
  telegram_update_id bigint,
  job_type text not null,
  source_kind text not null,
  source_file_id text,
  status text not null,
  requires_gpu boolean not null default false,
  progress int not null default 0,
  stage text not null default 'prepare',
  payload jsonb not null default '{}'::jsonb,
  source_object_key text,
  result_object_key text,
  provider_job_id text,
  result_file_id text,
  error_text text,
  created_at timestamptz not null default now(),
  started_at timestamptz,
  finished_at timestamptz
);

create unique index if not exists jobs_unique_telegram_update
  on jobs (telegram_update_id)
  where telegram_update_id is not null;

create table if not exists usage_events (
  id bigserial primary key,
  user_id bigint not null references users(id),
  job_id uuid references jobs(id),
  event_type text not null,
  quota_day date not null,
  created_at timestamptz not null default now()
);

create table if not exists daily_usage (
  user_id bigint not null references users(id),
  quota_day date not null,
  accepted_count int not null default 0,
  rejected_count int not null default 0,
  primary key (user_id, quota_day)
);

create table if not exists telegram_updates (
  update_id bigint primary key,
  tg_user_id bigint,
  update_type text,
  received_at timestamptz not null default now()
);

create table if not exists stars_payments (
  id bigserial primary key,
  user_id bigint not null references users(id),
  telegram_payment_charge_id text not null unique,
  provider_payment_charge_id text,
  invoice_payload text not null,
  amount int not null,
  currency text not null,
  status text not null default 'paid',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists user_entitlements (
  id bigserial primary key,
  user_id bigint not null references users(id),
  plan_code text not null,
  daily_limit int not null,
  starts_at timestamptz not null,
  ends_at timestamptz not null,
  is_active boolean not null default true,
  source_payment_id bigint references stars_payments(id),
  created_at timestamptz not null default now()
);

create index if not exists user_entitlements_active_idx
  on user_entitlements (user_id, is_active, starts_at, ends_at);
