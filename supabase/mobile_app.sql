-- Atratus mobile app backend: run once in the Supabase SQL editor.
-- Adds device push tokens plus four read-only snapshot tables for the
-- Flutter client. Gated tables reuse the same allow-list gate as `signals`.

create table if not exists device_tokens (
  token      text primary key,
  user_id    uuid not null references auth.users(id) on delete cascade,
  updated_at timestamptz default now()
);
alter table device_tokens enable row level security;
drop policy if exists dt_own on device_tokens;
create policy dt_own on device_tokens
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create table if not exists bars (
  asset text not null,
  date  date not null,
  open double precision, high double precision,
  low  double precision, close double precision,
  primary key (asset, date)
);

create table if not exists signal_history (
  asset text not null,
  date  date not null,
  signal text,
  prob double precision,
  actual_next_ret double precision,
  correct integer,
  primary key (asset, date)
);

-- Timing-policy annotation for the Today digest (nullable; populated only when
-- GTRADE_TIMING_POLICY is on). Idempotent so re-running the file is safe.
alter table signal_history add column if not exists timing_action text;
alter table signal_history add column if not exists timing_label text;

create table if not exists guru (
  asset text primary key,
  verdict text,
  council_pct double precision,
  lynch integer, buffett integer, graham integer, munger integer,
  source text,
  date date,
  correct_5d integer
);

create table if not exists guru_stats (
  id integer primary key,
  accuracy double precision,
  n integer,
  horizon text
);

alter table bars enable row level security;
alter table signal_history enable row level security;
alter table guru enable row level security;
alter table guru_stats enable row level security;

drop policy if exists bars_read on bars;
create policy bars_read on bars for select using (is_allowed());
drop policy if exists signal_history_read on signal_history;
create policy signal_history_read on signal_history for select using (is_allowed());
drop policy if exists guru_read on guru;
create policy guru_read on guru for select using (is_allowed());
drop policy if exists guru_stats_read on guru_stats;
create policy guru_stats_read on guru_stats for select using (is_allowed());

create or replace function allowed_device_tokens()
returns table (token text) language sql security definer as $$
  select dt.token
  from device_tokens dt
  join auth.users u on u.id = dt.user_id
  join access_list a on lower(a.email) = lower(u.email);
$$;
revoke execute on function allowed_device_tokens() from anon, authenticated;

-- News layer for the mobile app. `asset` is null for the general feed and set
-- for per-asset news, so the client fetches once and splits on it. `date` is
-- the trade date the row was exported for, not the article timestamp, which
-- keeps the existing client fetcher (it filters on `date`) working unchanged.
create table if not exists news (
  id        text primary key,
  asset     text,
  date      date not null,
  published text,
  title     text not null,
  link      text,
  source    text,
  category  text,
  sentiment double precision,
  label     text
);
alter table news enable row level security;
drop policy if exists news_read on news;
create policy news_read on news for select using (is_allowed());

-- One row per asset: whether its last move was unusual for that asset, and
-- whether the day's news sentiment agreed with the direction. Kept out of the
-- news table so it does not repeat on every article.
create table if not exists news_context (
  asset       text primary key,
  date        date not null,
  move_pct    double precision,
  notable     boolean,
  consistency text
);
alter table news_context enable row level security;
drop policy if exists news_context_read on news_context;
create policy news_context_read on news_context for select using (is_allowed());

-- Known-date event risk. `kind` is 'earnings' or 'macro'; `asset` is null for
-- macro. `confirmed` is false for a yfinance estimate, which the client labels
-- as expected rather than showing it as fact.
create table if not exists events (
  id         text primary key,
  kind       text not null,
  asset      text,
  date       date not null,
  name       text not null,
  importance text,
  confirmed  boolean
);
alter table events enable row level security;
drop policy if exists events_read on events;
create policy events_read on events for select using (is_allowed());

-- Trade levels: the prices to act on for today's setups, one row per asset.
-- `amount` is null until the real account is declared in RISK_CONFIG["equity"];
-- the client then shows `pct` instead, so a null here is a state, not a gap.
-- `status` is 'ok', 'stop_breached' for a position the price has already passed,
-- or the reason a row has no levels ('no_bars', 'short_history', 'flat_atr').
-- push_signals.py DELETEs the whole table before each insert: a stale entry
-- zone is a wrong price rather than merely an old one.
create table if not exists levels (
  asset      text primary key,
  date       date not null,
  side       text,
  entry_low  double precision,
  entry_high double precision,
  stop       double precision,
  -- not "trailing": that is a reserved word in Postgres (trim(trailing ...)),
  -- and a quoted identifier would need the quotes at every later reference.
  trailing_stop boolean,
  amount     double precision,
  pct        double precision,
  bound_by   text,
  held_days  integer,
  status     text
);
alter table levels enable row level security;
drop policy if exists levels_read on levels;
create policy levels_read on levels for select using (is_allowed());
