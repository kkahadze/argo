create extension if not exists pgcrypto;

create table if not exists public.translation_events (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    status text not null check (status in ('success', 'error')),
    source_text text not null,
    target_text text,
    source_language text not null,
    target_language text not null,
    provider text,
    model text,
    response_source text not null default 'unknown',
    duration_ms integer,
    source_text_length integer not null default 0,
    target_text_length integer,
    used_user_api_key boolean not null default false,
    prompt_metrics jsonb not null default '{}'::jsonb,
    error_message text,
    app_origin text,
    referer text,
    user_agent text
);

create index if not exists translation_events_created_at_idx
    on public.translation_events (created_at desc);

create index if not exists translation_events_status_idx
    on public.translation_events (status);

create index if not exists translation_events_provider_model_idx
    on public.translation_events (provider, model);

alter table public.translation_events enable row level security;

grant usage on schema public to anon, authenticated;
grant insert on public.translation_events to anon, authenticated;

drop policy if exists "translation_events_insert_only" on public.translation_events;
create policy "translation_events_insert_only"
    on public.translation_events
    for insert
    to anon, authenticated
    with check (true);
