create extension if not exists pgcrypto;

create table if not exists public.translation_events (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    visitor_id text,
    status text not null check (status in ('success', 'error')),
    source_text text not null,
    target_text text,
    source_language text not null,
    target_language text not null,
    provider text,
    model text,
    response_source text not null default 'unknown',
    translation_path text not null default 'unknown',
    used_llm boolean not null default false,
    used_evidence_bundle boolean not null default false,
    used_dictionary_entries boolean not null default false,
    used_grammar boolean not null default false,
    used_exact_candidate_shortlist boolean not null default false,
    exact_candidate_count integer,
    prompt_characters integer,
    dictionary_entries_characters integer,
    grammar_characters integer,
    llm_call_ms integer,
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

alter table public.translation_events
    add column if not exists visitor_id text;

create index if not exists translation_events_created_at_idx
    on public.translation_events (created_at desc);

create index if not exists translation_events_visitor_id_created_at_idx
    on public.translation_events (visitor_id, created_at desc)
    where visitor_id is not null;

create index if not exists translation_events_status_idx
    on public.translation_events (status);

create index if not exists translation_events_provider_model_idx
    on public.translation_events (provider, model);

create index if not exists translation_events_translation_path_idx
    on public.translation_events (translation_path, created_at desc);

create index if not exists translation_events_used_llm_idx
    on public.translation_events (used_llm, created_at desc);

alter table public.translation_events enable row level security;

grant usage on schema public to anon, authenticated;
revoke select, update, delete on public.translation_events from anon, authenticated;
grant insert on public.translation_events to anon, authenticated;

drop policy if exists "translation_events_insert_only" on public.translation_events;
create policy "translation_events_insert_only"
    on public.translation_events
    for insert
    to anon, authenticated
    with check (true);
