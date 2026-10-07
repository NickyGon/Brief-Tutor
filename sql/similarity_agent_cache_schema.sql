-- Similarity agent memory/cache for faster repeated enrichment.
-- Keyed by target A-/D- brief ID + account + candidate fingerprint.
-- Optional OEM family supports related-dealership reuse lookups.

create table if not exists public.similarity_agent_cache (
  id bigserial primary key,
  target_brief_id text not null,
  account_id text,
  oem_family text,
  candidate_fingerprint text not null default '',
  task_type text,
  pair_count integer not null default 0,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint uq_similarity_agent_cache_target_account_fp
    unique (target_brief_id, account_id, candidate_fingerprint)
);

create index if not exists idx_similarity_agent_cache_target
  on public.similarity_agent_cache(target_brief_id);

create index if not exists idx_similarity_agent_cache_account
  on public.similarity_agent_cache(account_id);

create index if not exists idx_similarity_agent_cache_oem
  on public.similarity_agent_cache(oem_family);

create index if not exists idx_similarity_agent_cache_updated
  on public.similarity_agent_cache(updated_at desc);
