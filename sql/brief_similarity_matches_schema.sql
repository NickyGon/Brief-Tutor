-- Brief-to-brief similarity matches. Campaign pair rows stay in campaign_similarity.

create table if not exists public.brief_similarity_matches (
  id bigserial primary key,
  target_brief_id text not null,
  candidate_brief_id text not null,
  target_file_name text,
  candidate_file_name text,
  brief_similarity_score double precision not null default 0,
  match_kind text not null default 'none',
  coverage double precision not null default 0,
  pair_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint uq_brief_similarity_matches_pair unique (target_brief_id, candidate_brief_id)
);

create index if not exists idx_brief_similarity_matches_target
  on public.brief_similarity_matches(target_brief_id);

alter table public.analysis_runs
  add column if not exists documents jsonb not null default '[]'::jsonb;
