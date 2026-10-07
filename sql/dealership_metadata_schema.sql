-- Dealership metadata schema for hierarchical similarity search accuracy.
-- Manual seed model:
--   - dealership_groups
--   - dealership_accounts
--   - dealership_account_oems (multi-OEM support)

create table if not exists public.dealership_groups (
  id bigserial primary key,
  group_name text not null,
  group_code text unique,
  notes text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.dealership_accounts (
  id bigserial primary key,
  account_id text not null unique,
  account_name text,
  group_fk bigint references public.dealership_groups(id) on delete set null,
  oem_family text not null default 'NA',
  handles_all_oems boolean not null default false,
  is_active boolean not null default true,
  notes text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.dealership_account_oems (
  id bigserial primary key,
  account_fk bigint not null references public.dealership_accounts(id) on delete cascade,
  oem text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint uq_dealership_account_oem unique (account_fk, oem)
);

create index if not exists idx_dealership_accounts_group on public.dealership_accounts(group_fk);
create index if not exists idx_dealership_accounts_family on public.dealership_accounts(oem_family);
create index if not exists idx_dealership_account_oems_account on public.dealership_account_oems(account_fk);

-- Optional seed guidance:
-- - Set handles_all_oems=true for broad accounts (equivalent to OEM value "All"/"NA").
-- - Use oem_family='NA' when no specific OEM family applies.
-- - oem may be a single brand or comma-separated brands in one row (e.g. "GMC, Buick").
--   Prefer one OEM per row when practical; the app also splits comma-separated values.
-- - OEM value "NA" or "All" means the account works with all OEMs.
