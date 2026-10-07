-- Journal de prévision : marché, séance d'origine, séance cible.
-- A appliquer sur Supabase. Ce fichier n'est pas exécuté par le dépôt.

alter table public.predictions
    add column if not exists market text,
    add column if not exists origin_date date,
    add column if not exists origin_price double precision,
    add column if not exists target_date date,
    add column if not exists feature_failure boolean,
    add column if not exists currency text;

alter table public.model_metrics
    add column if not exists market text,
    add column if not exists horizon integer;

create index if not exists predictions_market_target_idx
    on public.predictions (market, target_date);
