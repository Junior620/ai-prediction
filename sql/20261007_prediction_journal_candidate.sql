-- Prix candidat, distinct du prix publié.
-- Le prix publié est la clôture tant que l'horizon n'est pas validé.
-- A appliquer sur Supabase. Ce fichier n'est pas exécuté par le dépôt.

alter table public.predictions
    add column if not exists candidate_price double precision;
