-- Événements cacao versionnés. N'altère pas news_articles.
-- À appliquer sur Supabase. Le dépôt ne l'exécute pas.

CREATE TABLE IF NOT EXISTS public.cocoa_news_events (
    id BIGSERIAL PRIMARY KEY,
    event_key TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.cocoa_news_event_versions (
    id BIGSERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    event_key TEXT NOT NULL,
    source TEXT NOT NULL,
    source_weight NUMERIC,
    title TEXT NOT NULL,
    excerpt TEXT,
    category TEXT NOT NULL,
    country TEXT,
    zone TEXT,
    severity NUMERIC,
    reliability NUMERIC,
    novelty NUMERIC,
    duration_days INTEGER,
    sentiment_score NUMERIC,
    economic_direction TEXT NOT NULL,
    surprise TEXT,
    published_at TIMESTAMPTZ,
    published_at_verified BOOLEAN NOT NULL DEFAULT FALSE,
    observed_at TIMESTAMPTZ,
    observed_source TEXT,
    first_seen_at TIMESTAMPTZ,
    available_at TIMESTAMPTZ,
    available_source TEXT,
    content_hash TEXT NOT NULL,
    rules_version TEXT NOT NULL,
    classified_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (url, rules_version, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_cocoa_news_versions_available
    ON public.cocoa_news_event_versions (available_at);

COMMENT ON TABLE public.cocoa_news_event_versions IS
    'Versions append-only. Une correction ajoute une ligne et ne réécrit pas le passé.';
