-- Historique quotidien NASA POWER pour les ceintures cacao.
-- A appliquer sur Supabase. Le dépôt ne l'exécute pas.

CREATE TABLE IF NOT EXISTS public.cocoa_weather_daily (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    region TEXT NOT NULL,
    country TEXT NOT NULL,
    latitude DECIMAL(8, 4) NOT NULL,
    longitude DECIMAL(8, 4) NOT NULL,
    precip_mm DECIMAL(8, 2),
    temp_c DECIMAL(6, 2),
    temp_min_c DECIMAL(6, 2),
    temp_max_c DECIMAL(6, 2),
    source TEXT NOT NULL DEFAULT 'nasa_power',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, region)
);

CREATE INDEX IF NOT EXISTS idx_cocoa_weather_daily_date
    ON public.cocoa_weather_daily (date DESC);

COMMENT ON TABLE public.cocoa_weather_daily IS
    'Pluie et température quotidiennes Côte d''Ivoire et Ghana. Hors du prix publié.';
