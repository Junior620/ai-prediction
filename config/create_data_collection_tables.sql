-- SQL Script to create all tables for data collection system
-- Run this in your Supabase SQL Editor

-- 1. Cocoa Prices Table
CREATE TABLE IF NOT EXISTS public.cocoa_prices (
    id BIGSERIAL PRIMARY KEY,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source TEXT NOT NULL,
    symbol TEXT NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    volume BIGINT,
    date DATE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_cocoa_prices_collected_at ON public.cocoa_prices(collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_cocoa_prices_source ON public.cocoa_prices(source);
CREATE INDEX IF NOT EXISTS idx_cocoa_prices_date ON public.cocoa_prices(date DESC);

-- 2. Weather Data Table
CREATE TABLE IF NOT EXISTS public.weather_data (
    id BIGSERIAL PRIMARY KEY,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    location TEXT NOT NULL,
    name TEXT NOT NULL,
    country TEXT NOT NULL,
    temperature_c DECIMAL(5, 2) NOT NULL,
    condition TEXT NOT NULL,
    humidity INTEGER,
    precipitation_mm DECIMAL(6, 2),
    wind_kph DECIMAL(6, 2),
    cloud INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_weather_data_collected_at ON public.weather_data(collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_weather_data_location ON public.weather_data(location);
CREATE INDEX IF NOT EXISTS idx_weather_data_country ON public.weather_data(country);

-- 3. News Articles Table
CREATE TABLE IF NOT EXISTS public.news_articles (
    id BIGSERIAL PRIMARY KEY,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    title TEXT NOT NULL,
    description TEXT,
    source TEXT,
    published_at TIMESTAMPTZ,
    url TEXT,
    sentiment_label TEXT,
    sentiment_score DECIMAL(5, 4),
    sentiment_value DECIMAL(5, 4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_news_articles_collected_at ON public.news_articles(collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_articles_published_at ON public.news_articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_articles_sentiment_label ON public.news_articles(sentiment_label);

-- 4. Market Sentiment Table
CREATE TABLE IF NOT EXISTS public.market_sentiment (
    id BIGSERIAL PRIMARY KEY,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    total_articles INTEGER NOT NULL,
    analyzed_articles INTEGER NOT NULL,
    positive_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL,
    positive_ratio DECIMAL(5, 4) NOT NULL,
    negative_ratio DECIMAL(5, 4) NOT NULL,
    average_confidence DECIMAL(5, 4),
    average_sentiment DECIMAL(5, 4),
    market_sentiment_score DECIMAL(5, 4) NOT NULL,
    market_sentiment_label TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_market_sentiment_collected_at ON public.market_sentiment(collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_market_sentiment_label ON public.market_sentiment(market_sentiment_label);

-- 5. FX Rates Table
CREATE TABLE IF NOT EXISTS public.fx_rates (
    id BIGSERIAL PRIMARY KEY,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    base TEXT NOT NULL,
    eur DECIMAL(10, 6),
    gbp DECIMAL(10, 6),
    xof DECIMAL(10, 6),
    ghs DECIMAL(10, 6),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_fx_rates_collected_at ON public.fx_rates(collected_at DESC);

-- Enable Row Level Security (RLS) on all tables
ALTER TABLE public.cocoa_prices ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.weather_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.news_articles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.market_sentiment ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fx_rates ENABLE ROW LEVEL SECURITY;

-- Create policies to allow authenticated users to read and write
CREATE POLICY "Allow authenticated users to read cocoa_prices" ON public.cocoa_prices
    FOR SELECT USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

CREATE POLICY "Allow authenticated users to insert cocoa_prices" ON public.cocoa_prices
    FOR INSERT WITH CHECK (auth.role() = 'authenticated' OR auth.role() = 'service_role');

CREATE POLICY "Allow authenticated users to read weather_data" ON public.weather_data
    FOR SELECT USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

CREATE POLICY "Allow authenticated users to insert weather_data" ON public.weather_data
    FOR INSERT WITH CHECK (auth.role() = 'authenticated' OR auth.role() = 'service_role');

CREATE POLICY "Allow authenticated users to read news_articles" ON public.news_articles
    FOR SELECT USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

CREATE POLICY "Allow authenticated users to insert news_articles" ON public.news_articles
    FOR INSERT WITH CHECK (auth.role() = 'authenticated' OR auth.role() = 'service_role');

CREATE POLICY "Allow authenticated users to read market_sentiment" ON public.market_sentiment
    FOR SELECT USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

CREATE POLICY "Allow authenticated users to insert market_sentiment" ON public.market_sentiment
    FOR INSERT WITH CHECK (auth.role() = 'authenticated' OR auth.role() = 'service_role');

CREATE POLICY "Allow authenticated users to read fx_rates" ON public.fx_rates
    FOR SELECT USING (auth.role() = 'authenticated' OR auth.role() = 'anon');

CREATE POLICY "Allow authenticated users to insert fx_rates" ON public.fx_rates
    FOR INSERT WITH CHECK (auth.role() = 'authenticated' OR auth.role() = 'service_role');

-- Grant permissions
GRANT SELECT, INSERT ON public.cocoa_prices TO authenticated, anon, service_role;
GRANT SELECT, INSERT ON public.weather_data TO authenticated, anon, service_role;
GRANT SELECT, INSERT ON public.news_articles TO authenticated, anon, service_role;
GRANT SELECT, INSERT ON public.market_sentiment TO authenticated, anon, service_role;
GRANT SELECT, INSERT ON public.fx_rates TO authenticated, anon, service_role;

-- Grant sequence permissions
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated, anon, service_role;

-- Success message
SELECT 'All data collection tables created successfully!' AS status;
