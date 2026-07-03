-- SQL Script to create all tables for data collection system
-- This script DROPS existing tables first to ensure clean setup
-- Run this in your Supabase SQL Editor

-- ============================================================================
-- STEP 1: DROP EXISTING TABLES (if they exist)
-- ============================================================================

DROP TABLE IF EXISTS public.cocoa_prices CASCADE;
DROP TABLE IF EXISTS public.weather_data CASCADE;
DROP TABLE IF EXISTS public.news_articles CASCADE;
DROP TABLE IF EXISTS public.market_sentiment CASCADE;
DROP TABLE IF EXISTS public.fx_rates CASCADE;

-- ============================================================================
-- STEP 2: CREATE NEW TABLES WITH CORRECT SCHEMA
-- ============================================================================

-- 1. Cocoa Prices Table
CREATE TABLE public.cocoa_prices (
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
CREATE INDEX idx_cocoa_prices_collected_at ON public.cocoa_prices(collected_at DESC);
CREATE INDEX idx_cocoa_prices_source ON public.cocoa_prices(source);
CREATE INDEX idx_cocoa_prices_date ON public.cocoa_prices(date DESC);

COMMENT ON TABLE public.cocoa_prices IS 'Cocoa price data from Yahoo Finance and Investing.com';

-- 2. Weather Data Table
CREATE TABLE public.weather_data (
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
CREATE INDEX idx_weather_data_collected_at ON public.weather_data(collected_at DESC);
CREATE INDEX idx_weather_data_location ON public.weather_data(location);
CREATE INDEX idx_weather_data_country ON public.weather_data(country);

COMMENT ON TABLE public.weather_data IS 'Weather data from major cocoa producing regions';

-- 3. News Articles Table
CREATE TABLE public.news_articles (
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
CREATE INDEX idx_news_articles_collected_at ON public.news_articles(collected_at DESC);
CREATE INDEX idx_news_articles_published_at ON public.news_articles(published_at DESC);
CREATE INDEX idx_news_articles_sentiment_label ON public.news_articles(sentiment_label);

COMMENT ON TABLE public.news_articles IS 'News articles with sentiment analysis from NewsAPI';

-- 4. Market Sentiment Table
CREATE TABLE public.market_sentiment (
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
CREATE INDEX idx_market_sentiment_collected_at ON public.market_sentiment(collected_at DESC);
CREATE INDEX idx_market_sentiment_label ON public.market_sentiment(market_sentiment_label);

COMMENT ON TABLE public.market_sentiment IS 'Aggregated market sentiment from news analysis';

-- 5. FX Rates Table
CREATE TABLE public.fx_rates (
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
CREATE INDEX idx_fx_rates_collected_at ON public.fx_rates(collected_at DESC);

COMMENT ON TABLE public.fx_rates IS 'Foreign exchange rates for cocoa trading currencies';

-- ============================================================================
-- STEP 3: ENABLE ROW LEVEL SECURITY (RLS)
-- ============================================================================

ALTER TABLE public.cocoa_prices ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.weather_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.news_articles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.market_sentiment ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fx_rates ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- STEP 4: CREATE RLS POLICIES
-- ============================================================================

-- Cocoa Prices Policies
CREATE POLICY "Allow public read access to cocoa_prices" 
ON public.cocoa_prices FOR SELECT 
USING (true);

CREATE POLICY "Allow authenticated insert to cocoa_prices" 
ON public.cocoa_prices FOR INSERT 
WITH CHECK (auth.role() = 'authenticated' OR auth.role() = 'service_role');

-- Weather Data Policies
CREATE POLICY "Allow public read access to weather_data" 
ON public.weather_data FOR SELECT 
USING (true);

CREATE POLICY "Allow authenticated insert to weather_data" 
ON public.weather_data FOR INSERT 
WITH CHECK (auth.role() = 'authenticated' OR auth.role() = 'service_role');

-- News Articles Policies
CREATE POLICY "Allow public read access to news_articles" 
ON public.news_articles FOR SELECT 
USING (true);

CREATE POLICY "Allow authenticated insert to news_articles" 
ON public.news_articles FOR INSERT 
WITH CHECK (auth.role() = 'authenticated' OR auth.role() = 'service_role');

-- Market Sentiment Policies
CREATE POLICY "Allow public read access to market_sentiment" 
ON public.market_sentiment FOR SELECT 
USING (true);

CREATE POLICY "Allow authenticated insert to market_sentiment" 
ON public.market_sentiment FOR INSERT 
WITH CHECK (auth.role() = 'authenticated' OR auth.role() = 'service_role');

-- FX Rates Policies
CREATE POLICY "Allow public read access to fx_rates" 
ON public.fx_rates FOR SELECT 
USING (true);

CREATE POLICY "Allow authenticated insert to fx_rates" 
ON public.fx_rates FOR INSERT 
WITH CHECK (auth.role() = 'authenticated' OR auth.role() = 'service_role');

-- ============================================================================
-- STEP 5: GRANT PERMISSIONS
-- ============================================================================

GRANT SELECT, INSERT ON public.cocoa_prices TO authenticated, anon, service_role;
GRANT SELECT, INSERT ON public.weather_data TO authenticated, anon, service_role;
GRANT SELECT, INSERT ON public.news_articles TO authenticated, anon, service_role;
GRANT SELECT, INSERT ON public.market_sentiment TO authenticated, anon, service_role;
GRANT SELECT, INSERT ON public.fx_rates TO authenticated, anon, service_role;

-- Grant sequence permissions
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated, anon, service_role;

-- ============================================================================
-- STEP 6: VERIFY TABLES CREATED
-- ============================================================================

DO $$
DECLARE
    table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name IN ('cocoa_prices', 'weather_data', 'news_articles', 'market_sentiment', 'fx_rates');
    
    IF table_count = 5 THEN
        RAISE NOTICE '✅ SUCCESS: All 5 tables created successfully!';
        RAISE NOTICE '   - cocoa_prices';
        RAISE NOTICE '   - weather_data';
        RAISE NOTICE '   - news_articles';
        RAISE NOTICE '   - market_sentiment';
        RAISE NOTICE '   - fx_rates';
    ELSE
        RAISE EXCEPTION '❌ ERROR: Only % tables created. Expected 5.', table_count;
    END IF;
END $$;

-- Success message
SELECT 
    '✅ All data collection tables created successfully!' AS status,
    'Ready to collect and store cocoa market data' AS message;
