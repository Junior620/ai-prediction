-- Cocoa Price Prediction System - Supabase Database Schema
-- This script initializes the database schema for the hybrid prediction system

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Historical price data
CREATE TABLE IF NOT EXISTS price_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    market VARCHAR(50) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    volume DECIMAL(15, 2),
    currency VARCHAR(3) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(timestamp, market)
);

CREATE INDEX IF NOT EXISTS idx_price_timestamp ON price_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_price_market ON price_data(market);

-- Econometric data
CREATE TABLE IF NOT EXISTS econometric_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL UNIQUE,
    temperature DECIMAL(5, 2),
    rainfall DECIMAL(6, 2),
    stock_level DECIMAL(15, 2),
    production DECIMAL(15, 2),
    fx_rate_xaf_usd DECIMAL(10, 6),
    fx_rate_gbp_usd DECIMAL(10, 6),
    fx_rate_eur_usd DECIMAL(10, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_econ_timestamp ON econometric_data(timestamp);

-- News articles
CREATE TABLE IF NOT EXISTS news_articles (
    id VARCHAR(255) PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    published_at TIMESTAMP NOT NULL,
    url TEXT NOT NULL,
    keywords TEXT[],
    sentiment_score DECIMAL(3, 2),
    is_high_risk BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_news_sentiment ON news_articles(sentiment_score);
CREATE INDEX IF NOT EXISTS idx_news_high_risk ON news_articles(is_high_risk);

-- Predictions log
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    horizon INT NOT NULL,
    predicted_price DECIMAL(10, 2) NOT NULL,
    lower_bound DECIMAL(10, 2) NOT NULL,
    upper_bound DECIMAL(10, 2) NOT NULL,
    confidence_level DECIMAL(3, 2) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    baseline_component DECIMAL(10, 2),
    residual_component DECIMAL(10, 2),
    sentiment_component DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_pred_horizon ON predictions(horizon);
CREATE INDEX IF NOT EXISTS idx_pred_model_version ON predictions(model_version);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    rmse DECIMAL(10, 4) NOT NULL,
    mae DECIMAL(10, 4) NOT NULL,
    mape DECIMAL(6, 4) NOT NULL,
    directional_accuracy DECIMAL(5, 4) NOT NULL,
    coverage_rate DECIMAL(5, 4) NOT NULL,
    mean_interval_width DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_metrics_version ON model_metrics(model_version);
CREATE INDEX IF NOT EXISTS idx_metrics_created ON model_metrics(created_at);

-- Validation errors log
CREATE TABLE IF NOT EXISTS validation_errors (
    id SERIAL PRIMARY KEY,
    field VARCHAR(100) NOT NULL,
    value TEXT,
    error_type VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_errors_severity ON validation_errors(severity);
CREATE INDEX IF NOT EXISTS idx_errors_created ON validation_errors(created_at);

-- Comments for documentation
COMMENT ON TABLE price_data IS 'Historical cocoa price data from ICE London and ICE New York markets';
COMMENT ON TABLE econometric_data IS 'Econometric variables including weather, stocks, production, and FX rates';
COMMENT ON TABLE news_articles IS 'News articles with sentiment analysis for market shock detection';
COMMENT ON TABLE predictions IS 'Log of all predictions made by the hybrid model';
COMMENT ON TABLE model_metrics IS 'Performance metrics tracking for model monitoring';
COMMENT ON TABLE validation_errors IS 'Data validation errors and warnings';

-- Row Level Security (RLS) Policies
-- Enable RLS on all tables
ALTER TABLE price_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE econometric_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE news_articles ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE validation_errors ENABLE ROW LEVEL SECURITY;

-- Create policies for authenticated users
-- Note: Adjust these policies based on your specific security requirements

-- Price data: Read access for authenticated users
CREATE POLICY IF NOT EXISTS "Allow read access to price_data" ON price_data
    FOR SELECT
    USING (auth.role() = 'authenticated');

-- Econometric data: Read access for authenticated users
CREATE POLICY IF NOT EXISTS "Allow read access to econometric_data" ON econometric_data
    FOR SELECT
    USING (auth.role() = 'authenticated');

-- News articles: Read access for authenticated users
CREATE POLICY IF NOT EXISTS "Allow read access to news_articles" ON news_articles
    FOR SELECT
    USING (auth.role() = 'authenticated');

-- Predictions: Read access for authenticated users
CREATE POLICY IF NOT EXISTS "Allow read access to predictions" ON predictions
    FOR SELECT
    USING (auth.role() = 'authenticated');

-- Model metrics: Read access for authenticated users
CREATE POLICY IF NOT EXISTS "Allow read access to model_metrics" ON model_metrics
    FOR SELECT
    USING (auth.role() = 'authenticated');

-- Validation errors: Read access for authenticated users
CREATE POLICY IF NOT EXISTS "Allow read access to validation_errors" ON validation_errors
    FOR SELECT
    USING (auth.role() = 'authenticated');

-- Insert policies for service role (backend application)
-- These will be used by the application with service role key

CREATE POLICY IF NOT EXISTS "Allow insert to price_data for service role" ON price_data
    FOR INSERT
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY IF NOT EXISTS "Allow insert to econometric_data for service role" ON econometric_data
    FOR INSERT
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY IF NOT EXISTS "Allow insert to news_articles for service role" ON news_articles
    FOR INSERT
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY IF NOT EXISTS "Allow insert to predictions for service role" ON predictions
    FOR INSERT
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY IF NOT EXISTS "Allow insert to model_metrics for service role" ON model_metrics
    FOR INSERT
    WITH CHECK (auth.role() = 'service_role');

CREATE POLICY IF NOT EXISTS "Allow insert to validation_errors for service role" ON validation_errors
    FOR INSERT
    WITH CHECK (auth.role() = 'service_role');

-- Futures price data (historical snapshots)
CREATE TABLE IF NOT EXISTS cocoa_futures (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    data JSONB NOT NULL,
    source TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_futures_collected ON cocoa_futures(collected_at);
