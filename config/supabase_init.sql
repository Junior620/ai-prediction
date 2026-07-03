-- Cocoa Price Prediction System - Supabase Database Schema
-- Run this script in your Supabase SQL Editor to initialize the database

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Historical price data table
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

COMMENT ON TABLE price_data IS 'Historical cocoa price data from ICE London and ICE New York markets';

-- Econometric data table
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

COMMENT ON TABLE econometric_data IS 'Econometric data including weather, stocks, production, and exchange rates';

-- News articles table
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
CREATE INDEX IF NOT EXISTS idx_news_source ON news_articles(source);

COMMENT ON TABLE news_articles IS 'News articles from Reuters and Bloomberg with sentiment analysis';

-- Predictions log table
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

COMMENT ON TABLE predictions IS 'Log of all predictions made by the hybrid model';

-- Model performance metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    rmse DECIMAL(10, 4) NOT NULL,
    mae DECIMAL(10, 4) NOT NULL,
    mape DECIMAL(6, 4) NOT NULL,
    directional_accuracy DECIMAL(5, 4) NOT NULL,
    coverage_rate DECIMAL(5, 4) NOT NULL,
    mean_interval_width DECIMAL(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_metrics_version ON model_metrics(model_version);
CREATE INDEX IF NOT EXISTS idx_metrics_created ON model_metrics(created_at);

COMMENT ON TABLE model_metrics IS 'Performance metrics for model versions over time';

-- Validation errors log table
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
CREATE INDEX IF NOT EXISTS idx_errors_type ON validation_errors(error_type);

COMMENT ON TABLE validation_errors IS 'Log of data validation errors';

-- Create a view for recent predictions with actual prices (for performance monitoring)
CREATE OR REPLACE VIEW recent_predictions_with_actuals AS
SELECT 
    p.id,
    p.horizon,
    p.predicted_price,
    p.lower_bound,
    p.upper_bound,
    p.confidence_level,
    p.model_version,
    p.created_at as prediction_time,
    pd.price as actual_price,
    pd.timestamp as actual_time,
    ABS(p.predicted_price - pd.price) as absolute_error,
    ABS(p.predicted_price - pd.price) / pd.price * 100 as percentage_error,
    CASE 
        WHEN pd.price BETWEEN p.lower_bound AND p.upper_bound THEN true 
        ELSE false 
    END as within_confidence_interval
FROM predictions p
LEFT JOIN price_data pd ON 
    DATE(pd.timestamp) = DATE(p.created_at + (p.horizon || ' days')::INTERVAL)
WHERE p.created_at >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY p.created_at DESC;

COMMENT ON VIEW recent_predictions_with_actuals IS 'Recent predictions joined with actual prices for performance evaluation';

-- Create a function to clean old data (optional, for data retention)
CREATE OR REPLACE FUNCTION clean_old_data(retention_days INT DEFAULT 365)
RETURNS TABLE(
    table_name TEXT,
    rows_deleted BIGINT
) AS $$
DECLARE
    cutoff_date TIMESTAMP;
    deleted_count BIGINT;
BEGIN
    cutoff_date := CURRENT_TIMESTAMP - (retention_days || ' days')::INTERVAL;
    
    -- Clean old validation errors
    DELETE FROM validation_errors WHERE created_at < cutoff_date;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    table_name := 'validation_errors';
    rows_deleted := deleted_count;
    RETURN NEXT;
    
    -- Clean old predictions (keep at least 1 year)
    IF retention_days > 365 THEN
        DELETE FROM predictions WHERE created_at < cutoff_date;
        GET DIAGNOSTICS deleted_count = ROW_COUNT;
        table_name := 'predictions';
        rows_deleted := deleted_count;
        RETURN NEXT;
    END IF;
    
    RETURN;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION clean_old_data IS 'Clean old data based on retention policy';

-- Grant necessary permissions (adjust based on your Supabase setup)
-- Note: Supabase handles most permissions automatically, but you can customize here

-- Insert sample configuration data (optional)
-- You can add initial reference data here if needed

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Cocoa Price Prediction database schema initialized successfully!';
    RAISE NOTICE 'Tables created: price_data, econometric_data, news_articles, predictions, model_metrics, validation_errors';
    RAISE NOTICE 'View created: recent_predictions_with_actuals';
    RAISE NOTICE 'Function created: clean_old_data()';
END $$;
