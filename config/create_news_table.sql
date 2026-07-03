-- Create news_articles table for sentiment analysis
-- Execute this in Supabase SQL Editor

-- Drop existing table if needed (uncomment if you want to recreate)
-- DROP TABLE IF EXISTS news_articles CASCADE;

-- Create news_articles table
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

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_news_sentiment ON news_articles(sentiment_score);
CREATE INDEX IF NOT EXISTS idx_news_high_risk ON news_articles(is_high_risk);

-- Add comment for documentation
COMMENT ON TABLE news_articles IS 'News articles with sentiment analysis for market shock detection';

-- Enable Row Level Security
ALTER TABLE news_articles ENABLE ROW LEVEL SECURITY;

-- Drop existing policy if it exists
DROP POLICY IF EXISTS "Allow all for service role" ON news_articles;

-- Create policy for service role (bypass RLS)
CREATE POLICY "Allow all for service role" ON news_articles
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- Grant permissions
GRANT ALL ON news_articles TO postgres;
GRANT ALL ON news_articles TO service_role;
GRANT SELECT ON news_articles TO anon;
GRANT SELECT ON news_articles TO authenticated;
