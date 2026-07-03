-- Update existing news_articles table to add missing columns

-- Add missing columns if they don't exist
DO $$ 
BEGIN
    -- Add is_high_risk column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'news_articles' AND column_name = 'is_high_risk'
    ) THEN
        ALTER TABLE news_articles ADD COLUMN is_high_risk BOOLEAN;
    END IF;

    -- Add keywords column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'news_articles' AND column_name = 'keywords'
    ) THEN
        ALTER TABLE news_articles ADD COLUMN keywords TEXT[];
    END IF;

    -- Add sentiment_score column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'news_articles' AND column_name = 'sentiment_score'
    ) THEN
        ALTER TABLE news_articles ADD COLUMN sentiment_score DECIMAL(3, 2);
    END IF;

    -- Add created_at column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'news_articles' AND column_name = 'created_at'
    ) THEN
        ALTER TABLE news_articles ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
    END IF;
END $$;

-- Create indexes if they don't exist
CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_news_sentiment ON news_articles(sentiment_score);
CREATE INDEX IF NOT EXISTS idx_news_high_risk ON news_articles(is_high_risk);

-- Enable RLS if not already enabled
ALTER TABLE news_articles ENABLE ROW LEVEL SECURITY;

-- Drop and recreate policy
DROP POLICY IF EXISTS "Allow all for service role" ON news_articles;

CREATE POLICY "Allow all for service role" ON news_articles
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- Grant permissions
GRANT ALL ON news_articles TO postgres;
GRANT ALL ON news_articles TO service_role;
GRANT SELECT ON news_articles TO anon;
GRANT SELECT ON news_articles TO authenticated;
