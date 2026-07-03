-- Add content column to news_articles table

DO $$ 
BEGIN
    -- Add content column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'news_articles' AND column_name = 'content'
    ) THEN
        ALTER TABLE news_articles ADD COLUMN content TEXT NOT NULL DEFAULT '';
    END IF;
END $$;

-- Verify the table structure
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'news_articles'
ORDER BY ordinal_position;
