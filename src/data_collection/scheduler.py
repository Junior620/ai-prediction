"""
Automated Data Collection Scheduler.
Schedules periodic data collection from all sources.
"""

import os
import time
import logging
import schedule
from datetime import datetime
from typing import Optional

from src.data_collection.multi_source_collector import MultiSourceCollector
from src.nlp.sentiment_analyzer import SentimentAnalyzer
from src.data_collection.supabase_saver import SupabaseSaver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollectionScheduler:
    """
    Schedules and manages automated data collection.
    """
    
    def __init__(self, interval_minutes: int = 60):
        """
        Initialize the scheduler.
        
        Args:
            interval_minutes: Collection interval in minutes (default: 60 = hourly)
        """
        self.interval_minutes = interval_minutes
        self.collector = MultiSourceCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.saver = SupabaseSaver()
        
        self.collection_count = 0
        self.last_collection_time = None
        
        logger.info(f"📅 Scheduler initialized (interval: {interval_minutes} minutes)")
    
    def collect_and_save(self):
        """
        Collect data from all sources, analyze sentiment, and save to database.
        """
        try:
            logger.info("=" * 80)
            logger.info(f"🚀 Starting data collection #{self.collection_count + 1}")
            logger.info(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 80)
            
            # Step 1: Collect data from all sources
            collected_data = self.collector.collect_all_parallel()
            
            # Step 2: Analyze sentiment if news data available
            sentiment_data = None
            if collected_data.get("news") and collected_data["news"].get("articles"):
                logger.info("🧠 Analyzing news sentiment...")
                articles = collected_data["news"]["articles"]
                sentiment_data = self.sentiment_analyzer.analyze_articles(articles)
                
                # Add sentiment to news data
                if sentiment_data and "articles_with_sentiment" in sentiment_data:
                    collected_data["news"]["articles"] = sentiment_data["articles_with_sentiment"]
            
            # Step 3: Save to Supabase
            logger.info("💾 Saving data to Supabase...")
            save_results = self.saver.save_all_data(collected_data, sentiment_data)
            
            # Update stats
            self.collection_count += 1
            self.last_collection_time = datetime.now()
            
            # Summary
            logger.info("=" * 80)
            logger.info("✅ Data collection completed successfully")
            logger.info(f"📊 Collection #{self.collection_count}")
            logger.info(f"💾 Saved: {sum(1 for v in save_results.values() if v)}/{len(save_results)} data types")
            if sentiment_data:
                logger.info(f"🧠 Market Sentiment: {sentiment_data.get('market_sentiment_label', 'N/A')}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Data collection failed: {str(e)}")
            logger.exception(e)
    
    def start(self):
        """
        Start the scheduler and run indefinitely.
        """
        logger.info("=" * 80)
        logger.info("🚀 STARTING DATA COLLECTION SCHEDULER")
        logger.info("=" * 80)
        logger.info(f"⏰ Collection interval: Every {self.interval_minutes} minutes")
        logger.info(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Run immediately on start
        logger.info("🔄 Running initial collection...")
        self.collect_and_save()
        
        # Schedule periodic collections
        schedule.every(self.interval_minutes).minutes.do(self.collect_and_save)
        
        logger.info(f"✅ Scheduler started. Next collection in {self.interval_minutes} minutes.")
        logger.info("Press Ctrl+C to stop.")
        logger.info("=" * 80)
        
        # Run scheduler loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n" + "=" * 80)
            logger.info("🛑 Scheduler stopped by user")
            logger.info(f"📊 Total collections: {self.collection_count}")
            if self.last_collection_time:
                logger.info(f"⏰ Last collection: {self.last_collection_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 80)
    
    def run_once(self):
        """
        Run data collection once (for testing).
        """
        logger.info("🔄 Running single data collection...")
        self.collect_and_save()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "once":
            # Run once for testing
            scheduler = DataCollectionScheduler()
            scheduler.run_once()
        elif sys.argv[1].isdigit():
            # Custom interval
            interval = int(sys.argv[1])
            scheduler = DataCollectionScheduler(interval_minutes=interval)
            scheduler.start()
        else:
            print("Usage:")
            print("  python scheduler.py          # Run with default 60-minute interval")
            print("  python scheduler.py once     # Run once for testing")
            print("  python scheduler.py 30       # Run with custom interval (30 minutes)")
    else:
        # Default: hourly collection
        scheduler = DataCollectionScheduler(interval_minutes=60)
        scheduler.start()
