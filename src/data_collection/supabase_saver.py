"""
Supabase Data Saver for Multi-Source Collected Data.
Saves cocoa market data to Supabase database.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from supabase import create_client, Client

logger = logging.getLogger(__name__)


class SupabaseSaver:
    """
    Saves collected market data to Supabase database.
    """
    
    def __init__(self):
        """Initialize Supabase client."""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            logger.error("❌ Supabase credentials not configured")
            self.client = None
            return
        
        try:
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("✅ Supabase client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {str(e)}")
            self.client = None
    
    def save_price_data(self, data: Dict[str, Any]) -> bool:
        """
        Save cocoa price data to database.
        
        Args:
            data: Price data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            table_data = {
                "collected_at": data.get("timestamp"),
                "source": data.get("source"),
                "symbol": data.get("symbol"),
                "price": data.get("price"),
                "open": data.get("open"),
                "high": data.get("high"),
                "low": data.get("low"),
                "volume": data.get("volume"),
                "date": data.get("date")
            }
            
            self.client.table("cocoa_prices").insert(table_data).execute()
            logger.info(f"✅ Saved price data: {data['source']} - ${data['price']:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save price data: {str(e)}")
            return False
    
    def save_weather_data(self, data: Dict[str, Any]) -> bool:
        """
        Save weather data to database.
        
        Args:
            data: Weather data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            timestamp = data.get("timestamp")
            
            for location in data.get("locations", []):
                table_data = {
                    "collected_at": timestamp,
                    "location": location.get("location"),
                    "name": location.get("name"),
                    "country": location.get("country"),
                    "temperature_c": location.get("temperature_c"),
                    "condition": location.get("condition"),
                    "humidity": location.get("humidity"),
                    "precipitation_mm": location.get("precipitation_mm"),
                    "wind_kph": location.get("wind_kph"),
                    "cloud": location.get("cloud")
                }
                
                self.client.table("weather_data").insert(table_data).execute()
            
            logger.info(f"✅ Saved weather data for {len(data['locations'])} locations")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save weather data: {str(e)}")
            return False
    
    def save_news_data(self, data: Dict[str, Any]) -> bool:
        """
        Save news articles with sentiment to database.
        
        Args:
            data: News data dictionary with sentiment analysis
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            timestamp = data.get("timestamp")
            
            for article in data.get("articles", []):
                sentiment = article.get("sentiment", {})
                
                table_data = {
                    "collected_at": timestamp,
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "source": article.get("source"),
                    "published_at": article.get("published_at"),
                    "url": article.get("url"),
                    "sentiment_label": sentiment.get("label"),
                    "sentiment_score": sentiment.get("score"),
                    "sentiment_value": sentiment.get("sentiment_value")
                }
                
                self.client.table("news_articles").insert(table_data).execute()
            
            logger.info(f"✅ Saved {len(data['articles'])} news articles")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save news data: {str(e)}")
            return False
    
    def save_fx_rates(self, data: Dict[str, Any]) -> bool:
        """
        Save foreign exchange rates to database.
        
        Args:
            data: FX rates data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            table_data = {
                "collected_at": data.get("timestamp"),
                "base": data.get("base"),
                "eur": data.get("rates", {}).get("EUR"),
                "gbp": data.get("rates", {}).get("GBP"),
                "xof": data.get("rates", {}).get("XOF"),
                "ghs": data.get("rates", {}).get("GHS")
            }
            
            self.client.table("fx_rates").insert(table_data).execute()
            logger.info(f"✅ Saved FX rates")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save FX rates: {str(e)}")
            return False
    
    def save_market_sentiment(self, sentiment_data: Dict[str, Any]) -> bool:
        """
        Save aggregated market sentiment to database.
        
        Args:
            sentiment_data: Aggregated sentiment metrics
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            table_data = {
                "collected_at": datetime.now().isoformat(),
                "total_articles": sentiment_data.get("total_articles"),
                "analyzed_articles": sentiment_data.get("analyzed_articles"),
                "positive_count": sentiment_data.get("positive_count"),
                "negative_count": sentiment_data.get("negative_count"),
                "positive_ratio": sentiment_data.get("positive_ratio"),
                "negative_ratio": sentiment_data.get("negative_ratio"),
                "average_confidence": sentiment_data.get("average_confidence"),
                "average_sentiment": sentiment_data.get("average_sentiment"),
                "market_sentiment_score": sentiment_data.get("market_sentiment_score"),
                "market_sentiment_label": sentiment_data.get("market_sentiment_label")
            }
            
            self.client.table("market_sentiment").insert(table_data).execute()
            logger.info(f"✅ Saved market sentiment: {table_data['market_sentiment_label']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save market sentiment: {str(e)}")
            return False
    
    def save_all_data(self, collected_data: Dict[str, Any], sentiment_data: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """
        Save all collected data to Supabase.
        
        Args:
            collected_data: Dictionary with all collected data
            sentiment_data: Optional sentiment analysis results
            
        Returns:
            Dictionary with save status for each data type
        """
        results = {}
        
        # Save price data
        if collected_data.get("yahoo_finance"):
            results["yahoo_finance"] = self.save_price_data(collected_data["yahoo_finance"])
        
        if collected_data.get("investing_com"):
            results["investing_com"] = self.save_price_data(collected_data["investing_com"])
        
        # Save weather data
        if collected_data.get("weather"):
            results["weather"] = self.save_weather_data(collected_data["weather"])
        
        # Save news data
        if collected_data.get("news"):
            results["news"] = self.save_news_data(collected_data["news"])
        
        # Save FX rates
        if collected_data.get("fx_rates"):
            results["fx_rates"] = self.save_fx_rates(collected_data["fx_rates"])
        
        # Save market sentiment
        if sentiment_data:
            results["market_sentiment"] = self.save_market_sentiment(sentiment_data)
        
        successful = sum(1 for v in results.values() if v)
        total = len(results)
        
        logger.info(f"📊 Saved {successful}/{total} data types to Supabase")
        
        return results


if __name__ == "__main__":
    # Test the saver
    from dotenv import load_dotenv
    load_dotenv()
    
    saver = SupabaseSaver()
    
    # Test data
    test_price_data = {
        "source": "test",
        "symbol": "CC=F",
        "timestamp": datetime.now().isoformat(),
        "price": 4350.00,
        "open": 4300.00,
        "high": 4400.00,
        "low": 4250.00,
        "volume": 25000,
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    
    print("Testing Supabase saver...")
    result = saver.save_price_data(test_price_data)
    print(f"Save result: {'✅ Success' if result else '❌ Failed'}")
