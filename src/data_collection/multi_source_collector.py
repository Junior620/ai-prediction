"""
Multi-source data collector for cocoa price prediction.
Collects data from multiple sources in parallel: Yahoo Finance, Investing.com, NewsAPI, Weather, FX Rates.
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiSourceCollector:
    """
    Collects cocoa market data from multiple sources in parallel.
    """
    
    def __init__(self):
        """Initialize the multi-source collector with environment variables."""
        # Yahoo Finance
        self.yahoo_symbol = os.getenv("YAHOO_FINANCE_SYMBOL", "CC=F")
        self.yahoo_enabled = os.getenv("YAHOO_FINANCE_ENABLED", "true").lower() == "true"
        
        # Investing.com
        self.investing_url = os.getenv("INVESTING_COM_COCOA_URL", "https://www.investing.com/commodities/us-cocoa")
        self.investing_symbol = os.getenv("INVESTING_COM_COCOA_SYMBOL", "CCN6")
        
        # NewsAPI
        self.newsapi_url = os.getenv("NEWSAPI_URL", "https://newsapi.org/v2")
        self.newsapi_key = os.getenv("NEWSAPI_KEY", "")
        self.newsapi_enabled = os.getenv("NEWSAPI_ENABLED", "true").lower() == "true"
        
        # Weather API
        self.weather_url = os.getenv("WEATHER_API_URL", "https://api.weatherapi.com/v1")
        self.weather_key = os.getenv("WEATHER_API_KEY", "")
        
        # FX Rates API
        self.fx_url = os.getenv("FX_RATES_API_URL", "")
        self.fx_key = os.getenv("FX_RATES_API_KEY", "")
        
        logger.info("MultiSourceCollector initialized")
    
    def collect_yahoo_finance(self) -> Optional[Dict[str, Any]]:
        """
        Collect cocoa price data from Yahoo Finance.
        
        Returns:
            Dictionary with price data or None if failed
        """
        if not self.yahoo_enabled:
            logger.info("Yahoo Finance is disabled")
            return None
        
        try:
            logger.info(f"Collecting data from Yahoo Finance ({self.yahoo_symbol})...")
            
            ticker = yf.Ticker(self.yahoo_symbol)
            hist = ticker.history(period="1d")
            
            if hist.empty:
                logger.warning("No data returned from Yahoo Finance")
                return None
            
            latest = hist.iloc[-1]
            
            data = {
                "source": "yahoo_finance",
                "symbol": self.yahoo_symbol,
                "timestamp": datetime.now().isoformat(),
                "price": float(latest['Close']),
                "open": float(latest['Open']),
                "high": float(latest['High']),
                "low": float(latest['Low']),
                "volume": int(latest['Volume']),
                "date": hist.index[-1].strftime("%Y-%m-%d")
            }
            
            logger.info(f"✅ Yahoo Finance: ${data['price']:.2f}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Yahoo Finance failed: {str(e)}")
            return None
    
    def collect_investing_com(self) -> Optional[Dict[str, Any]]:
        """
        Collect cocoa price data from Investing.com via web scraping.
        
        Returns:
            Dictionary with price data or None if failed
        """
        driver = None
        try:
            logger.info(f"Collecting data from Investing.com ({self.investing_url})...")
            
            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            
            driver.get(self.investing_url)
            wait = WebDriverWait(driver, 10)
            
            # Try multiple selectors
            price = None
            selectors = [
                (By.CSS_SELECTOR, "[data-test='instrument-price-last']"),
                (By.CLASS_NAME, "instrument-price_last__KQzyA"),
                (By.CSS_SELECTOR, ".text-5xl"),
            ]
            
            for by, selector in selectors:
                try:
                    element = wait.until(EC.presence_of_element_located((by, selector)))
                    price_text = element.text
                    if price_text:
                        # Remove commas and convert to float
                        price = float(price_text.replace(',', ''))
                        break
                except:
                    continue
            
            driver.quit()
            
            if price:
                data = {
                    "source": "investing_com",
                    "symbol": self.investing_symbol,
                    "timestamp": datetime.now().isoformat(),
                    "price": price,
                    "date": datetime.now().strftime("%Y-%m-%d")
                }
                
                logger.info(f"✅ Investing.com: ${data['price']:.2f}")
                return data
            else:
                logger.warning("❌ Investing.com: Price not found")
                return None
                
        except Exception as e:
            logger.error(f"❌ Investing.com failed: {str(e)}")
            if driver:
                driver.quit()
            return None
    
    def collect_news_sentiment(self, query: str = "cocoa", days: int = 7) -> Optional[Dict[str, Any]]:
        """
        Collect news articles and sentiment from NewsAPI.
        
        Args:
            query: Search query for news articles
            days: Number of days to look back
            
        Returns:
            Dictionary with news data or None if failed
        """
        if not self.newsapi_enabled or not self.newsapi_key:
            logger.info("NewsAPI is disabled or no API key")
            return None
        
        try:
            logger.info(f"Collecting news from NewsAPI (query: {query})...")
            
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            
            url = f"{self.newsapi_url}/everything"
            params = {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "from": from_date,
                "to": to_date,
                "pageSize": 10,
                "apiKey": self.newsapi_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "ok":
                    articles = data.get("articles", [])
                    total = data.get("totalResults", 0)
                    
                    news_data = {
                        "source": "newsapi",
                        "timestamp": datetime.now().isoformat(),
                        "query": query,
                        "total_results": total,
                        "articles_count": len(articles),
                        "articles": [
                            {
                                "title": article.get("title"),
                                "description": article.get("description"),
                                "source": article.get("source", {}).get("name"),
                                "published_at": article.get("publishedAt"),
                                "url": article.get("url")
                            }
                            for article in articles
                        ]
                    }
                    
                    logger.info(f"✅ NewsAPI: {total} articles found")
                    return news_data
                else:
                    logger.warning(f"❌ NewsAPI error: {data.get('message')}")
                    return None
            else:
                logger.warning(f"❌ NewsAPI HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ NewsAPI failed: {str(e)}")
            return None
    
    def collect_weather_data(self, locations: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Collect weather data for multiple cocoa-producing regions and trading centers.
        
        Args:
            locations: List of locations to get weather for. If None, uses default locations.
            
        Returns:
            Dictionary with weather data for all locations or None if failed
        """
        if not self.weather_key:
            logger.info("Weather API key not configured")
            return None
        
        # Default locations: Major cocoa producers + Trading centers
        if locations is None:
            locations = [
                # AFRIQUE (70% production mondiale)
                # Côte d'Ivoire (40% mondial - 1er producteur)
                "Abidjan,Ivory Coast",
                "Yamoussoukro,Ivory Coast",
                
                # Ghana (20% mondial - 2ème producteur)
                "Accra,Ghana",
                "Kumasi,Ghana",
                
                # Cameroun (5% mondial - 5ème producteur)
                "Douala,Cameroon",
                "Yaounde,Cameroon",
                
                # Nigeria (4ème producteur)
                "Lagos,Nigeria",
                "Ibadan,Nigeria",
                
                # AMÉRIQUE
                # Équateur (3ème producteur)
                "Guayaquil,Ecuador",
                
                # Brésil
                "Salvador,Brazil",  # Bahia region
                
                # CENTRES DE TRADING
                "New York,USA",     # ICE NY Trading Center
                "London,UK"         # ICE London Trading Center
            ]
        
        try:
            logger.info(f"Collecting weather data for {len(locations)} locations...")
            
            weather_data = {
                "source": "weatherapi",
                "timestamp": datetime.now().isoformat(),
                "locations": []
            }
            
            for location in locations:
                try:
                    url = f"{self.weather_url}/current.json"
                    params = {
                        "key": self.weather_key,
                        "q": location,
                        "aqi": "no"
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        location_data = {
                            "location": location,
                            "name": data["location"]["name"],
                            "country": data["location"]["country"],
                            "temperature_c": data["current"]["temp_c"],
                            "condition": data["current"]["condition"]["text"],
                            "humidity": data["current"]["humidity"],
                            "precipitation_mm": data["current"]["precip_mm"],
                            "wind_kph": data["current"]["wind_kph"],
                            "cloud": data["current"]["cloud"]
                        }
                        
                        weather_data["locations"].append(location_data)
                        logger.info(f"✅ {location}: {location_data['temperature_c']}°C, {location_data['condition']}")
                    else:
                        logger.warning(f"❌ {location}: HTTP {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"❌ {location} failed: {str(e)}")
                    continue
            
            if weather_data["locations"]:
                logger.info(f"✅ Weather: Collected data for {len(weather_data['locations'])}/{len(locations)} locations")
                return weather_data
            else:
                logger.warning("❌ No weather data collected")
                return None
                
        except Exception as e:
            logger.error(f"❌ Weather API failed: {str(e)}")
            return None
    
    def collect_fx_rates(self) -> Optional[Dict[str, Any]]:
        """
        Collect foreign exchange rates (USD to other currencies).
        
        Returns:
            Dictionary with FX rates or None if failed
        """
        if not self.fx_url:
            logger.info("FX Rates API not configured")
            return None
        
        try:
            logger.info("Collecting FX rates...")
            
            response = requests.get(self.fx_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("result") == "success":
                    rates = data.get("conversion_rates", {})
                    
                    fx_data = {
                        "source": "exchangerate_api",
                        "timestamp": datetime.now().isoformat(),
                        "base": "USD",
                        "rates": {
                            "EUR": rates.get("EUR"),
                            "GBP": rates.get("GBP"),
                            "XOF": rates.get("XOF"),  # West African CFA franc (Côte d'Ivoire)
                            "GHS": rates.get("GHS"),  # Ghanaian cedi
                        }
                    }
                    
                    logger.info(f"✅ FX Rates: EUR={fx_data['rates']['EUR']}, GBP={fx_data['rates']['GBP']}")
                    return fx_data
                else:
                    logger.warning("❌ FX Rates API error")
                    return None
            else:
                logger.warning(f"❌ FX Rates HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ FX Rates failed: {str(e)}")
            return None
    
    def collect_all_parallel(self) -> Dict[str, Any]:
        """
        Collect data from all sources in parallel using ThreadPoolExecutor.
        
        Returns:
            Dictionary with all collected data
        """
        logger.info("=" * 80)
        logger.info("Starting parallel data collection from all sources...")
        logger.info("=" * 80)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "yahoo_finance": None,
            "investing_com": None,
            "news": None,
            "weather": None,
            "fx_rates": None
        }
        
        # Define tasks
        tasks = {
            "yahoo_finance": self.collect_yahoo_finance,
            "investing_com": self.collect_investing_com,
            "news": lambda: self.collect_news_sentiment("cocoa", days=7),
            "weather": lambda: self.collect_weather_data(),  # Now collects multiple locations
            "fx_rates": self.collect_fx_rates
        }
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_source = {executor.submit(func): source for source, func in tasks.items()}
            
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    result = future.result()
                    results[source] = result
                except Exception as e:
                    logger.error(f"❌ {source} failed with exception: {str(e)}")
                    results[source] = None
        
        logger.info("=" * 80)
        logger.info("Data collection completed")
        logger.info("=" * 80)
        
        # Summary
        successful = sum(1 for v in results.values() if v is not None and v != results["timestamp"])
        total = len(tasks)
        logger.info(f"Summary: {successful}/{total} sources successful")
        
        return results
    
    def collect_historical_data(self, days: int = 30) -> Dict[str, Any]:
        """
        Collect historical data from Yahoo Finance.
        
        Args:
            days: Number of days of historical data to collect
            
        Returns:
            Dictionary with historical data
        """
        try:
            logger.info(f"Collecting {days} days of historical data from Yahoo Finance...")
            
            ticker = yf.Ticker(self.yahoo_symbol)
            hist = ticker.history(period=f"{days}d")
            
            if hist.empty:
                logger.warning("No historical data returned")
                return {"error": "No data"}
            
            historical_data = {
                "source": "yahoo_finance",
                "symbol": self.yahoo_symbol,
                "timestamp": datetime.now().isoformat(),
                "days": days,
                "data_points": len(hist),
                "data": [
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "open": float(row['Open']),
                        "high": float(row['High']),
                        "low": float(row['Low']),
                        "close": float(row['Close']),
                        "volume": int(row['Volume'])
                    }
                    for date, row in hist.iterrows()
                ]
            }
            
            logger.info(f"✅ Collected {len(historical_data['data'])} historical data points")
            return historical_data
            
        except Exception as e:
            logger.error(f"❌ Historical data collection failed: {str(e)}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Test the collector
    from dotenv import load_dotenv
    load_dotenv()
    
    collector = MultiSourceCollector()
    
    # Test parallel collection
    print("\n" + "=" * 80)
    print("TESTING PARALLEL DATA COLLECTION")
    print("=" * 80 + "\n")
    
    results = collector.collect_all_parallel()
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for source, data in results.items():
        if source == "timestamp":
            continue
        status = "✅" if data else "❌"
        print(f"{status} {source}: {'Success' if data else 'Failed'}")
    
    # Test historical data collection
    print("\n" + "=" * 80)
    print("TESTING HISTORICAL DATA COLLECTION")
    print("=" * 80 + "\n")
    
    historical = collector.collect_historical_data(days=7)
    if "error" not in historical:
        print(f"✅ Collected {historical['data_points']} historical data points")
    else:
        print(f"❌ Failed: {historical['error']}")
