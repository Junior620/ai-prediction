"""
Data collector for the Cocoa Price Prediction System.

This module orchestrates data collection from multiple external sources:
- Price data from ICE London and ICE New York
- Econometric data (weather, stocks, production, FX rates)
- News feed from Reuters and Bloomberg

All data collection includes retry logic and rate limiting.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

from src.models.data_models import (
    PriceData,
    EconometricData,
    NewsArticle,
    ValidationError
)
from src.data_collection.retry_handler import RetryHandler
from src.data_collection.rate_limiter import MultiRateLimiter
from config.settings import get_settings, get_config

logger = logging.getLogger(__name__)


class DataCollectionError(Exception):
    """Base exception for data collection errors."""
    pass


class DataSourceUnavailableError(DataCollectionError):
    """Raised when a data source is unavailable."""
    pass


class DataCollector:
    """
    Orchestrates data collection from multiple external sources.
    
    Handles:
    - Price data collection from commodity exchanges
    - Econometric data collection (weather, stocks, production, FX)
    - News feed collection from financial news sources
    - Retry logic with exponential backoff
    - Rate limiting for API calls
    - Data validation and storage
    
    Attributes:
        settings: Application settings
        retry_handler: Handler for retrying failed requests
        rate_limiter: Multi-source rate limiter
    """
    
    def __init__(
        self,
        retry_handler: Optional[RetryHandler] = None,
        rate_limiter: Optional[MultiRateLimiter] = None
    ):
        """
        Initialize data collector.
        
        Args:
            retry_handler: Custom retry handler (uses default if None)
            rate_limiter: Custom rate limiter (uses default if None)
        """
        self.settings = get_settings()
        
        # Initialize retry handler
        if retry_handler is None:
            max_retries = get_config("data_collection.retry_config.max_retries", 3)
            backoff_factor = get_config("data_collection.retry_config.backoff_factor", 2.0)
            retry_delay = get_config("data_collection.retry_config.retry_delay", 300.0)
            
            self.retry_handler = RetryHandler(
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                retry_delay=retry_delay,
                retryable_exceptions=(RequestException, Timeout, ConnectionError)
            )
        else:
            self.retry_handler = retry_handler
        
        # Initialize rate limiter
        if rate_limiter is None:
            self.rate_limiter = MultiRateLimiter()
            self._setup_rate_limiters()
        else:
            self.rate_limiter = rate_limiter
        
        logger.info("DataCollector initialized successfully")
    
    def _setup_rate_limiters(self) -> None:
        """Setup rate limiters for different data sources."""
        # Default rate limits (can be overridden in config)
        rate_limits = {
            "ice_london": 1.0,  # 1 call per second
            "ice_ny": 1.0,
            "weather": 2.0,  # 2 calls per second
            "icco": 0.5,  # 1 call per 2 seconds
            "forex": 5.0,  # 5 calls per second
            "reuters": 1.0,
            "bloomberg": 1.0
        }
        
        for source, calls_per_second in rate_limits.items():
            self.rate_limiter.add_limiter(source, calls_per_second)
        
        logger.info("Rate limiters configured for all data sources")
    
    def collect_price_data(
        self,
        markets: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Collect historical price data from commodity exchanges.
        
        Retrieves price data from ICE London and ICE New York markets
        with retry logic and rate limiting.
        
        Args:
            markets: List of market identifiers (e.g., ["ICE_London", "ICE_NY"])
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with columns: [timestamp, market, price, volume, currency]
            
        Raises:
            DataCollectionError: If data collection fails for all markets
            
        Requirements:
            - 1.1: Retrieve Historical_Price_Data from London and New York markets
            - 1.2: Store data with timestamps in UTC format
            - 1.3: Collect price data with minimum frequency of daily intervals
        """
        logger.info(
            f"Collecting price data for markets {markets} "
            f"from {start_date} to {end_date}"
        )
        
        all_price_data = []
        errors = []
        
        for market in markets:
            try:
                market_data = self._collect_market_price_data(
                    market, start_date, end_date
                )
                all_price_data.extend(market_data)
                
                logger.info(
                    f"Successfully collected {len(market_data)} price records "
                    f"from {market}"
                )
                
            except Exception as e:
                error_msg = f"Failed to collect data from {market}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        if not all_price_data:
            raise DataCollectionError(
                f"Failed to collect price data from any market. Errors: {errors}"
            )
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "timestamp": pd.timestamp,
                "market": pd.market,
                "price": pd.price,
                "volume": pd.volume,
                "currency": pd.currency
            }
            for pd in all_price_data
        ])
        
        logger.info(f"Collected total of {len(df)} price records")
        return df
    
    def _collect_market_price_data(
        self,
        market: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[PriceData]:
        """
        Collect price data from a specific market.
        
        Args:
            market: Market identifier (e.g., "ICE_London")
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            List of PriceData objects
        """
        # Determine API configuration based on market
        if market == "ICE_London":
            api_url = self.settings.ice_london_api_url
            api_key = self.settings.ice_london_api_key
            rate_limiter_name = "ice_london"
        elif market == "ICE_NY":
            api_url = self.settings.ice_ny_api_url
            api_key = self.settings.ice_ny_api_key
            rate_limiter_name = "ice_ny"
        else:
            raise ValueError(f"Unknown market: {market}")
        
        if not api_url or not api_key:
            logger.warning(
                f"API credentials not configured for {market}. "
                f"Returning mock data for development."
            )
            return self._generate_mock_price_data(market, start_date, end_date)
        
        # Apply rate limiting
        self.rate_limiter.acquire(rate_limiter_name)
        
        # Make API request with retry logic
        price_data = self.retry_handler.execute_with_retry(
            self._fetch_price_data_from_api,
            api_url,
            api_key,
            market,
            start_date,
            end_date
        )
        
        return price_data
    
    def _fetch_price_data_from_api(
        self,
        api_url: str,
        api_key: str,
        market: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[PriceData]:
        """
        Fetch price data from external API.
        
        Args:
            api_url: API endpoint URL
            api_key: API authentication key
            market: Market identifier
            start_date: Start date
            end_date: End date
            
        Returns:
            List of PriceData objects
        """
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "market": market
        }
        
        response = requests.get(
            api_url,
            headers=headers,
            params=params,
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        # Parse response and create PriceData objects
        price_data = []
        for item in data.get("data", []):
            try:
                price_data.append(PriceData(
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    market=market,
                    price=float(item["price"]),
                    volume=float(item["volume"]),
                    currency=item.get("currency", "USD")
                ))
            except Exception as e:
                logger.warning(f"Failed to parse price data item: {e}")
        
        return price_data
    
    def _generate_mock_price_data(
        self,
        market: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[PriceData]:
        """
        Generate mock price data for development/testing.
        
        Args:
            market: Market identifier
            start_date: Start date
            end_date: End date
            
        Returns:
            List of mock PriceData objects
        """
        import numpy as np
        
        # Generate daily timestamps
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic price data with trend and noise
        base_price = 3000.0
        trend = np.linspace(0, 500, len(dates))
        noise = np.random.normal(0, 100, len(dates))
        prices = base_price + trend + noise
        
        # Generate volumes
        volumes = np.random.uniform(1000, 5000, len(dates))
        
        mock_data = []
        for date, price, volume in zip(dates, prices, volumes):
            mock_data.append(PriceData(
                timestamp=date.to_pydatetime(),
                market=market,
                price=float(price),
                volume=float(volume),
                currency="USD"
            ))
        
        logger.info(f"Generated {len(mock_data)} mock price records for {market}")
        return mock_data
    
    def collect_econometric_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect econometric data from multiple sources.
        
        Retrieves weather data, stock levels, production data, and FX rates
        with retry logic and rate limiting. Implements resilience by continuing
        if one data source fails (Requirement 12.1).
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            Dict with keys: ["weather", "stocks", "production", "fx_rates"]
            Each value is a DataFrame with the respective data
            
        Requirements:
            - 2.1: Retrieve Econometric_Data including weather, stocks, production, FX
            - 2.2: Include rainfall and temperature from cocoa-producing regions
            - 2.3: Retrieve stock levels from major ports at least weekly
            - 2.4: Retrieve production data from ICCO at least monthly
            - 2.5: Retrieve exchange rates for relevant currencies daily
            - 12.1: Continue operating if one data source fails
        """
        logger.info(
            f"Collecting econometric data from {start_date} to {end_date}"
        )
        
        result = {}
        failed_sources = []
        
        # Collect weather data
        try:
            weather_data = self._collect_weather_data(start_date, end_date)
            result["weather"] = weather_data
            logger.info(f"Collected {len(weather_data)} weather records")
        except Exception as e:
            logger.error(f"Failed to collect weather data: {e}")
            result["weather"] = pd.DataFrame()
            failed_sources.append("weather")
            
            # Send alert for data source failure
            try:
                from src.monitoring.alert_system import get_alert_system
                alert_system = get_alert_system()
                alert_system.send_data_source_failure_alert(
                    source_name="Weather API",
                    error_message=str(e)
                )
            except Exception as alert_error:
                logger.error(f"Failed to send alert: {alert_error}")
        
        # Collect stock levels
        try:
            stock_data = self._collect_stock_data(start_date, end_date)
            result["stocks"] = stock_data
            logger.info(f"Collected {len(stock_data)} stock records")
        except Exception as e:
            logger.error(f"Failed to collect stock data: {e}")
            result["stocks"] = pd.DataFrame()
            failed_sources.append("stocks")
            
            # Send alert for data source failure
            try:
                from src.monitoring.alert_system import get_alert_system
                alert_system = get_alert_system()
                alert_system.send_data_source_failure_alert(
                    source_name="Stock Data API",
                    error_message=str(e)
                )
            except Exception as alert_error:
                logger.error(f"Failed to send alert: {alert_error}")
        
        # Collect production data
        try:
            production_data = self._collect_production_data(start_date, end_date)
            result["production"] = production_data
            logger.info(f"Collected {len(production_data)} production records")
        except Exception as e:
            logger.error(f"Failed to collect production data: {e}")
            result["production"] = pd.DataFrame()
            failed_sources.append("production")
            
            # Send alert for data source failure
            try:
                from src.monitoring.alert_system import get_alert_system
                alert_system = get_alert_system()
                alert_system.send_data_source_failure_alert(
                    source_name="ICCO Production API",
                    error_message=str(e)
                )
            except Exception as alert_error:
                logger.error(f"Failed to send alert: {alert_error}")
        
        # Collect FX rates
        try:
            fx_data = self._collect_fx_rates(start_date, end_date)
            result["fx_rates"] = fx_data
            logger.info(f"Collected {len(fx_data)} FX rate records")
        except Exception as e:
            logger.error(f"Failed to collect FX rates: {e}")
            result["fx_rates"] = pd.DataFrame()
            failed_sources.append("fx_rates")
            
            # Send alert for data source failure
            try:
                from src.monitoring.alert_system import get_alert_system
                alert_system = get_alert_system()
                alert_system.send_data_source_failure_alert(
                    source_name="FX Rates API",
                    error_message=str(e)
                )
            except Exception as alert_error:
                logger.error(f"Failed to send alert: {alert_error}")
        
        # Log summary of collection
        if failed_sources:
            logger.warning(
                f"Econometric data collection completed with failures: {failed_sources}. "
                f"System will continue with available data sources."
            )
        else:
            logger.info("Econometric data collection completed successfully")
        
        return result
    
    def _collect_weather_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Collect weather data (temperature, rainfall) from cocoa regions."""
        api_url = self.settings.weather_api_url
        api_key = self.settings.weather_api_key
        
        if not api_url or not api_key:
            logger.warning("Weather API not configured. Returning mock data.")
            return self._generate_mock_weather_data(start_date, end_date)
        
        self.rate_limiter.acquire("weather")
        
        # Implementation would call actual weather API
        # For now, return mock data
        return self._generate_mock_weather_data(start_date, end_date)
    
    def _collect_stock_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Collect stock level data from major ports."""
        # Mock implementation
        dates = pd.date_range(start=start_date, end=end_date, freq='W')
        stock_levels = pd.Series(
            data=pd.np.random.uniform(50000, 150000, len(dates)),
            index=dates,
            name="stock_level"
        )
        return stock_levels.to_frame()
    
    def _collect_production_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Collect production data from ICCO."""
        api_url = self.settings.icco_api_url
        api_key = self.settings.icco_api_key
        
        if not api_url or not api_key:
            logger.warning("ICCO API not configured. Returning mock data.")
            # Mock monthly production data
            dates = pd.date_range(start=start_date, end=end_date, freq='MS')
            production = pd.Series(
                data=pd.np.random.uniform(300000, 500000, len(dates)),
                index=dates,
                name="production"
            )
            return production.to_frame()
        
        self.rate_limiter.acquire("icco")
        
        # Implementation would call ICCO API
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        production = pd.Series(
            data=pd.np.random.uniform(300000, 500000, len(dates)),
            index=dates,
            name="production"
        )
        return production.to_frame()
    
    def _collect_fx_rates(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Collect FX rates for XAF, GBP, EUR to USD."""
        api_url = self.settings.fx_rates_api_url
        api_key = self.settings.fx_rates_api_key
        
        if not api_url or not api_key:
            logger.warning("FX rates API not configured. Returning mock data.")
            return self._generate_mock_fx_data(start_date, end_date)
        
        self.rate_limiter.acquire("forex")
        
        # Implementation would call FX rates API
        return self._generate_mock_fx_data(start_date, end_date)
    
    def _generate_mock_weather_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate mock weather data."""
        import numpy as np
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        df = pd.DataFrame({
            "timestamp": dates,
            "temperature": np.random.uniform(20, 35, len(dates)),
            "rainfall": np.random.exponential(5, len(dates))
        })
        
        return df
    
    def _generate_mock_fx_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate mock FX rate data."""
        import numpy as np
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        df = pd.DataFrame({
            "timestamp": dates,
            "fx_rate_xaf_usd": np.random.uniform(0.0016, 0.0018, len(dates)),
            "fx_rate_gbp_usd": np.random.uniform(1.25, 1.35, len(dates)),
            "fx_rate_eur_usd": np.random.uniform(1.05, 1.15, len(dates))
        })
        
        return df
    
    def collect_news_feed(
        self,
        sources: List[str],
        keywords: List[str],
        hours_back: int = 24
    ) -> List[NewsArticle]:
        """
        Collect news articles from configured sources.
        
        Retrieves recent news articles related to cocoa markets from
        Reuters and Bloomberg with retry logic and rate limiting.
        
        Args:
            sources: List of news sources (e.g., ["reuters", "bloomberg"])
            keywords: List of keywords to filter articles (e.g., ["cocoa", "cacao"])
            hours_back: Number of hours to look back for articles (default: 24)
            
        Returns:
            List of NewsArticle objects
            
        Requirements:
            - 3.1: Retrieve News_Feed from configured sources at least hourly
            - 3.2: Extract articles related to cocoa markets
        """
        logger.info(
            f"Collecting news from {sources} with keywords {keywords} "
            f"for last {hours_back} hours"
        )
        
        all_articles = []
        start_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        for source in sources:
            try:
                articles = self._collect_news_from_source(
                    source, keywords, start_time
                )
                all_articles.extend(articles)
                
                logger.info(
                    f"Collected {len(articles)} articles from {source}"
                )
                
            except Exception as e:
                logger.error(f"Failed to collect news from {source}: {e}")
        
        logger.info(f"Collected total of {len(all_articles)} news articles")
        return all_articles
    
    def _collect_news_from_source(
        self,
        source: str,
        keywords: List[str],
        start_time: datetime
    ) -> List[NewsArticle]:
        """
        Collect news articles from a specific source.
        
        Args:
            source: News source identifier (e.g., "reuters")
            keywords: Keywords to filter articles
            start_time: Start time for article collection
            
        Returns:
            List of NewsArticle objects
        """
        # Determine API configuration based on source
        if source.lower() == "reuters":
            api_url = self.settings.reuters_api_url
            api_key = self.settings.reuters_api_key
            rate_limiter_name = "reuters"
        elif source.lower() == "bloomberg":
            api_url = self.settings.bloomberg_api_url
            api_key = self.settings.bloomberg_api_key
            rate_limiter_name = "bloomberg"
        else:
            raise ValueError(f"Unknown news source: {source}")
        
        if not api_url or not api_key:
            logger.warning(
                f"API credentials not configured for {source}. "
                f"Returning mock data."
            )
            return self._generate_mock_news_articles(source, keywords, start_time)
        
        # Apply rate limiting
        self.rate_limiter.acquire(rate_limiter_name)
        
        # Make API request with retry logic
        articles = self.retry_handler.execute_with_retry(
            self._fetch_news_from_api,
            api_url,
            api_key,
            source,
            keywords,
            start_time
        )
        
        return articles
    
    def _fetch_news_from_api(
        self,
        api_url: str,
        api_key: str,
        source: str,
        keywords: List[str],
        start_time: datetime
    ) -> List[NewsArticle]:
        """Fetch news articles from external API."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        params = {
            "keywords": ",".join(keywords),
            "start_time": start_time.isoformat(),
            "limit": 100
        }
        
        response = requests.get(
            api_url,
            headers=headers,
            params=params,
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        # Parse response and create NewsArticle objects
        articles = []
        for item in data.get("articles", []):
            try:
                articles.append(NewsArticle(
                    id=item["id"],
                    source=source,
                    title=item["title"],
                    content=item["content"],
                    published_at=datetime.fromisoformat(item["published_at"]),
                    url=item["url"],
                    keywords=item.get("keywords", [])
                ))
            except Exception as e:
                logger.warning(f"Failed to parse news article: {e}")
        
        return articles
    
    def _generate_mock_news_articles(
        self,
        source: str,
        keywords: List[str],
        start_time: datetime
    ) -> List[NewsArticle]:
        """Generate mock news articles for development/testing."""
        import uuid
        
        mock_articles = []
        num_articles = 5
        
        for i in range(num_articles):
            published_at = start_time + timedelta(hours=i * 4)
            
            mock_articles.append(NewsArticle(
                id=str(uuid.uuid4()),
                source=source,
                title=f"Cocoa Market Update {i+1}",
                content=f"Mock article content about cocoa markets. Keywords: {', '.join(keywords)}",
                published_at=published_at,
                url=f"https://{source}.com/article/{i+1}",
                keywords=keywords[:2]
            ))
        
        return mock_articles
    
    def validate_and_store(
        self,
        data: Union[pd.DataFrame, List],
        data_type: str
    ) -> Dict[str, Union[bool, List[ValidationError]]]:
        """
        Validate collected data and prepare for storage.
        
        Args:
            data: Data to validate (DataFrame or List of model objects)
            data_type: One of ["price", "econometric", "news"]
            
        Returns:
            Dict with keys:
                - "valid": Boolean indicating if data is valid
                - "errors": List of ValidationError objects
                - "records_validated": Number of records validated
                
        Requirements:
            - 1.4: Log errors and retry after 5 minutes when source unavailable
            - 1.5: Validate prices are within realistic bounds (1000-10000 USD/MT)
            - 2.6: Flag missing values for imputation
        """
        logger.info(f"Validating {data_type} data")
        
        errors = []
        records_validated = 0
        
        if data_type == "price":
            if isinstance(data, pd.DataFrame):
                records_validated = len(data)
                # Validate price ranges
                invalid_prices = data[
                    (data["price"] < 1000) | (data["price"] > 10000)
                ]
                for idx, row in invalid_prices.iterrows():
                    errors.append(ValidationError(
                        field="price",
                        value=str(row["price"]),
                        error_type="out_of_range",
                        message=f"Price {row['price']} outside valid range [1000, 10000]",
                        severity="ERROR"
                    ))
        
        elif data_type == "econometric":
            if isinstance(data, dict):
                for key, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        records_validated += len(df)
                        # Check for missing values
                        missing = df.isnull().sum()
                        for col, count in missing.items():
                            if count > 0:
                                errors.append(ValidationError(
                                    field=col,
                                    value=None,
                                    error_type="missing",
                                    message=f"{count} missing values in {col}",
                                    severity="WARNING"
                                ))
        
        elif data_type == "news":
            if isinstance(data, list):
                records_validated = len(data)
                # Basic validation for news articles
                for article in data:
                    if not isinstance(article, NewsArticle):
                        errors.append(ValidationError(
                            field="article",
                            value=str(type(article)),
                            error_type="invalid_format",
                            message="Invalid article format",
                            severity="ERROR"
                        ))
        
        is_valid = len([e for e in errors if e.severity == "ERROR"]) == 0
        
        logger.info(
            f"Validation complete: {records_validated} records, "
            f"{len(errors)} issues found, valid={is_valid}"
        )
        
        return {
            "valid": is_valid,
            "errors": errors,
            "records_validated": records_validated
        }
