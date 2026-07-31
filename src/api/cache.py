"""
Redis cache module for the FastAPI application.

This module provides caching functionality for predictions with 1-hour TTL.
"""

import json
from typing import Optional
from datetime import datetime, timedelta
import redis
from loguru import logger

from config.settings import get_settings
from src.api.models import PredictionResponse


class RedisCache:
    """
    Redis cache manager for predictions.
    
    Provides methods to:
    - Store predictions with TTL
    - Retrieve cached predictions
    - Invalidate cache entries
    
    Attributes:
        redis_client: Redis client instance
        default_ttl: Default TTL in seconds (1 hour)
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize Redis cache manager.
        
        Args:
            redis_client: Optional Redis client. If None, creates a new client.
        """
        if redis_client is None:
            settings = get_settings()
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password if settings.redis_password else None,
                db=settings.redis_db,
                decode_responses=True
            )
        else:
            self.redis_client = redis_client
        
        # Default TTL: 1 hour (3600 seconds)
        self.default_ttl = 3600
        
        logger.info("RedisCache initialized")
    
    def _generate_cache_key(
        self,
        market: str,
        horizons: list[int],
        include_sentiment: bool
    ) -> str:
        """
        Generate cache key for a prediction request.
        
        Args:
            market: Market identifier
            horizons: List of prediction horizons
            include_sentiment: Whether sentiment is included
        
        Returns:
            Cache key string
        """
        horizons_str = "_".join(str(h) for h in sorted(horizons))
        sentiment_flag = "with_sentiment" if include_sentiment else "no_sentiment"
        return f"prediction:{market}:{horizons_str}:{sentiment_flag}"
    
    def get_prediction(
        self,
        market: str,
        horizons: list[int],
        include_sentiment: bool
    ) -> Optional[PredictionResponse]:
        """
        Retrieve cached prediction if available.
        
        Args:
            market: Market identifier
            horizons: List of prediction horizons
            include_sentiment: Whether sentiment is included
        
        Returns:
            PredictionResponse if cache hit, None if cache miss
        """
        cache_key = self._generate_cache_key(market, horizons, include_sentiment)
        
        try:
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data is None:
                logger.debug(f"Cache miss for key: {cache_key}")
                return None
            
            # Deserialize JSON data
            data_dict = json.loads(cached_data)
            
            # Convert back to PredictionResponse
            prediction_response = PredictionResponse(**data_dict)
            
            logger.info(f"Cache hit for key: {cache_key}")
            return prediction_response
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None
    
    def set_prediction(
        self,
        market: str,
        horizons: list[int],
        include_sentiment: bool,
        prediction_response: PredictionResponse,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store prediction in cache with TTL.
        
        Args:
            market: Market identifier
            horizons: List of prediction horizons
            include_sentiment: Whether sentiment is included
            prediction_response: Prediction response to cache
            ttl: Optional custom TTL in seconds (default: 1 hour)
        
        Returns:
            True if successful, False otherwise
        """
        cache_key = self._generate_cache_key(market, horizons, include_sentiment)
        
        if ttl is None:
            ttl = self.default_ttl
        
        try:
            # Serialize to JSON
            data_dict = prediction_response.model_dump(mode='json')
            cached_data = json.dumps(data_dict, default=str)
            
            # Store in Redis with TTL
            self.redis_client.setex(
                name=cache_key,
                time=ttl,
                value=cached_data
            )
            
            logger.info(f"Cached prediction with key: {cache_key}, TTL: {ttl}s")
            return True
            
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False
    
    def invalidate_prediction(
        self,
        market: str,
        horizons: list[int],
        include_sentiment: bool
    ) -> bool:
        """
        Invalidate a cached prediction.
        
        Args:
            market: Market identifier
            horizons: List of prediction horizons
            include_sentiment: Whether sentiment is included
        
        Returns:
            True if successful, False otherwise
        """
        cache_key = self._generate_cache_key(market, horizons, include_sentiment)
        
        try:
            deleted = self.redis_client.delete(cache_key)
            
            if deleted:
                logger.info(f"Invalidated cache for key: {cache_key}")
            else:
                logger.debug(f"No cache entry found for key: {cache_key}")
            
            return bool(deleted)
            
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return False
    
    def invalidate_all_predictions(self) -> int:
        """
        Invalidate all cached predictions.
        
        Returns:
            Number of cache entries deleted
        """
        try:
            # Find all prediction cache keys
            keys = self.redis_client.keys("prediction:*")
            
            if not keys:
                logger.info("No prediction cache entries to invalidate")
                return 0
            
            # Delete all keys
            deleted = self.redis_client.delete(*keys)
            
            logger.info(f"Invalidated {deleted} prediction cache entries")
            return deleted
            
        except Exception as e:
            logger.error(f"Error invalidating all predictions: {e}")
            return 0
    
    def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.
        
        Returns:
            True if Redis is accessible, False otherwise
        """
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def set_latest_tv_alert(self, market: str, alert: dict, ttl: int = 7 * 24 * 3600) -> bool:
        """Store latest TradingView alert snapshot for dashboard polling."""
        try:
            key = f"tv_alert:latest:{market.upper()}"
            self.redis_client.setex(key, ttl, json.dumps(alert, default=str))
            return True
        except Exception as e:
            logger.error(f"Error storing latest TV alert: {e}")
            return False

    def get_latest_tv_alert(self, market: str) -> Optional[dict]:
        """Return latest TradingView alert snapshot, or None."""
        try:
            key = f"tv_alert:latest:{market.upper()}"
            raw = self.redis_client.get(key)
            if not raw:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Error reading latest TV alert: {e}")
            return None
