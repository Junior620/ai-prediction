"""
Tests for Redis cache functionality.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch

from src.api.cache import RedisCache
from src.api.models import PredictionResponse, PredictionItem


class TestRedisCache:
    """Tests for RedisCache class."""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Create mock Redis client."""
        return Mock()
    
    @pytest.fixture
    def cache(self, mock_redis_client):
        """Create RedisCache instance with mock client."""
        return RedisCache(redis_client=mock_redis_client)
    
    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache.default_ttl == 3600
        assert cache.redis_client is not None
    
    def test_generate_cache_key(self, cache):
        """Test cache key generation."""
        key = cache._generate_cache_key(
            market="ICE_London",
            horizons=[1, 7, 30],
            include_sentiment=True
        )
        
        assert "prediction:ICE_London" in key
        assert "1_7_30" in key
        assert "with_sentiment" in key
    
    def test_generate_cache_key_no_sentiment(self, cache):
        """Test cache key generation without sentiment."""
        key = cache._generate_cache_key(
            market="ICE_NY",
            horizons=[1],
            include_sentiment=False
        )
        
        assert "prediction:ICE_NY" in key
        assert "no_sentiment" in key
    
    def test_set_prediction_success(self, cache, mock_redis_client):
        """Test successful prediction caching."""
        prediction_response = PredictionResponse(
            predictions=[
                PredictionItem(
                    horizon=1,
                    price=3010.0,
                    confidence_interval=[2910.0, 3110.0],
                    confidence_level=0.95,
                    timestamp=datetime.utcnow()
                )
            ],
            model_version="1.0.0",
            sentiment_score=-0.15,
            market="ICE_London"
        )
        
        mock_redis_client.setex.return_value = True
        
        result = cache.set_prediction(
            market="ICE_London",
            horizons=[1],
            include_sentiment=True,
            prediction_response=prediction_response
        )
        
        assert result is True
        mock_redis_client.setex.assert_called_once()
        
        # Verify TTL was set
        call_args = mock_redis_client.setex.call_args
        assert call_args[1]['time'] == 3600  # Default TTL
    
    def test_set_prediction_custom_ttl(self, cache, mock_redis_client):
        """Test prediction caching with custom TTL."""
        prediction_response = PredictionResponse(
            predictions=[
                PredictionItem(
                    horizon=1,
                    price=3010.0,
                    confidence_interval=[2910.0, 3110.0],
                    confidence_level=0.95,
                    timestamp=datetime.utcnow()
                )
            ],
            model_version="1.0.0",
            market="ICE_London"
        )
        
        mock_redis_client.setex.return_value = True
        
        result = cache.set_prediction(
            market="ICE_London",
            horizons=[1],
            include_sentiment=True,
            prediction_response=prediction_response,
            ttl=7200  # 2 hours
        )
        
        assert result is True
        call_args = mock_redis_client.setex.call_args
        assert call_args[1]['time'] == 7200
    
    def test_get_prediction_cache_hit(self, cache, mock_redis_client):
        """Test retrieving cached prediction."""
        cached_data = {
            "predictions": [
                {
                    "horizon": 1,
                    "price": 3010.0,
                    "confidence_interval": [2910.0, 3110.0],
                    "confidence_level": 0.95,
                    "timestamp": datetime.utcnow().isoformat()
                }
            ],
            "model_version": "1.0.0",
            "sentiment_score": -0.15,
            "market": "ICE_London"
        }
        
        mock_redis_client.get.return_value = json.dumps(cached_data, default=str)
        
        result = cache.get_prediction(
            market="ICE_London",
            horizons=[1],
            include_sentiment=True
        )
        
        assert result is not None
        assert result.market == "ICE_London"
        assert len(result.predictions) == 1
        assert result.predictions[0].price == 3010.0
    
    def test_get_prediction_cache_miss(self, cache, mock_redis_client):
        """Test cache miss returns None."""
        mock_redis_client.get.return_value = None
        
        result = cache.get_prediction(
            market="ICE_London",
            horizons=[1],
            include_sentiment=True
        )
        
        assert result is None
    
    def test_get_prediction_error_handling(self, cache, mock_redis_client):
        """Test error handling during cache retrieval."""
        mock_redis_client.get.side_effect = Exception("Redis error")
        
        result = cache.get_prediction(
            market="ICE_London",
            horizons=[1],
            include_sentiment=True
        )
        
        assert result is None
    
    def test_invalidate_prediction_success(self, cache, mock_redis_client):
        """Test successful cache invalidation."""
        mock_redis_client.delete.return_value = 1
        
        result = cache.invalidate_prediction(
            market="ICE_London",
            horizons=[1],
            include_sentiment=True
        )
        
        assert result is True
        mock_redis_client.delete.assert_called_once()
    
    def test_invalidate_prediction_not_found(self, cache, mock_redis_client):
        """Test cache invalidation when entry doesn't exist."""
        mock_redis_client.delete.return_value = 0
        
        result = cache.invalidate_prediction(
            market="ICE_London",
            horizons=[1],
            include_sentiment=True
        )
        
        assert result is False
    
    def test_invalidate_all_predictions(self, cache, mock_redis_client):
        """Test invalidating all cached predictions."""
        mock_redis_client.keys.return_value = [
            "prediction:ICE_London:1:with_sentiment",
            "prediction:ICE_NY:7:no_sentiment"
        ]
        mock_redis_client.delete.return_value = 2
        
        result = cache.invalidate_all_predictions()
        
        assert result == 2
        mock_redis_client.keys.assert_called_once_with("prediction:*")
        mock_redis_client.delete.assert_called_once()
    
    def test_invalidate_all_predictions_empty(self, cache, mock_redis_client):
        """Test invalidating all predictions when cache is empty."""
        mock_redis_client.keys.return_value = []
        
        result = cache.invalidate_all_predictions()
        
        assert result == 0
        mock_redis_client.delete.assert_not_called()
    
    def test_health_check_success(self, cache, mock_redis_client):
        """Test successful health check."""
        mock_redis_client.ping.return_value = True
        
        result = cache.health_check()
        
        assert result is True
        mock_redis_client.ping.assert_called_once()
    
    def test_health_check_failure(self, cache, mock_redis_client):
        """Test failed health check."""
        mock_redis_client.ping.side_effect = Exception("Connection error")
        
        result = cache.health_check()
        
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
