"""
Comprehensive tests for the FastAPI application.

Tests cover:
- All API endpoints (predict, performance, models, retrain)
- Authentication and authorization
- Redis caching functionality
- Error handling
- Request/response validation
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import json

from src.api.app import app
from src.api.models import (
    PredictionRequest,
    PredictionResponse,
    PredictionItem,
    PerformanceResponse,
    ModelsResponse,
    RetrainingRequest
)
from src.models.data_models import Prediction, NewsArticle


TEST_SECRET_KEY = "test_secret_key_for_jwt"


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_settings():
    """Mock settings with a shared JWT secret for auth + app."""
    settings = Mock()
    settings.secret_key = TEST_SECRET_KEY
    settings.supabase_url = "http://test-supabase.com"
    settings.supabase_key = "test_key"
    settings.redis_host = "localhost"
    settings.redis_port = 6379
    settings.redis_password = ""
    settings.redis_db = 0
    settings.mlflow_tracking_uri = "http://localhost:5000"
    settings.mlflow_registry_uri = "http://localhost:5000"
    with patch("src.api.auth.get_settings", return_value=settings), \
         patch("src.api.app.get_settings", return_value=settings):
        yield settings


@pytest.fixture
def auth_headers(mock_settings):
    """JWT auth headers for a regular user."""
    from src.api.auth import create_user_token
    token = create_user_token("test_user", role="user")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers(mock_settings):
    """JWT auth headers for an admin user."""
    from src.api.auth import create_user_token
    token = create_user_token("admin_user", role="admin")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_redis_cache():
    """Mock Redis cache object (patch into app module in each test)."""
    cache = Mock()
    cache.health_check.return_value = True
    cache.get_prediction.return_value = None
    cache.set_prediction.return_value = True
    cache.invalidate_all_predictions.return_value = 5
    return cache


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client object."""
    return Mock()


def _make_mock_predictor():
    """Build a mock multi-market predictor used by /api/v1/predict."""
    predictor = Mock()
    predictor.model_version = "1.0.0"

    def mock_predict(horizons, exog_features, recent_news, **kwargs):
        predictions = []
        for horizon in horizons:
            pred = Prediction(
                horizon=horizon,
                price=3000.0 + horizon * 10,
                confidence_interval=(2900.0 + horizon * 10, 3100.0 + horizon * 10),
                confidence_level=0.95,
                timestamp=datetime.utcnow(),
                model_version="1.0.0",
                components={
                    "baseline": 2950.0,
                    "residual": 50.0,
                    "sentiment": 0.0
                }
            )
            predictions.append(pred)
        return predictions

    predictor.predict = mock_predict
    nlp_analyzer = Mock()
    nlp_analyzer.aggregate_sentiment.return_value = -0.15
    predictor.nlp_analyzer = nlp_analyzer
    return predictor


@pytest.fixture
def mock_price_predictor():
    """Mock cocoa predictor registered in app.predictors (multi-market)."""
    predictor = _make_mock_predictor()
    with patch("src.api.app.predictors", {"cocoa": predictor}), \
         patch("src.api.app.price_predictor", predictor):
        yield predictor


@pytest.fixture
def mock_model_manager():
    """Mock model manager object."""
    manager = Mock()

    mock_version = Mock()
    mock_version.version = "1"
    mock_version.current_stage = "Production"
    mock_version.creation_timestamp = datetime.utcnow().timestamp() * 1000
    manager.list_model_versions.return_value = [mock_version]

    manager.get_model_info.return_value = {
        "name": "cocoa_prophet",
        "version": "1",
        "stage": "Production",
        "metrics": {
            "rmse": 45.2,
            "mae": 32.1,
            "mape": 0.025
        }
    }
    return manager


class TestRootEndpoints:
    """Tests for root and health check endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Cocoa Price Prediction API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
    
    def test_health_check_healthy(self, client, mock_redis_cache):
        """Test health check when all services are healthy."""
        with patch('src.api.app.redis_cache', mock_redis_cache):
            with patch('src.api.app.supabase_client', Mock()):
                with patch('src.api.app.model_manager', Mock()):
                    with patch('src.api.app.price_predictor', Mock()):
                        response = client.get("/health")
                        assert response.status_code == 200
                        data = response.json()
                        assert data["status"] == "healthy"
                        assert data["services"]["redis"] is True
    
    def test_health_check_degraded(self, client):
        """Test health check when some services are unavailable."""
        with patch('src.api.app.redis_cache', None):
            response = client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "degraded"


class TestPredictEndpoint:
    """Tests for the /api/v1/predict endpoint."""
    
    def test_predict_success(
        self,
        client,
        auth_headers,
        mock_price_predictor,
        mock_supabase_client,
        mock_redis_cache
    ):
        """Test successful prediction request."""
        with patch('src.api.app.supabase_client', mock_supabase_client):
            with patch('src.api.app.redis_cache', mock_redis_cache):
                # Mock Supabase query for news
                mock_response = Mock()
                mock_response.data = []
                mock_supabase_client.table.return_value.select.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
                
                # Mock Supabase insert for predictions
                mock_supabase_client.table.return_value.insert.return_value.execute.return_value = Mock()
                
                request_data = {
                    "horizons": [1, 7, 30],
                    "market": "ICE_London",
                    "include_sentiment": True
                }
                
                response = client.post(
                    "/api/v1/predict",
                    json=request_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "predictions" in data
                assert len(data["predictions"]) == 3
                assert data["model_version"] == "1.0.0"
                assert data["market"] == "ICE_London"
                assert "sentiment_score" in data
    
    def test_predict_without_authentication(self, client):
        """Test prediction request without authentication fails."""
        request_data = {
            "horizons": [1, 7, 30],
            "market": "ICE_London",
            "include_sentiment": True
        }
        
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code in (401, 403)
    
    def test_predict_invalid_token(self, client, mock_settings):
        """Test prediction request with invalid token fails."""
        request_data = {
            "horizons": [1, 7, 30],
            "market": "ICE_London",
            "include_sentiment": True
        }
        
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.post("/api/v1/predict", json=request_data, headers=headers)
        assert response.status_code == 401
    
    def test_predict_cache_hit(
        self,
        client,
        auth_headers,
        mock_price_predictor,
        mock_redis_cache
    ):
        """Test prediction returns cached result on cache hit."""
        # Mock cached prediction
        cached_response = PredictionResponse(
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
        mock_redis_cache.get_prediction.return_value = cached_response
        
        with patch('src.api.app.redis_cache', mock_redis_cache):
            request_data = {
                "horizons": [1],
                "market": "ICE_London",
                "include_sentiment": True
            }
            
            response = client.post(
                "/api/v1/predict",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 1
            assert data["predictions"][0]["price"] == 3010.0
    
    def test_predict_invalid_request(self, client, auth_headers):
        """Test prediction with invalid request data."""
        request_data = {
            "horizons": [],  # Empty horizons list
            "market": "ICE_London",
            "include_sentiment": True
        }
        
        response = client.post(
            "/api/v1/predict",
            json=request_data,
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
    
    def test_predict_service_unavailable(self, client, auth_headers):
        """Test prediction when service is unavailable."""
        with patch('src.api.app.predictors', {}), \
             patch('src.api.app.price_predictor', None):
            request_data = {
                "horizons": [1, 7, 30],
                "market": "ICE_London",
                "include_sentiment": True
            }
            
            response = client.post(
                "/api/v1/predict",
                json=request_data,
                headers=auth_headers
            )
            assert response.status_code == 503


class TestPerformanceEndpoint:
    """Tests for the /api/v1/performance endpoint."""
    
    def test_get_performance_success(
        self,
        client,
        auth_headers,
        mock_supabase_client
    ):
        """Test successful performance metrics retrieval."""
        # Mock Supabase query
        mock_response = Mock()
        mock_response.data = [
            {
                "created_at": datetime.utcnow().isoformat(),
                "model_version": "1.0.0",
                "rmse": 45.2,
                "mae": 32.1,
                "mape": 0.025,
                "directional_accuracy": 0.75,
                "coverage_rate": 0.95,
                "mean_interval_width": 200.0
            }
        ]
        
        mock_supabase_client.table.return_value.select.return_value.gte.return_value.lte.return_value.order.return_value.execute.return_value = mock_response
        
        with patch('src.api.app.supabase_client', mock_supabase_client):
            start_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
            end_date = datetime.utcnow().isoformat()
            
            response = client.get(
                f"/api/v1/performance?start_date={start_date}&end_date={end_date}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "metrics" in data
            assert len(data["metrics"]) == 1
            assert data["model_version"] == "1.0.0"
    
    def test_get_performance_without_authentication(self, client):
        """Test performance request without authentication fails."""
        start_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
        end_date = datetime.utcnow().isoformat()
        
        response = client.get(
            f"/api/v1/performance?start_date={start_date}&end_date={end_date}"
        )
        assert response.status_code in (401, 403)
    
    def test_get_performance_invalid_date_range(self, client, auth_headers):
        """Test performance request with invalid date range."""
        start_date = datetime.utcnow().isoformat()
        end_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
        
        response = client.get(
            f"/api/v1/performance?start_date={start_date}&end_date={end_date}",
            headers=auth_headers
        )
        assert response.status_code == 400
    
    def test_get_performance_no_data(
        self,
        client,
        auth_headers,
        mock_supabase_client
    ):
        """Test performance request when no data is available."""
        # Mock empty response
        mock_response = Mock()
        mock_response.data = []
        
        mock_supabase_client.table.return_value.select.return_value.gte.return_value.lte.return_value.order.return_value.execute.return_value = mock_response
        
        with patch('src.api.app.supabase_client', mock_supabase_client):
            start_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
            end_date = datetime.utcnow().isoformat()
            
            response = client.get(
                f"/api/v1/performance?start_date={start_date}&end_date={end_date}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["metrics"]) == 0


class TestModelsEndpoint:
    """Tests for the /api/v1/models endpoint."""
    
    def test_list_models_success(
        self,
        client,
        auth_headers,
        mock_model_manager
    ):
        """Test successful models listing."""
        with patch('src.api.app.model_manager', mock_model_manager):
            response = client.get("/api/v1/models", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert "current_production_version" in data
    
    def test_list_models_without_authentication(self, client):
        """Test models listing without authentication fails."""
        response = client.get("/api/v1/models")
        assert response.status_code in (401, 403)
    
    def test_list_models_empty(self, client, auth_headers):
        """Test models listing when no models are available."""
        mock_manager = Mock()
        mock_manager.list_model_versions.side_effect = Exception("No models found")
        
        with patch('src.api.app.model_manager', mock_manager):
            response = client.get("/api/v1/models", headers=auth_headers)
            
            # Should still return 200 with empty list
            assert response.status_code == 200
            data = response.json()
            assert len(data["models"]) == 0


class TestRetrainEndpoint:
    """Tests for the /api/v1/retrain endpoint."""
    
    def test_retrain_success(
        self,
        client,
        admin_auth_headers,
        mock_redis_cache
    ):
        """Test successful retraining trigger."""
        with patch('src.api.app.redis_cache', mock_redis_cache):
            request_data = {
                "model_type": "all",
                "reason": "Performance degradation detected"
            }
            
            response = client.post(
                "/api/v1/retrain",
                json=request_data,
                headers=admin_auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "accepted"
            assert "job_id" in data
            assert "estimated_completion" in data
    
    def test_retrain_without_authentication(self, client):
        """Test retraining without authentication fails."""
        request_data = {
            "model_type": "all",
            "reason": "Performance degradation detected"
        }
        
        response = client.post("/api/v1/retrain", json=request_data)
        assert response.status_code in (401, 403)
    
    def test_retrain_without_admin_privileges(self, client, auth_headers):
        """Test retraining without admin privileges fails."""
        request_data = {
            "model_type": "all",
            "reason": "Performance degradation detected"
        }
        
        response = client.post(
            "/api/v1/retrain",
            json=request_data,
            headers=auth_headers
        )
        assert response.status_code == 403
    
    def test_retrain_invalid_model_type(self, client, admin_auth_headers):
        """Test retraining with invalid model type."""
        request_data = {
            "model_type": "invalid_model",
            "reason": "Testing"
        }
        
        response = client.post(
            "/api/v1/retrain",
            json=request_data,
            headers=admin_auth_headers
        )
        assert response.status_code == 422  # Validation error
    
    def test_retrain_specific_model(
        self,
        client,
        admin_auth_headers,
        mock_redis_cache
    ):
        """Test retraining specific model type."""
        with patch('src.api.app.redis_cache', mock_redis_cache):
            request_data = {
                "model_type": "prophet",
                "reason": "Updating time series model"
            }
            
            response = client.post(
                "/api/v1/retrain",
                json=request_data,
                headers=admin_auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "prophet" in data["message"]


class TestRedisCaching:
    """Tests for Redis caching functionality."""
    
    def test_cache_stores_prediction(self, mock_redis_cache):
        """Test that predictions are stored in cache."""
        from src.api.cache import RedisCache
        
        cache = RedisCache(redis_client=Mock())
        cache.redis_client.setex = Mock(return_value=True)
        
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
        
        result = cache.set_prediction(
            market="ICE_London",
            horizons=[1],
            include_sentiment=True,
            prediction_response=prediction_response
        )
        
        assert result is True
        cache.redis_client.setex.assert_called_once()
    
    def test_cache_retrieves_prediction(self):
        """Test that predictions are retrieved from cache."""
        from src.api.cache import RedisCache
        
        mock_redis = Mock()
        cache = RedisCache(redis_client=mock_redis)
        
        # Mock cached data
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
        mock_redis.get.return_value = json.dumps(cached_data, default=str)
        
        result = cache.get_prediction(
            market="ICE_London",
            horizons=[1],
            include_sentiment=True
        )
        
        assert result is not None
        assert result.market == "ICE_London"
        assert len(result.predictions) == 1
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        from src.api.cache import RedisCache
        
        mock_redis = Mock()
        mock_redis.delete.return_value = 1
        cache = RedisCache(redis_client=mock_redis)
        
        result = cache.invalidate_prediction(
            market="ICE_London",
            horizons=[1],
            include_sentiment=True
        )
        
        assert result is True
        mock_redis.delete.assert_called_once()


class TestAuthentication:
    """Tests for authentication functionality."""
    
    def test_verify_token_success(self, mock_settings):
        """Test successful token verification with a real JWT."""
        from src.api.auth import verify_token, create_user_token
        from fastapi.security import HTTPAuthorizationCredentials
        
        token = create_user_token("authenticated_user", role="user")
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token
        )
        
        result = verify_token(credentials)
        assert result == "authenticated_user"
    
    def test_verify_token_failure(self, mock_settings):
        """Test failed token verification."""
        from src.api.auth import verify_token
        from fastapi.security import HTTPAuthorizationCredentials
        from fastapi import HTTPException
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="invalid_token"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            verify_token(credentials)
        
        assert exc_info.value.status_code == 401
    
    def test_verify_admin_token_success(self, mock_settings):
        """Test successful admin token verification with a real JWT."""
        from src.api.auth import verify_admin_token, create_user_token
        from fastapi.security import HTTPAuthorizationCredentials
        
        token = create_user_token("admin_user", role="admin")
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token
        )
        
        result = verify_admin_token(credentials)
        assert result == "admin_user"
    
    def test_verify_admin_token_non_admin(self, mock_settings):
        """Test admin token verification with non-admin token."""
        from src.api.auth import verify_admin_token, create_user_token
        from fastapi.security import HTTPAuthorizationCredentials
        from fastapi import HTTPException
        
        token = create_user_token("regular_user", role="user")
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token
        )
        
        with pytest.raises(HTTPException) as exc_info:
            verify_admin_token(credentials)
        
        assert exc_info.value.status_code == 403


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
