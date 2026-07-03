"""
Tests for API Pydantic models.

Tests request/response validation without requiring full app initialization.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.api.models import (
    PredictionRequest,
    PredictionResponse,
    PredictionItem,
    PerformanceMetricsItem,
    PerformanceResponse,
    ModelInfo,
    ModelsResponse,
    RetrainingRequest,
    RetrainingResponse,
    ErrorResponse
)


class TestPredictionRequest:
    """Tests for PredictionRequest model."""
    
    def test_valid_prediction_request(self):
        """Test valid prediction request."""
        request = PredictionRequest(
            horizons=[1, 7, 30],
            market="ICE_London",
            include_sentiment=True
        )
        
        assert request.horizons == [1, 7, 30]
        assert request.market == "ICE_London"
        assert request.include_sentiment is True
    
    def test_prediction_request_defaults(self):
        """Test prediction request with defaults."""
        request = PredictionRequest(
            horizons=[1],
            market="ICE_NY"
        )
        
        assert request.include_sentiment is True  # Default value
    
    def test_prediction_request_empty_horizons(self):
        """Test prediction request with empty horizons fails."""
        with pytest.raises(ValidationError):
            PredictionRequest(
                horizons=[],
                market="ICE_London"
            )
    
    def test_prediction_request_missing_fields(self):
        """Test prediction request with missing required fields fails."""
        with pytest.raises(ValidationError):
            PredictionRequest(horizons=[1])


class TestPredictionResponse:
    """Tests for PredictionResponse model."""
    
    def test_valid_prediction_response(self):
        """Test valid prediction response."""
        response = PredictionResponse(
            predictions=[
                PredictionItem(
                    horizon=1,
                    price=3000.0,
                    confidence_interval=[2900.0, 3100.0],
                    confidence_level=0.95,
                    timestamp=datetime.utcnow()
                )
            ],
            model_version="1.0.0",
            sentiment_score=-0.15,
            market="ICE_London"
        )
        
        assert len(response.predictions) == 1
        assert response.model_version == "1.0.0"
        assert response.sentiment_score == -0.15
        assert response.market == "ICE_London"
    
    def test_prediction_response_without_sentiment(self):
        """Test prediction response without sentiment score."""
        response = PredictionResponse(
            predictions=[
                PredictionItem(
                    horizon=1,
                    price=3000.0,
                    confidence_interval=[2900.0, 3100.0],
                    confidence_level=0.95,
                    timestamp=datetime.utcnow()
                )
            ],
            model_version="1.0.0",
            market="ICE_London"
        )
        
        assert response.sentiment_score is None


class TestPredictionItem:
    """Tests for PredictionItem model."""
    
    def test_valid_prediction_item(self):
        """Test valid prediction item."""
        item = PredictionItem(
            horizon=7,
            price=3050.0,
            confidence_interval=[2950.0, 3150.0],
            confidence_level=0.95,
            timestamp=datetime.utcnow()
        )
        
        assert item.horizon == 7
        assert item.price == 3050.0
        assert len(item.confidence_interval) == 2
    
    def test_prediction_item_invalid_confidence_interval(self):
        """Test prediction item with invalid confidence interval."""
        with pytest.raises(ValidationError):
            PredictionItem(
                horizon=7,
                price=3050.0,
                confidence_interval=[2950.0],  # Only one value
                confidence_level=0.95,
                timestamp=datetime.utcnow()
            )


class TestPerformanceMetricsItem:
    """Tests for PerformanceMetricsItem model."""
    
    def test_valid_performance_metrics(self):
        """Test valid performance metrics item."""
        metrics = PerformanceMetricsItem(
            timestamp=datetime.utcnow(),
            rmse=45.2,
            mae=32.1,
            mape=0.025,
            directional_accuracy=0.75,
            coverage_rate=0.95,
            mean_interval_width=200.0
        )
        
        assert metrics.rmse == 45.2
        assert metrics.mae == 32.1
        assert metrics.directional_accuracy == 0.75


class TestPerformanceResponse:
    """Tests for PerformanceResponse model."""
    
    def test_valid_performance_response(self):
        """Test valid performance response."""
        response = PerformanceResponse(
            model_version="1.0.0",
            metrics=[
                PerformanceMetricsItem(
                    timestamp=datetime.utcnow(),
                    rmse=45.2,
                    mae=32.1,
                    mape=0.025,
                    directional_accuracy=0.75,
                    coverage_rate=0.95,
                    mean_interval_width=200.0
                )
            ],
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow()
        )
        
        assert response.model_version == "1.0.0"
        assert len(response.metrics) == 1


class TestModelInfo:
    """Tests for ModelInfo model."""
    
    def test_valid_model_info(self):
        """Test valid model info."""
        info = ModelInfo(
            name="cocoa_prophet",
            version="1",
            stage="Production",
            created_at=datetime.utcnow(),
            metrics={"rmse": 45.2, "mae": 32.1}
        )
        
        assert info.name == "cocoa_prophet"
        assert info.version == "1"
        assert info.stage == "Production"
        assert info.metrics["rmse"] == 45.2


class TestModelsResponse:
    """Tests for ModelsResponse model."""
    
    def test_valid_models_response(self):
        """Test valid models response."""
        response = ModelsResponse(
            models=[
                ModelInfo(
                    name="cocoa_prophet",
                    version="1",
                    stage="Production",
                    created_at=datetime.utcnow()
                )
            ],
            current_production_version="cocoa_prophet:1"
        )
        
        assert len(response.models) == 1
        assert response.current_production_version == "cocoa_prophet:1"


class TestRetrainingRequest:
    """Tests for RetrainingRequest model."""
    
    def test_valid_retraining_request(self):
        """Test valid retraining request."""
        request = RetrainingRequest(
            model_type="all",
            reason="Performance degradation detected"
        )
        
        assert request.model_type == "all"
        assert request.reason == "Performance degradation detected"
    
    def test_retraining_request_specific_model(self):
        """Test retraining request for specific model."""
        request = RetrainingRequest(
            model_type="prophet",
            reason="Updating time series model"
        )
        
        assert request.model_type == "prophet"
    
    def test_retraining_request_invalid_model_type(self):
        """Test retraining request with invalid model type."""
        with pytest.raises(ValidationError):
            RetrainingRequest(
                model_type="invalid_model",
                reason="Testing"
            )
    
    def test_retraining_request_default_model_type(self):
        """Test retraining request with default model type."""
        request = RetrainingRequest(
            reason="Scheduled retraining"
        )
        
        assert request.model_type == "all"


class TestRetrainingResponse:
    """Tests for RetrainingResponse model."""
    
    def test_valid_retraining_response(self):
        """Test valid retraining response."""
        response = RetrainingResponse(
            status="accepted",
            message="Retraining job created",
            job_id="12345",
            estimated_completion=datetime.utcnow()
        )
        
        assert response.status == "accepted"
        assert response.job_id == "12345"


class TestErrorResponse:
    """Tests for ErrorResponse model."""
    
    def test_valid_error_response(self):
        """Test valid error response."""
        error = ErrorResponse(
            error="validation_error",
            message="Invalid request data",
            detail="Field 'horizons' cannot be empty"
        )
        
        assert error.error == "validation_error"
        assert error.message == "Invalid request data"
        assert error.detail == "Field 'horizons' cannot be empty"
    
    def test_error_response_without_detail(self):
        """Test error response without detail."""
        error = ErrorResponse(
            error="internal_error",
            message="An error occurred"
        )
        
        assert error.detail is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
