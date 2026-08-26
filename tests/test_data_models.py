"""
Unit tests for data models.

Tests validation logic, field constraints, and edge cases for all Pydantic models.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError as PydanticValidationError

from src.models.data_models import (
    PriceData,
    EconometricData,
    NewsArticle,
    Prediction,
    ModelMetrics,
    ValidationError,
)


class TestPriceData:
    """Tests for PriceData model."""
    
    def test_valid_price_data(self):
        """Test creating valid PriceData instance."""
        data = PriceData(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            market="ICE_London",
            price=3250.50,
            volume=1500.0,
            currency="USD"
        )
        assert data.price == 3250.50
        assert data.market == "ICE_London"
        assert data.currency == "USD"
    
    def test_valid_price_data_ice_ny(self):
        """Test creating valid PriceData instance for ICE_NY market."""
        data = PriceData(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            market="ICE_NY",
            price=3100.00,
            volume=2000.0,
            currency="USD"
        )
        assert data.market == "ICE_NY"
        assert data.price == 3100.00
    
    def test_price_boundary_minimum(self):
        """Test that price at minimum boundary (1000) is accepted."""
        data = PriceData(
            timestamp=datetime.now(timezone.utc),
            market="ICE_London",
            price=1000.0,  # Minimum boundary
            volume=1000.0,
            currency="USD"
        )
        assert data.price == 1000.0
    
    def test_price_boundary_maximum(self):
        """Test that price at maximum boundary (10000) is accepted."""
        data = PriceData(
            timestamp=datetime.now(timezone.utc),
            market="ICE_London",
            price=10000.0,  # Maximum boundary
            volume=1000.0,
            currency="USD"
        )
        assert data.price == 10000.0
    
    def test_price_out_of_range_low(self):
        """Test that price below 1000 is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            PriceData(
                timestamp=datetime.now(timezone.utc),
                market="ICE_London",
                price=500.0,  # Below minimum
                volume=1000.0,
                currency="USD"
            )
        assert "price" in str(exc_info.value)
    
    def test_price_out_of_range_high(self):
        """Test that price above 10000 is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            PriceData(
                timestamp=datetime.now(timezone.utc),
                market="ICE_London",
                price=15000.0,  # Above maximum
                volume=1000.0,
                currency="USD"
            )
        assert "price" in str(exc_info.value)
    
    def test_invalid_market(self):
        """Test that invalid market identifier is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            PriceData(
                timestamp=datetime.now(timezone.utc),
                market="INVALID_MARKET",
                price=3000.0,
                volume=1000.0,
                currency="USD"
            )
        assert "Market must be one of" in str(exc_info.value)
    
    def test_invalid_currency(self):
        """Test that invalid currency code is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            PriceData(
                timestamp=datetime.now(timezone.utc),
                market="ICE_London",
                price=3000.0,
                volume=1000.0,
                currency="JPY"  # Not in valid list
            )
        assert "Currency must be one of" in str(exc_info.value)
    
    def test_negative_volume(self):
        """Test that negative volume is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            PriceData(
                timestamp=datetime.now(timezone.utc),
                market="ICE_London",
                price=3000.0,
                volume=-100.0,
                currency="USD"
            )
        assert "volume" in str(exc_info.value)
    
    def test_zero_volume(self):
        """Test that zero volume is accepted."""
        data = PriceData(
            timestamp=datetime.now(timezone.utc),
            market="ICE_London",
            price=3000.0,
            volume=0.0,
            currency="USD"
        )
        assert data.volume == 0.0
    
    def test_currency_case_normalization(self):
        """Test that currency code is normalized to uppercase."""
        data = PriceData(
            timestamp=datetime.now(timezone.utc),
            market="ICE_London",
            price=3000.0,
            volume=1000.0,
            currency="usd"  # Lowercase
        )
        assert data.currency == "USD"  # Should be normalized to uppercase
    
    def test_missing_required_field(self):
        """Test that missing required field is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            PriceData(
                timestamp=datetime.now(timezone.utc),
                market="ICE_London",
                # price is missing
                volume=1000.0,
                currency="USD"
            )
        assert "price" in str(exc_info.value)


class TestEconometricData:
    """Tests for EconometricData model."""
    
    def test_valid_econometric_data(self):
        """Test creating valid EconometricData instance."""
        data = EconometricData(
            timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            temperature=25.5,
            rainfall=10.2,
            stock_level=50000.0,
            production=10000.0,
            fx_rate_xaf_usd=0.0017,
            fx_rate_gbp_usd=1.27,
            fx_rate_eur_usd=1.10
        )
        assert data.temperature == 25.5
        assert data.fx_rate_xaf_usd == 0.0017
    
    def test_temperature_out_of_range_low(self):
        """Test that temperature below -10°C is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            EconometricData(
                timestamp=datetime.now(timezone.utc),
                temperature=-15.0,  # Below minimum
                rainfall=10.0,
                stock_level=1000.0,
                production=500.0
            )
        assert "temperature" in str(exc_info.value)
    
    def test_temperature_out_of_range_high(self):
        """Test that temperature above 50°C is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            EconometricData(
                timestamp=datetime.now(timezone.utc),
                temperature=60.0,  # Above maximum
                rainfall=10.0,
                stock_level=1000.0,
                production=500.0
            )
        assert "temperature" in str(exc_info.value)
    
    def test_rainfall_negative(self):
        """Test that negative rainfall is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            EconometricData(
                timestamp=datetime.now(timezone.utc),
                temperature=25.0,
                rainfall=-5.0,  # Negative
                stock_level=1000.0,
                production=500.0
            )
        assert "rainfall" in str(exc_info.value)
    
    def test_rainfall_out_of_range_high(self):
        """Test that rainfall above 500mm is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            EconometricData(
                timestamp=datetime.now(timezone.utc),
                temperature=25.0,
                rainfall=600.0,  # Above maximum
                stock_level=1000.0,
                production=500.0
            )
        assert "rainfall" in str(exc_info.value)
    
    def test_negative_fx_rate(self):
        """Test that negative FX rate is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            EconometricData(
                timestamp=datetime.now(timezone.utc),
                temperature=25.0,
                rainfall=10.0,
                stock_level=1000.0,
                production=500.0,
                fx_rate_xaf_usd=-0.001  # Negative rate
            )
        assert "fx_rate_xaf_usd" in str(exc_info.value)
    
    def test_zero_fx_rate(self):
        """Test that zero FX rate is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            EconometricData(
                timestamp=datetime.now(timezone.utc),
                temperature=25.0,
                rainfall=10.0,
                stock_level=1000.0,
                production=500.0,
                fx_rate_gbp_usd=0.0  # Zero rate
            )
        assert "fx_rate_gbp_usd" in str(exc_info.value)
    
    def test_optional_fields(self):
        """Test that optional fields can be None."""
        data = EconometricData(
            timestamp=datetime.now(timezone.utc),
            temperature=None,
            rainfall=None,
            stock_level=None,
            production=None
        )
        assert data.temperature is None
        assert data.rainfall is None
    
    def test_negative_stock_level(self):
        """Test that negative stock level is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            EconometricData(
                timestamp=datetime.now(timezone.utc),
                temperature=25.0,
                rainfall=10.0,
                stock_level=-1000.0,  # Negative
                production=500.0
            )
        assert "stock_level" in str(exc_info.value)
    
    def test_negative_production(self):
        """Test that negative production is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            EconometricData(
                timestamp=datetime.now(timezone.utc),
                temperature=25.0,
                rainfall=10.0,
                stock_level=1000.0,
                production=-500.0  # Negative
            )
        assert "production" in str(exc_info.value)
    
    def test_temperature_boundary_minimum(self):
        """Test that temperature at minimum boundary (-10°C) is accepted."""
        data = EconometricData(
            timestamp=datetime.now(timezone.utc),
            temperature=-10.0,  # Minimum boundary
            rainfall=10.0,
            stock_level=1000.0,
            production=500.0
        )
        assert data.temperature == -10.0
    
    def test_temperature_boundary_maximum(self):
        """Test that temperature at maximum boundary (50°C) is accepted."""
        data = EconometricData(
            timestamp=datetime.now(timezone.utc),
            temperature=50.0,  # Maximum boundary
            rainfall=10.0,
            stock_level=1000.0,
            production=500.0
        )
        assert data.temperature == 50.0
    
    def test_rainfall_boundary_minimum(self):
        """Test that rainfall at minimum boundary (0mm) is accepted."""
        data = EconometricData(
            timestamp=datetime.now(timezone.utc),
            temperature=25.0,
            rainfall=0.0,  # Minimum boundary
            stock_level=1000.0,
            production=500.0
        )
        assert data.rainfall == 0.0
    
    def test_rainfall_boundary_maximum(self):
        """Test that rainfall at maximum boundary (500mm) is accepted."""
        data = EconometricData(
            timestamp=datetime.now(timezone.utc),
            temperature=25.0,
            rainfall=500.0,  # Maximum boundary
            stock_level=1000.0,
            production=500.0
        )
        assert data.rainfall == 500.0
    
    def test_all_fx_rates_valid(self):
        """Test that all FX rates can be set with valid positive values."""
        data = EconometricData(
            timestamp=datetime.now(timezone.utc),
            temperature=25.0,
            rainfall=10.0,
            stock_level=1000.0,
            production=500.0,
            fx_rate_xaf_usd=0.0017,
            fx_rate_gbp_usd=1.27,
            fx_rate_eur_usd=1.10
        )
        assert data.fx_rate_xaf_usd == 0.0017
        assert data.fx_rate_gbp_usd == 1.27
        assert data.fx_rate_eur_usd == 1.10


class TestNewsArticle:
    """Tests for NewsArticle model."""
    
    def test_valid_news_article(self):
        """Test creating valid NewsArticle instance."""
        article = NewsArticle(
            id="article_123",
            source="reuters",
            title="Cocoa prices surge on supply concerns",
            content="Full article content here...",
            published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            url="https://reuters.com/article/123",
            keywords=["cocoa", "supply", "prices"],
            sentiment_score=-0.3,
            is_high_risk=False
        )
        assert article.source == "reuters"
        assert article.sentiment_score == -0.3
    
    def test_sentiment_score_out_of_range_low(self):
        """Test that sentiment score below -1 is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            NewsArticle(
                id="article_123",
                source="reuters",
                title="Test article",
                content="Content",
                published_at=datetime.now(timezone.utc),
                url="https://example.com",
                sentiment_score=-1.5  # Below minimum
            )
        assert "sentiment_score" in str(exc_info.value)
    
    def test_sentiment_score_out_of_range_high(self):
        """Test that sentiment score above +1 is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            NewsArticle(
                id="article_123",
                source="reuters",
                title="Test article",
                content="Content",
                published_at=datetime.now(timezone.utc),
                url="https://example.com",
                sentiment_score=1.5  # Above maximum
            )
        assert "sentiment_score" in str(exc_info.value)
    
    def test_arbitrary_source_accepted(self):
        """Test that source is a free-form string (any value accepted)."""
        article = NewsArticle(
            id="article_123",
            source="invalid_source",
            title="Test article",
            content="Content",
            published_at=datetime.now(timezone.utc),
            url="https://example.com"
        )
        assert article.source == "invalid_source"
    
    def test_empty_title(self):
        """Test that empty title is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            NewsArticle(
                id="article_123",
                source="reuters",
                title="",  # Empty
                content="Content",
                published_at=datetime.now(timezone.utc),
                url="https://example.com"
            )
        assert "title" in str(exc_info.value)
    
    def test_empty_content_allowed(self):
        """Test that empty content is allowed (defaults to empty string)."""
        article = NewsArticle(
            id="article_123",
            source="reuters",
            title="Test article",
            content="",
            published_at=datetime.now(timezone.utc),
            url="https://example.com"
        )
        assert article.content == ""
    
    def test_sentiment_score_boundary_minimum(self):
        """Test that sentiment score at minimum boundary (-1) is accepted."""
        article = NewsArticle(
            id="article_123",
            source="reuters",
            title="Test article",
            content="Content",
            published_at=datetime.now(timezone.utc),
            url="https://example.com",
            sentiment_score=-1.0  # Minimum boundary
        )
        assert article.sentiment_score == -1.0
    
    def test_sentiment_score_boundary_maximum(self):
        """Test that sentiment score at maximum boundary (+1) is accepted."""
        article = NewsArticle(
            id="article_123",
            source="reuters",
            title="Test article",
            content="Content",
            published_at=datetime.now(timezone.utc),
            url="https://example.com",
            sentiment_score=1.0  # Maximum boundary
        )
        assert article.sentiment_score == 1.0
    
    def test_source_case_preserved(self):
        """Test that source case is preserved (no forced lowercase)."""
        article = NewsArticle(
            id="article_123",
            source="REUTERS",
            title="Test article",
            content="Content",
            published_at=datetime.now(timezone.utc),
            url="https://example.com"
        )
        assert article.source == "REUTERS"
    
    def test_optional_sentiment_fields(self):
        """Test that sentiment_score and is_high_risk can be None."""
        article = NewsArticle(
            id="article_123",
            source="reuters",
            title="Test article",
            content="Content",
            published_at=datetime.now(timezone.utc),
            url="https://example.com"
        )
        assert article.sentiment_score is None
        assert article.is_high_risk is None
    
    def test_keywords_default_empty_list(self):
        """Test that keywords defaults to empty list."""
        article = NewsArticle(
            id="article_123",
            source="reuters",
            title="Test article",
            content="Content",
            published_at=datetime.now(timezone.utc),
            url="https://example.com"
        )
        assert article.keywords == []


class TestPrediction:
    """Tests for Prediction model."""
    
    def test_valid_prediction(self):
        """Test creating valid Prediction instance."""
        pred = Prediction(
            horizon=7,
            price=3250.0,
            confidence_interval=(3100.0, 3400.0),
            confidence_level=0.95,
            timestamp=datetime.now(timezone.utc),
            model_version="v1.0.0",
            components={"baseline": 3200.0, "residual": 30.0, "sentiment": 20.0}
        )
        assert pred.horizon == 7
        assert pred.confidence_interval == (3100.0, 3400.0)
    
    def test_invalid_confidence_interval_order(self):
        """Test that reversed confidence interval bounds are rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            Prediction(
                horizon=7,
                price=3250.0,
                confidence_interval=(3400.0, 3100.0),  # Reversed
                confidence_level=0.95,
                timestamp=datetime.now(timezone.utc),
                model_version="v1.0.0",
                components={"baseline": 3200.0, "residual": 30.0, "sentiment": 20.0}
            )
        assert "Lower bound must be less than upper bound" in str(exc_info.value)
    
    def test_confidence_interval_out_of_range(self):
        """Test that confidence interval outside valid price range is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            Prediction(
                horizon=7,
                price=3250.0,
                confidence_interval=(500.0, 3400.0),  # Lower bound too low
                confidence_level=0.95,
                timestamp=datetime.now(timezone.utc),
                model_version="v1.0.0",
                components={"baseline": 3200.0, "residual": 30.0, "sentiment": 20.0}
            )
        assert "Confidence interval bounds must be within" in str(exc_info.value)
    
    def test_missing_components(self):
        """Test that missing required components are rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            Prediction(
                horizon=7,
                price=3250.0,
                confidence_interval=(3100.0, 3400.0),
                confidence_level=0.95,
                timestamp=datetime.now(timezone.utc),
                model_version="v1.0.0",
                components={"baseline": 3200.0}  # Missing residual and sentiment
            )
        assert "Missing required components" in str(exc_info.value)
    
    def test_confidence_level_out_of_range(self):
        """Test that confidence level outside [0, 1] is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            Prediction(
                horizon=7,
                price=3250.0,
                confidence_interval=(3100.0, 3400.0),
                confidence_level=1.5,  # Above maximum
                timestamp=datetime.now(timezone.utc),
                model_version="v1.0.0",
                components={"baseline": 3200.0, "residual": 30.0, "sentiment": 20.0}
            )
        assert "confidence_level" in str(exc_info.value)
    
    def test_negative_horizon(self):
        """Test that negative horizon is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            Prediction(
                horizon=-1,  # Negative
                price=3250.0,
                confidence_interval=(3100.0, 3400.0),
                confidence_level=0.95,
                timestamp=datetime.now(timezone.utc),
                model_version="v1.0.0",
                components={"baseline": 3200.0, "residual": 30.0, "sentiment": 20.0}
            )
        assert "horizon" in str(exc_info.value)
    
    def test_zero_horizon(self):
        """Test that zero horizon is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            Prediction(
                horizon=0,  # Zero
                price=3250.0,
                confidence_interval=(3100.0, 3400.0),
                confidence_level=0.95,
                timestamp=datetime.now(timezone.utc),
                model_version="v1.0.0",
                components={"baseline": 3200.0, "residual": 30.0, "sentiment": 20.0}
            )
        assert "horizon" in str(exc_info.value)
    
    def test_price_out_of_range(self):
        """Test that predicted price outside valid range (500-15000) is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            Prediction(
                horizon=7,
                price=100.0,  # Below minimum (ge=500)
                confidence_interval=(1000.0, 2000.0),
                confidence_level=0.95,
                timestamp=datetime.now(timezone.utc),
                model_version="v1.0.0",
                components={"baseline": 100.0, "residual": 0.0, "sentiment": 0.0}
            )
        assert "price" in str(exc_info.value)
    
    def test_empty_components_dict(self):
        """Test that empty components dict is accepted."""
        pred = Prediction(
            horizon=7,
            price=3250.0,
            confidence_interval=(3100.0, 3400.0),
            confidence_level=0.95,
            timestamp=datetime.now(timezone.utc),
            model_version="v1.0.0",
            components={}  # Empty dict
        )
        assert pred.components == {}
    
    def test_confidence_interval_equal_bounds(self):
        """Test that confidence interval with equal bounds is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            Prediction(
                horizon=7,
                price=3250.0,
                confidence_interval=(3250.0, 3250.0),  # Equal bounds
                confidence_level=0.95,
                timestamp=datetime.now(timezone.utc),
                model_version="v1.0.0",
                components={"baseline": 3200.0, "residual": 30.0, "sentiment": 20.0}
            )
        assert "Lower bound must be less than upper bound" in str(exc_info.value)


class TestModelMetrics:
    """Tests for ModelMetrics model."""
    
    def test_valid_model_metrics(self):
        """Test creating valid ModelMetrics instance."""
        metrics = ModelMetrics(
            rmse=150.5,
            mae=120.3,
            mape=0.045,
            directional_accuracy=0.75,
            coverage_rate=0.95,
            mean_interval_width=300.0,
            timestamp=datetime.now(timezone.utc),
            model_version="v1.0.0"
        )
        assert metrics.rmse == 150.5
        assert metrics.directional_accuracy == 0.75
    
    def test_negative_rmse(self):
        """Test that negative RMSE is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            ModelMetrics(
                rmse=-10.0,  # Negative
                mae=120.3,
                mape=0.045,
                directional_accuracy=0.75,
                coverage_rate=0.95,
                mean_interval_width=300.0,
                timestamp=datetime.now(timezone.utc),
                model_version="v1.0.0"
            )
        assert "rmse" in str(exc_info.value)
    
    def test_directional_accuracy_out_of_range(self):
        """Test that directional accuracy outside [0, 1] is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            ModelMetrics(
                rmse=150.5,
                mae=120.3,
                mape=0.045,
                directional_accuracy=1.5,  # Above maximum
                coverage_rate=0.95,
                mean_interval_width=300.0,
                timestamp=datetime.now(timezone.utc),
                model_version="v1.0.0"
            )
        assert "directional_accuracy" in str(exc_info.value)
    
    def test_negative_mae(self):
        """Test that negative MAE is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            ModelMetrics(
                rmse=150.5,
                mae=-120.3,  # Negative
                mape=0.045,
                directional_accuracy=0.75,
                coverage_rate=0.95,
                mean_interval_width=300.0,
                timestamp=datetime.now(timezone.utc),
                model_version="v1.0.0"
            )
        assert "mae" in str(exc_info.value)
    
    def test_negative_mape(self):
        """Test that negative MAPE is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            ModelMetrics(
                rmse=150.5,
                mae=120.3,
                mape=-0.045,  # Negative
                directional_accuracy=0.75,
                coverage_rate=0.95,
                mean_interval_width=300.0,
                timestamp=datetime.now(timezone.utc),
                model_version="v1.0.0"
            )
        assert "mape" in str(exc_info.value)
    
    def test_coverage_rate_out_of_range(self):
        """Test that coverage rate outside [0, 1] is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            ModelMetrics(
                rmse=150.5,
                mae=120.3,
                mape=0.045,
                directional_accuracy=0.75,
                coverage_rate=1.5,  # Above maximum
                mean_interval_width=300.0,
                timestamp=datetime.now(timezone.utc),
                model_version="v1.0.0"
            )
        assert "coverage_rate" in str(exc_info.value)
    
    def test_negative_mean_interval_width(self):
        """Test that negative mean interval width is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            ModelMetrics(
                rmse=150.5,
                mae=120.3,
                mape=0.045,
                directional_accuracy=0.75,
                coverage_rate=0.95,
                mean_interval_width=-300.0,  # Negative
                timestamp=datetime.now(timezone.utc),
                model_version="v1.0.0"
            )
        assert "mean_interval_width" in str(exc_info.value)
    
    def test_zero_metrics_valid(self):
        """Test that zero values for metrics are accepted where valid."""
        metrics = ModelMetrics(
            rmse=0.0,
            mae=0.0,
            mape=0.0,
            directional_accuracy=0.0,
            coverage_rate=0.0,
            mean_interval_width=0.0,
            timestamp=datetime.now(timezone.utc),
            model_version="v1.0.0"
        )
        assert metrics.rmse == 0.0
        assert metrics.directional_accuracy == 0.0


class TestValidationError:
    """Tests for ValidationError model."""
    
    def test_valid_validation_error(self):
        """Test creating valid ValidationError instance."""
        error = ValidationError(
            field="price",
            value="15000.0",
            error_type="out_of_range",
            message="Price exceeds maximum allowed value",
            severity="ERROR"
        )
        assert error.field == "price"
        assert error.severity == "ERROR"
    
    def test_invalid_error_type(self):
        """Test that invalid error type is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            ValidationError(
                field="price",
                value="15000.0",
                error_type="invalid_type",  # Not in valid list
                message="Error message",
                severity="ERROR"
            )
        assert "Error type must be one of" in str(exc_info.value)
    
    def test_invalid_severity(self):
        """Test that invalid severity level is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            ValidationError(
                field="price",
                value="15000.0",
                error_type="out_of_range",
                message="Error message",
                severity="INVALID"  # Not in valid list
            )
        assert "Severity must be one of" in str(exc_info.value)
    
    def test_severity_case_normalization(self):
        """Test that severity is normalized to uppercase."""
        error = ValidationError(
            field="price",
            value="15000.0",
            error_type="out_of_range",
            message="Error message",
            severity="error"  # Lowercase
        )
        assert error.severity == "ERROR"  # Should be normalized to uppercase
    
    def test_empty_message(self):
        """Test that empty message is rejected."""
        with pytest.raises(PydanticValidationError) as exc_info:
            ValidationError(
                field="price",
                value="15000.0",
                error_type="out_of_range",
                message="",  # Empty
                severity="ERROR"
            )
        assert "message" in str(exc_info.value)
    
    def test_optional_value_field(self):
        """Test that value field can be None."""
        error = ValidationError(
            field="price",
            value=None,  # Optional
            error_type="missing",
            message="Price field is missing",
            severity="ERROR"
        )
        assert error.value is None
    
    def test_all_error_types_valid(self):
        """Test that all valid error types are accepted."""
        error_types = ["out_of_range", "missing", "duplicate", "invalid_format", "constraint_violation"]
        for error_type in error_types:
            error = ValidationError(
                field="test_field",
                value="test_value",
                error_type=error_type,
                message="Test message",
                severity="ERROR"
            )
            assert error.error_type == error_type
    
    def test_all_severity_levels_valid(self):
        """Test that all valid severity levels are accepted."""
        severity_levels = ["INFO", "WARNING", "ERROR", "CRITICAL"]
        for severity in severity_levels:
            error = ValidationError(
                field="test_field",
                value="test_value",
                error_type="out_of_range",
                message="Test message",
                severity=severity
            )
            assert error.severity == severity
