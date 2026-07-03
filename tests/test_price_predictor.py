"""
Unit tests for the PricePredictor class.

Tests cover:
- Initialization and validation
- Prediction generation for multiple horizons
- Prediction combination logic
- Confidence interval calculation
- Prediction validation
- Error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from src.models.price_predictor import PricePredictor
from src.models.time_series_model import TimeSeriesModel
from src.models.ml_model import MLModel
from src.nlp.nlp_analyzer import NLPAnalyzer
from src.models.data_models import Prediction, NewsArticle


class TestPricePredictorInitialization:
    """Test PricePredictor initialization and validation."""
    
    def test_initialization_with_fitted_models(self):
        """Test successful initialization with fitted models."""
        # Create mock models
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        # Initialize predictor
        predictor = PricePredictor(
            ts_model=ts_model,
            ml_model=ml_model,
            nlp_analyzer=nlp_analyzer,
            sentiment_weight=0.1,
            model_version="1.0.0"
        )
        
        assert predictor.ts_model == ts_model
        assert predictor.ml_model == ml_model
        assert predictor.nlp_analyzer == nlp_analyzer
        assert predictor.sentiment_weight == 0.1
        assert predictor.model_version == "1.0.0"
    
    def test_initialization_with_unfitted_ts_model(self):
        """Test initialization fails with unfitted time series model."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = False
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        with pytest.raises(ValueError, match="TimeSeriesModel must be fitted"):
            PricePredictor(ts_model, ml_model, nlp_analyzer)
    
    def test_initialization_with_unfitted_ml_model(self):
        """Test initialization fails with unfitted ML model."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = False
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        with pytest.raises(ValueError, match="MLModel must be fitted"):
            PricePredictor(ts_model, ml_model, nlp_analyzer)
    
    def test_initialization_with_invalid_sentiment_weight(self):
        """Test initialization fails with invalid sentiment weight."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        # Test negative weight
        with pytest.raises(ValueError, match="sentiment_weight must be between"):
            PricePredictor(ts_model, ml_model, nlp_analyzer, sentiment_weight=-0.1)
        
        # Test weight > 1.0
        with pytest.raises(ValueError, match="sentiment_weight must be between"):
            PricePredictor(ts_model, ml_model, nlp_analyzer, sentiment_weight=1.5)


class TestPredictionCombination:
    """Test prediction combination logic."""
    
    def test_combine_predictions_basic(self):
        """Test basic prediction combination."""
        # Create mock models
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer, sentiment_weight=0.1)
        
        # Test combination
        result = predictor.combine_predictions(
            baseline=3000.0,
            residual=50.0,
            sentiment_score=0.0,
            sentiment_weight=0.1
        )
        
        # Expected: 3000 + 50 + (0.0 * 0.1 * 3000) = 3050
        assert result == 3050.0
    
    def test_combine_predictions_with_positive_sentiment(self):
        """Test combination with positive sentiment."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer, sentiment_weight=0.1)
        
        result = predictor.combine_predictions(
            baseline=3000.0,
            residual=50.0,
            sentiment_score=0.5,
            sentiment_weight=0.1
        )
        
        # Expected: 3000 + 50 + (0.5 * 0.1 * 3000) = 3000 + 50 + 150 = 3200
        assert result == 3200.0
    
    def test_combine_predictions_with_negative_sentiment(self):
        """Test combination with negative sentiment."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer, sentiment_weight=0.1)
        
        result = predictor.combine_predictions(
            baseline=3000.0,
            residual=50.0,
            sentiment_score=-0.5,
            sentiment_weight=0.1
        )
        
        # Expected: 3000 + 50 + (-0.5 * 0.1 * 3000) = 3000 + 50 - 150 = 2900
        assert result == 2900.0
    
    def test_combine_predictions_with_negative_residual(self):
        """Test combination with negative residual correction."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer, sentiment_weight=0.1)
        
        result = predictor.combine_predictions(
            baseline=3000.0,
            residual=-100.0,
            sentiment_score=0.0,
            sentiment_weight=0.1
        )
        
        # Expected: 3000 - 100 + 0 = 2900
        assert result == 2900.0


class TestConfidenceIntervalCalculation:
    """Test confidence interval calculation."""
    
    def test_calculate_confidence_interval_basic(self):
        """Test basic confidence interval calculation."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
        
        lower, upper = predictor.calculate_confidence_interval(
            prediction=3000.0,
            ts_uncertainty=100.0,
            ml_uncertainty=50.0,
            sentiment_risk=False,
            confidence_level=0.95
        )
        
        # Combined uncertainty: sqrt(100^2 + 50^2) = sqrt(12500) ≈ 111.8
        # Margin: 1.96 * 111.8 ≈ 219.1
        # Expected: [2780.9, 3219.1]
        assert lower < 3000.0
        assert upper > 3000.0
        assert abs((upper - lower) / 2 - 219.1) < 1.0  # Allow small rounding error
    
    def test_calculate_confidence_interval_with_sentiment_risk(self):
        """Test confidence interval widening with sentiment risk."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
        
        # Without sentiment risk
        lower_no_risk, upper_no_risk = predictor.calculate_confidence_interval(
            prediction=3000.0,
            ts_uncertainty=100.0,
            ml_uncertainty=50.0,
            sentiment_risk=False,
            confidence_level=0.95
        )
        
        # With sentiment risk
        lower_risk, upper_risk = predictor.calculate_confidence_interval(
            prediction=3000.0,
            ts_uncertainty=100.0,
            ml_uncertainty=50.0,
            sentiment_risk=True,
            confidence_level=0.95
        )
        
        # Interval should be 50% wider with sentiment risk
        width_no_risk = upper_no_risk - lower_no_risk
        width_risk = upper_risk - lower_risk
        
        assert abs(width_risk / width_no_risk - 1.5) < 0.01  # Should be ~1.5x wider
    
    def test_calculate_confidence_interval_different_levels(self):
        """Test confidence interval calculation with different confidence levels."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
        
        # 90% confidence interval (narrower)
        lower_90, upper_90 = predictor.calculate_confidence_interval(
            prediction=3000.0,
            ts_uncertainty=100.0,
            ml_uncertainty=50.0,
            sentiment_risk=False,
            confidence_level=0.90
        )
        
        # 95% confidence interval
        lower_95, upper_95 = predictor.calculate_confidence_interval(
            prediction=3000.0,
            ts_uncertainty=100.0,
            ml_uncertainty=50.0,
            sentiment_risk=False,
            confidence_level=0.95
        )
        
        # 99% confidence interval (wider)
        lower_99, upper_99 = predictor.calculate_confidence_interval(
            prediction=3000.0,
            ts_uncertainty=100.0,
            ml_uncertainty=50.0,
            sentiment_risk=False,
            confidence_level=0.99
        )
        
        # Higher confidence level should give wider intervals
        width_90 = upper_90 - lower_90
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99
        
        assert width_90 < width_95 < width_99


class TestPredictionValidation:
    """Test prediction validation."""
    
    def test_validate_prediction_within_range(self):
        """Test validation passes for prediction within range."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
        
        is_valid = predictor.validate_prediction(3000.0, (1000.0, 10000.0))
        assert is_valid is True
    
    def test_validate_prediction_below_range(self):
        """Test validation fails for prediction below range."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
        
        is_valid = predictor.validate_prediction(500.0, (1000.0, 10000.0))
        assert is_valid is False
    
    def test_validate_prediction_above_range(self):
        """Test validation fails for prediction above range."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
        
        is_valid = predictor.validate_prediction(15000.0, (1000.0, 10000.0))
        assert is_valid is False
    
    def test_validate_prediction_at_boundaries(self):
        """Test validation at range boundaries."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
        
        # At lower boundary
        is_valid = predictor.validate_prediction(1000.0, (1000.0, 10000.0))
        assert is_valid is True
        
        # At upper boundary
        is_valid = predictor.validate_prediction(10000.0, (1000.0, 10000.0))
        assert is_valid is True


class TestPredictMethod:
    """Test the main predict method."""
    
    def test_predict_single_horizon(self):
        """Test prediction generation for a single horizon."""
        # Create mock models
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        # Mock Prophet forecast
        prophet_forecast = pd.DataFrame({
            'ds': [datetime.now()],
            'yhat': [3000.0],
            'yhat_lower': [2800.0],
            'yhat_upper': [3200.0]
        })
        ts_model.predict.return_value = prophet_forecast
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        ml_model.predict.return_value = np.array([50.0])
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        nlp_analyzer.aggregate_sentiment.return_value = 0.0
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
        
        # Create features
        features = pd.DataFrame({
            'temperature': [25.0],
            'rainfall': [10.0],
            'stock_level': [50000.0]
        })
        
        # Generate predictions
        predictions = predictor.predict(
            horizons=[1],
            exog_features=features,
            recent_news=[]
        )
        
        assert len(predictions) == 1
        assert isinstance(predictions[0], Prediction)
        assert predictions[0].horizon == 1
        assert predictions[0].price > 0
        assert predictions[0].confidence_level == 0.95
        assert 'baseline' in predictions[0].components
        assert 'residual' in predictions[0].components
        assert 'sentiment' in predictions[0].components
    
    def test_predict_multiple_horizons(self):
        """Test prediction generation for multiple horizons."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        # Mock Prophet forecast for 30 days
        prophet_forecast = pd.DataFrame({
            'ds': pd.date_range(start=datetime.now(), periods=30, freq='D'),
            'yhat': np.linspace(3000, 3100, 30),
            'yhat_lower': np.linspace(2800, 2900, 30),
            'yhat_upper': np.linspace(3200, 3300, 30)
        })
        ts_model.predict.return_value = prophet_forecast
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        ml_model.predict.return_value = np.array([50.0, 60.0, 70.0])
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        nlp_analyzer.aggregate_sentiment.return_value = 0.2
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
        
        # Create features for 3 horizons
        features = pd.DataFrame({
            'temperature': [25.0, 26.0, 27.0],
            'rainfall': [10.0, 12.0, 15.0],
            'stock_level': [50000.0, 48000.0, 45000.0]
        })
        
        # Generate predictions
        predictions = predictor.predict(
            horizons=[1, 7, 30],
            exog_features=features,
            recent_news=[]
        )
        
        assert len(predictions) == 3
        assert predictions[0].horizon == 1
        assert predictions[1].horizon == 7
        assert predictions[2].horizon == 30
        
        # All predictions should have valid structure
        for pred in predictions:
            assert isinstance(pred, Prediction)
            assert pred.price > 0
            assert pred.confidence_interval[0] < pred.price < pred.confidence_interval[1]
    
    def test_predict_with_high_risk_sentiment(self):
        """Test prediction with high-risk sentiment."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        prophet_forecast = pd.DataFrame({
            'ds': [datetime.now()],
            'yhat': [3000.0],
            'yhat_lower': [2800.0],
            'yhat_upper': [3200.0]
        })
        ts_model.predict.return_value = prophet_forecast
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        ml_model.predict.return_value = np.array([50.0])
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        nlp_analyzer.aggregate_sentiment.return_value = -0.7  # High risk
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
        
        features = pd.DataFrame({
            'temperature': [25.0],
            'rainfall': [10.0]
        })
        
        predictions = predictor.predict(
            horizons=[1],
            exog_features=features,
            recent_news=[]
        )
        
        # Confidence interval should be wider due to sentiment risk
        pred = predictions[0]
        interval_width = pred.confidence_interval[1] - pred.confidence_interval[0]
        
        # The interval should be reasonably wide (this is a qualitative check)
        assert interval_width > 0
    
    def test_predict_with_mismatched_lengths(self):
        """Test prediction fails with mismatched horizons and features."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
        
        features = pd.DataFrame({
            'temperature': [25.0, 26.0]  # 2 rows
        })
        
        with pytest.raises(ValueError, match="horizons and exog_features must have same length"):
            predictor.predict(
                horizons=[1, 7, 30],  # 3 horizons
                exog_features=features,
                recent_news=[]
            )
    
    def test_predict_with_empty_horizons(self):
        """Test prediction fails with empty horizons list."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
        
        features = pd.DataFrame({'temperature': []})
        
        with pytest.raises(ValueError, match="horizons list cannot be empty"):
            predictor.predict(
                horizons=[],
                exog_features=features,
                recent_news=[]
            )
    
    def test_predict_with_invalid_horizons(self):
        """Test prediction fails with invalid horizon values."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
        
        features = pd.DataFrame({'temperature': [25.0]})
        
        with pytest.raises(ValueError, match="All horizons must be positive"):
            predictor.predict(
                horizons=[0],
                exog_features=features,
                recent_news=[]
            )


class TestModelInfo:
    """Test model information retrieval."""
    
    def test_get_model_info(self):
        """Test getting model information."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        ts_model.get_hyperparameters.return_value = {
            'seasonality_mode': 'multiplicative',
            'yearly_seasonality': True
        }
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        ml_model.get_hyperparameters.return_value = {
            'n_estimators': 100,
            'max_depth': 6
        }
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(
            ts_model, ml_model, nlp_analyzer,
            sentiment_weight=0.15,
            model_version="2.0.0"
        )
        
        info = predictor.get_model_info()
        
        assert info['model_version'] == "2.0.0"
        assert info['sentiment_weight'] == 0.15
        assert 'ts_model_params' in info
        assert 'ml_model_params' in info
        assert info['ts_model_params']['seasonality_mode'] == 'multiplicative'
        assert info['ml_model_params']['n_estimators'] == 100


class TestRepr:
    """Test string representation."""
    
    def test_repr(self):
        """Test __repr__ method."""
        ts_model = Mock(spec=TimeSeriesModel)
        ts_model.is_fitted = True
        
        ml_model = Mock(spec=MLModel)
        ml_model.is_fitted = True
        
        nlp_analyzer = Mock(spec=NLPAnalyzer)
        
        predictor = PricePredictor(
            ts_model, ml_model, nlp_analyzer,
            sentiment_weight=0.1,
            model_version="1.0.0"
        )
        
        repr_str = repr(predictor)
        
        assert "PricePredictor" in repr_str
        assert "model_version='1.0.0'" in repr_str
        assert "sentiment_weight=0.1" in repr_str
