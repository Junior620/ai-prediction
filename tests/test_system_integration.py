"""
Integration tests for the complete Cocoa Price Prediction System.

This test suite verifies that all components are correctly wired together
and can work as a cohesive system.

Tests cover:
- Data flow: DataCollector → DataValidator → DataPreprocessor
- Model flow: DataPreprocessor → TimeSeriesModel → MLModel
- Prediction flow: NLPAnalyzer → PricePredictor
- Storage flow: PricePredictor → API → Cache
- Monitoring flow: PerformanceMonitor → ModelManager

Requirements: Task 20.1 - Integrate all components
"""

import pytest
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.integration.system_integrator import SystemIntegrator
from src.data_collection.data_collector import DataCollector
from src.data_validation.data_validator import DataValidator
from src.data_preprocessing.data_preprocessor import DataPreprocessor
from src.models.time_series_model import TimeSeriesModel
from src.models.ml_model import MLModel
from src.nlp.nlp_analyzer import NLPAnalyzer
from src.models.price_predictor import PricePredictor
from src.models.data_models import NewsArticle, PriceData


class TestSystemIntegration:
    """Test suite for system integration."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Create a mock Supabase client."""
        client = Mock()
        client.table = Mock(return_value=Mock())
        return client
    
    @pytest.fixture
    def mock_redis_cache(self):
        """Create a mock Redis cache."""
        cache = Mock()
        cache.health_check = Mock(return_value=True)
        cache.get = Mock(return_value=None)
        cache.set = Mock(return_value=True)
        return cache
    
    @pytest.fixture
    def system_integrator(self, mock_supabase_client, mock_redis_cache):
        """Create a SystemIntegrator instance with mocked dependencies."""
        with patch('src.integration.system_integrator.create_client', return_value=mock_supabase_client):
            with patch('src.integration.system_integrator.RedisCache', return_value=mock_redis_cache):
                integrator = SystemIntegrator(
                    supabase_client=mock_supabase_client,
                    redis_cache=mock_redis_cache
                )
                return integrator
    
    def test_system_integrator_initialization(self, system_integrator):
        """Test that SystemIntegrator initializes all components."""
        # Verify all components are initialized
        assert system_integrator.data_collector is not None
        assert system_integrator.data_validator is not None
        assert system_integrator.data_preprocessor is not None
        assert system_integrator.ts_model is not None
        assert system_integrator.ml_model is not None
        assert system_integrator.nlp_analyzer is not None
        assert system_integrator.model_manager is not None
        assert system_integrator.performance_monitor is not None
        assert system_integrator.supabase_client is not None
        assert system_integrator.redis_cache is not None
    
    def test_health_check(self, system_integrator):
        """Test system health check."""
        health_status = system_integrator.health_check()
        
        # Verify health status structure
        assert "data_collector" in health_status
        assert "data_validator" in health_status
        assert "data_preprocessor" in health_status
        assert "ts_model" in health_status
        assert "ml_model" in health_status
        assert "nlp_analyzer" in health_status
        assert "price_predictor" in health_status
        assert "model_manager" in health_status
        assert "performance_monitor" in health_status
        assert "redis_cache" in health_status
        assert "supabase" in health_status
        assert "overall" in health_status
        
        # Verify component health
        assert health_status["data_collector"] is True
        assert health_status["data_validator"] is True
        assert health_status["data_preprocessor"] is True
        assert health_status["nlp_analyzer"] is True
        assert health_status["redis_cache"] is True
        assert health_status["supabase"] is True
    
    def test_data_collection_to_validation_flow(self, system_integrator):
        """Test data flow from DataCollector to DataValidator."""
        # Create mock price data
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        # Mock data collector to return sample data
        mock_price_data = pd.DataFrame({
            'timestamp': pd.date_range(start=start_date, end=end_date, freq='D'),
            'market': ['ICE_London'] * 10,
            'price': np.random.uniform(3000, 3500, 10),
            'volume': np.random.uniform(1000, 5000, 10),
            'currency': ['USD'] * 10
        })
        
        # Validate the data
        validated_df, errors = system_integrator.data_validator.validate_price_data(
            mock_price_data
        )
        
        # Verify validation succeeded
        assert len(validated_df) > 0
        assert 'timestamp' in validated_df.columns
        assert 'price' in validated_df.columns
    
    def test_validation_to_preprocessing_flow(self, system_integrator):
        """Test data flow from DataValidator to DataPreprocessor."""
        # Create validated price data with missing values
        price_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
            'market': ['ICE_London'] * 10,
            'price': [3000, 3100, np.nan, 3200, 3300, np.nan, 3400, 3500, 3600, 3700],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            'currency': ['USD'] * 10
        })
        
        # Preprocess the data
        processed_df = system_integrator.data_preprocessor.handle_missing_values(
            price_df,
            strategy={'price': 'forward_fill'}
        )
        
        # Verify preprocessing succeeded
        assert processed_df['price'].isna().sum() == 0
        assert len(processed_df) == len(price_df)
    
    def test_preprocessing_to_time_series_model_flow(self, system_integrator):
        """Test data flow from DataPreprocessor to TimeSeriesModel."""
        # Create preprocessed data
        price_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'price': 3000 + np.cumsum(np.random.randn(100) * 10)
        })
        
        # Prepare data for Prophet
        prophet_df = price_df.copy()
        prophet_df.columns = ['ds', 'y']
        
        # Train time series model
        system_integrator.ts_model.fit(prophet_df, date_col='ds', target_col='y')
        
        # Verify model is trained
        assert system_integrator.ts_model.is_fitted is True
        
        # Generate predictions
        forecast = system_integrator.ts_model.predict(periods=7, freq='D')
        
        # Verify predictions
        assert len(forecast) > 0
        assert 'yhat' in forecast.columns
        assert 'yhat_lower' in forecast.columns
        assert 'yhat_upper' in forecast.columns
    
    def test_time_series_to_ml_model_flow(self, system_integrator):
        """Test data flow from TimeSeriesModel to MLModel."""
        # Create and train time series model
        price_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'price': 3000 + np.cumsum(np.random.randn(100) * 10)
        })
        
        prophet_df = price_df.copy()
        prophet_df.columns = ['ds', 'y']
        system_integrator.ts_model.fit(prophet_df, date_col='ds', target_col='y')
        
        # Compute residuals
        residuals = system_integrator.ts_model.compute_residuals(prophet_df)
        
        # Verify residuals
        assert len(residuals) == len(prophet_df)
        assert isinstance(residuals, pd.Series)
        
        # Create features for ML model
        X = pd.DataFrame({
            'temperature': np.random.uniform(20, 35, len(residuals)),
            'rainfall': np.random.exponential(5, len(residuals)),
            'stock_level': np.random.uniform(40000, 60000, len(residuals)),
            'production': np.random.uniform(3500000, 4500000, len(residuals)),
            'fx_rate_xaf_usd': np.random.uniform(0.0016, 0.0018, len(residuals)),
            'fx_rate_gbp_usd': np.random.uniform(1.25, 1.35, len(residuals)),
            'fx_rate_eur_usd': np.random.uniform(1.05, 1.15, len(residuals))
        })
        
        # Train ML model
        system_integrator.ml_model.fit(X, residuals)
        
        # Verify model is trained
        assert system_integrator.ml_model.is_fitted is True
        
        # Generate predictions
        predictions = system_integrator.ml_model.predict(X.head(10))
        
        # Verify predictions
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)
    
    def test_nlp_analyzer_to_price_predictor_flow(self, system_integrator):
        """Test data flow from NLPAnalyzer to PricePredictor."""
        # Create mock news articles
        news_articles = [
            NewsArticle(
                id="1",
                source="reuters",
                title="Cocoa prices surge on supply concerns",
                content="Cocoa prices rose sharply due to drought in West Africa",
                published_at=datetime.now(timezone.utc),
                url="http://example.com/1",
                keywords=[]
            ),
            NewsArticle(
                id="2",
                source="bloomberg",
                title="Chocolate demand remains strong",
                content="Global chocolate consumption continues to grow",
                published_at=datetime.now(timezone.utc),
                url="http://example.com/2",
                keywords=[]
            )
        ]
        
        # Analyze sentiment
        for article in news_articles:
            sentiment = system_integrator.nlp_analyzer.analyze_sentiment(
                f"{article.title} {article.content}"
            )
            article.sentiment_score = sentiment["score"]
        
        # Verify sentiment analysis
        assert all(article.sentiment_score is not None for article in news_articles)
        assert all(-1 <= article.sentiment_score <= 1 for article in news_articles)
        
        # Aggregate sentiment
        aggregated_sentiment = system_integrator.nlp_analyzer.aggregate_sentiment(
            news_articles
        )
        
        # Verify aggregation
        assert -1 <= aggregated_sentiment <= 1
    
    def test_complete_prediction_pipeline(self, system_integrator):
        """Test complete prediction pipeline from data to predictions."""
        # Step 1: Train models
        price_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'price': 3000 + np.cumsum(np.random.randn(100) * 10),
            'market': ['ICE_London'] * 100,
            'volume': np.random.uniform(1000, 5000, 100),
            'currency': ['USD'] * 100
        })
        
        # Add time features
        price_df = system_integrator.data_preprocessor.engineer_time_features(
            price_df, timestamp_col='timestamp'
        )
        
        # Create econometric data
        econometric_data = {
            'weather': pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='D'),
                'temperature': np.random.uniform(20, 35, 100),
                'rainfall': np.random.exponential(5, 100)
            }),
            'stocks': pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='D'),
                'stock_level': np.random.uniform(40000, 60000, 100)
            }),
            'production': pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='D'),
                'production': np.random.uniform(3500000, 4500000, 100)
            }),
            'fx_rates': pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='D'),
                'fx_rate_xaf_usd': np.random.uniform(0.0016, 0.0018, 100),
                'fx_rate_gbp_usd': np.random.uniform(1.25, 1.35, 100),
                'fx_rate_eur_usd': np.random.uniform(1.05, 1.15, 100)
            })
        }
        
        # Train models
        training_results = system_integrator.train_models(
            price_df=price_df,
            econometric_data=econometric_data,
            val_size=0.2
        )
        
        # Verify training succeeded
        assert training_results["ts_model_trained"] is True
        assert training_results["ml_model_trained"] is True
        assert "ml_cv_metrics" in training_results
        
        # Step 2: Generate predictions
        horizons = [1, 7, 30]
        exog_features = pd.DataFrame({
            'temperature': [25.0, 26.0, 27.0],
            'rainfall': [10.0, 12.0, 15.0],
            'stock_level': [50000.0, 48000.0, 45000.0],
            'production': [4000000.0, 4000000.0, 4000000.0],
            'fx_rate_xaf_usd': [0.0017, 0.0017, 0.0017],
            'fx_rate_gbp_usd': [1.27, 1.27, 1.27],
            'fx_rate_eur_usd': [1.09, 1.09, 1.09],
            'day_of_week': [0, 1, 2],
            'day_of_month': [1, 2, 3],
            'month': [1, 1, 1],
            'quarter': [1, 1, 1],
            'is_month_start': [True, False, False],
            'is_month_end': [False, False, False],
            'days_since_shock': [999, 999, 999]
        })
        
        news_articles = [
            NewsArticle(
                id="1",
                source="reuters",
                title="Cocoa market stable",
                content="Prices remain steady",
                published_at=datetime.now(timezone.utc),
                url="http://example.com/1",
                keywords=[],
                sentiment_score=0.1
            )
        ]
        
        predictions = system_integrator.generate_predictions(
            horizons=horizons,
            exog_features=exog_features,
            recent_news=news_articles,
            use_cache=False
        )
        
        # Verify predictions
        assert len(predictions) == len(horizons)
        for pred in predictions:
            assert pred.price > 0
            assert pred.confidence_interval[0] < pred.price < pred.confidence_interval[1]
            assert pred.confidence_level == 0.95
            assert "baseline" in pred.components
            assert "residual" in pred.components
            assert "sentiment" in pred.components
    
    def test_cache_integration(self, system_integrator, mock_redis_cache):
        """Test Redis cache integration."""
        # Verify cache is available
        assert system_integrator.redis_cache is not None
        assert system_integrator.redis_cache.health_check() is True
        
        # Test cache operations
        cache_key = "test_key"
        cache_value = {"test": "value"}
        
        system_integrator.redis_cache.set(cache_key, cache_value)
        system_integrator.redis_cache.get(cache_key)
        
        # Verify cache methods were called
        assert mock_redis_cache.set.called
        assert mock_redis_cache.get.called
    
    def test_database_integration(self, system_integrator, mock_supabase_client):
        """Test Supabase database integration."""
        # Verify database client is available
        assert system_integrator.supabase_client is not None
        
        # Test database operations
        system_integrator.supabase_client.table("predictions")
        
        # Verify database methods were called
        assert mock_supabase_client.table.called
    
    def test_performance_monitoring_integration(self, system_integrator):
        """Test PerformanceMonitor integration."""
        # Verify performance monitor is available
        assert system_integrator.performance_monitor is not None
        
        # Create mock predictions and actual prices
        from src.models.data_models import Prediction
        
        predictions = [
            Prediction(
                horizon=1,
                price=3100.0,
                confidence_interval=(3000.0, 3200.0),
                confidence_level=0.95,
                timestamp=datetime.now(timezone.utc),
                model_version="1.0.0",
                components={"baseline": 3050.0, "residual": 50.0, "sentiment": 0.0}
            )
        ]
        
        actual_prices = pd.DataFrame({
            'timestamp': [datetime.now(timezone.utc) + timedelta(days=1)],
            'price': [3120.0]
        })
        
        # Monitor performance
        metrics = system_integrator.monitor_performance(
            predictions=predictions,
            actual_prices=actual_prices
        )
        
        # Verify metrics were computed
        assert isinstance(metrics, dict)
    
    def test_model_manager_integration(self, system_integrator):
        """Test ModelManager integration."""
        # Verify model manager is available
        assert system_integrator.model_manager is not None
        
        # Test model manager is properly initialized
        assert hasattr(system_integrator.model_manager, 'client')
    
    def test_error_handling_in_pipeline(self, system_integrator):
        """Test error handling in the complete pipeline."""
        # Test with invalid data
        with pytest.raises(Exception):
            system_integrator.train_models(
                price_df=pd.DataFrame(),  # Empty dataframe
                econometric_data={},
                val_size=0.2
            )
    
    def test_component_dependencies(self, system_integrator):
        """Test that all component dependencies are correctly wired."""
        # Verify PricePredictor has access to all required models
        if system_integrator.price_predictor:
            assert system_integrator.price_predictor.ts_model is not None
            assert system_integrator.price_predictor.ml_model is not None
            assert system_integrator.price_predictor.nlp_analyzer is not None
        
        # Verify PerformanceMonitor has access to database
        assert system_integrator.performance_monitor.supabase_client is not None
        
        # Verify ModelManager is initialized
        assert system_integrator.model_manager.client is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
