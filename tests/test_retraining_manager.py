"""
Unit tests for the RetrainingManager.

Tests cover:
- Automatic retraining trigger logic (time-based and data-based)
- Model validation on validation set
- Model comparison and promotion logic
- Model version history management
- Error handling and edge cases
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, call

# Note: We mock MLflow components to avoid protobuf compatibility issues
# from mlflow.entities.model_registry import ModelVersion


@pytest.fixture
def mock_model_manager():
    """Create mock ModelManager."""
    manager = Mock(spec=ModelManager)
    manager.list_model_versions = Mock(return_value=[])
    manager.promote_model = Mock()
    manager.log_model = Mock(return_value="1")
    manager.client = Mock()
    return manager


@pytest.fixture
def mock_performance_monitor():
    """Create mock PerformanceMonitor."""
    monitor = Mock(spec=PerformanceMonitor)
    monitor.compute_metrics = Mock(return_value={
        'rmse': 50.0,
        'mae': 40.0,
        'mape': 0.02,
        'directional_accuracy': 0.75,
        'coverage_rate': 0.95,
        'mean_interval_width': 200.0
    })
    monitor.compare_models = Mock(return_value="model_a")
    return monitor


@pytest.fixture
def mock_data_preprocessor():
    """Create mock DataPreprocessor."""
    preprocessor = Mock(spec=DataPreprocessor)
    
    # Mock handle_missing_values to return input unchanged
    preprocessor.handle_missing_values = Mock(side_effect=lambda df, strategy: df)
    
    # Mock create_train_val_split
    def mock_split(df, val_size, shuffle):
        split_idx = int(len(df) * (1 - val_size))
        return df.iloc[:split_idx], df.iloc[split_idx:]
    
    preprocessor.create_train_val_split = Mock(side_effect=mock_split)
    
    return preprocessor


@pytest.fixture
def mock_supabase_client():
    """Create mock Supabase client."""
    client = Mock()
    
    # Mock table queries
    table_mock = Mock()
    table_mock.select = Mock(return_value=table_mock)
    table_mock.gte = Mock(return_value=table_mock)
    table_mock.order = Mock(return_value=table_mock)
    table_mock.execute = Mock(return_value=Mock(data=[]))
    
    client.table = Mock(return_value=table_mock)
    
    return client


@pytest.fixture
def retraining_manager(
    mock_model_manager,
    mock_performance_monitor,
    mock_data_preprocessor,
    mock_supabase_client
):
    """Create RetrainingManager instance with mocked dependencies."""
    return RetrainingManager(
        model_manager=mock_model_manager,
        performance_monitor=mock_performance_monitor,
        data_preprocessor=mock_data_preprocessor,
        supabase_client=mock_supabase_client,
        retraining_frequency_days=30,
        min_new_data_days=30,
        max_model_versions=5
    )


class TestRetrainingManagerInitialization:
    """Test RetrainingManager initialization."""
    
    def test_initialization_with_valid_parameters(
        self,
        mock_model_manager,
        mock_performance_monitor,
        mock_data_preprocessor,
        mock_supabase_client
    ):
        """Test successful initialization with valid parameters."""
        manager = RetrainingManager(
            model_manager=mock_model_manager,
            performance_monitor=mock_performance_monitor,
            data_preprocessor=mock_data_preprocessor,
            supabase_client=mock_supabase_client,
            retraining_frequency_days=30,
            min_new_data_days=30,
            max_model_versions=5
        )
        
        assert manager.retraining_frequency_days == 30
        assert manager.min_new_data_days == 30
        assert manager.max_model_versions == 5
    
    def test_initialization_with_invalid_retraining_frequency(
        self,
        mock_model_manager,
        mock_performance_monitor,
        mock_data_preprocessor,
        mock_supabase_client
    ):
        """Test initialization fails with invalid retraining frequency."""
        with pytest.raises(ValueError, match="retraining_frequency_days must be positive"):
            RetrainingManager(
                model_manager=mock_model_manager,
                performance_monitor=mock_performance_monitor,
                data_preprocessor=mock_data_preprocessor,
                supabase_client=mock_supabase_client,
                retraining_frequency_days=0
            )
    
    def test_initialization_with_invalid_min_new_data_days(
        self,
        mock_model_manager,
        mock_performance_monitor,
        mock_data_preprocessor,
        mock_supabase_client
    ):
        """Test initialization fails with invalid min_new_data_days."""
        with pytest.raises(ValueError, match="min_new_data_days must be positive"):
            RetrainingManager(
                model_manager=mock_model_manager,
                performance_monitor=mock_performance_monitor,
                data_preprocessor=mock_data_preprocessor,
                supabase_client=mock_supabase_client,
                min_new_data_days=-1
            )
    
    def test_initialization_with_invalid_max_model_versions(
        self,
        mock_model_manager,
        mock_performance_monitor,
        mock_data_preprocessor,
        mock_supabase_client
    ):
        """Test initialization fails with invalid max_model_versions."""
        with pytest.raises(ValueError, match="max_model_versions must be at least 1"):
            RetrainingManager(
                model_manager=mock_model_manager,
                performance_monitor=mock_performance_monitor,
                data_preprocessor=mock_data_preprocessor,
                supabase_client=mock_supabase_client,
                max_model_versions=0
            )


class TestRetrainingTriggerLogic:
    """Test automatic retraining trigger logic."""
    
    def test_should_trigger_retraining_no_existing_model(self, retraining_manager):
        """Test retraining is triggered when no existing model exists."""
        # Mock no existing model
        retraining_manager.model_manager.list_model_versions.return_value = []
        
        should_retrain, reason = retraining_manager.should_trigger_retraining()
        
        assert should_retrain is True
        assert reason == "no_existing_model"
    
    def test_should_trigger_retraining_time_based(self, retraining_manager):
        """Test time-based retraining trigger (>= 30 days since last training)."""
        # Mock existing model from 35 days ago
        old_timestamp = (datetime.now() - timedelta(days=35)).timestamp() * 1000
        mock_version = Mock()
        mock_version.creation_timestamp = old_timestamp
        
        retraining_manager.model_manager.list_model_versions.return_value = [mock_version]
        
        should_retrain, reason = retraining_manager.should_trigger_retraining()
        
        assert should_retrain is True
        assert "time_based" in reason
        assert "35" in reason
    
    def test_should_trigger_retraining_data_based(self, retraining_manager):
        """Test data-based retraining trigger (>= 30 days of new data)."""
        # Mock existing model from 20 days ago (not enough for time-based)
        recent_timestamp = (datetime.now() - timedelta(days=20)).timestamp() * 1000
        mock_version = Mock()
        mock_version.creation_timestamp = recent_timestamp
        
        retraining_manager.model_manager.list_model_versions.return_value = [mock_version]
        
        # Mock 35 days of new data available
        retraining_manager._count_new_data_days = Mock(return_value=35)
        
        should_retrain, reason = retraining_manager.should_trigger_retraining()
        
        assert should_retrain is True
        assert "data_based" in reason
        assert "35" in reason
    
    def test_should_not_trigger_retraining(self, retraining_manager):
        """Test retraining is not triggered when conditions not met."""
        # Mock existing model from 10 days ago
        recent_timestamp = (datetime.now() - timedelta(days=10)).timestamp() * 1000
        mock_version = Mock()
        mock_version.creation_timestamp = recent_timestamp
        
        retraining_manager.model_manager.list_model_versions.return_value = [mock_version]
        
        # Mock only 10 days of new data
        retraining_manager._count_new_data_days = Mock(return_value=10)
        
        should_retrain, reason = retraining_manager.should_trigger_retraining()
        
        assert should_retrain is False
        assert reason == "no_trigger"
    
    def test_count_new_data_days(self, retraining_manager):
        """Test counting days of new data."""
        since_date = datetime.now() - timedelta(days=10)
        
        # Mock database response with 8 unique dates
        mock_data = [
            {"timestamp": (since_date + timedelta(days=i)).isoformat()}
            for i in range(8)
        ]
        
        mock_response = Mock()
        mock_response.data = mock_data
        
        retraining_manager.supabase_client.table().select().gte().execute.return_value = mock_response
        
        count = retraining_manager._count_new_data_days(since_date)
        
        assert count == 8
    
    def test_count_new_data_days_no_data(self, retraining_manager):
        """Test counting new data days when no data available."""
        since_date = datetime.now() - timedelta(days=10)
        
        # Mock empty database response
        mock_response = Mock()
        mock_response.data = []
        
        retraining_manager.supabase_client.table().select().gte().execute.return_value = mock_response
        
        count = retraining_manager._count_new_data_days(since_date)
        
        assert count == 0


class TestModelValidation:
    """Test model validation on validation set."""
    
    @patch('src.models.retraining_manager.PricePredictor')
    def test_validate_model_success(self, mock_predictor_class, retraining_manager):
        """Test successful model validation."""
        # Create mock predictor
        mock_predictor = Mock(spec=PricePredictor)
        
        # Mock predictions
        mock_prediction = Mock()
        mock_prediction.price = 3000.0
        mock_prediction.confidence_interval = (2900.0, 3100.0)
        
        mock_predictor.predict = Mock(return_value=[mock_prediction])
        
        # Create validation data
        val_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'price': np.random.uniform(2800, 3200, 10),
            'temperature': np.random.uniform(20, 30, 10),
            'rainfall': np.random.uniform(0, 50, 10),
            'stock_level': np.random.uniform(40000, 60000, 10),
            'production': np.random.uniform(1000, 2000, 10),
            'fx_rate_xaf_usd': np.random.uniform(0.0016, 0.0018, 10),
            'fx_rate_gbp_usd': np.random.uniform(1.25, 1.30, 10),
            'fx_rate_eur_usd': np.random.uniform(1.08, 1.12, 10)
        })
        
        econometric_data = pd.DataFrame()
        
        # Validate model
        metrics = retraining_manager._validate_model(
            mock_predictor,
            val_data,
            econometric_data
        )
        
        # Check that metrics were computed
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert metrics['rmse'] == 50.0  # From mock
    
    @patch('src.models.retraining_manager.PricePredictor')
    def test_validate_model_no_successful_predictions(
        self,
        mock_predictor_class,
        retraining_manager
    ):
        """Test validation fails when no successful predictions."""
        # Create mock predictor that always fails
        mock_predictor = Mock(spec=PricePredictor)
        mock_predictor.predict = Mock(side_effect=Exception("Prediction failed"))
        
        # Create validation data
        val_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'price': [3000.0] * 5,
            'temperature': [25.0] * 5,
            'rainfall': [10.0] * 5,
            'stock_level': [50000.0] * 5,
            'production': [1500.0] * 5,
            'fx_rate_xaf_usd': [0.0017] * 5,
            'fx_rate_gbp_usd': [1.27] * 5,
            'fx_rate_eur_usd': [1.10] * 5
        })
        
        econometric_data = pd.DataFrame()
        
        # Validation should fail
        with pytest.raises(ValueError, match="No successful predictions"):
            retraining_manager._validate_model(
                mock_predictor,
                val_data,
                econometric_data
            )


class TestModelComparison:
    """Test model comparison and promotion logic."""
    
    def test_should_promote_model_new_is_better(self, retraining_manager):
        """Test new model is promoted when it performs better."""
        # Mock current production model exists
        mock_prophet = Mock()
        mock_xgboost = Mock()
        
        retraining_manager.model_manager.load_model = Mock(
            side_effect=[mock_prophet, mock_xgboost]
        )
        
        # Mock validation
        retraining_manager._validate_model = Mock(return_value={
            'rmse': 60.0,  # Current model has worse RMSE
            'mae': 50.0,
            'mape': 0.025,
            'directional_accuracy': 0.70,
            'coverage_rate': 0.93,
            'mean_interval_width': 220.0
        })
        
        # Mock comparison (new model wins)
        retraining_manager.performance_monitor.compare_models.return_value = "model_a"
        
        new_metrics = {
            'rmse': 50.0,  # New model has better RMSE
            'mae': 40.0,
            'mape': 0.02,
            'directional_accuracy': 0.75,
            'coverage_rate': 0.95,
            'mean_interval_width': 200.0
        }
        
        val_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'price': [3000.0] * 5,
            'temperature': [25.0] * 5,
            'rainfall': [10.0] * 5,
            'stock_level': [50000.0] * 5,
            'production': [1500.0] * 5,
            'fx_rate_xaf_usd': [0.0017] * 5,
            'fx_rate_gbp_usd': [1.27] * 5,
            'fx_rate_eur_usd': [1.10] * 5
        })
        
        should_promote = retraining_manager._should_promote_model(
            "cocoa_price_predictor",
            new_metrics,
            val_data,
            pd.DataFrame()
        )
        
        assert should_promote is True
    
    def test_should_not_promote_model_current_is_better(self, retraining_manager):
        """Test new model is not promoted when current performs better."""
        # Mock current production model exists
        mock_prophet = Mock()
        mock_xgboost = Mock()
        
        retraining_manager.model_manager.load_model = Mock(
            side_effect=[mock_prophet, mock_xgboost]
        )
        
        # Mock validation (current model is better)
        retraining_manager._validate_model = Mock(return_value={
            'rmse': 45.0,  # Current model has better RMSE
            'mae': 35.0,
            'mape': 0.018,
            'directional_accuracy': 0.78,
            'coverage_rate': 0.96,
            'mean_interval_width': 190.0
        })
        
        # Mock comparison (current model wins)
        retraining_manager.performance_monitor.compare_models.return_value = "model_b"
        
        new_metrics = {
            'rmse': 50.0,
            'mae': 40.0,
            'mape': 0.02,
            'directional_accuracy': 0.75,
            'coverage_rate': 0.95,
            'mean_interval_width': 200.0
        }
        
        val_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'price': [3000.0] * 5,
            'temperature': [25.0] * 5,
            'rainfall': [10.0] * 5,
            'stock_level': [50000.0] * 5,
            'production': [1500.0] * 5,
            'fx_rate_xaf_usd': [0.0017] * 5,
            'fx_rate_gbp_usd': [1.27] * 5,
            'fx_rate_eur_usd': [1.10] * 5
        })
        
        should_promote = retraining_manager._should_promote_model(
            "cocoa_price_predictor",
            new_metrics,
            val_data,
            pd.DataFrame()
        )
        
        assert should_promote is False
    
    def test_should_promote_model_no_current_model(self, retraining_manager):
        """Test new model is promoted when no current production model exists."""
        # Mock no current production model
        retraining_manager.model_manager.load_model = Mock(
            side_effect=ValueError("No model found")
        )
        
        new_metrics = {
            'rmse': 50.0,
            'mae': 40.0,
            'mape': 0.02,
            'directional_accuracy': 0.75,
            'coverage_rate': 0.95,
            'mean_interval_width': 200.0
        }
        
        val_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'price': [3000.0] * 5,
            'temperature': [25.0] * 5,
            'rainfall': [10.0] * 5,
            'stock_level': [50000.0] * 5,
            'production': [1500.0] * 5,
            'fx_rate_xaf_usd': [0.0017] * 5,
            'fx_rate_gbp_usd': [1.27] * 5,
            'fx_rate_eur_usd': [1.10] * 5
        })
        
        should_promote = retraining_manager._should_promote_model(
            "cocoa_price_predictor",
            new_metrics,
            val_data,
            pd.DataFrame()
        )
        
        assert should_promote is True


class TestModelVersionManagement:
    """Test model version history management."""
    
    def test_cleanup_old_versions(self, retraining_manager):
        """Test cleanup of old model versions."""
        # Mock 10 model versions
        mock_versions = []
        for i in range(10):
            version = Mock()
            version.version = str(i + 1)
            version.current_stage = "None" if i < 8 else "Production"
            mock_versions.append(version)
        
        retraining_manager.model_manager.list_model_versions = Mock(
            return_value=mock_versions
        )
        
        # Cleanup should archive versions 6-10 (keeping 5 most recent: 1-5)
        retraining_manager._cleanup_old_versions("cocoa_price_predictor")
        
        # Check that old versions were archived
        # Should be called for both prophet and xgboost models
        assert retraining_manager.model_manager.client.transition_model_version_stage.call_count > 0
    
    def test_cleanup_old_versions_no_cleanup_needed(self, retraining_manager):
        """Test cleanup when number of versions is within limit."""
        # Mock only 3 model versions
        mock_versions = []
        for i in range(3):
            version = Mock()
            version.version = str(i + 1)
            version.current_stage = "None"
            mock_versions.append(version)
        
        retraining_manager.model_manager.list_model_versions = Mock(
            return_value=mock_versions
        )
        
        # Cleanup should not archive anything
        retraining_manager._cleanup_old_versions("cocoa_price_predictor")
        
        # No archiving should occur
        assert retraining_manager.model_manager.client.transition_model_version_stage.call_count == 0
    
    def test_generate_version_string(self, retraining_manager):
        """Test version string generation."""
        version = retraining_manager._generate_version_string()
        
        # Check format: YYYY.MM.DD.HHMM
        assert len(version) == 15  # "2024.01.15.1430"
        assert version.count('.') == 3
        
        # Check it's a valid datetime format
        parts = version.split('.')
        assert len(parts) == 4
        assert len(parts[0]) == 4  # Year
        assert len(parts[1]) == 2  # Month
        assert len(parts[2]) == 2  # Day
        assert len(parts[3]) == 4  # HHMM


class TestRetrainingStatus:
    """Test retraining status reporting."""
    
    def test_get_retraining_status_with_model(self, retraining_manager):
        """Test getting retraining status when model exists."""
        # Mock existing model
        mock_version = Mock()
        mock_version.version = "2024.01.15.1430"
        mock_version.creation_timestamp = (datetime.now() - timedelta(days=15)).timestamp() * 1000
        
        retraining_manager.model_manager.list_model_versions.return_value = [mock_version]
        retraining_manager._count_new_data_days = Mock(return_value=20)
        
        status = retraining_manager.get_retraining_status()
        
        assert status['has_model'] is True
        assert status['latest_version'] == "2024.01.15.1430"
        assert status['days_since_training'] == 15
        assert status['new_data_days'] == 20
        assert 'should_retrain' in status
        assert 'reason' in status
    
    def test_get_retraining_status_no_model(self, retraining_manager):
        """Test getting retraining status when no model exists."""
        # Mock no existing model
        retraining_manager.model_manager.list_model_versions.return_value = []
        
        status = retraining_manager.get_retraining_status()
        
        assert status['has_model'] is False
        assert status['should_retrain'] is True
        assert status['reason'] == "no_existing_model"


class TestRetrainingWorkflow:
    """Test complete retraining workflow."""
    
    @patch('src.models.retraining_manager.NLPAnalyzer')
    @patch('src.models.retraining_manager.PricePredictor')
    def test_retrain_models_no_data(
        self,
        mock_predictor_class,
        mock_nlp_class,
        retraining_manager
    ):
        """Test retraining fails gracefully when no data available."""
        # Mock empty data
        retraining_manager._fetch_training_data = Mock(
            return_value=(pd.DataFrame(), pd.DataFrame())
        )
        
        success, message, version = retraining_manager.retrain_models()
        
        assert success is False
        assert "no_data_available" in message
        assert version is None
    
    def test_repr(self, retraining_manager):
        """Test string representation."""
        repr_str = repr(retraining_manager)
        
        assert "RetrainingManager" in repr_str
        assert "retraining_frequency=30d" in repr_str
        assert "min_new_data=30d" in repr_str
        assert "max_versions=5" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
