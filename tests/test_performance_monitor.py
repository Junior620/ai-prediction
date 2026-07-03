"""
Unit tests for the PerformanceMonitor class.

Tests cover:
- Computation of performance metrics (RMSE, MAE, MAPE, directional accuracy, coverage rate)
- Tracking performance metrics in database
- Detection of performance degradation
- Triggering retraining alerts
- Comparison of model versions
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from src.monitoring.performance_monitor import PerformanceMonitor


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor class."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Create a mock Supabase client."""
        mock_client = Mock()
        mock_table = Mock()
        mock_client.table.return_value = mock_table
        return mock_client
    
    @pytest.fixture
    def performance_monitor(self, mock_supabase_client):
        """Create a PerformanceMonitor instance with mocked Supabase client."""
        return PerformanceMonitor(
            supabase_client=mock_supabase_client,
            degradation_threshold=0.15
        )
    
    def test_initialization(self, mock_supabase_client):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor(
            supabase_client=mock_supabase_client,
            degradation_threshold=0.2
        )
        
        assert monitor.degradation_threshold == 0.2
        assert monitor.supabase_client == mock_supabase_client
    
    def test_initialization_invalid_threshold(self, mock_supabase_client):
        """Test initialization with invalid degradation threshold."""
        with pytest.raises(ValueError, match="degradation_threshold must be between 0 and 1"):
            PerformanceMonitor(
                supabase_client=mock_supabase_client,
                degradation_threshold=1.5
            )
        
        with pytest.raises(ValueError, match="degradation_threshold must be between 0 and 1"):
            PerformanceMonitor(
                supabase_client=mock_supabase_client,
                degradation_threshold=0.0
            )
    
    def test_compute_metrics_basic(self, performance_monitor):
        """Test basic computation of performance metrics."""
        # Create synthetic data
        y_true = np.array([3000.0, 3100.0, 3200.0, 3150.0, 3250.0])
        y_pred = np.array([2950.0, 3120.0, 3180.0, 3160.0, 3230.0])
        y_pred_lower = np.array([2850.0, 3020.0, 3080.0, 3060.0, 3130.0])
        y_pred_upper = np.array([3050.0, 3220.0, 3280.0, 3260.0, 3330.0])
        
        metrics = performance_monitor.compute_metrics(
            y_true, y_pred, y_pred_lower, y_pred_upper
        )
        
        # Verify all required metrics are present
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert "directional_accuracy" in metrics
        assert "coverage_rate" in metrics
        assert "mean_interval_width" in metrics
        
        # Verify metrics are non-negative
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["mape"] >= 0
        
        # Verify directional accuracy and coverage rate are between 0 and 1
        assert 0 <= metrics["directional_accuracy"] <= 1
        assert 0 <= metrics["coverage_rate"] <= 1
        
        # Verify mean interval width is positive
        assert metrics["mean_interval_width"] > 0
    
    def test_compute_metrics_perfect_predictions(self, performance_monitor):
        """Test metrics computation with perfect predictions."""
        y_true = np.array([3000.0, 3100.0, 3200.0, 3300.0])
        y_pred = y_true.copy()
        y_pred_lower = y_true - 50.0
        y_pred_upper = y_true + 50.0
        
        metrics = performance_monitor.compute_metrics(
            y_true, y_pred, y_pred_lower, y_pred_upper
        )
        
        # Perfect predictions should have zero error
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["mape"] == 0.0
        
        # Coverage rate should be 100% (all values within CI)
        assert metrics["coverage_rate"] == 1.0
    
    def test_compute_metrics_directional_accuracy(self, performance_monitor):
        """Test directional accuracy computation."""
        # Create data with known directional patterns
        # Actual: up, up, down
        y_true = np.array([3000.0, 3100.0, 3200.0, 3150.0])
        # Predicted: up, up, down (all correct)
        y_pred = np.array([2950.0, 3050.0, 3150.0, 3100.0])
        y_pred_lower = y_pred - 100.0
        y_pred_upper = y_pred + 100.0
        
        metrics = performance_monitor.compute_metrics(
            y_true, y_pred, y_pred_lower, y_pred_upper
        )
        
        # All directions predicted correctly
        assert metrics["directional_accuracy"] == 1.0
    
    def test_compute_metrics_coverage_rate(self, performance_monitor):
        """Test coverage rate computation."""
        y_true = np.array([3000.0, 3100.0, 3200.0, 3300.0])
        y_pred = np.array([3000.0, 3100.0, 3200.0, 3300.0])
        
        # First 3 values within CI, last value outside
        y_pred_lower = np.array([2900.0, 3000.0, 3100.0, 3350.0])
        y_pred_upper = np.array([3100.0, 3200.0, 3300.0, 3400.0])
        
        metrics = performance_monitor.compute_metrics(
            y_true, y_pred, y_pred_lower, y_pred_upper
        )
        
        # 3 out of 4 values within CI = 75% coverage
        assert metrics["coverage_rate"] == 0.75
    
    def test_compute_metrics_invalid_input_shapes(self, performance_monitor):
        """Test metrics computation with mismatched array shapes."""
        y_true = np.array([3000.0, 3100.0, 3200.0])
        y_pred = np.array([2950.0, 3120.0])  # Different length
        y_pred_lower = np.array([2850.0, 3020.0, 3080.0])
        y_pred_upper = np.array([3050.0, 3220.0, 3280.0])
        
        with pytest.raises(ValueError, match="All input arrays must have the same length"):
            performance_monitor.compute_metrics(
                y_true, y_pred, y_pred_lower, y_pred_upper
            )
    
    def test_compute_metrics_empty_arrays(self, performance_monitor):
        """Test metrics computation with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        y_pred_lower = np.array([])
        y_pred_upper = np.array([])
        
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            performance_monitor.compute_metrics(
                y_true, y_pred, y_pred_lower, y_pred_upper
            )
    
    def test_compute_metrics_nan_values(self, performance_monitor):
        """Test metrics computation with NaN values."""
        y_true = np.array([3000.0, np.nan, 3200.0])
        y_pred = np.array([2950.0, 3120.0, 3180.0])
        y_pred_lower = np.array([2850.0, 3020.0, 3080.0])
        y_pred_upper = np.array([3050.0, 3220.0, 3280.0])
        
        with pytest.raises(ValueError, match="Input arrays contain NaN or infinite values"):
            performance_monitor.compute_metrics(
                y_true, y_pred, y_pred_lower, y_pred_upper
            )
    
    def test_compute_metrics_single_observation(self, performance_monitor):
        """Test metrics computation with single observation."""
        y_true = np.array([3000.0])
        y_pred = np.array([2950.0])
        y_pred_lower = np.array([2850.0])
        y_pred_upper = np.array([3050.0])
        
        metrics = performance_monitor.compute_metrics(
            y_true, y_pred, y_pred_lower, y_pred_upper
        )
        
        # Should compute error metrics
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        
        # Directional accuracy requires at least 2 observations
        assert metrics["directional_accuracy"] == 0.0
    
    def test_track_performance(self, performance_monitor, mock_supabase_client):
        """Test tracking performance metrics to database."""
        metrics = {
            "rmse": 45.5,
            "mae": 35.2,
            "mape": 0.012,
            "directional_accuracy": 0.85,
            "coverage_rate": 0.92,
            "mean_interval_width": 150.0
        }
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        model_version = "v1.2.3"
        
        # Mock the database response
        mock_response = Mock()
        mock_response.data = [{"id": 1}]
        mock_table = mock_supabase_client.table.return_value
        mock_table.insert.return_value.execute.return_value = mock_response
        
        # Track performance
        performance_monitor.track_performance(metrics, timestamp, model_version)
        
        # Verify database call
        mock_supabase_client.table.assert_called_with("model_metrics")
        mock_table.insert.assert_called_once()
        
        # Verify inserted data
        call_args = mock_table.insert.call_args[0][0]
        assert call_args["model_version"] == model_version
        assert call_args["rmse"] == 45.5
        assert call_args["mae"] == 35.2
        assert call_args["mape"] == 0.012
        assert call_args["directional_accuracy"] == 0.85
        assert call_args["coverage_rate"] == 0.92
        assert call_args["mean_interval_width"] == 150.0
    
    def test_track_performance_missing_metrics(self, performance_monitor):
        """Test tracking performance with missing required metrics."""
        incomplete_metrics = {
            "rmse": 45.5,
            "mae": 35.2
            # Missing other required metrics
        }
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        model_version = "v1.2.3"
        
        with pytest.raises(ValueError, match="Missing required metrics"):
            performance_monitor.track_performance(
                incomplete_metrics, timestamp, model_version
            )
    
    def test_detect_degradation_no_degradation(self, performance_monitor):
        """Test degradation detection when performance is stable."""
        baseline_metrics = {
            "rmse": 50.0,
            "mae": 40.0,
            "mape": 0.015
        }
        current_metrics = {
            "rmse": 52.0,  # 4% increase (below 15% threshold)
            "mae": 41.0,   # 2.5% increase
            "mape": 0.016  # 6.7% increase
        }
        
        degraded = performance_monitor.detect_degradation(
            current_metrics, baseline_metrics
        )
        
        assert degraded is False
    
    def test_detect_degradation_with_degradation(self, performance_monitor):
        """Test degradation detection when performance has degraded."""
        baseline_metrics = {
            "rmse": 50.0,
            "mae": 40.0,
            "mape": 0.015
        }
        current_metrics = {
            "rmse": 60.0,  # 20% increase (above 15% threshold)
            "mae": 42.0,   # 5% increase
            "mape": 0.016  # 6.7% increase
        }
        
        degraded = performance_monitor.detect_degradation(
            current_metrics, baseline_metrics
        )
        
        assert degraded is True
    
    def test_detect_degradation_custom_threshold(self, performance_monitor):
        """Test degradation detection with custom threshold."""
        baseline_metrics = {
            "rmse": 50.0,
            "mae": 40.0,
            "mape": 0.015
        }
        current_metrics = {
            "rmse": 55.0,  # 10% increase
            "mae": 42.0,
            "mape": 0.016
        }
        
        # With 15% threshold: no degradation
        degraded = performance_monitor.detect_degradation(
            current_metrics, baseline_metrics, threshold=0.15
        )
        assert degraded is False
        
        # With 5% threshold: degradation detected
        degraded = performance_monitor.detect_degradation(
            current_metrics, baseline_metrics, threshold=0.05
        )
        assert degraded is True
    
    def test_detect_degradation_zero_baseline(self, performance_monitor):
        """Test degradation detection with zero baseline values."""
        baseline_metrics = {
            "rmse": 0.0,  # Zero baseline
            "mae": 40.0,
            "mape": 0.015
        }
        current_metrics = {
            "rmse": 10.0,
            "mae": 42.0,
            "mape": 0.016
        }
        
        # Should handle zero baseline gracefully
        degraded = performance_monitor.detect_degradation(
            current_metrics, baseline_metrics
        )
        
        # Should not raise error, result depends on other metrics
        assert isinstance(degraded, bool)
    
    def test_trigger_retraining_alert(self, performance_monitor, caplog):
        """Test triggering retraining alert."""
        import logging
        caplog.set_level(logging.CRITICAL)
        
        metrics = {
            "rmse": 60.0,
            "mae": 50.0,
            "mape": 0.020,
            "directional_accuracy": 0.70,
            "coverage_rate": 0.85
        }
        reason = "performance_degradation"
        
        performance_monitor.trigger_retraining_alert(reason, metrics)
        
        # Verify critical log was created
        assert any(
            record.levelname == "CRITICAL" and "MODEL RETRAINING ALERT" in record.message
            for record in caplog.records
        )
    
    def test_compare_models_model_a_wins(self, performance_monitor):
        """Test model comparison when model A performs better."""
        model_a_metrics = {
            "rmse": 40.0,
            "mae": 30.0,
            "mape": 0.010,
            "directional_accuracy": 0.90,
            "coverage_rate": 0.95
        }
        model_b_metrics = {
            "rmse": 50.0,
            "mae": 40.0,
            "mape": 0.015,
            "directional_accuracy": 0.85,
            "coverage_rate": 0.90
        }
        
        result = performance_monitor.compare_models(model_a_metrics, model_b_metrics)
        
        assert result == "model_a"
    
    def test_compare_models_model_b_wins(self, performance_monitor):
        """Test model comparison when model B performs better."""
        model_a_metrics = {
            "rmse": 50.0,
            "mae": 40.0,
            "mape": 0.015,
            "directional_accuracy": 0.85,
            "coverage_rate": 0.90
        }
        model_b_metrics = {
            "rmse": 40.0,
            "mae": 30.0,
            "mape": 0.010,
            "directional_accuracy": 0.90,
            "coverage_rate": 0.95
        }
        
        result = performance_monitor.compare_models(model_a_metrics, model_b_metrics)
        
        assert result == "model_b"
    
    def test_compare_models_tie(self, performance_monitor):
        """Test model comparison when models perform equally."""
        model_metrics = {
            "rmse": 45.0,
            "mae": 35.0,
            "mape": 0.012,
            "directional_accuracy": 0.87,
            "coverage_rate": 0.92
        }
        
        result = performance_monitor.compare_models(model_metrics, model_metrics)
        
        assert result == "tie"
    
    def test_compare_models_mixed_performance(self, performance_monitor):
        """Test model comparison with mixed performance characteristics."""
        model_a_metrics = {
            "rmse": 40.0,  # Better
            "mae": 45.0,   # Worse
            "mape": 0.010, # Better
            "directional_accuracy": 0.85,  # Worse
            "coverage_rate": 0.95  # Better
        }
        model_b_metrics = {
            "rmse": 50.0,
            "mae": 35.0,
            "mape": 0.015,
            "directional_accuracy": 0.90,
            "coverage_rate": 0.90
        }
        
        result = performance_monitor.compare_models(model_a_metrics, model_b_metrics)
        
        # Result should be one of the valid options
        assert result in ["model_a", "model_b", "tie"]
    
    def test_compare_models_missing_metrics(self, performance_monitor):
        """Test model comparison with missing required metrics."""
        incomplete_metrics = {
            "rmse": 45.0,
            "mae": 35.0
            # Missing other required metrics
        }
        complete_metrics = {
            "rmse": 50.0,
            "mae": 40.0,
            "mape": 0.015,
            "directional_accuracy": 0.85,
            "coverage_rate": 0.90
        }
        
        with pytest.raises(ValueError, match="missing required keys"):
            performance_monitor.compare_models(incomplete_metrics, complete_metrics)
    
    def test_get_recent_metrics(self, performance_monitor, mock_supabase_client):
        """Test retrieving recent metrics from database."""
        model_version = "v1.2.3"
        
        # Mock database response
        mock_data = [
            {
                "rmse": 45.5,
                "mae": 35.2,
                "mape": 0.012,
                "directional_accuracy": 0.85,
                "coverage_rate": 0.92,
                "mean_interval_width": 150.0,
                "created_at": "2024-01-15T10:30:00",
                "model_version": model_version
            },
            {
                "rmse": 46.0,
                "mae": 36.0,
                "mape": 0.013,
                "directional_accuracy": 0.84,
                "coverage_rate": 0.91,
                "mean_interval_width": 155.0,
                "created_at": "2024-01-14T10:30:00",
                "model_version": model_version
            }
        ]
        
        mock_response = Mock()
        mock_response.data = mock_data
        
        mock_table = mock_supabase_client.table.return_value
        mock_select = mock_table.select.return_value
        mock_eq = mock_select.eq.return_value
        mock_order = mock_eq.order.return_value
        mock_limit = mock_order.limit.return_value
        mock_limit.execute.return_value = mock_response
        
        # Get recent metrics
        metrics_list = performance_monitor.get_recent_metrics(model_version, limit=2)
        
        # Verify results
        assert len(metrics_list) == 2
        assert all(m.model_version == model_version for m in metrics_list)
        assert metrics_list[0].rmse == 45.5
        assert metrics_list[1].rmse == 46.0
    
    def test_get_baseline_metrics(self, performance_monitor, mock_supabase_client):
        """Test retrieving baseline metrics from database."""
        model_version = "v1.2.3"
        
        # Mock database response
        mock_data = [
            {
                "rmse": 50.0,
                "mae": 40.0,
                "mape": 0.015,
                "directional_accuracy": 0.80,
                "coverage_rate": 0.90,
                "mean_interval_width": 160.0,
                "created_at": "2024-01-01T00:00:00",
                "model_version": model_version
            }
        ]
        
        mock_response = Mock()
        mock_response.data = mock_data
        
        mock_table = mock_supabase_client.table.return_value
        mock_select = mock_table.select.return_value
        mock_eq = mock_select.eq.return_value
        mock_order = mock_eq.order.return_value
        mock_limit = mock_order.limit.return_value
        mock_limit.execute.return_value = mock_response
        
        # Get baseline metrics
        baseline = performance_monitor.get_baseline_metrics(model_version)
        
        # Verify results
        assert baseline is not None
        assert baseline["rmse"] == 50.0
        assert baseline["mae"] == 40.0
        assert baseline["mape"] == 0.015
    
    def test_get_baseline_metrics_not_found(self, performance_monitor, mock_supabase_client):
        """Test retrieving baseline metrics when none exist."""
        model_version = "v1.2.3"
        
        # Mock empty database response
        mock_response = Mock()
        mock_response.data = []
        
        mock_table = mock_supabase_client.table.return_value
        mock_select = mock_table.select.return_value
        mock_eq = mock_select.eq.return_value
        mock_order = mock_eq.order.return_value
        mock_limit = mock_order.limit.return_value
        mock_limit.execute.return_value = mock_response
        
        # Get baseline metrics
        baseline = performance_monitor.get_baseline_metrics(model_version)
        
        # Should return None when no metrics found
        assert baseline is None
