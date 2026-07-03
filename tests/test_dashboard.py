"""
Unit tests for the Dashboard visualization module.

Tests cover:
- Prediction chart generation
- Performance dashboard generation
- Weekly report generation
- Market shock detection
- Gauge plotting
- Recommendation generation
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

from src.visualization.dashboard import Dashboard, MATPLOTLIB_AVAILABLE


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client."""
    return Mock()


@pytest.fixture
def dashboard(mock_supabase_client):
    """Create a Dashboard instance with mocked Supabase client."""
    if not MATPLOTLIB_AVAILABLE:
        pytest.skip("matplotlib not available")
    return Dashboard(supabase_client=mock_supabase_client)


@pytest.fixture
def sample_price_data():
    """Create sample price data."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'price': [3000 + i * 10 + np.random.randn() * 5 for i in range(len(dates))]
    })


@pytest.fixture
def sample_predictions():
    """Create sample prediction data."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'predicted_price': [3000 + i * 10 for i in range(len(dates))],
        'lower_bound': [2900 + i * 10 for i in range(len(dates))],
        'upper_bound': [3100 + i * 10 for i in range(len(dates))]
    })


@pytest.fixture
def sample_metrics_history():
    """Create sample metrics history."""
    dates = [datetime(2024, 1, i) for i in range(1, 11)]
    return [
        {
            'rmse': 50.0 + i,
            'mae': 40.0 + i,
            'mape': 0.015 + i * 0.001,
            'directional_accuracy': 0.75 - i * 0.01,
            'coverage_rate': 0.95 - i * 0.005,
            'mean_interval_width': 200.0 + i * 5,
            'timestamp': date
        }
        for i, date in enumerate(dates)
    ]


class TestDashboardInitialization:
    """Test Dashboard initialization."""
    
    def test_init_with_default_params(self, mock_supabase_client):
        """Test initialization with default parameters."""
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        dashboard = Dashboard(supabase_client=mock_supabase_client)
        
        assert dashboard.figure_size == (12, 6)
        assert dashboard.dpi == 100
        assert dashboard.supabase_client == mock_supabase_client
    
    def test_init_with_custom_params(self, mock_supabase_client):
        """Test initialization with custom parameters."""
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")
        
        dashboard = Dashboard(
            supabase_client=mock_supabase_client,
            figure_size=(16, 8),
            dpi=150
        )
        
        assert dashboard.figure_size == (16, 8)
        assert dashboard.dpi == 150
    
    def test_init_without_matplotlib(self):
        """Test that initialization fails gracefully without matplotlib."""
        if MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib is available")
        
        with pytest.raises(ImportError, match="matplotlib is required"):
            Dashboard()


class TestPredictionChart:
    """Test prediction chart generation."""
    
    def test_generate_prediction_chart_success(
        self,
        dashboard,
        sample_price_data,
        sample_predictions
    ):
        """Test successful prediction chart generation."""
        # Mock database queries
        dashboard._fetch_actual_prices = Mock(return_value=sample_price_data)
        dashboard._fetch_predictions = Mock(return_value=sample_predictions)
        dashboard._detect_shock_periods = Mock(return_value=[])
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        fig = dashboard.generate_prediction_chart(
            start_date=start_date,
            end_date=end_date,
            horizon=1,
            market="ICE_London"
        )
        
        assert fig is not None
        assert len(fig.axes) == 1
        
        # Verify mocks were called
        dashboard._fetch_actual_prices.assert_called_once_with(
            start_date, end_date, "ICE_London"
        )
        dashboard._fetch_predictions.assert_called_once_with(
            start_date, end_date, 1
        )
    
    def test_generate_prediction_chart_no_actual_data(self, dashboard):
        """Test chart generation with no actual price data."""
        dashboard._fetch_actual_prices = Mock(return_value=pd.DataFrame())
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        with pytest.raises(ValueError, match="No actual price data found"):
            dashboard.generate_prediction_chart(
                start_date=start_date,
                end_date=end_date
            )
    
    def test_generate_prediction_chart_with_shocks(
        self,
        dashboard,
        sample_price_data,
        sample_predictions
    ):
        """Test chart generation with market shock periods."""
        shock_periods = [
            (datetime(2024, 1, 3), datetime(2024, 1, 4)),
            (datetime(2024, 1, 7), datetime(2024, 1, 7))
        ]
        
        dashboard._fetch_actual_prices = Mock(return_value=sample_price_data)
        dashboard._fetch_predictions = Mock(return_value=sample_predictions)
        dashboard._detect_shock_periods = Mock(return_value=shock_periods)
        
        fig = dashboard.generate_prediction_chart(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10)
        )
        
        assert fig is not None
        dashboard._detect_shock_periods.assert_called_once()
    
    def test_generate_prediction_chart_save_to_file(
        self,
        dashboard,
        sample_price_data,
        sample_predictions,
        tmp_path
    ):
        """Test saving chart to file."""
        dashboard._fetch_actual_prices = Mock(return_value=sample_price_data)
        dashboard._fetch_predictions = Mock(return_value=sample_predictions)
        dashboard._detect_shock_periods = Mock(return_value=[])
        
        save_path = tmp_path / "chart.png"
        
        fig = dashboard.generate_prediction_chart(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
            save_path=str(save_path)
        )
        
        assert fig is not None
        assert save_path.exists()


class TestPerformanceDashboard:
    """Test performance dashboard generation."""
    
    def test_generate_performance_dashboard_success(
        self,
        dashboard,
        sample_metrics_history
    ):
        """Test successful dashboard generation."""
        dashboard._fetch_metrics_history = Mock(return_value=sample_metrics_history)
        
        fig = dashboard.generate_performance_dashboard(model_version="v1.0.0")
        
        assert fig is not None
        assert len(fig.axes) == 4  # 4 subplots
        
        dashboard._fetch_metrics_history.assert_called_once_with("v1.0.0", limit=30)
    
    def test_generate_performance_dashboard_no_metrics(self, dashboard):
        """Test dashboard generation with no metrics."""
        dashboard._fetch_metrics_history = Mock(return_value=[])
        
        with pytest.raises(ValueError, match="No metrics found"):
            dashboard.generate_performance_dashboard(model_version="v1.0.0")
    
    def test_generate_performance_dashboard_save_to_file(
        self,
        dashboard,
        sample_metrics_history,
        tmp_path
    ):
        """Test saving dashboard to file."""
        dashboard._fetch_metrics_history = Mock(return_value=sample_metrics_history)
        
        save_path = tmp_path / "dashboard.png"
        
        fig = dashboard.generate_performance_dashboard(
            model_version="v1.0.0",
            save_path=str(save_path)
        )
        
        assert fig is not None
        assert save_path.exists()


class TestWeeklyReport:
    """Test weekly report generation."""
    
    def test_generate_weekly_report_success(
        self,
        dashboard,
        sample_price_data,
        sample_predictions
    ):
        """Test successful weekly report generation."""
        dashboard._fetch_predictions = Mock(return_value=sample_predictions)
        dashboard._fetch_actual_prices = Mock(return_value=sample_price_data)
        
        week_start = datetime(2024, 1, 1)
        report = dashboard.generate_weekly_report(
            model_version="v1.0.0",
            week_start=week_start
        )
        
        assert report is not None
        assert "week_start" in report
        assert "week_end" in report
        assert "model_version" in report
        assert "total_predictions" in report
        assert "mean_absolute_error" in report
        assert "directional_accuracy" in report
        assert "coverage_rate" in report
        assert "daily_breakdown" in report
        assert "recommendations" in report
        
        assert report["model_version"] == "v1.0.0"
        assert report["total_predictions"] > 0
    
    def test_generate_weekly_report_insufficient_data(self, dashboard):
        """Test report generation with insufficient data."""
        dashboard._fetch_predictions = Mock(return_value=pd.DataFrame())
        dashboard._fetch_actual_prices = Mock(return_value=pd.DataFrame())
        
        report = dashboard.generate_weekly_report(model_version="v1.0.0")
        
        assert report["status"] == "insufficient_data"
    
    def test_generate_weekly_report_no_matches(
        self,
        dashboard,
        sample_price_data
    ):
        """Test report generation with no matching predictions and actuals."""
        # Create predictions with different dates
        different_dates = pd.date_range(start='2024-02-01', end='2024-02-10', freq='D')
        predictions = pd.DataFrame({
            'timestamp': different_dates,
            'predicted_price': [3000] * len(different_dates),
            'lower_bound': [2900] * len(different_dates),
            'upper_bound': [3100] * len(different_dates)
        })
        
        dashboard._fetch_predictions = Mock(return_value=predictions)
        dashboard._fetch_actual_prices = Mock(return_value=sample_price_data)
        
        report = dashboard.generate_weekly_report(
            model_version="v1.0.0",
            week_start=datetime(2024, 1, 1)
        )
        
        assert report["status"] == "no_matches"
    
    def test_generate_weekly_report_save_to_file(
        self,
        dashboard,
        sample_price_data,
        sample_predictions,
        tmp_path
    ):
        """Test saving report to file."""
        dashboard._fetch_predictions = Mock(return_value=sample_predictions)
        dashboard._fetch_actual_prices = Mock(return_value=sample_price_data)
        
        save_path = tmp_path / "report.txt"
        
        report = dashboard.generate_weekly_report(
            model_version="v1.0.0",
            week_start=datetime(2024, 1, 1),
            save_path=str(save_path)
        )
        
        assert report is not None
        assert save_path.exists()
        
        # Verify file content
        content = save_path.read_text()
        assert "WEEKLY PERFORMANCE REPORT" in content
        assert "v1.0.0" in content


class TestShockDetection:
    """Test market shock detection."""
    
    def test_detect_shock_periods_with_shocks(self, dashboard):
        """Test detection of market shock periods."""
        # Create data with a shock (>5% change)
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        prices = [3000, 3010, 3020, 3200, 3210, 3220, 3230, 3240, 3250, 3260]
        price_data = pd.DataFrame({
            'timestamp': dates,
            'price': prices
        })
        
        shock_periods = dashboard._detect_shock_periods(price_data, threshold=0.05)
        
        assert len(shock_periods) > 0
        assert all(isinstance(period, tuple) for period in shock_periods)
        assert all(len(period) == 2 for period in shock_periods)
    
    def test_detect_shock_periods_no_shocks(self, dashboard):
        """Test detection with no market shocks."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        prices = [3000 + i * 10 for i in range(len(dates))]
        price_data = pd.DataFrame({
            'timestamp': dates,
            'price': prices
        })
        
        shock_periods = dashboard._detect_shock_periods(price_data, threshold=0.05)
        
        assert len(shock_periods) == 0
    
    def test_detect_shock_periods_insufficient_data(self, dashboard):
        """Test detection with insufficient data."""
        price_data = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1)],
            'price': [3000]
        })
        
        shock_periods = dashboard._detect_shock_periods(price_data)
        
        assert len(shock_periods) == 0


class TestRecommendations:
    """Test recommendation generation."""
    
    def test_generate_recommendations_high_mae(self, dashboard):
        """Test recommendations for high MAE."""
        summary = {
            'mean_absolute_error': 150.0,
            'directional_accuracy': 0.7,
            'coverage_rate': 0.95,
            'mean_percentage_error': 3.0
        }
        
        recommendations = dashboard._generate_recommendations(summary)
        
        assert len(recommendations) > 0
        assert any("mean absolute error" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_low_directional_accuracy(self, dashboard):
        """Test recommendations for low directional accuracy."""
        summary = {
            'mean_absolute_error': 50.0,
            'directional_accuracy': 0.5,
            'coverage_rate': 0.95,
            'mean_percentage_error': 3.0
        }
        
        recommendations = dashboard._generate_recommendations(summary)
        
        assert len(recommendations) > 0
        assert any("directional accuracy" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_low_coverage(self, dashboard):
        """Test recommendations for low coverage rate."""
        summary = {
            'mean_absolute_error': 50.0,
            'directional_accuracy': 0.7,
            'coverage_rate': 0.85,
            'mean_percentage_error': 3.0
        }
        
        recommendations = dashboard._generate_recommendations(summary)
        
        assert len(recommendations) > 0
        assert any("coverage rate" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_high_coverage(self, dashboard):
        """Test recommendations for very high coverage rate."""
        summary = {
            'mean_absolute_error': 50.0,
            'directional_accuracy': 0.7,
            'coverage_rate': 0.99,
            'mean_percentage_error': 3.0
        }
        
        recommendations = dashboard._generate_recommendations(summary)
        
        assert len(recommendations) > 0
        assert any("too wide" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_good_performance(self, dashboard):
        """Test recommendations for good performance."""
        summary = {
            'mean_absolute_error': 50.0,
            'directional_accuracy': 0.7,
            'coverage_rate': 0.95,
            'mean_percentage_error': 3.0
        }
        
        recommendations = dashboard._generate_recommendations(summary)
        
        assert len(recommendations) > 0
        assert any("acceptable" in rec.lower() for rec in recommendations)


class TestDatabaseQueries:
    """Test database query methods."""
    
    def test_fetch_actual_prices_success(self, dashboard, mock_supabase_client):
        """Test successful fetching of actual prices."""
        mock_response = Mock()
        mock_response.data = [
            {'timestamp': '2024-01-01T00:00:00', 'price': 3000.0},
            {'timestamp': '2024-01-02T00:00:00', 'price': 3010.0}
        ]
        
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.lte.return_value.order.return_value.execute.return_value = mock_response
        
        result = dashboard._fetch_actual_prices(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            market="ICE_London"
        )
        
        assert not result.empty
        assert len(result) == 2
        assert 'timestamp' in result.columns
        assert 'price' in result.columns
    
    def test_fetch_actual_prices_no_data(self, dashboard, mock_supabase_client):
        """Test fetching actual prices with no data."""
        mock_response = Mock()
        mock_response.data = []
        
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.lte.return_value.order.return_value.execute.return_value = mock_response
        
        result = dashboard._fetch_actual_prices(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            market="ICE_London"
        )
        
        assert result.empty
    
    def test_fetch_predictions_success(self, dashboard, mock_supabase_client):
        """Test successful fetching of predictions."""
        mock_response = Mock()
        mock_response.data = [
            {
                'created_at': '2024-01-01T00:00:00',
                'predicted_price': 3000.0,
                'lower_bound': 2900.0,
                'upper_bound': 3100.0
            }
        ]
        
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.lte.return_value.order.return_value.execute.return_value = mock_response
        
        result = dashboard._fetch_predictions(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            horizon=1
        )
        
        assert not result.empty
        assert 'timestamp' in result.columns
        assert 'predicted_price' in result.columns
    
    def test_fetch_metrics_history_success(self, dashboard, mock_supabase_client):
        """Test successful fetching of metrics history."""
        mock_response = Mock()
        mock_response.data = [
            {
                'rmse': 50.0,
                'mae': 40.0,
                'mape': 0.015,
                'directional_accuracy': 0.75,
                'coverage_rate': 0.95,
                'mean_interval_width': 200.0,
                'created_at': '2024-01-01T00:00:00'
            }
        ]
        
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        result = dashboard._fetch_metrics_history(
            model_version="v1.0.0",
            limit=10
        )
        
        assert len(result) == 1
        assert result[0]['rmse'] == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
