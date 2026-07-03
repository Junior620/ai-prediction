"""
Comprehensive unit tests for TimeSeriesModel.

Tests cover:
- Model initialization with different hyperparameters
- Training with synthetic data
- Prediction generation for future periods
- Residual computation
- Component extraction (trend, seasonality)
- Error handling for edge cases

Requirements: 5.2, 5.4, 5.5
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.models.time_series_model import TimeSeriesModel


class TestTimeSeriesModelInitialization:
    """Test TimeSeriesModel initialization with various hyperparameters."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        model = TimeSeriesModel()
        
        assert model.seasonality_mode == "multiplicative"
        assert model.yearly_seasonality is True
        assert model.weekly_seasonality is False
        assert model.changepoint_prior_scale == 0.05
        assert model.is_fitted is False
        assert model.model is not None
    
    def test_custom_initialization(self):
        """Test initialization with custom hyperparameters."""
        model = TimeSeriesModel(
            seasonality_mode="additive",
            yearly_seasonality=False,
            weekly_seasonality=True,
            changepoint_prior_scale=0.1
        )
        
        assert model.seasonality_mode == "additive"
        assert model.yearly_seasonality is False
        assert model.weekly_seasonality is True
        assert model.changepoint_prior_scale == 0.1
        assert model.is_fitted is False
    
    def test_get_hyperparameters(self):
        """Test retrieval of hyperparameters."""
        model = TimeSeriesModel(
            seasonality_mode="additive",
            changepoint_prior_scale=0.08
        )
        
        params = model.get_hyperparameters()
        
        assert params['seasonality_mode'] == "additive"
        assert params['yearly_seasonality'] is True
        assert params['weekly_seasonality'] is False
        assert params['changepoint_prior_scale'] == 0.08
    
    def test_repr_not_fitted(self):
        """Test string representation when model is not fitted."""
        model = TimeSeriesModel()
        
        repr_str = repr(model)
        
        assert "TimeSeriesModel" in repr_str
        assert "not fitted" in repr_str
        assert "multiplicative" in repr_str


class TestTimeSeriesModelFit:
    """Test model fitting with various data scenarios."""
    
    def test_fit_with_synthetic_data(self):
        """Test fitting with synthetic time series data."""
        model = TimeSeriesModel()
        
        # Create synthetic data with trend and seasonality
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        trend = np.linspace(3000, 3500, 365)
        seasonality = 200 * np.sin(2 * np.pi * np.arange(365) / 365)
        noise = np.random.normal(0, 50, 365)
        prices = trend + seasonality + noise
        
        df = pd.DataFrame({
            'ds': dates,
            'y': prices
        })
        
        # Fit the model
        model.fit(df)
        
        assert model.is_fitted is True
    
    def test_fit_with_custom_column_names(self):
        """Test fitting with custom column names."""
        model = TimeSeriesModel()
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = np.random.uniform(3000, 3500, 100)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'price': prices
        })
        
        # Fit with custom column names
        model.fit(df, date_col='timestamp', target_col='price')
        
        assert model.is_fitted is True
    
    def test_fit_with_missing_values(self):
        """Test fitting with missing values (Prophet handles internally)."""
        model = TimeSeriesModel()
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = np.random.uniform(3000, 3500, 100)
        prices[10:15] = np.nan  # Introduce missing values
        
        df = pd.DataFrame({
            'ds': dates,
            'y': prices
        })
        
        # Should fit successfully (Prophet handles NaN)
        model.fit(df)
        
        assert model.is_fitted is True
    
    def test_fit_with_unsorted_data(self):
        """Test fitting with unsorted data (should be sorted internally)."""
        model = TimeSeriesModel()
        
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        prices = np.random.uniform(3000, 3500, 50)
        
        df = pd.DataFrame({
            'ds': dates,
            'y': prices
        })
        
        # Shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Should fit successfully (sorted internally)
        model.fit(df)
        
        assert model.is_fitted is True
    
    def test_fit_with_string_dates(self):
        """Test fitting with string dates (should be converted)."""
        model = TimeSeriesModel()
        
        df = pd.DataFrame({
            'ds': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'],
            'y': [3000.0, 3100.0, 3050.0, 3150.0, 3200.0]
        })
        
        # Should fit successfully (dates converted internally)
        model.fit(df)
        
        assert model.is_fitted is True
    
    def test_fit_missing_date_column_raises_error(self):
        """Test that missing date column raises ValueError."""
        model = TimeSeriesModel()
        
        df = pd.DataFrame({
            'y': [3000.0, 3100.0, 3200.0]
        })
        
        with pytest.raises(ValueError, match="Date column 'ds' not found"):
            model.fit(df)
    
    def test_fit_missing_target_column_raises_error(self):
        """Test that missing target column raises ValueError."""
        model = TimeSeriesModel()
        
        df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=3)
        })
        
        with pytest.raises(ValueError, match="Target column 'y' not found"):
            model.fit(df)
    
    def test_fit_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        model = TimeSeriesModel()
        
        df = pd.DataFrame({
            'ds': [pd.Timestamp('2020-01-01')],
            'y': [3000.0]
        })
        
        with pytest.raises(ValueError, match="Need at least 2 data points"):
            model.fit(df)
    
    def test_repr_after_fitting(self):
        """Test string representation after model is fitted."""
        model = TimeSeriesModel()
        
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        prices = np.random.uniform(3000, 3500, 50)
        df = pd.DataFrame({'ds': dates, 'y': prices})
        
        model.fit(df)
        repr_str = repr(model)
        
        assert "fitted" in repr_str
        assert "not fitted" not in repr_str


class TestTimeSeriesModelPredict:
    """Test prediction generation for future periods."""
    
    def test_predict_future_periods(self):
        """Test generating predictions for future periods."""
        model = TimeSeriesModel()
        
        # Create and fit with synthetic data
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        trend = np.linspace(3000, 3500, 365)
        seasonality = 200 * np.sin(2 * np.pi * np.arange(365) / 365)
        prices = trend + seasonality
        
        df = pd.DataFrame({'ds': dates, 'y': prices})
        model.fit(df)
        
        # Predict 30 days ahead
        forecast = model.predict(periods=30, freq='D')
        
        # Check forecast structure
        assert len(forecast) == 365 + 30  # Historical + future
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns
        assert 'yhat_lower' in forecast.columns
        assert 'yhat_upper' in forecast.columns
        assert 'trend' in forecast.columns
        assert 'yearly' in forecast.columns  # yearly_seasonality=True by default
    
    def test_predict_different_horizons(self):
        """Test predictions for different time horizons."""
        model = TimeSeriesModel()
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = np.random.uniform(3000, 3500, 100)
        df = pd.DataFrame({'ds': dates, 'y': prices})
        
        model.fit(df)
        
        # Test different horizons
        forecast_1 = model.predict(periods=1)
        forecast_7 = model.predict(periods=7)
        forecast_30 = model.predict(periods=30)
        
        assert len(forecast_1) == 101  # 100 historical + 1 future
        assert len(forecast_7) == 107  # 100 historical + 7 future
        assert len(forecast_30) == 130  # 100 historical + 30 future
    
    def test_predict_weekly_frequency(self):
        """Test predictions with weekly frequency."""
        model = TimeSeriesModel()
        
        dates = pd.date_range('2020-01-01', periods=52, freq='W')
        prices = np.random.uniform(3000, 3500, 52)
        df = pd.DataFrame({'ds': dates, 'y': prices})
        
        model.fit(df)
        
        # Predict 4 weeks ahead
        forecast = model.predict(periods=4, freq='W')
        
        assert len(forecast) == 56  # 52 historical + 4 future
    
    def test_predict_confidence_intervals(self):
        """Test that confidence intervals are generated."""
        model = TimeSeriesModel()
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = np.random.uniform(3000, 3500, 100)
        df = pd.DataFrame({'ds': dates, 'y': prices})
        
        model.fit(df)
        forecast = model.predict(periods=10)
        
        # Check confidence intervals exist and are reasonable
        assert (forecast['yhat_lower'] <= forecast['yhat']).all()
        assert (forecast['yhat'] <= forecast['yhat_upper']).all()
        assert (forecast['yhat_upper'] > forecast['yhat_lower']).all()
    
    def test_predict_without_yearly_seasonality(self):
        """Test predictions when yearly seasonality is disabled."""
        model = TimeSeriesModel(yearly_seasonality=False)
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = np.random.uniform(3000, 3500, 100)
        df = pd.DataFrame({'ds': dates, 'y': prices})
        
        model.fit(df)
        forecast = model.predict(periods=10)
        
        # yearly column should not be present
        assert 'yearly' not in forecast.columns
        assert 'trend' in forecast.columns
    
    def test_predict_before_fit_raises_error(self):
        """Test that predicting before fitting raises RuntimeError."""
        model = TimeSeriesModel()
        
        with pytest.raises(RuntimeError, match="Model must be fitted before making predictions"):
            model.predict(periods=10)
    
    def test_predict_invalid_periods_raises_error(self):
        """Test that invalid periods raises ValueError."""
        model = TimeSeriesModel()
        
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        prices = np.random.uniform(3000, 3500, 50)
        df = pd.DataFrame({'ds': dates, 'y': prices})
        
        model.fit(df)
        
        with pytest.raises(ValueError, match="periods must be positive"):
            model.predict(periods=0)
        
        with pytest.raises(ValueError, match="periods must be positive"):
            model.predict(periods=-5)


class TestTimeSeriesModelComputeResiduals:
    """Test residual computation."""
    
    def test_compute_residuals_basic(self):
        """Test basic residual computation."""
        model = TimeSeriesModel()
        
        # Create synthetic data with known pattern
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = np.linspace(3000, 3500, 100)  # Linear trend
        
        df = pd.DataFrame({'ds': dates, 'y': prices})
        model.fit(df)
        
        # Compute residuals
        residuals = model.compute_residuals(df)
        
        # Check residuals structure
        assert len(residuals) == 100
        assert isinstance(residuals, pd.Series)
        assert isinstance(residuals.index[0], pd.Timestamp)
    
    def test_residuals_mean_near_zero(self):
        """Test that residuals have mean close to zero for good fit."""
        model = TimeSeriesModel()
        
        # Create data that Prophet should fit well
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        trend = np.linspace(3000, 3500, 365)
        seasonality = 200 * np.sin(2 * np.pi * np.arange(365) / 365)
        prices = trend + seasonality
        
        df = pd.DataFrame({'ds': dates, 'y': prices})
        model.fit(df)
        
        residuals = model.compute_residuals(df)
        
        # Mean should be close to zero (within reasonable tolerance)
        assert abs(residuals.mean()) < 50  # Tolerance for synthetic data
    
    def test_residuals_capture_unexplained_variation(self):
        """Test that residuals capture variation not explained by the model."""
        model = TimeSeriesModel()
        
        # Create data with trend + noise
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        trend = np.linspace(3000, 3500, 100)
        noise = np.random.normal(0, 100, 100)
        prices = trend + noise
        
        df = pd.DataFrame({'ds': dates, 'y': prices})
        model.fit(df)
        
        residuals = model.compute_residuals(df)
        
        # Residuals should have non-zero standard deviation
        assert residuals.std() > 0
        # Residuals should be smaller than original price variation
        assert residuals.std() < df['y'].std()
    
    def test_residuals_indexed_by_date(self):
        """Test that residuals are indexed by date."""
        model = TimeSeriesModel()
        
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        prices = np.random.uniform(3000, 3500, 50)
        df = pd.DataFrame({'ds': dates, 'y': prices})
        
        model.fit(df)
        residuals = model.compute_residuals(df)
        
        # Check index is datetime
        assert isinstance(residuals.index, pd.DatetimeIndex)
        # Check dates match
        assert (residuals.index == dates).all()
    
    def test_compute_residuals_before_fit_raises_error(self):
        """Test that computing residuals before fitting raises RuntimeError."""
        model = TimeSeriesModel()
        
        df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=10),
            'y': np.random.uniform(3000, 3500, 10)
        })
        
        with pytest.raises(RuntimeError, match="Model must be fitted before computing residuals"):
            model.compute_residuals(df)
    
    def test_compute_residuals_missing_columns_raises_error(self):
        """Test that missing columns raises ValueError."""
        model = TimeSeriesModel()
        
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        prices = np.random.uniform(3000, 3500, 50)
        df = pd.DataFrame({'ds': dates, 'y': prices})
        
        model.fit(df)
        
        # Try with missing columns
        df_invalid = pd.DataFrame({'timestamp': dates})
        
        with pytest.raises(ValueError, match="DataFrame must have 'ds' and 'y' columns"):
            model.compute_residuals(df_invalid)
    
    def test_residuals_positive_and_negative(self):
        """Test that residuals include both positive and negative values."""
        model = TimeSeriesModel()
        
        # Create data with random noise
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        trend = np.linspace(3000, 3500, 100)
        noise = np.random.normal(0, 50, 100)
        prices = trend + noise
        
        df = pd.DataFrame({'ds': dates, 'y': prices})
        model.fit(df)
        
        residuals = model.compute_residuals(df)
        
        # Should have both positive and negative residuals
        assert (residuals > 0).any()
        assert (residuals < 0).any()


class TestTimeSeriesModelGetComponents:
    """Test extraction of trend and seasonal components."""
    
    def test_get_components_basic(self):
        """Test basic component extraction."""
        model = TimeSeriesModel()
        
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        trend = np.linspace(3000, 3500, 365)
        seasonality = 200 * np.sin(2 * np.pi * np.arange(365) / 365)
        prices = trend + seasonality
        
        df = pd.DataFrame({'ds': dates, 'y': prices})
        model.fit(df)
        
        components = model.get_components()
        
        # Check structure
        assert 'ds' in components.columns
        assert 'trend' in components.columns
        assert 'yearly' in components.columns
        assert 'yhat' in components.columns
        assert len(components) == 365
    
    def test_get_components_without_yearly_seasonality(self):
        """Test component extraction when yearly seasonality is disabled."""
        model = TimeSeriesModel(yearly_seasonality=False)
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = np.random.uniform(3000, 3500, 100)
        df = pd.DataFrame({'ds': dates, 'y': prices})
        
        model.fit(df)
        components = model.get_components()
        
        # yearly column should not be present
        assert 'yearly' not in components.columns
        assert 'trend' in components.columns
        assert 'yhat' in components.columns
    
    def test_get_components_with_weekly_seasonality(self):
        """Test component extraction with weekly seasonality enabled."""
        model = TimeSeriesModel(weekly_seasonality=True)
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = np.random.uniform(3000, 3500, 100)
        df = pd.DataFrame({'ds': dates, 'y': prices})
        
        model.fit(df)
        components = model.get_components()
        
        # weekly column should be present
        assert 'weekly' in components.columns
        assert 'trend' in components.columns
    
    def test_trend_component_captures_direction(self):
        """Test that trend component captures overall direction."""
        model = TimeSeriesModel()
        
        # Create data with clear upward trend
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        trend = np.linspace(3000, 4000, 365)  # Clear upward trend
        noise = np.random.normal(0, 20, 365)
        prices = trend + noise
        
        df = pd.DataFrame({'ds': dates, 'y': prices})
        model.fit(df)
        
        components = model.get_components()
        
        # Trend should be increasing
        assert components['trend'].iloc[-1] > components['trend'].iloc[0]
        # Trend should be relatively smooth
        trend_diff = components['trend'].diff().dropna()
        assert trend_diff.std() < 50  # Low variation in trend changes
    
    def test_get_components_before_fit_raises_error(self):
        """Test that getting components before fitting raises RuntimeError."""
        model = TimeSeriesModel()
        
        with pytest.raises(RuntimeError, match="Model must be fitted before extracting components"):
            model.get_components()
    
    def test_components_sum_to_prediction(self):
        """Test that components approximately sum to the prediction."""
        model = TimeSeriesModel(seasonality_mode='additive')
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = np.random.uniform(3000, 3500, 100)
        df = pd.DataFrame({'ds': dates, 'y': prices})
        
        model.fit(df)
        components = model.get_components()
        
        # For additive model: yhat ≈ trend + yearly
        # Check a few points
        for i in [0, 50, 99]:
            expected = components['trend'].iloc[i] + components['yearly'].iloc[i]
            actual = components['yhat'].iloc[i]
            # Allow some tolerance due to other components
            assert abs(expected - actual) < 100


class TestTimeSeriesModelIntegration:
    """Test integration scenarios combining multiple operations."""
    
    def test_full_workflow_fit_predict_residuals(self):
        """Test complete workflow: fit, predict, compute residuals."""
        model = TimeSeriesModel()
        
        # Create synthetic data
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        trend = np.linspace(3000, 3500, 365)
        seasonality = 200 * np.sin(2 * np.pi * np.arange(365) / 365)
        noise = np.random.normal(0, 30, 365)
        prices = trend + seasonality + noise
        
        df = pd.DataFrame({'ds': dates, 'y': prices})
        
        # Fit the model
        model.fit(df)
        assert model.is_fitted is True
        
        # Generate predictions
        forecast = model.predict(periods=30)
        assert len(forecast) == 395  # 365 + 30
        assert 'yhat' in forecast.columns
        
        # Compute residuals
        residuals = model.compute_residuals(df)
        assert len(residuals) == 365
        assert abs(residuals.mean()) < 50
        
        # Get components
        components = model.get_components()
        assert len(components) == 365
        assert 'trend' in components.columns
    
    def test_multiple_predictions_consistent(self):
        """Test that multiple prediction calls produce consistent point predictions."""
        model = TimeSeriesModel()
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = np.random.uniform(3000, 3500, 100)
        df = pd.DataFrame({'ds': dates, 'y': prices})
        
        model.fit(df)
        
        # Make two predictions
        forecast1 = model.predict(periods=10)
        forecast2 = model.predict(periods=10)
        
        # Point predictions (yhat) should be identical
        # Note: Confidence intervals may vary slightly due to Prophet's internal sampling
        pd.testing.assert_series_equal(forecast1['yhat'], forecast2['yhat'])
        pd.testing.assert_series_equal(forecast1['ds'], forecast2['ds'])
        pd.testing.assert_series_equal(forecast1['trend'], forecast2['trend'])
    
    def test_model_with_realistic_cocoa_data_pattern(self):
        """Test model with realistic cocoa price patterns."""
        model = TimeSeriesModel()
        
        # Simulate realistic cocoa price data
        # - Base price around 3000-3500 USD/MT
        # - Yearly seasonality (harvest cycles)
        # - Some volatility
        dates = pd.date_range('2018-01-01', periods=1095, freq='D')  # 3 years
        
        # Base trend with slight increase
        trend = np.linspace(3000, 3300, 1095)
        
        # Yearly seasonality (harvest in Oct-Mar)
        yearly_cycle = 300 * np.sin(2 * np.pi * np.arange(1095) / 365 - np.pi/2)
        
        # Market volatility
        volatility = np.random.normal(0, 80, 1095)
        
        # Occasional shocks
        shocks = np.zeros(1095)
        shock_indices = [200, 500, 800]
        for idx in shock_indices:
            shocks[idx:idx+30] = np.random.uniform(200, 400)
        
        prices = trend + yearly_cycle + volatility + shocks
        
        df = pd.DataFrame({'ds': dates, 'y': prices})
        
        # Fit and predict
        model.fit(df)
        forecast = model.predict(periods=90)  # 3 months ahead
        
        # Verify reasonable predictions
        future_predictions = forecast.tail(90)
        assert future_predictions['yhat'].min() > 2500  # Reasonable lower bound
        assert future_predictions['yhat'].max() < 4500  # Reasonable upper bound
        
        # Verify confidence intervals exist and are reasonable
        # Note: CI width may not always strictly increase due to seasonal patterns
        future_predictions = forecast.tail(90)
        ci_widths = future_predictions['yhat_upper'] - future_predictions['yhat_lower']
        assert ci_widths.min() > 0  # All CIs should be positive
        assert ci_widths.mean() > 100  # Reasonable width for cocoa prices


class TestTimeSeriesModelEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_short_time_series(self):
        """Test with minimal data points."""
        model = TimeSeriesModel()
        
        # Minimum 2 points
        df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=2),
            'y': [3000.0, 3100.0]
        })
        
        model.fit(df)
        forecast = model.predict(periods=1)
        
        assert len(forecast) == 3  # 2 historical + 1 future
    
    def test_constant_values(self):
        """Test with constant price values."""
        model = TimeSeriesModel()
        
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        prices = np.full(50, 3000.0)  # All same value
        
        df = pd.DataFrame({'ds': dates, 'y': prices})
        model.fit(df)
        
        forecast = model.predict(periods=10)
        
        # Predictions should be close to constant value
        future_predictions = forecast.tail(10)
        assert abs(future_predictions['yhat'].mean() - 3000.0) < 100
    
    def test_high_volatility_data(self):
        """Test with highly volatile data."""
        model = TimeSeriesModel()
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        # High volatility: random walk with large steps
        prices = 3000 + np.cumsum(np.random.normal(0, 200, 100))
        
        df = pd.DataFrame({'ds': dates, 'y': prices})
        model.fit(df)
        
        forecast = model.predict(periods=10)
        
        # Should still produce predictions with wide confidence intervals
        future_predictions = forecast.tail(10)
        ci_width = future_predictions['yhat_upper'] - future_predictions['yhat_lower']
        assert ci_width.mean() > 100  # Wide confidence intervals


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
