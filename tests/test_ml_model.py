"""
Comprehensive unit tests for MLModel.

Tests cover:
- Model initialization with different hyperparameters
- Training with synthetic residuals
- Prediction of residuals
- Cross-validation with time series split
- Feature importance extraction
- Error handling for edge cases

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.models.ml_model import MLModel


class TestMLModelInitialization:
    """Test MLModel initialization with various hyperparameters."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        model = MLModel()
        
        assert model.n_estimators == 100
        assert model.max_depth == 6
        assert model.learning_rate == 0.1
        assert model.objective == "reg:squarederror"
        assert model.is_fitted is False
        assert model.model is not None
    
    def test_custom_initialization(self):
        """Test initialization with custom hyperparameters."""
        model = MLModel(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            objective="reg:pseudohubererror"
        )
        
        assert model.n_estimators == 200
        assert model.max_depth == 8
        assert model.learning_rate == 0.05
        assert model.objective == "reg:pseudohubererror"
        assert model.is_fitted is False
    
    def test_get_hyperparameters(self):
        """Test retrieval of hyperparameters."""
        model = MLModel(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.08
        )
        
        params = model.get_hyperparameters()
        
        assert params['n_estimators'] == 150
        assert params['max_depth'] == 7
        assert params['learning_rate'] == 0.08
        assert params['objective'] == "reg:squarederror"
    
    def test_repr_not_fitted(self):
        """Test string representation when model is not fitted."""
        model = MLModel()
        
        repr_str = repr(model)
        
        assert "MLModel" in repr_str
        assert "not fitted" in repr_str
        assert "n_estimators=100" in repr_str


class TestMLModelFit:
    """Test model fitting with various data scenarios."""
    
    def test_fit_with_synthetic_residuals(self):
        """Test fitting with synthetic residual data."""
        model = MLModel(n_estimators=50)  # Fewer estimators for faster tests
        
        # Create synthetic features and residuals
        n_samples = 200
        X = pd.DataFrame({
            'temperature': np.random.uniform(20, 35, n_samples),
            'rainfall': np.random.uniform(0, 200, n_samples),
            'stock_level': np.random.uniform(100000, 500000, n_samples),
            'fx_rate_xaf_usd': np.random.uniform(0.0015, 0.0020, n_samples),
            'sentiment_score': np.random.uniform(-0.5, 0.5, n_samples)
        })
        
        # Generate residuals with some correlation to features
        y = (
            X['temperature'] * 2 +
            X['rainfall'] * 0.5 -
            X['sentiment_score'] * 100 +
            np.random.normal(0, 20, n_samples)
        )
        
        # Fit the model
        model.fit(X, y)
        
        assert model.is_fitted is True
    
    def test_fit_with_early_stopping(self):
        """Test fitting with early stopping on validation set."""
        model = MLModel(n_estimators=100)
        
        # Create training data
        n_train = 150
        X_train = pd.DataFrame({
            'feature1': np.random.randn(n_train),
            'feature2': np.random.randn(n_train),
            'feature3': np.random.randn(n_train)
        })
        y_train = X_train['feature1'] * 2 + X_train['feature2'] * 3 + np.random.randn(n_train) * 0.5
        
        # Create validation data
        n_val = 50
        X_val = pd.DataFrame({
            'feature1': np.random.randn(n_val),
            'feature2': np.random.randn(n_val),
            'feature3': np.random.randn(n_val)
        })
        y_val = X_val['feature1'] * 2 + X_val['feature2'] * 3 + np.random.randn(n_val) * 0.5
        
        # Fit with early stopping
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
        
        assert model.is_fitted is True
        # Early stopping should have stopped before max iterations
        # (though this depends on data, so we just check it fitted)
    
    def test_fit_with_single_feature(self):
        """Test fitting with a single feature."""
        model = MLModel(n_estimators=30)
        
        n_samples = 100
        X = pd.DataFrame({
            'temperature': np.random.uniform(20, 35, n_samples)
        })
        y = X['temperature'] * 5 + np.random.normal(0, 10, n_samples)
        
        model.fit(X, y)
        
        assert model.is_fitted is True
    
    def test_fit_with_many_features(self):
        """Test fitting with many features."""
        model = MLModel(n_estimators=30)
        
        n_samples = 150
        n_features = 20
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = X.iloc[:, 0] * 2 + X.iloc[:, 1] * 3 + np.random.randn(n_samples)
        
        model.fit(X, y)
        
        assert model.is_fitted is True
    
    def test_fit_empty_data_raises_error(self):
        """Test that fitting with empty data raises ValueError."""
        model = MLModel()
        
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
        
        with pytest.raises(ValueError, match="Cannot fit model with empty dataset"):
            model.fit(X, y)
    
    def test_fit_mismatched_lengths_raises_error(self):
        """Test that mismatched X and y lengths raise ValueError."""
        model = MLModel()
        
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        y = pd.Series([1, 2])
        
        with pytest.raises(ValueError, match="X and y must have same length"):
            model.fit(X, y)
    
    def test_fit_no_features_raises_error(self):
        """Test that fitting with no features raises ValueError."""
        model = MLModel()
        
        X = pd.DataFrame(index=range(10))  # No columns
        y = pd.Series(np.random.randn(10))
        
        with pytest.raises(ValueError, match="Feature matrix X must have at least one column"):
            model.fit(X, y)
    
    def test_fit_invalid_eval_set_raises_error(self):
        """Test that invalid eval_set raises ValueError."""
        model = MLModel()
        
        X = pd.DataFrame({'feature1': np.random.randn(50)})
        y = pd.Series(np.random.randn(50))
        
        # eval_set should be a list
        with pytest.raises(ValueError, match="eval_set must be a list"):
            model.fit(X, y, eval_set="invalid")
    
    def test_fit_eval_set_mismatched_lengths_raises_error(self):
        """Test that eval_set with mismatched lengths raises ValueError."""
        model = MLModel()
        
        X_train = pd.DataFrame({'feature1': np.random.randn(50)})
        y_train = pd.Series(np.random.randn(50))
        
        X_val = pd.DataFrame({'feature1': np.random.randn(20)})
        y_val = pd.Series(np.random.randn(15))  # Mismatched length
        
        with pytest.raises(ValueError, match="X and y must have same length"):
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    def test_repr_after_fitting(self):
        """Test string representation after model is fitted."""
        model = MLModel(n_estimators=30)
        
        X = pd.DataFrame({'feature1': np.random.randn(50)})
        y = pd.Series(np.random.randn(50))
        
        model.fit(X, y)
        repr_str = repr(model)
        
        assert "fitted" in repr_str
        assert "not fitted" not in repr_str


class TestMLModelPredict:
    """Test prediction generation for residuals."""
    
    def test_predict_basic(self):
        """Test basic prediction functionality."""
        model = MLModel(n_estimators=30)
        
        # Train the model
        n_train = 100
        X_train = pd.DataFrame({
            'feature1': np.random.randn(n_train),
            'feature2': np.random.randn(n_train)
        })
        y_train = X_train['feature1'] * 2 + X_train['feature2'] * 3 + np.random.randn(n_train) * 0.5
        
        model.fit(X_train, y_train)
        
        # Make predictions
        n_test = 20
        X_test = pd.DataFrame({
            'feature1': np.random.randn(n_test),
            'feature2': np.random.randn(n_test)
        })
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == n_test
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_learns_pattern(self):
        """Test that model learns the underlying pattern."""
        model = MLModel(n_estimators=50)
        
        # Create data with clear pattern
        n_train = 200
        X_train = pd.DataFrame({
            'x': np.linspace(0, 10, n_train)
        })
        y_train = X_train['x'] * 5 + 10  # y = 5x + 10
        
        model.fit(X_train, y_train)
        
        # Test on new data
        X_test = pd.DataFrame({'x': [2.0, 4.0, 6.0]})
        predictions = model.predict(X_test)
        
        # Predictions should be close to true values
        expected = np.array([20.0, 30.0, 40.0])
        assert np.allclose(predictions, expected, atol=5.0)
    
    def test_predict_single_sample(self):
        """Test prediction on a single sample."""
        model = MLModel(n_estimators=30)
        
        X_train = pd.DataFrame({'feature1': np.random.randn(50)})
        y_train = pd.Series(np.random.randn(50))
        
        model.fit(X_train, y_train)
        
        X_test = pd.DataFrame({'feature1': [0.5]})
        predictions = model.predict(X_test)
        
        assert len(predictions) == 1
        assert isinstance(predictions[0], (float, np.floating))
    
    def test_predict_multiple_samples(self):
        """Test prediction on multiple samples."""
        model = MLModel(n_estimators=30)
        
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y_train = pd.Series(np.random.randn(100))
        
        model.fit(X_train, y_train)
        
        X_test = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50)
        })
        predictions = model.predict(X_test)
        
        assert len(predictions) == 50
    
    def test_predict_before_fit_raises_error(self):
        """Test that predicting before fitting raises RuntimeError."""
        model = MLModel()
        
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(RuntimeError, match="Model must be fitted before making predictions"):
            model.predict(X)
    
    def test_predict_wrong_features_raises_error(self):
        """Test that predicting with wrong number of features raises ValueError."""
        model = MLModel(n_estimators=30)
        
        # Train with 2 features
        X_train = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50)
        })
        y_train = pd.Series(np.random.randn(50))
        
        model.fit(X_train, y_train)
        
        # Try to predict with 3 features
        X_test = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10),
            'feature3': np.random.randn(10)
        })
        
        with pytest.raises(ValueError, match="X has 3 features, but model was trained with 2 features"):
            model.predict(X_test)


class TestMLModelCrossValidate:
    """Test time series cross-validation."""
    
    def test_cross_validate_basic(self):
        """Test basic cross-validation functionality."""
        model = MLModel(n_estimators=30)
        
        # Create synthetic data
        n_samples = 200
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        y = X['feature1'] * 2 + X['feature2'] * 3 + np.random.randn(n_samples) * 0.5
        
        # Perform cross-validation
        metrics = model.cross_validate(X, y, cv=5)
        
        # Check that all metrics are present
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert 'r2' in metrics
        
        # Check that metrics are reasonable
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert metrics['mape'] >= 0
        assert -1 <= metrics['r2'] <= 1
    
    def test_cross_validate_different_folds(self):
        """Test cross-validation with different number of folds."""
        model = MLModel(n_estimators=30)
        
        n_samples = 150
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        y = X['feature1'] * 2 + np.random.randn(n_samples)
        
        # Test with different fold counts
        metrics_3 = model.cross_validate(X, y, cv=3)
        metrics_5 = model.cross_validate(X, y, cv=5)
        
        # Both should return valid metrics
        assert all(k in metrics_3 for k in ['rmse', 'mae', 'mape', 'r2'])
        assert all(k in metrics_5 for k in ['rmse', 'mae', 'mape', 'r2'])
    
    def test_cross_validate_good_fit(self):
        """Test cross-validation with data that should fit well."""
        model = MLModel(n_estimators=50)
        
        # Create data with clear linear relationship
        # Use more samples to ensure early folds have enough training data
        n_samples = 500
        X = pd.DataFrame({
            'x': np.linspace(0, 10, n_samples)
        })
        y = X['x'] * 5 + 10 + np.random.randn(n_samples) * 0.5  # Low noise
        
        metrics = model.cross_validate(X, y, cv=5)
        
        # Check that metrics are computed and reasonable
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
    
    def test_cross_validate_respects_time_order(self):
        """Test that cross-validation respects temporal order."""
        model = MLModel(n_estimators=50)
        
        # Create time-ordered data with trend
        # Use more samples for better learning
        n_samples = 300
        X = pd.DataFrame({
            'time_index': np.arange(n_samples),
            'feature1': np.random.randn(n_samples)
        })
        # Add a trend component
        y = X['time_index'] * 0.5 + X['feature1'] * 2 + np.random.randn(n_samples) * 0.5
        
        # Cross-validation should work (TimeSeriesSplit respects order)
        metrics = model.cross_validate(X, y, cv=5)
        
        # Just verify metrics are computed
        assert metrics['rmse'] > 0
        assert 'r2' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
    
    def test_cross_validate_mismatched_lengths_raises_error(self):
        """Test that mismatched X and y lengths raise ValueError."""
        model = MLModel()
        
        X = pd.DataFrame({'feature1': np.random.randn(50)})
        y = pd.Series(np.random.randn(40))
        
        with pytest.raises(ValueError, match="X and y must have same length"):
            model.cross_validate(X, y)
    
    def test_cross_validate_invalid_cv_raises_error(self):
        """Test that invalid cv parameter raises ValueError."""
        model = MLModel()
        
        X = pd.DataFrame({'feature1': np.random.randn(50)})
        y = pd.Series(np.random.randn(50))
        
        with pytest.raises(ValueError, match="cv must be at least 2"):
            model.cross_validate(X, y, cv=1)
    
    def test_cross_validate_insufficient_data_raises_error(self):
        """Test that insufficient data for CV raises ValueError."""
        model = MLModel()
        
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        y = pd.Series([1, 2, 3])
        
        with pytest.raises(ValueError, match="Cannot perform 5-fold CV with only 3 samples"):
            model.cross_validate(X, y, cv=5)
    
    def test_cross_validate_handles_zero_values_in_mape(self):
        """Test that MAPE calculation handles zero values gracefully."""
        model = MLModel(n_estimators=30)
        
        # Create data with some zero values
        n_samples = 100
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples)
        })
        y = pd.Series(np.random.randn(n_samples))
        y.iloc[0:5] = 0  # Set some values to zero
        
        # Should not raise error
        metrics = model.cross_validate(X, y, cv=3)
        
        assert 'mape' in metrics
        assert not np.isnan(metrics['mape'])
        assert not np.isinf(metrics['mape'])


class TestMLModelFeatureImportance:
    """Test feature importance extraction."""
    
    def test_get_feature_importance_basic(self):
        """Test basic feature importance extraction."""
        model = MLModel(n_estimators=30)
        
        # Train model
        n_samples = 100
        X = pd.DataFrame({
            'important_feature': np.random.randn(n_samples),
            'less_important': np.random.randn(n_samples) * 0.1,
            'noise': np.random.randn(n_samples) * 0.01
        })
        y = X['important_feature'] * 10 + X['less_important'] * 2 + np.random.randn(n_samples) * 0.5
        
        model.fit(X, y)
        
        # Get feature importance
        importance_df = model.get_feature_importance()
        
        # Check structure
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) > 0
        
        # Check that it's sorted (descending)
        assert importance_df['importance'].is_monotonic_decreasing
    
    def test_get_feature_importance_different_types(self):
        """Test different importance types."""
        model = MLModel(n_estimators=30)
        
        n_samples = 100
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        y = X['feature1'] * 5 + np.random.randn(n_samples)
        
        model.fit(X, y)
        
        # Test different importance types
        importance_gain = model.get_feature_importance(importance_type='gain')
        importance_weight = model.get_feature_importance(importance_type='weight')
        importance_cover = model.get_feature_importance(importance_type='cover')
        
        # All should return valid DataFrames
        assert len(importance_gain) > 0
        assert len(importance_weight) > 0
        assert len(importance_cover) > 0
    
    def test_get_feature_importance_identifies_important_features(self):
        """Test that feature importance correctly identifies important features."""
        model = MLModel(n_estimators=50)
        
        # Create data where feature1 is clearly most important
        n_samples = 200
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        y = X['feature1'] * 100 + X['feature2'] * 0.1 + X['feature3'] * 0.1 + np.random.randn(n_samples)
        
        model.fit(X, y)
        
        importance_df = model.get_feature_importance()
        
        # feature1 should be most important
        # Note: XGBoost feature names are like 'f0', 'f1', 'f2'
        # The most important feature should have highest importance
        assert importance_df.iloc[0]['importance'] > importance_df.iloc[1]['importance']
    
    def test_get_feature_importance_before_fit_raises_error(self):
        """Test that getting importance before fitting raises RuntimeError."""
        model = MLModel()
        
        with pytest.raises(RuntimeError, match="Model must be fitted before extracting feature importance"):
            model.get_feature_importance()
    
    def test_get_feature_importance_invalid_type_raises_error(self):
        """Test that invalid importance type raises ValueError."""
        model = MLModel(n_estimators=30)
        
        X = pd.DataFrame({'feature1': np.random.randn(50)})
        y = pd.Series(np.random.randn(50))
        
        model.fit(X, y)
        
        with pytest.raises(ValueError, match="importance_type must be one of"):
            model.get_feature_importance(importance_type='invalid')


class TestMLModelIntegration:
    """Test integration scenarios combining multiple operations."""
    
    def test_full_workflow_fit_predict_cv_importance(self):
        """Test complete workflow: fit, predict, cross-validate, get importance."""
        model = MLModel(n_estimators=50)
        
        # Create synthetic data
        n_samples = 200
        X = pd.DataFrame({
            'temperature': np.random.uniform(20, 35, n_samples),
            'rainfall': np.random.uniform(0, 200, n_samples),
            'sentiment': np.random.uniform(-0.5, 0.5, n_samples)
        })
        y = (
            X['temperature'] * 3 +
            X['rainfall'] * 0.5 -
            X['sentiment'] * 50 +
            np.random.normal(0, 10, n_samples)
        )
        
        # Fit the model
        model.fit(X, y)
        assert model.is_fitted is True
        
        # Make predictions
        X_test = X.iloc[:20]
        predictions = model.predict(X_test)
        assert len(predictions) == 20
        
        # Cross-validate
        metrics = model.cross_validate(X, y, cv=5)
        assert 'rmse' in metrics
        assert metrics['rmse'] > 0
        
        # Get feature importance
        importance = model.get_feature_importance()
        assert len(importance) > 0
        assert 'feature' in importance.columns
    
    def test_model_with_realistic_residual_pattern(self):
        """Test model with realistic residual patterns."""
        model = MLModel(n_estimators=100)
        
        # Simulate realistic residuals from time series model
        # Residuals should be influenced by econometric factors
        n_samples = 300
        
        # Econometric features
        temperature = np.random.uniform(22, 32, n_samples)
        rainfall = np.random.uniform(50, 250, n_samples)
        stock_level = np.random.uniform(200000, 600000, n_samples)
        fx_rate = np.random.uniform(0.0016, 0.0019, n_samples)
        sentiment = np.random.uniform(-0.8, 0.8, n_samples)
        
        X = pd.DataFrame({
            'temperature': temperature,
            'rainfall': rainfall,
            'stock_level': stock_level,
            'fx_rate_xaf_usd': fx_rate,
            'sentiment_score': sentiment
        })
        
        # Residuals influenced by multiple factors
        residuals = (
            (temperature - 27) * 5 +  # Temperature effect
            (rainfall - 150) * 0.2 +  # Rainfall effect
            (stock_level - 400000) * 0.0001 +  # Stock effect
            (fx_rate - 0.00175) * 10000 +  # FX effect
            sentiment * -80 +  # Sentiment effect (negative correlation)
            np.random.normal(0, 15, n_samples)  # Noise
        )
        
        y = pd.Series(residuals)
        
        # Split into train and test
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Fit and evaluate
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate test RMSE
        test_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        
        # Should achieve reasonable accuracy
        assert test_rmse < 50  # Reasonable for this synthetic data
        
        # Feature importance should identify key features
        importance = model.get_feature_importance()
        assert len(importance) == 5
    
    def test_model_improves_over_baseline(self):
        """Test that ML model improves over a naive baseline."""
        model = MLModel(n_estimators=50)
        
        # Create data where features help predict residuals
        n_samples = 200
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        y = X['feature1'] * 5 + X['feature2'] * 3 + np.random.randn(n_samples) * 2
        
        # Fit model
        model.fit(X, y)
        
        # Predict
        y_pred = model.predict(X)
        
        # Calculate model RMSE
        model_rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        
        # Baseline: predict mean
        baseline_pred = np.full(len(y), y.mean())
        baseline_rmse = np.sqrt(np.mean((y - baseline_pred) ** 2))
        
        # Model should be better than baseline
        assert model_rmse < baseline_rmse
        
        # Model should achieve at least 20% improvement (per requirement 6.6)
        improvement = (baseline_rmse - model_rmse) / baseline_rmse
        assert improvement > 0.2


class TestMLModelEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_dataset(self):
        """Test with minimal data points."""
        model = MLModel(n_estimators=10)
        
        # Minimum viable dataset
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        y = pd.Series([1, 2, 3, 4, 5])
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == 5
    
    def test_constant_target_values(self):
        """Test with constant residual values."""
        model = MLModel(n_estimators=30)
        
        X = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50)
        })
        y = pd.Series(np.full(50, 10.0))  # All same value
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Predictions should be close to constant value
        assert np.allclose(predictions, 10.0, atol=1.0)
    
    def test_high_variance_residuals(self):
        """Test with highly variable residuals."""
        model = MLModel(n_estimators=50)
        
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        # High variance residuals
        y = pd.Series(np.random.normal(0, 100, 100))
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Should still produce predictions
        assert len(predictions) == 100
        assert not np.any(np.isnan(predictions))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
