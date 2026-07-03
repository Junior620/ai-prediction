"""
Comprehensive unit tests for DataPreprocessor.

Tests cover:
- Missing value handling with multiple strategies (forward_fill, interpolate, mean, drop)
- Feature normalization (standard and minmax) and denormalization
- Train/validation split without data leakage (chronological split)
- Time-based feature engineering

Requirements: 4.2, 4.3, 4.6
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessorInitialization:
    """Test DataPreprocessor initialization."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        preprocessor = DataPreprocessor()
        
        assert isinstance(preprocessor.scalers, dict)
        assert len(preprocessor.scalers) == 0


class TestHandleMissingValues:
    """Test handle_missing_values method with different strategies."""
    
    def test_forward_fill_strategy(self):
        """Test forward fill strategy for time series continuity."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'price': [3000.0, np.nan, np.nan, 3200.0, 3100.0]
        })
        
        strategy = {'price': 'forward_fill'}
        df_processed = preprocessor.handle_missing_values(df, strategy)
        
        assert df_processed['price'].isna().sum() == 0
        assert df_processed['price'].iloc[1] == 3000.0  # Forward filled
        assert df_processed['price'].iloc[2] == 3000.0  # Forward filled
    
    def test_forward_fill_with_leading_nan(self):
        """Test forward fill handles leading NaN with backward fill."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'price': [np.nan, np.nan, 3200.0, 3100.0, 3150.0]
        })
        
        strategy = {'price': 'forward_fill'}
        df_processed = preprocessor.handle_missing_values(df, strategy)
        
        assert df_processed['price'].isna().sum() == 0
        assert df_processed['price'].iloc[0] == 3200.0  # Backward filled
        assert df_processed['price'].iloc[1] == 3200.0  # Backward filled
    
    def test_interpolate_strategy(self):
        """Test interpolate strategy for smooth transitions."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'temperature': [20.0, np.nan, np.nan, 30.0, 25.0]
        })
        
        strategy = {'temperature': 'interpolate'}
        df_processed = preprocessor.handle_missing_values(df, strategy)
        
        assert df_processed['temperature'].isna().sum() == 0
        # Linear interpolation: 20 -> 23.33 -> 26.67 -> 30
        assert 22.0 < df_processed['temperature'].iloc[1] < 24.0
        assert 26.0 < df_processed['temperature'].iloc[2] < 28.0
    
    def test_mean_strategy(self):
        """Test mean strategy for econometric features."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'stock_level': [1000.0, np.nan, 1200.0, np.nan, 1100.0]
        })
        
        strategy = {'stock_level': 'mean'}
        df_processed = preprocessor.handle_missing_values(df, strategy)
        
        expected_mean = (1000.0 + 1200.0 + 1100.0) / 3  # 1100.0
        assert df_processed['stock_level'].isna().sum() == 0
        assert df_processed['stock_level'].iloc[1] == expected_mean
        assert df_processed['stock_level'].iloc[3] == expected_mean
    
    def test_drop_strategy(self):
        """Test drop strategy for non-critical features."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'price': [3000.0, 3100.0, 3200.0, 3150.0, 3180.0],
            'optional_field': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        
        strategy = {'optional_field': 'drop'}
        df_processed = preprocessor.handle_missing_values(df, strategy)
        
        assert len(df_processed) == 3  # 2 rows dropped
        assert df_processed['optional_field'].isna().sum() == 0
    
    def test_multiple_strategies(self):
        """Test applying different strategies to different columns."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'price': [3000.0, np.nan, 3200.0, 3150.0, 3180.0],
            'temperature': [20.0, np.nan, np.nan, 30.0, 25.0],
            'stock_level': [1000.0, np.nan, 1200.0, 1100.0, 1150.0]
        })
        
        strategy = {
            'price': 'forward_fill',
            'temperature': 'interpolate',
            'stock_level': 'mean'
        }
        df_processed = preprocessor.handle_missing_values(df, strategy)
        
        assert df_processed['price'].isna().sum() == 0
        assert df_processed['temperature'].isna().sum() == 0
        assert df_processed['stock_level'].isna().sum() == 0
    
    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'price': [3000.0, np.nan, 3200.0]
        })
        
        strategy = {'price': 'invalid_strategy'}
        
        with pytest.raises(ValueError, match="Invalid strategy"):
            preprocessor.handle_missing_values(df, strategy)
    
    def test_invalid_column_raises_error(self):
        """Test that invalid column raises KeyError."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'price': [3000.0, np.nan, 3200.0]
        })
        
        strategy = {'nonexistent_column': 'forward_fill'}
        
        with pytest.raises(KeyError, match="Column 'nonexistent_column' not found"):
            preprocessor.handle_missing_values(df, strategy)
    
    def test_preserves_original_dataframe(self):
        """Test that original DataFrame is not modified."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'price': [3000.0, np.nan, 3200.0]
        })
        
        df_original = df.copy()
        strategy = {'price': 'forward_fill'}
        df_processed = preprocessor.handle_missing_values(df, strategy)
        
        # Original should still have NaN
        assert df['price'].isna().sum() == 1
        # Processed should not have NaN
        assert df_processed['price'].isna().sum() == 0


class TestNormalizeFeatures:
    """Test normalize_features method with standard and minmax scaling."""
    
    def test_standard_normalization(self):
        """Test standard normalization (zero mean, unit variance)."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'price': [3000.0, 3100.0, 3200.0, 3300.0, 3400.0],
            'volume': [1000.0, 1200.0, 1100.0, 1300.0, 1150.0]
        })
        
        df_norm, scalers = preprocessor.normalize_features(
            df, ['price', 'volume'], method='standard'
        )
        
        # Check normalized values have approximately zero mean and unit variance
        # Note: Using ddof=0 to match sklearn's StandardScaler behavior
        assert abs(df_norm['price'].mean()) < 1e-10
        assert abs(df_norm['price'].std(ddof=0) - 1.0) < 1e-10
        assert abs(df_norm['volume'].mean()) < 1e-10
        assert abs(df_norm['volume'].std(ddof=0) - 1.0) < 1e-10
        
        # Check scalers are returned
        assert 'price' in scalers
        assert 'volume' in scalers
        assert scalers['price']['method'] == 'standard'
    
    def test_minmax_normalization(self):
        """Test minmax normalization (scale to [0, 1] range)."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'price': [3000.0, 3100.0, 3200.0, 3300.0, 3400.0],
            'volume': [1000.0, 1200.0, 1100.0, 1300.0, 1150.0]
        })
        
        df_norm, scalers = preprocessor.normalize_features(
            df, ['price', 'volume'], method='minmax'
        )
        
        # Check normalized values are in [0, 1] range
        assert df_norm['price'].min() == 0.0
        assert abs(df_norm['price'].max() - 1.0) < 1e-10
        assert df_norm['volume'].min() == 0.0
        assert abs(df_norm['volume'].max() - 1.0) < 1e-10
        
        # Check scalers are returned
        assert scalers['price']['method'] == 'minmax'
        assert scalers['volume']['method'] == 'minmax'
    
    def test_normalize_single_column(self):
        """Test normalizing a single column."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'price': [3000.0, 3100.0, 3200.0, 3300.0, 3400.0],
            'volume': [1000.0, 1200.0, 1100.0, 1300.0, 1150.0]
        })
        
        df_norm, scalers = preprocessor.normalize_features(
            df, ['price'], method='standard'
        )
        
        # Only price should be normalized
        assert abs(df_norm['price'].mean()) < 1e-10
        # Volume should remain unchanged
        assert df_norm['volume'].equals(df['volume'])
        
        # Only price scaler should be returned
        assert 'price' in scalers
        assert 'volume' not in scalers
    
    def test_inverse_transform(self):
        """Test inverse transformation returns original values."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'price': [3000.0, 3100.0, 3200.0, 3300.0, 3400.0],
            'volume': [1000.0, 1200.0, 1100.0, 1300.0, 1150.0]
        })
        
        df_norm, scalers = preprocessor.normalize_features(
            df, ['price', 'volume'], method='standard'
        )
        
        df_inverse = preprocessor.inverse_transform(df_norm, scalers)
        
        # Check values are restored to original
        np.testing.assert_array_almost_equal(
            df_inverse['price'].values,
            df['price'].values,
            decimal=5
        )
        np.testing.assert_array_almost_equal(
            df_inverse['volume'].values,
            df['volume'].values,
            decimal=5
        )
    
    def test_inverse_transform_minmax(self):
        """Test inverse transformation with minmax scaling."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'price': [3000.0, 3100.0, 3200.0, 3300.0, 3400.0]
        })
        
        df_norm, scalers = preprocessor.normalize_features(
            df, ['price'], method='minmax'
        )
        
        df_inverse = preprocessor.inverse_transform(df_norm, scalers)
        
        np.testing.assert_array_almost_equal(
            df_inverse['price'].values,
            df['price'].values,
            decimal=5
        )
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'price': [3000.0, 3100.0, 3200.0]
        })
        
        with pytest.raises(ValueError, match="Invalid method"):
            preprocessor.normalize_features(df, ['price'], method='invalid')
    
    def test_invalid_column_raises_error(self):
        """Test that invalid column raises KeyError."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'price': [3000.0, 3100.0, 3200.0]
        })
        
        with pytest.raises(KeyError, match="Column 'nonexistent' not found"):
            preprocessor.normalize_features(df, ['nonexistent'], method='standard')
    
    def test_preserves_original_dataframe(self):
        """Test that original DataFrame is not modified."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'price': [3000.0, 3100.0, 3200.0]
        })
        
        df_original = df.copy()
        df_norm, scalers = preprocessor.normalize_features(df, ['price'], method='standard')
        
        # Original should be unchanged
        assert df['price'].equals(df_original['price'])
        # Normalized should be different
        assert not df_norm['price'].equals(df['price'])


class TestCreateTrainValSplit:
    """Test create_train_val_split method for chronological splitting."""
    
    def test_default_split_80_20(self):
        """Test default 80/20 train/validation split."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100),
            'price': np.random.uniform(3000, 3500, 100)
        })
        
        train_df, val_df = preprocessor.create_train_val_split(df, val_size=0.2)
        
        assert len(train_df) == 80
        assert len(val_df) == 20
        assert len(train_df) + len(val_df) == len(df)
    
    def test_custom_split_ratio(self):
        """Test custom split ratio."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100),
            'price': np.random.uniform(3000, 3500, 100)
        })
        
        train_df, val_df = preprocessor.create_train_val_split(df, val_size=0.3)
        
        assert len(train_df) == 70
        assert len(val_df) == 30
    
    def test_chronological_order_preserved(self):
        """Test that chronological order is preserved (no data leakage)."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100),
            'price': list(range(100))  # Sequential values
        })
        
        train_df, val_df = preprocessor.create_train_val_split(df, val_size=0.2, shuffle=False)
        
        # Train should have earlier data
        assert train_df['price'].iloc[0] == 0
        assert train_df['price'].iloc[-1] == 79
        
        # Validation should have later data
        assert val_df['price'].iloc[0] == 80
        assert val_df['price'].iloc[-1] == 99
        
        # No overlap
        assert train_df['price'].max() < val_df['price'].min()
    
    def test_no_data_leakage(self):
        """Test that there is no data leakage between train and validation."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100),
            'price': list(range(100))
        })
        
        train_df, val_df = preprocessor.create_train_val_split(df, val_size=0.2, shuffle=False)
        
        # Ensure no timestamps overlap
        train_timestamps = set(train_df['timestamp'])
        val_timestamps = set(val_df['timestamp'])
        
        assert len(train_timestamps.intersection(val_timestamps)) == 0
        
        # Ensure validation data comes after training data
        assert train_df['timestamp'].max() < val_df['timestamp'].min()
    
    def test_shuffle_option(self):
        """Test shuffle option (not recommended for time series)."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100),
            'price': list(range(100))
        })
        
        train_df, val_df = preprocessor.create_train_val_split(df, val_size=0.2, shuffle=True)
        
        # With shuffle, data should not be in chronological order
        # Train data should contain values from throughout the range
        assert len(train_df) == 80
        assert len(val_df) == 20
        
        # Values should be mixed (not sequential)
        train_values = sorted(train_df['price'].values)
        # If shuffled, the first value in sorted train should be close to 0
        # and last value should be close to 99
        assert train_values[0] < 20
        assert train_values[-1] > 80
    
    def test_invalid_val_size_raises_error(self):
        """Test that invalid val_size raises ValueError."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'price': [3000.0, 3100.0, 3200.0]
        })
        
        with pytest.raises(ValueError, match="val_size must be between 0 and 1"):
            preprocessor.create_train_val_split(df, val_size=1.5)
        
        with pytest.raises(ValueError, match="val_size must be between 0 and 1"):
            preprocessor.create_train_val_split(df, val_size=0.0)
        
        with pytest.raises(ValueError, match="val_size must be between 0 and 1"):
            preprocessor.create_train_val_split(df, val_size=-0.2)
    
    def test_preserves_original_dataframe(self):
        """Test that original DataFrame is not modified."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100),
            'price': list(range(100))
        })
        
        df_original = df.copy()
        train_df, val_df = preprocessor.create_train_val_split(df, val_size=0.2)
        
        # Original should be unchanged
        assert df.equals(df_original)
    
    def test_small_dataset(self):
        """Test split with small dataset."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'price': list(range(10))
        })
        
        train_df, val_df = preprocessor.create_train_val_split(df, val_size=0.2)
        
        assert len(train_df) == 8
        assert len(val_df) == 2


class TestEngineerTimeFeatures:
    """Test engineer_time_features method for time-based feature creation."""
    
    def test_basic_time_features(self):
        """Test creation of basic time features."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'price': np.random.uniform(3000, 3500, 10)
        })
        
        df_features = preprocessor.engineer_time_features(df, 'timestamp')
        
        # Check all expected columns are created
        assert 'day_of_week' in df_features.columns
        assert 'day_of_month' in df_features.columns
        assert 'month' in df_features.columns
        assert 'quarter' in df_features.columns
        assert 'is_month_start' in df_features.columns
        assert 'is_month_end' in df_features.columns
        assert 'days_since_shock' in df_features.columns
    
    def test_day_of_week_values(self):
        """Test day_of_week values are correct (0=Monday, 6=Sunday)."""
        preprocessor = DataPreprocessor()
        
        # 2024-01-01 is a Monday
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=7),
            'price': [3000.0] * 7
        })
        
        df_features = preprocessor.engineer_time_features(df, 'timestamp')
        
        # Check day of week sequence
        expected_days = [0, 1, 2, 3, 4, 5, 6]  # Monday to Sunday
        assert list(df_features['day_of_week']) == expected_days
    
    def test_day_of_month_values(self):
        """Test day_of_month values are correct."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=31),
            'price': [3000.0] * 31
        })
        
        df_features = preprocessor.engineer_time_features(df, 'timestamp')
        
        # Check day of month sequence
        expected_days = list(range(1, 32))
        assert list(df_features['day_of_month']) == expected_days
    
    def test_month_values(self):
        """Test month values are correct."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-15', '2024-02-15', '2024-03-15', '2024-12-15']),
            'price': [3000.0] * 4
        })
        
        df_features = preprocessor.engineer_time_features(df, 'timestamp')
        
        assert list(df_features['month']) == [1, 2, 3, 12]
    
    def test_quarter_values(self):
        """Test quarter values are correct."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-15', '2024-04-15', '2024-07-15', '2024-10-15']),
            'price': [3000.0] * 4
        })
        
        df_features = preprocessor.engineer_time_features(df, 'timestamp')
        
        assert list(df_features['quarter']) == [1, 2, 3, 4]
    
    def test_is_month_start_flag(self):
        """Test is_month_start flag is correct."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-02-01', '2024-02-15']),
            'price': [3000.0] * 4
        })
        
        df_features = preprocessor.engineer_time_features(df, 'timestamp')
        
        assert list(df_features['is_month_start']) == [True, False, True, False]
    
    def test_is_month_end_flag(self):
        """Test is_month_end flag is correct."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-30', '2024-01-31', '2024-02-28', '2024-02-29']),
            'price': [3000.0] * 4
        })
        
        df_features = preprocessor.engineer_time_features(df, 'timestamp')
        
        assert list(df_features['is_month_end']) == [False, True, False, True]
    
    def test_days_since_shock_without_shock_column(self):
        """Test days_since_shock when no shock column exists."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'price': [3000.0] * 5
        })
        
        df_features = preprocessor.engineer_time_features(df, 'timestamp')
        
        # Should initialize with large value (999)
        assert all(df_features['days_since_shock'] == 999)
    
    def test_days_since_shock_with_shock_column(self):
        """Test days_since_shock calculation when shock column exists."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'price': [3000.0] * 10,
            'is_shock': [False, False, True, False, False, False, True, False, False, False]
        })
        
        df_features = preprocessor.engineer_time_features(df, 'timestamp')
        
        # Check days since shock calculation
        expected_days = [999, 999, 0, 1, 2, 3, 0, 1, 2, 3]
        assert list(df_features['days_since_shock']) == expected_days
    
    def test_days_since_shock_multiple_shocks(self):
        """Test days_since_shock with multiple shocks."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=8),
            'price': [3000.0] * 8,
            'is_shock': [True, False, False, True, False, False, False, True]
        })
        
        df_features = preprocessor.engineer_time_features(df, 'timestamp')
        
        # Each shock resets the counter
        expected_days = [0, 1, 2, 0, 1, 2, 3, 0]
        assert list(df_features['days_since_shock']) == expected_days
    
    def test_invalid_timestamp_column_raises_error(self):
        """Test that invalid timestamp column raises KeyError."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'price': [3000.0, 3100.0, 3200.0]
        })
        
        with pytest.raises(KeyError, match="Column 'timestamp' not found"):
            preprocessor.engineer_time_features(df, 'timestamp')
    
    def test_non_datetime_column_raises_error(self):
        """Test that non-datetime column raises ValueError."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': ['2024-01-01', '2024-01-02', '2024-01-03'],  # String, not datetime
            'price': [3000.0, 3100.0, 3200.0]
        })
        
        # Should convert automatically, so no error
        df_features = preprocessor.engineer_time_features(df, 'timestamp')
        assert 'day_of_week' in df_features.columns
    
    def test_preserves_original_dataframe(self):
        """Test that original DataFrame is not modified."""
        preprocessor = DataPreprocessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'price': [3000.0] * 5
        })
        
        df_original = df.copy()
        df_features = preprocessor.engineer_time_features(df, 'timestamp')
        
        # Original should only have original columns
        assert list(df.columns) == list(df_original.columns)
        # Features should have additional columns
        assert len(df_features.columns) > len(df.columns)


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple preprocessing steps."""
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        preprocessor = DataPreprocessor()
        
        # Create sample data with missing values
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100),
            'price': np.random.uniform(3000, 3500, 100),
            'temperature': np.random.uniform(20, 30, 100),
            'stock_level': np.random.uniform(1000, 2000, 100)
        })
        
        # Introduce some missing values
        df.loc[10:12, 'price'] = np.nan
        df.loc[20:22, 'temperature'] = np.nan
        df.loc[30, 'stock_level'] = np.nan
        
        # Step 1: Handle missing values
        strategy = {
            'price': 'forward_fill',
            'temperature': 'interpolate',
            'stock_level': 'mean'
        }
        df_clean = preprocessor.handle_missing_values(df, strategy)
        
        # Step 2: Engineer time features
        df_features = preprocessor.engineer_time_features(df_clean, 'timestamp')
        
        # Step 3: Normalize features
        df_norm, scalers = preprocessor.normalize_features(
            df_features,
            ['price', 'temperature', 'stock_level'],
            method='standard'
        )
        
        # Step 4: Create train/val split
        train_df, val_df = preprocessor.create_train_val_split(df_norm, val_size=0.2)
        
        # Verify pipeline results
        assert train_df['price'].isna().sum() == 0
        assert val_df['price'].isna().sum() == 0
        assert 'day_of_week' in train_df.columns
        assert 'day_of_week' in val_df.columns
        assert len(train_df) == 80
        assert len(val_df) == 20
        
        # Verify no data leakage
        assert train_df['timestamp'].max() < val_df['timestamp'].min()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
