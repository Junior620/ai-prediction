"""
Comprehensive unit tests for DataValidator.

Tests cover:
- Detection of out-of-bounds values (price ranges, temperature, rainfall, etc.)
- Outlier detection using both zscore and IQR methods
- Market shock detection (>5% daily price changes)

Requirements: 1.5, 4.4
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.data_validation import DataValidator
from src.models.data_models import ValidationError


class TestDataValidatorInitialization:
    """Test DataValidator initialization."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        validator = DataValidator()
        
        assert validator.price_min == 1000.0
        assert validator.price_max == 10000.0
        assert validator.shock_threshold == 0.05
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        validator = DataValidator(
            price_min=2000.0,
            price_max=8000.0,
            shock_threshold=0.10
        )
        
        assert validator.price_min == 2000.0
        assert validator.price_max == 8000.0
        assert validator.shock_threshold == 0.10


class TestValidatePriceData:
    """Test validate_price_data method for out-of-bounds detection."""
    
    def test_valid_price_data(self):
        """Test validation with all valid data."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            'market': ['ICE_London', 'ICE_London', 'ICE_London'],
            'price': [3000.0, 3100.0, 3200.0],
            'volume': [1000.0, 1200.0, 1100.0],
            'currency': ['USD', 'USD', 'USD']
        })
        
        valid_df, errors = validator.validate_price_data(df)
        
        assert len(valid_df) == 3
        assert len(errors) == 0
    
    def test_price_below_minimum(self):
        """Test detection of prices below minimum bound (1000 USD/MT)."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'market': ['ICE_London', 'ICE_London'],
            'price': [500.0, 3000.0],  # 500 is below minimum
            'volume': [1000.0, 1200.0],
            'currency': ['USD', 'USD']
        })
        
        valid_df, errors = validator.validate_price_data(df)
        
        assert len(valid_df) == 1  # Only one valid record
        assert len(errors) >= 1
        assert any(e.field == 'price' and e.error_type == 'out_of_range' for e in errors)
        assert any('500' in e.message for e in errors)
    
    def test_price_above_maximum(self):
        """Test detection of prices above maximum bound (10000 USD/MT)."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'market': ['ICE_London', 'ICE_London'],
            'price': [3000.0, 15000.0],  # 15000 is above maximum
            'volume': [1000.0, 1200.0],
            'currency': ['USD', 'USD']
        })
        
        valid_df, errors = validator.validate_price_data(df)
        
        assert len(valid_df) == 1
        assert len(errors) >= 1
        assert any(e.field == 'price' and e.error_type == 'out_of_range' for e in errors)
        assert any('15000' in e.message for e in errors)
    
    def test_multiple_invalid_prices(self):
        """Test detection of multiple out-of-bounds prices."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            'market': ['ICE_London', 'ICE_London', 'ICE_London'],
            'price': [500.0, 3000.0, 12000.0],  # Two invalid
            'volume': [1000.0, 1200.0, 1100.0],
            'currency': ['USD', 'USD', 'USD']
        })
        
        valid_df, errors = validator.validate_price_data(df)
        
        assert len(valid_df) == 1
        price_errors = [e for e in errors if e.field == 'price']
        assert len(price_errors) == 2
    
    def test_negative_volume(self):
        """Test detection of negative volumes."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'market': ['ICE_London', 'ICE_London'],
            'price': [3000.0, 3100.0],
            'volume': [-100.0, 1200.0],  # Negative volume
            'currency': ['USD', 'USD']
        })
        
        valid_df, errors = validator.validate_price_data(df)
        
        assert len(valid_df) == 1
        assert any(e.field == 'volume' and e.error_type == 'out_of_range' for e in errors)
    
    def test_invalid_market_identifier(self):
        """Test detection of invalid market identifiers."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'market': ['INVALID_MARKET', 'ICE_London'],
            'price': [3000.0, 3100.0],
            'volume': [1000.0, 1200.0],
            'currency': ['USD', 'USD']
        })
        
        valid_df, errors = validator.validate_price_data(df)
        
        assert len(valid_df) == 1
        assert any(e.field == 'market' and e.error_type == 'invalid_format' for e in errors)
    
    def test_duplicate_timestamps(self):
        """Test detection of duplicate timestamps per market."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'market': ['ICE_London', 'ICE_London', 'ICE_London'],
            'price': [3000.0, 3050.0, 3100.0],
            'volume': [1000.0, 1100.0, 1200.0],
            'currency': ['USD', 'USD', 'USD']
        })
        
        valid_df, errors = validator.validate_price_data(df)
        
        assert len(valid_df) == 2  # First occurrence kept
        assert any(e.field == 'timestamp' and e.error_type == 'duplicate' for e in errors)
    
    def test_missing_required_columns(self):
        """Test handling of missing required columns."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1)],
            'market': ['ICE_London'],
            'price': [3000.0]
            # Missing 'volume' and 'currency'
        })
        
        valid_df, errors = validator.validate_price_data(df)
        
        assert len(valid_df) == 0
        assert len(errors) >= 1
        assert any(e.error_type == 'missing' and 'columns' in e.field for e in errors)


class TestValidateEconometricData:
    """Test validate_econometric_data method for range validation."""
    
    def test_valid_weather_data(self):
        """Test validation with valid weather data."""
        validator = DataValidator()
        
        data = {
            'weather': pd.DataFrame({
                'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                'temperature': [25.0, 28.0],
                'rainfall': [10.0, 15.0]
            })
        }
        
        valid_data, errors = validator.validate_econometric_data(data)
        
        assert 'weather' in valid_data
        assert len(valid_data['weather']) == 2
        error_count = len([e for e in errors if e.severity == 'ERROR'])
        assert error_count == 0
    
    def test_temperature_out_of_range_low(self):
        """Test detection of temperature below valid range (-10°C)."""
        validator = DataValidator()
        
        data = {
            'weather': pd.DataFrame({
                'timestamp': [datetime(2024, 1, 1)],
                'temperature': [-15.0],  # Below minimum
                'rainfall': [10.0]
            })
        }
        
        valid_data, errors = validator.validate_econometric_data(data)
        
        assert any(e.field == 'temperature' and e.error_type == 'out_of_range' for e in errors)
        assert any('-15' in e.message for e in errors)
    
    def test_temperature_out_of_range_high(self):
        """Test detection of temperature above valid range (50°C)."""
        validator = DataValidator()
        
        data = {
            'weather': pd.DataFrame({
                'timestamp': [datetime(2024, 1, 1)],
                'temperature': [55.0],  # Above maximum
                'rainfall': [10.0]
            })
        }
        
        valid_data, errors = validator.validate_econometric_data(data)
        
        assert any(e.field == 'temperature' and e.error_type == 'out_of_range' for e in errors)
        assert any('55' in e.message for e in errors)
    
    def test_rainfall_out_of_range_negative(self):
        """Test detection of negative rainfall."""
        validator = DataValidator()
        
        data = {
            'weather': pd.DataFrame({
                'timestamp': [datetime(2024, 1, 1)],
                'temperature': [25.0],
                'rainfall': [-5.0]  # Negative rainfall
            })
        }
        
        valid_data, errors = validator.validate_econometric_data(data)
        
        assert any(e.field == 'rainfall' and e.error_type == 'out_of_range' for e in errors)
    
    def test_rainfall_out_of_range_high(self):
        """Test detection of rainfall above valid range (500mm/day)."""
        validator = DataValidator()
        
        data = {
            'weather': pd.DataFrame({
                'timestamp': [datetime(2024, 1, 1)],
                'temperature': [25.0],
                'rainfall': [600.0]  # Above maximum
            })
        }
        
        valid_data, errors = validator.validate_econometric_data(data)
        
        assert any(e.field == 'rainfall' and e.error_type == 'out_of_range' for e in errors)
    
    def test_negative_stock_level(self):
        """Test detection of negative stock levels."""
        validator = DataValidator()
        
        data = {
            'stocks': pd.DataFrame({
                'timestamp': [datetime(2024, 1, 1)],
                'stock_level': [-100.0]  # Negative stock
            })
        }
        
        valid_data, errors = validator.validate_econometric_data(data)
        
        assert any(e.field == 'stock_level' and e.error_type == 'out_of_range' for e in errors)
    
    def test_negative_production(self):
        """Test detection of negative production values."""
        validator = DataValidator()
        
        data = {
            'production': pd.DataFrame({
                'timestamp': [datetime(2024, 1, 1)],
                'production': [-50.0]  # Negative production
            })
        }
        
        valid_data, errors = validator.validate_econometric_data(data)
        
        assert any(e.field == 'production' and e.error_type == 'out_of_range' for e in errors)
    
    def test_invalid_fx_rates(self):
        """Test detection of non-positive FX rates."""
        validator = DataValidator()
        
        data = {
            'fx_rates': pd.DataFrame({
                'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                'fx_rate_xaf_usd': [0.0, 0.0017],  # 0.0 is invalid
                'fx_rate_gbp_usd': [1.25, -0.5],  # Negative is invalid
                'fx_rate_eur_usd': [1.10, 1.12]
            })
        }
        
        valid_data, errors = validator.validate_econometric_data(data)
        
        fx_errors = [e for e in errors if 'fx_rate' in e.field and e.error_type == 'out_of_range']
        assert len(fx_errors) >= 2
    
    def test_missing_values_flagged(self):
        """Test that missing values are flagged as warnings."""
        validator = DataValidator()
        
        data = {
            'weather': pd.DataFrame({
                'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                'temperature': [25.0, np.nan],  # Missing value
                'rainfall': [10.0, 15.0]
            })
        }
        
        valid_data, errors = validator.validate_econometric_data(data)
        
        assert any(e.field == 'temperature' and e.error_type == 'missing' for e in errors)
    
    def test_multiple_data_categories(self):
        """Test validation across multiple econometric data categories."""
        validator = DataValidator()
        
        data = {
            'weather': pd.DataFrame({
                'timestamp': [datetime(2024, 1, 1)],
                'temperature': [25.0],
                'rainfall': [10.0]
            }),
            'stocks': pd.DataFrame({
                'timestamp': [datetime(2024, 1, 1)],
                'stock_level': [50000.0]
            }),
            'production': pd.DataFrame({
                'timestamp': [datetime(2024, 1, 1)],
                'production': [100000.0]
            }),
            'fx_rates': pd.DataFrame({
                'timestamp': [datetime(2024, 1, 1)],
                'fx_rate_xaf_usd': [0.0017],
                'fx_rate_gbp_usd': [1.25],
                'fx_rate_eur_usd': [1.10]
            })
        }
        
        valid_data, errors = validator.validate_econometric_data(data)
        
        assert len(valid_data) == 4
        assert 'weather' in valid_data
        assert 'stocks' in valid_data
        assert 'production' in valid_data
        assert 'fx_rates' in valid_data


class TestDetectOutliers:
    """Test detect_outliers method with zscore and IQR methods."""
    
    def test_zscore_method_with_outlier(self):
        """Test zscore method detects clear outliers."""
        validator = DataValidator()
        
        # Create data with a clear outlier (using threshold of 2.5 to detect the outlier)
        df = pd.DataFrame({
            'price': [3000, 3100, 3050, 3080, 3120, 3090, 3110, 3070, 9000, 3095]
        })
        
        outliers = validator.detect_outliers(df, 'price', method='zscore', threshold=2.5)
        
        assert isinstance(outliers, pd.Series)
        assert len(outliers) == 10
        assert outliers.iloc[8] == True  # 9000 should be detected
        assert outliers.iloc[0] == False  # Normal values should not be detected
    
    def test_zscore_method_no_outliers(self):
        """Test zscore method with no outliers."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'price': [3000, 3100, 3050, 3080, 3120, 3090, 3110, 3070, 3095, 3105]
        })
        
        outliers = validator.detect_outliers(df, 'price', method='zscore', threshold=3.0)
        
        assert outliers.sum() == 0  # No outliers detected
    
    def test_zscore_method_multiple_outliers(self):
        """Test zscore method detects multiple outliers."""
        validator = DataValidator()
        
        # Create data with tight cluster and extreme outliers
        # Using a lower threshold (1.5) to detect both outliers
        df = pd.DataFrame({
            'price': [3000, 3010, 3020, 3030, 3040, 3050, 3060, 3070, 3080, 3090,
                     15000, 20000]  # Two very extreme outliers
        })
        
        outliers = validator.detect_outliers(df, 'price', method='zscore', threshold=1.5)
        
        assert outliers.sum() >= 2  # At least 2 outliers
    
    def test_iqr_method_with_outlier(self):
        """Test IQR method detects outliers."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'price': [3000, 3100, 3050, 3080, 3120, 9000]
        })
        
        outliers = validator.detect_outliers(df, 'price', method='iqr', threshold=1.5)
        
        assert isinstance(outliers, pd.Series)
        assert outliers.iloc[5] == True  # 9000 should be detected
    
    def test_iqr_method_no_outliers(self):
        """Test IQR method with no outliers."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'price': [3000, 3100, 3050, 3080, 3120, 3090]
        })
        
        outliers = validator.detect_outliers(df, 'price', method='iqr', threshold=1.5)
        
        assert outliers.sum() == 0
    
    def test_iqr_method_custom_threshold(self):
        """Test IQR method with custom threshold."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'price': [3000, 3100, 3050, 3080, 3120, 3500]
        })
        
        # More lenient threshold
        outliers_lenient = validator.detect_outliers(df, 'price', method='iqr', threshold=3.0)
        # Stricter threshold
        outliers_strict = validator.detect_outliers(df, 'price', method='iqr', threshold=1.0)
        
        assert outliers_strict.sum() >= outliers_lenient.sum()
    
    def test_outliers_with_nan_values(self):
        """Test outlier detection handles NaN values correctly."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'price': [3000, np.nan, 3010, 3020, 3030, 3040, 3050, 15000]  # Tight cluster + outlier
        })
        
        outliers = validator.detect_outliers(df, 'price', method='zscore', threshold=2.0)
        
        assert len(outliers) == 8
        assert outliers.iloc[1] == False  # NaN should not be marked as outlier
        assert outliers.iloc[7] == True  # 15000 should be detected
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        validator = DataValidator()
        
        df = pd.DataFrame({'price': [3000, 3100, 3050]})
        
        with pytest.raises(ValueError, match="Method must be 'zscore' or 'iqr'"):
            validator.detect_outliers(df, 'price', method='invalid')
    
    def test_invalid_column_raises_error(self):
        """Test that invalid column raises KeyError."""
        validator = DataValidator()
        
        df = pd.DataFrame({'price': [3000, 3100, 3050]})
        
        with pytest.raises(KeyError, match="Column 'nonexistent' not found"):
            validator.detect_outliers(df, 'nonexistent', method='zscore')
    
    def test_zero_standard_deviation(self):
        """Test handling of zero standard deviation."""
        validator = DataValidator()
        
        df = pd.DataFrame({'price': [3000, 3000, 3000, 3000]})
        
        outliers = validator.detect_outliers(df, 'price', method='zscore', threshold=3.0)
        
        assert outliers.sum() == 0  # No outliers when all values are the same
    
    def test_zero_iqr(self):
        """Test handling of zero IQR."""
        validator = DataValidator()
        
        df = pd.DataFrame({'price': [3000, 3000, 3000, 3000]})
        
        outliers = validator.detect_outliers(df, 'price', method='iqr', threshold=1.5)
        
        assert outliers.sum() == 0


class TestFlagMarketShocks:
    """Test flag_market_shocks method for detecting >5% price changes."""
    
    def test_detect_single_shock(self):
        """Test detection of a single market shock."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'market': ['ICE_London'] * 5,
            'price': [3000, 3100, 3300, 3150, 3200]  # 3100 -> 3300 is ~6.45% increase
        })
        
        df_with_shocks = validator.flag_market_shocks(df, threshold=0.05)
        
        assert 'is_shock' in df_with_shocks.columns
        assert df_with_shocks['is_shock'].iloc[2] == True  # Should detect the shock
        assert df_with_shocks['is_shock'].iloc[0] == False  # First row has no previous value
    
    def test_detect_multiple_shocks(self):
        """Test detection of multiple market shocks."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=6),
            'market': ['ICE_London'] * 6,
            'price': [3000, 3200, 3100, 3300, 3150, 3350]  # Multiple >5% changes
        })
        
        df_with_shocks = validator.flag_market_shocks(df, threshold=0.05)
        
        shock_count = df_with_shocks['is_shock'].sum()
        assert shock_count >= 2
    
    def test_no_shocks_detected(self):
        """Test when no market shocks are present."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'market': ['ICE_London'] * 5,
            'price': [3000, 3050, 3100, 3120, 3150]  # All changes < 5%
        })
        
        df_with_shocks = validator.flag_market_shocks(df, threshold=0.05)
        
        # First row is always False (no previous value)
        assert df_with_shocks['is_shock'].iloc[1:].sum() == 0
    
    def test_shock_detection_multiple_markets(self):
        """Test shock detection across multiple markets."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=4).tolist() * 2,
            'market': ['ICE_London'] * 4 + ['ICE_NY'] * 4,
            'price': [3000, 3200, 3100, 3150,  # London: shock at index 1
                     2800, 2850, 3000, 2950]   # NY: shock at index 6
        })
        
        df_with_shocks = validator.flag_market_shocks(df, threshold=0.05)
        
        london_shocks = df_with_shocks[df_with_shocks['market'] == 'ICE_London']['is_shock'].sum()
        ny_shocks = df_with_shocks[df_with_shocks['market'] == 'ICE_NY']['is_shock'].sum()
        
        assert london_shocks >= 1
        assert ny_shocks >= 1
    
    def test_shock_detection_custom_threshold(self):
        """Test shock detection with custom threshold."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=4),
            'market': ['ICE_London'] * 4,
            'price': [3000, 3180, 3100, 3200]  # 6% increase
        })
        
        # With 5% threshold, should detect shock
        df_5pct = validator.flag_market_shocks(df, threshold=0.05)
        # With 10% threshold, should not detect shock
        df_10pct = validator.flag_market_shocks(df, threshold=0.10)
        
        assert df_5pct['is_shock'].sum() > 0
        assert df_10pct['is_shock'].sum() == 0
    
    def test_shock_detection_negative_change(self):
        """Test shock detection for negative price changes."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=4),
            'market': ['ICE_London'] * 4,
            'price': [3000, 2800, 3100, 3050]  # -6.67% decrease
        })
        
        df_with_shocks = validator.flag_market_shocks(df, threshold=0.05)
        
        assert df_with_shocks['is_shock'].iloc[1] == True  # Should detect negative shock
    
    def test_shock_detection_preserves_data(self):
        """Test that shock detection preserves original data."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3),
            'market': ['ICE_London'] * 3,
            'price': [3000, 3200, 3100],
            'volume': [1000, 1100, 1050]
        })
        
        df_with_shocks = validator.flag_market_shocks(df, threshold=0.05)
        
        assert len(df_with_shocks) == len(df)
        assert all(df_with_shocks['price'] == df['price'])
        assert all(df_with_shocks['volume'] == df['volume'])
        assert 'is_shock' in df_with_shocks.columns
    
    def test_shock_detection_missing_columns_raises_error(self):
        """Test that missing required columns raises KeyError."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3),
            'price': [3000, 3200, 3100]
            # Missing 'market' column
        })
        
        with pytest.raises(KeyError, match="Missing required columns"):
            validator.flag_market_shocks(df, threshold=0.05)
    
    def test_shock_detection_unsorted_data(self):
        """Test shock detection with unsorted timestamps."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 3), datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'market': ['ICE_London'] * 3,
            'price': [3300, 3000, 3100]
        })
        
        df_with_shocks = validator.flag_market_shocks(df, threshold=0.05)
        
        # Should handle unsorted data by sorting internally
        assert 'is_shock' in df_with_shocks.columns
        assert len(df_with_shocks) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
