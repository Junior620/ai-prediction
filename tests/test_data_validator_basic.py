"""
Basic tests for DataValidator to verify task 4.1 implementation.

This test file provides a quick verification that the DataValidator class
has been implemented with all required methods.
"""

import pytest
from datetime import datetime
import pandas as pd
import numpy as np

from src.data_validation import DataValidator
from src.models.data_models import ValidationError


class TestDataValidatorBasic:
    """Basic tests to verify DataValidator implementation."""
    
    def test_validator_initialization(self):
        """Test that DataValidator can be initialized with default parameters."""
        validator = DataValidator()
        
        assert validator.price_min == 1000.0
        assert validator.price_max == 10000.0
        assert validator.shock_threshold == 0.05
    
    def test_validator_custom_initialization(self):
        """Test that DataValidator can be initialized with custom parameters."""
        validator = DataValidator(
            price_min=2000.0,
            price_max=8000.0,
            shock_threshold=0.10
        )
        
        assert validator.price_min == 2000.0
        assert validator.price_max == 8000.0
        assert validator.shock_threshold == 0.10
    
    def test_validate_price_data_method_exists(self):
        """Test that validate_price_data method exists."""
        validator = DataValidator()
        assert hasattr(validator, 'validate_price_data')
        assert callable(validator.validate_price_data)
    
    def test_validate_econometric_data_method_exists(self):
        """Test that validate_econometric_data method exists."""
        validator = DataValidator()
        assert hasattr(validator, 'validate_econometric_data')
        assert callable(validator.validate_econometric_data)
    
    def test_detect_outliers_method_exists(self):
        """Test that detect_outliers method exists."""
        validator = DataValidator()
        assert hasattr(validator, 'detect_outliers')
        assert callable(validator.detect_outliers)
    
    def test_flag_market_shocks_method_exists(self):
        """Test that flag_market_shocks method exists."""
        validator = DataValidator()
        assert hasattr(validator, 'flag_market_shocks')
        assert callable(validator.flag_market_shocks)
    
    def test_validate_price_data_valid_data(self):
        """Test validate_price_data with valid data."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'market': ['ICE_London', 'ICE_London'],
            'price': [3000.0, 3100.0],
            'volume': [1000.0, 1200.0],
            'currency': ['USD', 'USD']
        })
        
        valid_df, errors = validator.validate_price_data(df)
        
        assert len(valid_df) == 2
        assert len(errors) == 0
    
    def test_validate_price_data_invalid_price(self):
        """Test validate_price_data with invalid price."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1)],
            'market': ['ICE_London'],
            'price': [500.0],  # Below minimum
            'volume': [1000.0],
            'currency': ['USD']
        })
        
        valid_df, errors = validator.validate_price_data(df)
        
        assert len(valid_df) == 0
        assert len(errors) > 0
        assert any(e.field == 'price' for e in errors)
    
    def test_detect_outliers_zscore(self):
        """Test detect_outliers with zscore method."""
        validator = DataValidator()
        
        # Create data with a clear outlier (more data points for better statistics)
        df = pd.DataFrame({
            'price': [3000, 3100, 3050, 3080, 3120, 3090, 3110, 3070, 9000, 3095]
        })
        
        outliers = validator.detect_outliers(df, 'price', method='zscore', threshold=2.0)
        
        assert isinstance(outliers, pd.Series)
        assert len(outliers) == 10
        assert outliers.iloc[8] == True  # 9000 should be detected as outlier
    
    def test_detect_outliers_iqr(self):
        """Test detect_outliers with IQR method."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'price': [3000, 3100, 3050, 9000, 3200]  # 9000 is an outlier
        })
        
        outliers = validator.detect_outliers(df, 'price', method='iqr', threshold=1.5)
        
        assert isinstance(outliers, pd.Series)
        assert len(outliers) == 5
        assert outliers.iloc[3] == True  # 9000 should be detected as outlier
    
    def test_flag_market_shocks(self):
        """Test flag_market_shocks with significant price change."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'market': ['ICE_London'] * 5,
            'price': [3000, 3100, 3300, 3150, 3200]  # 3100 -> 3300 is ~6.45% increase
        })
        
        df_with_shocks = validator.flag_market_shocks(df, threshold=0.05)
        
        assert 'is_shock' in df_with_shocks.columns
        assert len(df_with_shocks) == 5
        assert df_with_shocks['is_shock'].iloc[2] == True  # Should detect the shock
    
    def test_validate_econometric_data_valid(self):
        """Test validate_econometric_data with valid data."""
        validator = DataValidator()
        
        data = {
            'weather': pd.DataFrame({
                'timestamp': [datetime(2024, 1, 1)],
                'temperature': [25.0],
                'rainfall': [10.0]
            })
        }
        
        valid_data, errors = validator.validate_econometric_data(data)
        
        assert 'weather' in valid_data
        assert len(valid_data['weather']) == 1
        # Warnings for missing values are acceptable
        error_count = len([e for e in errors if e.severity == 'ERROR'])
        assert error_count == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
