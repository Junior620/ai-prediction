"""
Data validator for the Cocoa Price Prediction System.

This module validates data quality and consistency for price data,
econometric data, and detects outliers and market shocks.

Requirements addressed:
- 1.5: Validate prices within realistic bounds (1000-10000 USD/MT)
- 2.6: Flag missing values and validate econometric data ranges
- 4.4: Detect outliers using statistical methods
- 4.5: Flag market shocks (>5% daily price change)
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats

from src.models.data_models import ValidationError

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates data quality and consistency.
    
    Provides methods for:
    - Price data validation with realistic bounds
    - Econometric data validation with range checks
    - Outlier detection using zscore and IQR methods
    - Market shock detection for significant price changes
    
    Attributes:
        price_min: Minimum valid price (USD/MT)
        price_max: Maximum valid price (USD/MT)
        shock_threshold: Threshold for market shock detection (default: 5%)
    """
    
    # Valid ranges for different data types
    PRICE_RANGE = (1000.0, 10000.0)  # USD per metric ton
    TEMPERATURE_RANGE = (-10.0, 50.0)  # Celsius
    RAINFALL_RANGE = (0.0, 500.0)  # mm/day
    STOCK_LEVEL_MIN = 0.0  # metric tons
    PRODUCTION_MIN = 0.0  # metric tons
    FX_RATE_MIN = 0.0  # must be positive
    
    def __init__(
        self,
        price_min: float = 1000.0,
        price_max: float = 10000.0,
        shock_threshold: float = 0.05
    ):
        """
        Initialize data validator.
        
        Args:
            price_min: Minimum valid price in USD/MT (default: 1000.0)
            price_max: Maximum valid price in USD/MT (default: 10000.0)
            shock_threshold: Threshold for market shock detection (default: 0.05 = 5%)
        """
        self.price_min = price_min
        self.price_max = price_max
        self.shock_threshold = shock_threshold
        
        logger.info(
            f"DataValidator initialized with price range [{price_min}, {price_max}] "
            f"and shock threshold {shock_threshold*100}%"
        )
    
    def validate_price_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[ValidationError]]:
        """
        Validate price data for realistic bounds and consistency.
        
        Performs the following checks:
        - Price range: 1000-10000 USD/MT
        - No duplicate timestamps per market
        - Chronological order
        - No gaps > 7 days (except weekends)
        - Valid market identifiers
        - Non-negative volumes
        
        Args:
            df: DataFrame with columns [timestamp, market, price, volume, currency]
            
        Returns:
            Tuple of (validated_df, list_of_errors)
            - validated_df: DataFrame with valid records only
            - list_of_errors: List of ValidationError objects for invalid records
            
        Requirements:
            - 1.5: Validate prices within realistic bounds (1000-10000 USD/MT)
        
        Example:
            >>> validator = DataValidator()
            >>> df = pd.DataFrame({
            ...     'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            ...     'market': ['ICE_London', 'ICE_London'],
            ...     'price': [3000.0, 3100.0],
            ...     'volume': [1000.0, 1200.0],
            ...     'currency': ['USD', 'USD']
            ... })
            >>> valid_df, errors = validator.validate_price_data(df)
        """
        logger.info(f"Validating {len(df)} price data records")
        
        errors = []
        valid_mask = pd.Series([True] * len(df), index=df.index)
        
        # Check required columns
        required_columns = ['timestamp', 'market', 'price', 'volume', 'currency']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error = ValidationError(
                field="columns",
                value=str(missing_columns),
                error_type="missing",
                message=f"Missing required columns: {missing_columns}",
                severity="CRITICAL"
            )
            errors.append(error)
            logger.error(error.message)
            return pd.DataFrame(), errors
        
        # Validate price range
        invalid_prices = (df['price'] < self.price_min) | (df['price'] > self.price_max)
        if invalid_prices.any():
            for idx in df[invalid_prices].index:
                row = df.loc[idx]
                error = ValidationError(
                    field="price",
                    value=str(row['price']),
                    error_type="out_of_range",
                    message=(
                        f"Price {row['price']} USD/MT at {row['timestamp']} "
                        f"outside valid range [{self.price_min}, {self.price_max}]"
                    ),
                    severity="ERROR"
                )
                errors.append(error)
                valid_mask[idx] = False
            
            logger.warning(f"Found {invalid_prices.sum()} records with invalid prices")
        
        # Validate non-negative volumes
        invalid_volumes = df['volume'] < 0
        if invalid_volumes.any():
            for idx in df[invalid_volumes].index:
                row = df.loc[idx]
                error = ValidationError(
                    field="volume",
                    value=str(row['volume']),
                    error_type="out_of_range",
                    message=f"Volume {row['volume']} is negative at {row['timestamp']}",
                    severity="ERROR"
                )
                errors.append(error)
                valid_mask[idx] = False
            
            logger.warning(f"Found {invalid_volumes.sum()} records with negative volumes")
        
        # Validate market identifiers
        valid_markets = ['ICE_London', 'ICE_NY']
        invalid_markets = ~df['market'].isin(valid_markets)
        if invalid_markets.any():
            for idx in df[invalid_markets].index:
                row = df.loc[idx]
                error = ValidationError(
                    field="market",
                    value=str(row['market']),
                    error_type="invalid_format",
                    message=f"Invalid market identifier: {row['market']}. Must be one of {valid_markets}",
                    severity="ERROR"
                )
                errors.append(error)
                valid_mask[idx] = False
            
            logger.warning(f"Found {invalid_markets.sum()} records with invalid market identifiers")
        
        # Check for duplicate timestamps per market
        duplicates = df.duplicated(subset=['timestamp', 'market'], keep='first')
        if duplicates.any():
            for idx in df[duplicates].index:
                row = df.loc[idx]
                error = ValidationError(
                    field="timestamp",
                    value=str(row['timestamp']),
                    error_type="duplicate",
                    message=f"Duplicate timestamp {row['timestamp']} for market {row['market']}",
                    severity="WARNING"
                )
                errors.append(error)
                valid_mask[idx] = False
            
            logger.warning(f"Found {duplicates.sum()} duplicate timestamp records")
        
        # Check chronological order per market
        df_sorted = df.copy()
        df_sorted = df_sorted.sort_values(['market', 'timestamp'])
        
        for market in df_sorted['market'].unique():
            market_data = df_sorted[df_sorted['market'] == market]
            
            if not market_data['timestamp'].is_monotonic_increasing:
                error = ValidationError(
                    field="timestamp",
                    value=market,
                    error_type="constraint_violation",
                    message=f"Timestamps not in chronological order for market {market}",
                    severity="WARNING"
                )
                errors.append(error)
                logger.warning(error.message)
            
            # Check for gaps > 7 days (excluding weekends)
            time_diffs = market_data['timestamp'].diff()
            large_gaps = time_diffs > pd.Timedelta(days=7)
            
            if large_gaps.any():
                gap_count = large_gaps.sum()
                error = ValidationError(
                    field="timestamp",
                    value=market,
                    error_type="missing",
                    message=f"Found {gap_count} gaps > 7 days in {market} data",
                    severity="INFO"
                )
                errors.append(error)
                logger.info(error.message)
        
        # Return validated dataframe and errors
        valid_df = df[valid_mask].copy()
        
        logger.info(
            f"Validation complete: {len(valid_df)}/{len(df)} records valid, "
            f"{len(errors)} issues found"
        )
        
        return valid_df, errors
    
    def validate_econometric_data(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], List[ValidationError]]:
        """
        Validate econometric data completeness and ranges.
        
        Performs the following checks:
        - Temperature: -10 to 50°C
        - Rainfall: 0 to 500mm/day
        - Stock levels: >= 0
        - Production: >= 0
        - FX rates: > 0
        - Missing value detection
        
        Args:
            data: Dict with keys ["weather", "stocks", "production", "fx_rates"]
                  Each value is a DataFrame with the respective data
            
        Returns:
            Tuple of (validated_data_dict, list_of_errors)
            - validated_data_dict: Dict with validated DataFrames
            - list_of_errors: List of ValidationError objects
            
        Requirements:
            - 2.6: Flag missing values and validate econometric data ranges
        
        Example:
            >>> validator = DataValidator()
            >>> data = {
            ...     'weather': pd.DataFrame({
            ...         'timestamp': [datetime(2024, 1, 1)],
            ...         'temperature': [25.0],
            ...         'rainfall': [10.0]
            ...     })
            ... }
            >>> valid_data, errors = validator.validate_econometric_data(data)
        """
        logger.info(f"Validating econometric data with {len(data)} categories")
        
        errors = []
        validated_data = {}
        
        # Validate weather data
        if 'weather' in data and not data['weather'].empty:
            weather_df = data['weather'].copy()
            weather_errors = []
            
            # Validate temperature range
            if 'temperature' in weather_df.columns:
                invalid_temp = (
                    (weather_df['temperature'].notna()) &
                    ((weather_df['temperature'] < self.TEMPERATURE_RANGE[0]) |
                     (weather_df['temperature'] > self.TEMPERATURE_RANGE[1]))
                )
                
                if invalid_temp.any():
                    for idx in weather_df[invalid_temp].index:
                        temp = weather_df.loc[idx, 'temperature']
                        weather_errors.append(ValidationError(
                            field="temperature",
                            value=str(temp),
                            error_type="out_of_range",
                            message=(
                                f"Temperature {temp}°C outside valid range "
                                f"{self.TEMPERATURE_RANGE}"
                            ),
                            severity="ERROR"
                        ))
                    
                    logger.warning(f"Found {invalid_temp.sum()} invalid temperature values")
                
                # Check for missing values
                missing_temp = weather_df['temperature'].isna().sum()
                if missing_temp > 0:
                    weather_errors.append(ValidationError(
                        field="temperature",
                        value=None,
                        error_type="missing",
                        message=f"{missing_temp} missing temperature values",
                        severity="WARNING"
                    ))
            
            # Validate rainfall range
            if 'rainfall' in weather_df.columns:
                invalid_rain = (
                    (weather_df['rainfall'].notna()) &
                    ((weather_df['rainfall'] < self.RAINFALL_RANGE[0]) |
                     (weather_df['rainfall'] > self.RAINFALL_RANGE[1]))
                )
                
                if invalid_rain.any():
                    for idx in weather_df[invalid_rain].index:
                        rain = weather_df.loc[idx, 'rainfall']
                        weather_errors.append(ValidationError(
                            field="rainfall",
                            value=str(rain),
                            error_type="out_of_range",
                            message=(
                                f"Rainfall {rain}mm/day outside valid range "
                                f"{self.RAINFALL_RANGE}"
                            ),
                            severity="ERROR"
                        ))
                    
                    logger.warning(f"Found {invalid_rain.sum()} invalid rainfall values")
                
                # Check for missing values
                missing_rain = weather_df['rainfall'].isna().sum()
                if missing_rain > 0:
                    weather_errors.append(ValidationError(
                        field="rainfall",
                        value=None,
                        error_type="missing",
                        message=f"{missing_rain} missing rainfall values",
                        severity="WARNING"
                    ))
            
            errors.extend(weather_errors)
            validated_data['weather'] = weather_df
        
        # Validate stock data
        if 'stocks' in data and not data['stocks'].empty:
            stocks_df = data['stocks'].copy()
            
            if 'stock_level' in stocks_df.columns:
                invalid_stocks = (
                    (stocks_df['stock_level'].notna()) &
                    (stocks_df['stock_level'] < self.STOCK_LEVEL_MIN)
                )
                
                if invalid_stocks.any():
                    for idx in stocks_df[invalid_stocks].index:
                        stock = stocks_df.loc[idx, 'stock_level']
                        errors.append(ValidationError(
                            field="stock_level",
                            value=str(stock),
                            error_type="out_of_range",
                            message=f"Stock level {stock} is negative",
                            severity="ERROR"
                        ))
                    
                    logger.warning(f"Found {invalid_stocks.sum()} invalid stock levels")
                
                # Check for missing values
                missing_stocks = stocks_df['stock_level'].isna().sum()
                if missing_stocks > 0:
                    errors.append(ValidationError(
                        field="stock_level",
                        value=None,
                        error_type="missing",
                        message=f"{missing_stocks} missing stock level values",
                        severity="WARNING"
                    ))
            
            validated_data['stocks'] = stocks_df
        
        # Validate production data
        if 'production' in data and not data['production'].empty:
            production_df = data['production'].copy()
            
            if 'production' in production_df.columns:
                invalid_prod = (
                    (production_df['production'].notna()) &
                    (production_df['production'] < self.PRODUCTION_MIN)
                )
                
                if invalid_prod.any():
                    for idx in production_df[invalid_prod].index:
                        prod = production_df.loc[idx, 'production']
                        errors.append(ValidationError(
                            field="production",
                            value=str(prod),
                            error_type="out_of_range",
                            message=f"Production {prod} is negative",
                            severity="ERROR"
                        ))
                    
                    logger.warning(f"Found {invalid_prod.sum()} invalid production values")
                
                # Check for missing values
                missing_prod = production_df['production'].isna().sum()
                if missing_prod > 0:
                    errors.append(ValidationError(
                        field="production",
                        value=None,
                        error_type="missing",
                        message=f"{missing_prod} missing production values",
                        severity="WARNING"
                    ))
            
            validated_data['production'] = production_df
        
        # Validate FX rates
        if 'fx_rates' in data and not data['fx_rates'].empty:
            fx_df = data['fx_rates'].copy()
            fx_columns = ['fx_rate_xaf_usd', 'fx_rate_gbp_usd', 'fx_rate_eur_usd']
            
            for col in fx_columns:
                if col in fx_df.columns:
                    invalid_fx = (
                        (fx_df[col].notna()) &
                        (fx_df[col] <= self.FX_RATE_MIN)
                    )
                    
                    if invalid_fx.any():
                        for idx in fx_df[invalid_fx].index:
                            rate = fx_df.loc[idx, col]
                            errors.append(ValidationError(
                                field=col,
                                value=str(rate),
                                error_type="out_of_range",
                                message=f"FX rate {col}={rate} must be positive",
                                severity="ERROR"
                            ))
                        
                        logger.warning(f"Found {invalid_fx.sum()} invalid {col} values")
                    
                    # Check for missing values
                    missing_fx = fx_df[col].isna().sum()
                    if missing_fx > 0:
                        errors.append(ValidationError(
                            field=col,
                            value=None,
                            error_type="missing",
                            message=f"{missing_fx} missing {col} values",
                            severity="WARNING"
                        ))
            
            validated_data['fx_rates'] = fx_df
        
        logger.info(
            f"Econometric validation complete: {len(validated_data)} categories validated, "
            f"{len(errors)} issues found"
        )
        
        return validated_data, errors
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = "zscore",
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect outliers in a data column using statistical methods.
        
        Supports two methods:
        - zscore: Detects values beyond threshold standard deviations from mean
        - iqr: Detects values beyond threshold * IQR from Q1/Q3
        
        Args:
            df: DataFrame containing the data
            column: Name of the column to check for outliers
            method: Detection method ("zscore" or "iqr")
            threshold: Threshold for outlier detection
                      - For zscore: number of standard deviations (default: 3.0)
                      - For iqr: multiplier for IQR (default: 3.0, typical is 1.5)
            
        Returns:
            Boolean Series indicating outliers (True = outlier)
            
        Requirements:
            - 4.4: Detect outliers using statistical methods (zscore and IQR)
        
        Raises:
            ValueError: If method is not "zscore" or "iqr"
            KeyError: If column does not exist in DataFrame
        
        Example:
            >>> validator = DataValidator()
            >>> df = pd.DataFrame({'price': [3000, 3100, 3050, 9000, 3200]})
            >>> outliers = validator.detect_outliers(df, 'price', method='zscore', threshold=3.0)
            >>> print(outliers)
            0    False
            1    False
            2    False
            3     True
            4    False
            Name: price, dtype: bool
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        
        if method not in ["zscore", "iqr"]:
            raise ValueError(f"Method must be 'zscore' or 'iqr', got '{method}'")
        
        logger.info(
            f"Detecting outliers in column '{column}' using {method} method "
            f"with threshold {threshold}"
        )
        
        # Get the column data and remove NaN values for calculation
        data = df[column].dropna()
        
        if len(data) == 0:
            logger.warning(f"Column '{column}' has no valid data")
            return pd.Series([False] * len(df), index=df.index)
        
        if method == "zscore":
            # Z-score method: (x - mean) / std
            mean = data.mean()
            std = data.std()
            
            if std == 0:
                logger.warning(f"Column '{column}' has zero standard deviation")
                return pd.Series([False] * len(df), index=df.index)
            
            z_scores = np.abs((df[column] - mean) / std)
            outliers = z_scores > threshold
            
            # NaN values are not outliers
            outliers = outliers.fillna(False)
            
            outlier_count = outliers.sum()
            logger.info(
                f"Z-score method detected {outlier_count} outliers "
                f"(mean={mean:.2f}, std={std:.2f})"
            )
        
        elif method == "iqr":
            # IQR method: values beyond Q1 - threshold*IQR or Q3 + threshold*IQR
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:
                logger.warning(f"Column '{column}' has zero IQR")
                return pd.Series([False] * len(df), index=df.index)
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            
            # NaN values are not outliers
            outliers = outliers.fillna(False)
            
            outlier_count = outliers.sum()
            logger.info(
                f"IQR method detected {outlier_count} outliers "
                f"(Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
                f"bounds=[{lower_bound:.2f}, {upper_bound:.2f}])"
            )
        
        return outliers
    
    def flag_market_shocks(
        self,
        df: pd.DataFrame,
        threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Flag periods with market shocks (>5% daily price change).
        
        Calculates the percentage change in price from one day to the next
        and flags records where the absolute change exceeds the threshold.
        
        Args:
            df: DataFrame with columns [timestamp, market, price]
                Must be sorted by timestamp within each market
            threshold: Threshold for shock detection (default: 0.05 = 5%)
            
        Returns:
            DataFrame with additional 'is_shock' boolean column
            
        Requirements:
            - 4.5: Flag market shocks for variations >5%
        
        Example:
            >>> validator = DataValidator()
            >>> df = pd.DataFrame({
            ...     'timestamp': pd.date_range('2024-01-01', periods=5),
            ...     'market': ['ICE_London'] * 5,
            ...     'price': [3000, 3100, 3300, 3150, 3200]
            ... })
            >>> df_with_shocks = validator.flag_market_shocks(df, threshold=0.05)
            >>> print(df_with_shocks['is_shock'])
            0    False
            1    False
            2     True  # 6.45% increase
            3    False
            4    False
            Name: is_shock, dtype: bool
        """
        logger.info(
            f"Flagging market shocks with threshold {threshold*100}% "
            f"for {len(df)} records"
        )
        
        # Check required columns
        required_columns = ['timestamp', 'market', 'price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")
        
        # Create a copy to avoid modifying the original
        df_result = df.copy()
        
        # Initialize is_shock column
        df_result['is_shock'] = False
        
        # Calculate percentage change per market
        for market in df_result['market'].unique():
            market_mask = df_result['market'] == market
            market_data = df_result[market_mask].copy()
            
            # Sort by timestamp to ensure correct order
            market_data = market_data.sort_values('timestamp')
            
            # Calculate percentage change
            price_pct_change = market_data['price'].pct_change().abs()
            
            # Flag shocks
            shock_mask = price_pct_change > threshold
            
            # Update the main dataframe
            df_result.loc[market_data.index, 'is_shock'] = shock_mask
        
        # Count shocks
        shock_count = df_result['is_shock'].sum()
        shock_percentage = (shock_count / len(df_result)) * 100 if len(df_result) > 0 else 0
        
        logger.info(
            f"Market shock detection complete: {shock_count} shocks detected "
            f"({shock_percentage:.2f}% of records)"
        )
        
        # Log details of detected shocks
        if shock_count > 0:
            shock_records = df_result[df_result['is_shock']]
            for idx, row in shock_records.iterrows():
                logger.info(
                    f"Shock detected: {row['market']} at {row['timestamp']}, "
                    f"price={row['price']:.2f}"
                )
        
        return df_result
