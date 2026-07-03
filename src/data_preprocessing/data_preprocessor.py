"""
Data Preprocessor for the Cocoa Price Prediction System.

This module provides the DataPreprocessor class which handles data cleaning,
transformation, and preparation for model training. It includes functionality for:
- Missing value handling with multiple strategies
- Feature normalization (standard and minmax)
- Train/validation splitting for time series
- Time-based feature engineering

Requirements: 4.1, 4.2, 4.3, 4.6
"""

from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    """
    Preprocesses data for model training.
    
    This class provides methods for handling missing values, normalizing features,
    creating train/validation splits, and engineering time-based features.
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.scalers: Dict[str, Any] = {}
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Handle missing values using specified strategies.
        
        This method applies different imputation strategies to different columns
        based on the provided strategy mapping. It preserves the time series
        nature of the data when appropriate.
        
        Args:
            df: DataFrame with potential missing values
            strategy: Dictionary mapping column names to strategy names
                     Valid strategies:
                     - "forward_fill": For time series continuity
                     - "interpolate": For smooth transitions
                     - "mean": For econometric features
                     - "drop": For non-critical features
        
        Returns:
            DataFrame with missing values handled according to strategies
        
        Raises:
            ValueError: If an invalid strategy is specified
            KeyError: If a column in strategy doesn't exist in df
        
        Example:
            >>> preprocessor = DataPreprocessor()
            >>> strategy = {
            ...     "price": "forward_fill",
            ...     "temperature": "interpolate",
            ...     "stock_level": "mean"
            ... }
            >>> df_clean = preprocessor.handle_missing_values(df, strategy)
        """
        valid_strategies = ["forward_fill", "interpolate", "mean", "drop"]
        
        # Validate strategies
        for col, strat in strategy.items():
            if strat not in valid_strategies:
                raise ValueError(
                    f"Invalid strategy '{strat}' for column '{col}'. "
                    f"Valid strategies: {valid_strategies}"
                )
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")
        
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Apply strategies
        for col, strat in strategy.items():
            if strat == "forward_fill":
                df_processed[col] = df_processed[col].ffill()
                # Backward fill any remaining NaN at the start
                df_processed[col] = df_processed[col].bfill()
            
            elif strat == "interpolate":
                df_processed[col] = df_processed[col].interpolate(
                    method='linear',
                    limit_direction='both'
                )
            
            elif strat == "mean":
                mean_value = df_processed[col].mean()
                df_processed[col] = df_processed[col].fillna(mean_value)
            
            elif strat == "drop":
                # Drop rows where this column has missing values
                df_processed = df_processed.dropna(subset=[col])
        
        return df_processed
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "standard"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalize numerical features.
        
        This method normalizes specified columns using either standard scaling
        (zero mean, unit variance) or min-max scaling (scale to [0, 1] range).
        The scaler parameters are returned to enable inverse transformation.
        
        Args:
            df: DataFrame with features to normalize
            columns: List of column names to normalize
            method: Normalization method, either "standard" or "minmax"
        
        Returns:
            Tuple containing:
            - DataFrame with normalized features
            - Dictionary of scaler parameters for inverse transform
              Format: {column_name: {"scaler": scaler_object, "method": method}}
        
        Raises:
            ValueError: If an invalid method is specified
            KeyError: If a column doesn't exist in df
        
        Example:
            >>> preprocessor = DataPreprocessor()
            >>> df_norm, scalers = preprocessor.normalize_features(
            ...     df, ["price", "volume"], method="standard"
            ... )
            >>> # Later, to inverse transform:
            >>> df_original = preprocessor.inverse_transform(df_norm, scalers)
        """
        valid_methods = ["standard", "minmax"]
        
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Valid methods: {valid_methods}"
            )
        
        # Validate columns exist
        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")
        
        # Create a copy to avoid modifying the original
        df_normalized = df.copy()
        scaler_params = {}
        
        # Normalize each column
        for col in columns:
            # Create appropriate scaler
            if method == "standard":
                scaler = StandardScaler()
            else:  # minmax
                scaler = MinMaxScaler()
            
            # Fit and transform
            values = df_normalized[col].values.reshape(-1, 1)
            df_normalized[col] = scaler.fit_transform(values).flatten()
            
            # Store scaler for inverse transform
            scaler_params[col] = {
                "scaler": scaler,
                "method": method
            }
        
        return df_normalized, scaler_params
    
    def create_train_val_split(
        self,
        df: pd.DataFrame,
        val_size: float = 0.2,
        shuffle: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create chronological train/validation split.
        
        This method splits the data into training and validation sets while
        preserving the time series order. By default, it does not shuffle
        to prevent data leakage in time series forecasting.
        
        Args:
            df: DataFrame to split
            val_size: Fraction of data to use for validation (default: 0.2)
            shuffle: Whether to shuffle data before splitting (default: False)
                    Should be False for time series to preserve chronological order
        
        Returns:
            Tuple containing:
            - Training DataFrame (first 1-val_size fraction)
            - Validation DataFrame (last val_size fraction)
        
        Raises:
            ValueError: If val_size is not between 0 and 1
        
        Example:
            >>> preprocessor = DataPreprocessor()
            >>> train_df, val_df = preprocessor.create_train_val_split(
            ...     df, val_size=0.2, shuffle=False
            ... )
            >>> print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
        """
        if not 0 < val_size < 1:
            raise ValueError(
                f"val_size must be between 0 and 1, got {val_size}"
            )
        
        # Create a copy to avoid modifying the original
        df_split = df.copy()
        
        # Shuffle if requested (not recommended for time series)
        if shuffle:
            df_split = df_split.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        # Calculate split index
        split_idx = int(len(df_split) * (1 - val_size))
        
        # Split chronologically
        train_df = df_split.iloc[:split_idx].copy()
        val_df = df_split.iloc[split_idx:].copy()
        
        return train_df, val_df
    
    def engineer_time_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Engineer time-based features.
        
        This method creates various time-based features from a timestamp column,
        including day of week, day of month, month, quarter, and flags for
        month start/end. It also calculates days since market shocks if detected.
        
        Args:
            df: DataFrame with a timestamp column
            timestamp_col: Name of the timestamp column (default: "timestamp")
        
        Returns:
            DataFrame with additional time-based feature columns:
            - day_of_week: Day of week (0=Monday, 6=Sunday)
            - day_of_month: Day of month (1-31)
            - month: Month (1-12)
            - quarter: Quarter (1-4)
            - is_month_start: Boolean flag for first day of month
            - is_month_end: Boolean flag for last day of month
            - days_since_shock: Days since last market shock (if detected)
        
        Raises:
            KeyError: If timestamp_col doesn't exist in df
            ValueError: If timestamp_col is not datetime type
        
        Example:
            >>> preprocessor = DataPreprocessor()
            >>> df_features = preprocessor.engineer_time_features(df, "timestamp")
            >>> print(df_features[["timestamp", "day_of_week", "month"]].head())
        """
        if timestamp_col not in df.columns:
            raise KeyError(f"Column '{timestamp_col}' not found in DataFrame")
        
        # Create a copy to avoid modifying the original
        df_features = df.copy()
        
        # Ensure timestamp column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df_features[timestamp_col]):
            try:
                df_features[timestamp_col] = pd.to_datetime(df_features[timestamp_col])
            except Exception as e:
                raise ValueError(
                    f"Could not convert '{timestamp_col}' to datetime: {e}"
                )
        
        # Extract time features
        df_features['day_of_week'] = df_features[timestamp_col].dt.dayofweek
        df_features['day_of_month'] = df_features[timestamp_col].dt.day
        df_features['month'] = df_features[timestamp_col].dt.month
        df_features['quarter'] = df_features[timestamp_col].dt.quarter
        df_features['is_month_start'] = df_features[timestamp_col].dt.is_month_start
        df_features['is_month_end'] = df_features[timestamp_col].dt.is_month_end
        
        # Calculate days since shock if shock column exists
        if 'is_shock' in df_features.columns:
            df_features['days_since_shock'] = self._calculate_days_since_shock(
                df_features, timestamp_col
            )
        else:
            # Initialize with a large value if no shocks detected
            df_features['days_since_shock'] = 999
        
        return df_features
    
    def _calculate_days_since_shock(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> pd.Series:
        """
        Calculate days since the last market shock.
        
        This is a helper method that computes the number of days since the
        most recent market shock for each row.
        
        Args:
            df: DataFrame with 'is_shock' column
            timestamp_col: Name of the timestamp column
        
        Returns:
            Series with days since last shock for each row
        """
        days_since_shock = []
        last_shock_date = None
        
        for idx, row in df.iterrows():
            if row.get('is_shock', False):
                last_shock_date = row[timestamp_col]
                days_since_shock.append(0)
            elif last_shock_date is not None:
                days_diff = (row[timestamp_col] - last_shock_date).days
                days_since_shock.append(days_diff)
            else:
                # No shock has occurred yet
                days_since_shock.append(999)
        
        return pd.Series(days_since_shock, index=df.index)
    
    def inverse_transform(
        self,
        df: pd.DataFrame,
        scaler_params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Inverse transform normalized features back to original scale.
        
        This method reverses the normalization applied by normalize_features()
        using the stored scaler parameters.
        
        Args:
            df: DataFrame with normalized features
            scaler_params: Dictionary of scaler parameters from normalize_features()
        
        Returns:
            DataFrame with features transformed back to original scale
        
        Example:
            >>> df_norm, scalers = preprocessor.normalize_features(df, ["price"])
            >>> df_original = preprocessor.inverse_transform(df_norm, scalers)
        """
        df_inverse = df.copy()
        
        for col, params in scaler_params.items():
            if col in df_inverse.columns:
                scaler = params["scaler"]
                values = df_inverse[col].values.reshape(-1, 1)
                df_inverse[col] = scaler.inverse_transform(values).flatten()
        
        return df_inverse
