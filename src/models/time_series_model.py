"""
Time Series Model using Prophet for cocoa price prediction.

This module implements the baseline forecasting model that captures
trends and seasonal patterns in cocoa prices.
"""

from typing import Optional
import pandas as pd
import numpy as np
from prophet import Prophet
from loguru import logger


class TimeSeriesModel:
    """Prophet-based time series forecasting model.
    
    This model captures long-term trends and seasonal patterns in cocoa prices.
    It serves as the baseline prediction in the hybrid system, with residuals
    being further refined by the ML model.
    
    Attributes:
        model: Prophet model instance
        seasonality_mode: Type of seasonality ('additive' or 'multiplicative')
        yearly_seasonality: Whether to include yearly seasonality
        weekly_seasonality: Whether to include weekly seasonality
        changepoint_prior_scale: Flexibility of trend changes
        is_fitted: Whether the model has been trained
    """
    
    def __init__(
        self,
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05
    ):
        """Initialize Prophet model with hyperparameters.
        
        Args:
            seasonality_mode: 'additive' or 'multiplicative'. Multiplicative is
                better for data where seasonal variations scale with the trend.
            yearly_seasonality: If True, model yearly seasonal patterns (12 months).
            weekly_seasonality: If True, model weekly patterns. Usually False for
                commodity prices which don't have strong weekly patterns.
            changepoint_prior_scale: Controls flexibility of trend. Lower values
                (0.001-0.05) make trend more conservative, higher values (0.05-0.5)
                allow more flexibility. Default 0.05 is good for cocoa prices.
        """
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.is_fitted = False
        
        # Initialize Prophet model
        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            daily_seasonality=False  # Not relevant for commodity prices
        )
        
        logger.info(
            f"TimeSeriesModel initialized with seasonality_mode={seasonality_mode}, "
            f"yearly_seasonality={yearly_seasonality}, "
            f"changepoint_prior_scale={changepoint_prior_scale}"
        )
    
    def fit(
        self,
        df: pd.DataFrame,
        date_col: str = "ds",
        target_col: str = "y"
    ) -> None:
        """Fit Prophet model to historical price data.
        
        Prophet expects a DataFrame with columns 'ds' (date) and 'y' (target).
        If your DataFrame has different column names, specify them via parameters.
        
        Args:
            df: DataFrame with historical price data. Must contain date and price columns.
            date_col: Name of the date column (default: 'ds' for Prophet format)
            target_col: Name of the target/price column (default: 'y' for Prophet format)
        
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        # Validate input
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        if len(df) < 2:
            raise ValueError("Need at least 2 data points to fit the model")
        
        # Prepare data in Prophet format
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Ensure ds is datetime
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        # Sort by date
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        # Check for missing values
        if prophet_df['y'].isna().any():
            logger.warning(
                f"Found {prophet_df['y'].isna().sum()} missing values in target. "
                "Prophet will handle these internally."
            )
        
        logger.info(f"Fitting Prophet model on {len(prophet_df)} data points")
        logger.info(f"Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
        
        # Fit the model
        self.model.fit(prophet_df)
        self.is_fitted = True
        
        logger.info("Prophet model fitted successfully")
    
    def predict(
        self,
        periods: int,
        freq: str = "D"
    ) -> pd.DataFrame:
        """Generate baseline predictions for future periods.
        
        Args:
            periods: Number of periods to forecast into the future
            freq: Frequency of predictions. 'D' for daily, 'W' for weekly, 'M' for monthly.
        
        Returns:
            DataFrame with columns:
                - ds: Date
                - yhat: Predicted value (baseline prediction)
                - yhat_lower: Lower bound of confidence interval
                - yhat_upper: Upper bound of confidence interval
                - trend: Trend component
                - yearly: Yearly seasonal component (if enabled)
        
        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions. Call fit() first.")
        
        if periods <= 0:
            raise ValueError(f"periods must be positive, got {periods}")
        
        logger.info(f"Generating predictions for {periods} periods with frequency '{freq}'")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        
        # Generate predictions
        forecast = self.model.predict(future)
        
        # Select relevant columns
        result_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']
        if self.yearly_seasonality:
            result_cols.append('yearly')
        
        result = forecast[result_cols].copy()
        
        logger.info(f"Generated {len(result)} predictions")
        
        return result
    
    def compute_residuals(
        self,
        df: pd.DataFrame
    ) -> pd.Series:
        """Compute residuals (actual - predicted) for training data.
        
        Residuals represent the portion of price variation not captured by
        the time series model. These will be predicted by the ML model using
        econometric features.
        
        Args:
            df: DataFrame with actual values. Must have 'ds' (date) and 'y' (actual) columns.
        
        Returns:
            Series of residual values (actual - predicted), indexed by date
        
        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing residuals. Call fit() first.")
        
        # Validate input
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must have 'ds' and 'y' columns")
        
        # Prepare data
        prophet_df = df[['ds', 'y']].copy()
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        # Generate predictions for the same dates
        forecast = self.model.predict(prophet_df[['ds']])
        
        # Compute residuals: actual - predicted
        residuals = prophet_df['y'] - forecast['yhat']
        residuals.index = prophet_df['ds']
        
        logger.info(
            f"Computed residuals: mean={residuals.mean():.2f}, "
            f"std={residuals.std():.2f}, "
            f"min={residuals.min():.2f}, "
            f"max={residuals.max():.2f}"
        )
        
        return residuals
    
    def get_components(self) -> pd.DataFrame:
        """Extract trend and seasonal components from the fitted model.
        
        This is useful for understanding what the model has learned and for
        debugging. The components show how much of the price variation is
        attributed to trend vs seasonality.
        
        Returns:
            DataFrame with columns:
                - ds: Date
                - trend: Trend component
                - yearly: Yearly seasonal component (if enabled)
                - weekly: Weekly seasonal component (if enabled)
                - yhat: Overall prediction (sum of components)
        
        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before extracting components. Call fit() first.")
        
        # Get the training data predictions to extract components
        # Prophet stores the history internally
        history = self.model.history.copy()
        forecast = self.model.predict(history[['ds']])
        
        # Select component columns
        component_cols = ['ds', 'trend', 'yhat']
        if self.yearly_seasonality:
            component_cols.insert(2, 'yearly')
        if self.weekly_seasonality:
            component_cols.insert(2, 'weekly')
        
        components = forecast[component_cols].copy()
        
        logger.info(f"Extracted {len(components)} component values")
        
        return components
    
    def get_hyperparameters(self) -> dict:
        """Get the current hyperparameters of the model.
        
        Returns:
            Dictionary with hyperparameter names and values
        """
        return {
            'seasonality_mode': self.seasonality_mode,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'changepoint_prior_scale': self.changepoint_prior_scale
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"TimeSeriesModel(seasonality_mode='{self.seasonality_mode}', "
            f"yearly_seasonality={self.yearly_seasonality}, "
            f"changepoint_prior_scale={self.changepoint_prior_scale}, "
            f"status='{status}')"
        )
