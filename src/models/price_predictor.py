"""
Price Predictor for hybrid cocoa price forecasting.

This module implements the PricePredictor class that combines predictions from
Prophet (time series baseline), XGBoost (residual corrections), and FinBERT
(sentiment adjustments) to generate final price predictions with confidence intervals.
"""

from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from src.models.time_series_model import TimeSeriesModel
from src.models.ml_model import MLModel
from src.nlp.nlp_analyzer import NLPAnalyzer
from src.models.data_models import Prediction, NewsArticle


class PricePredictor:
    """
    Combines predictions from multiple models to generate hybrid forecasts.
    
    The PricePredictor orchestrates the hybrid prediction system by:
    1. Getting baseline predictions from Prophet (time series model)
    2. Getting residual corrections from XGBoost (ML model)
    3. Getting sentiment adjustments from FinBERT (NLP analyzer)
    4. Combining all components into final predictions
    5. Computing confidence intervals with sentiment-based adjustments
    
    The hybrid approach follows the additive formula:
    Final_Price = Baseline + Residual_Correction + Sentiment_Adjustment
    
    Attributes:
        ts_model: Fitted TimeSeriesModel instance
        ml_model: Fitted MLModel instance
        nlp_analyzer: NLPAnalyzer instance
        sentiment_weight: Weight for sentiment adjustment (default: 0.1)
        model_version: Version identifier for tracking
    """
    
    def __init__(
        self,
        ts_model: TimeSeriesModel,
        ml_model: MLModel,
        nlp_analyzer: NLPAnalyzer,
        sentiment_weight: float = 0.1,
        model_version: str = "1.0.0"
    ):
        """
        Initialize PricePredictor with trained models.
        
        Args:
            ts_model: Fitted TimeSeriesModel for baseline predictions
            ml_model: Fitted MLModel for residual corrections
            nlp_analyzer: NLPAnalyzer for sentiment analysis
            sentiment_weight: Weight for sentiment adjustment (0.0-1.0).
                Higher values give more influence to sentiment. Default 0.1
                means sentiment can adjust price by up to ±10%.
            model_version: Version identifier for tracking predictions
        
        Raises:
            ValueError: If models are not fitted or sentiment_weight is invalid
        """
        # Validate models are fitted
        if not ts_model.is_fitted:
            raise ValueError("TimeSeriesModel must be fitted before use")
        if not ml_model.is_fitted:
            raise ValueError("MLModel must be fitted before use")
        
        # Validate sentiment_weight
        if not 0.0 <= sentiment_weight <= 1.0:
            raise ValueError(
                f"sentiment_weight must be between 0.0 and 1.0, got {sentiment_weight}"
            )
        
        self.ts_model = ts_model
        self.ml_model = ml_model
        self.nlp_analyzer = nlp_analyzer
        self.sentiment_weight = sentiment_weight
        self.model_version = model_version
        
        logger.info(
            f"PricePredictor initialized with sentiment_weight={sentiment_weight}, "
            f"model_version={model_version}"
        )
    
    def predict(
        self,
        horizons: List[int],
        exog_features: pd.DataFrame,
        recent_news: List[NewsArticle],
        historical_range: Optional[Tuple[float, float]] = None
    ) -> List[Prediction]:
        """
        Generate hybrid predictions for multiple horizons.
        
        This is the main prediction method that orchestrates the entire hybrid
        forecasting pipeline:
        1. Get baseline predictions from Prophet for each horizon
        2. Get residual corrections from XGBoost using econometric features
        3. Aggregate sentiment from recent news articles
        4. Combine all components into final predictions
        5. Calculate confidence intervals with sentiment-based adjustments
        6. Validate predictions are within realistic bounds
        
        Args:
            horizons: List of prediction horizons in days (e.g., [1, 7, 30])
            exog_features: DataFrame with econometric features for each horizon.
                Expected columns: temperature, rainfall, stock_level, production,
                fx_rate_xaf_usd, fx_rate_gbp_usd, fx_rate_eur_usd, etc.
                Must have same number of rows as len(horizons).
            recent_news: List of NewsArticle objects for sentiment analysis
            historical_range: Optional tuple of (min_price, max_price) from
                historical data for validation. If None, uses default bounds
                (1000, 10000).
        
        Returns:
            List of Prediction objects, one for each horizon, containing:
                - horizon: Days ahead
                - price: Final predicted price
                - confidence_interval: (lower_bound, upper_bound)
                - confidence_level: 0.95
                - timestamp: When prediction was made
                - model_version: Version identifier
                - components: Dict with baseline, residual, sentiment values
        
        Raises:
            ValueError: If horizons and exog_features have mismatched lengths
            RuntimeError: If prediction generation fails
        
        Example:
            >>> predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
            >>> horizons = [1, 7, 30]
            >>> features = pd.DataFrame({
            ...     'temperature': [25.0, 26.0, 27.0],
            ...     'rainfall': [10.0, 12.0, 15.0],
            ...     'stock_level': [50000, 48000, 45000],
            ...     # ... other features
            ... })
            >>> news = [article1, article2, article3]
            >>> predictions = predictor.predict(horizons, features, news)
            >>> print(predictions[0].price)
            3250.50
        """
        # Validate inputs
        if len(horizons) != len(exog_features):
            raise ValueError(
                f"horizons and exog_features must have same length. "
                f"Got horizons: {len(horizons)}, features: {len(exog_features)}"
            )
        
        if not horizons:
            raise ValueError("horizons list cannot be empty")
        
        if any(h <= 0 for h in horizons):
            raise ValueError("All horizons must be positive integers")
        
        logger.info(f"Generating predictions for horizons: {horizons}")
        
        # Set default historical range if not provided
        if historical_range is None:
            historical_range = (1000.0, 10000.0)
        
        # Step 1: Get baseline predictions from Prophet
        max_horizon = max(horizons)
        logger.info(f"Getting baseline predictions from Prophet for {max_horizon} periods")
        
        try:
            prophet_forecast = self.ts_model.predict(periods=max_horizon, freq="D")
        except Exception as e:
            logger.error(f"Failed to get Prophet predictions: {e}")
            raise RuntimeError(f"Prophet prediction failed: {e}")
        
        # Step 2: Get residual corrections from XGBoost with fallback
        logger.info(f"Getting residual corrections from XGBoost for {len(exog_features)} horizons")
        
        residual_corrections = None
        xgboost_failed = False
        
        try:
            residual_corrections = self.ml_model.predict(exog_features)
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            logger.warning("Falling back to Prophet baseline (zero residuals)")
            xgboost_failed = True
            # Use zero residuals as fallback (Requirement 12.2)
            residual_corrections = np.zeros(len(exog_features))
            
            # Send alert for model failure
            try:
                from src.monitoring.alert_system import get_alert_system
                alert_system = get_alert_system()
                alert_system.send_model_failure_alert(
                    model_name="XGBoost",
                    error_message=str(e),
                    context={"horizons": horizons}
                )
            except Exception as alert_error:
                logger.error(f"Failed to send alert: {alert_error}")
        
        # Step 3: Aggregate sentiment from recent news
        logger.info(f"Aggregating sentiment from {len(recent_news)} news articles")
        
        try:
            sentiment_score = self.nlp_analyzer.aggregate_sentiment(recent_news)
            logger.info(f"Aggregated sentiment score: {sentiment_score:.3f}")
        except Exception as e:
            logger.warning(f"Failed to aggregate sentiment, using neutral (0.0): {e}")
            sentiment_score = 0.0
        
        # Check if sentiment indicates high risk
        sentiment_risk = sentiment_score < -0.6
        if sentiment_risk:
            logger.warning(
                f"High-risk sentiment detected (score: {sentiment_score:.3f}). "
                "Confidence intervals will be widened."
            )
        
        # Step 4: Generate predictions for each horizon
        predictions = []
        current_time = datetime.now()
        
        for i, horizon in enumerate(horizons):
            # Get baseline from Prophet (horizon-1 because Prophet is 0-indexed from last training point)
            baseline = prophet_forecast.iloc[horizon - 1]['yhat']
            ts_lower = prophet_forecast.iloc[horizon - 1]['yhat_lower']
            ts_upper = prophet_forecast.iloc[horizon - 1]['yhat_upper']
            ts_uncertainty = (ts_upper - ts_lower) / 2.0
            
            # Get residual correction from XGBoost
            residual = residual_corrections[i]
            
            # Estimate ML model uncertainty (use std of residuals as proxy)
            # In production, this could be computed from cross-validation results
            ml_uncertainty = abs(residual) * 0.2  # Assume 20% uncertainty
            
            # Combine predictions
            final_price = self.combine_predictions(
                baseline=baseline,
                residual=residual,
                sentiment_score=sentiment_score,
                sentiment_weight=self.sentiment_weight
            )
            
            # Calculate confidence interval
            lower_bound, upper_bound = self.calculate_confidence_interval(
                prediction=final_price,
                ts_uncertainty=ts_uncertainty,
                ml_uncertainty=ml_uncertainty,
                sentiment_risk=sentiment_risk,
                confidence_level=0.95
            )
            
            # Validate prediction
            is_valid = self.validate_prediction(final_price, historical_range)
            if not is_valid:
                logger.warning(
                    f"Prediction for horizon {horizon} ({final_price:.2f}) "
                    f"is outside historical range {historical_range}. "
                    "Clamping to valid range."
                )
                final_price = np.clip(final_price, historical_range[0], historical_range[1])
                lower_bound = np.clip(lower_bound, historical_range[0], historical_range[1])
                upper_bound = np.clip(upper_bound, historical_range[0], historical_range[1])
            
            # Calculate sentiment adjustment component
            sentiment_adjustment = sentiment_score * self.sentiment_weight * baseline
            
            # Create Prediction object
            prediction = Prediction(
                horizon=horizon,
                price=float(final_price),
                confidence_interval=(float(lower_bound), float(upper_bound)),
                confidence_level=0.95,
                timestamp=current_time,
                model_version=self.model_version,
                components={
                    "baseline": float(baseline),
                    "residual": float(residual),
                    "sentiment": float(sentiment_adjustment)
                }
            )
            
            predictions.append(prediction)
            
            logger.info(
                f"Horizon {horizon}d: price={final_price:.2f}, "
                f"CI=[{lower_bound:.2f}, {upper_bound:.2f}], "
                f"baseline={baseline:.2f}, residual={residual:.2f}, "
                f"sentiment_adj={sentiment_adjustment:.2f}"
            )
        
        logger.info(f"Successfully generated {len(predictions)} predictions")
        
        return predictions
    
    def combine_predictions(
        self,
        baseline: float,
        residual: float,
        sentiment_score: float,
        sentiment_weight: float = 0.1
    ) -> float:
        """
        Combine predictions using additive approach.
        
        The hybrid prediction formula is:
        final = baseline + residual + (sentiment_score * sentiment_weight * baseline)
        
        Where:
        - baseline: Prophet's time series prediction
        - residual: XGBoost's correction to the baseline
        - sentiment_score: Aggregated sentiment from news (-1 to +1)
        - sentiment_weight: How much sentiment can adjust the price (0.0 to 1.0)
        
        The sentiment adjustment is proportional to the baseline price, so a
        sentiment_score of -0.5 with sentiment_weight of 0.1 would reduce the
        price by 5% of the baseline.
        
        Args:
            baseline: Baseline prediction from time series model
            residual: Residual correction from ML model
            sentiment_score: Sentiment score from NLP analyzer (-1 to +1)
            sentiment_weight: Weight for sentiment adjustment (0.0 to 1.0)
        
        Returns:
            Combined final prediction
        
        Example:
            >>> predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
            >>> final = predictor.combine_predictions(
            ...     baseline=3000.0,
            ...     residual=50.0,
            ...     sentiment_score=-0.5,
            ...     sentiment_weight=0.1
            ... )
            >>> print(final)
            2900.0  # 3000 + 50 + (-0.5 * 0.1 * 3000)
        """
        # Calculate sentiment adjustment as percentage of baseline
        sentiment_adjustment = sentiment_score * sentiment_weight * baseline
        
        # Combine all components
        final_prediction = baseline + residual + sentiment_adjustment
        
        logger.debug(
            f"Combining: baseline={baseline:.2f}, residual={residual:.2f}, "
            f"sentiment_adj={sentiment_adjustment:.2f} -> final={final_prediction:.2f}"
        )
        
        return final_prediction
    
    def calculate_confidence_interval(
        self,
        prediction: float,
        ts_uncertainty: float,
        ml_uncertainty: float,
        sentiment_risk: bool,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate combined confidence interval.
        
        The confidence interval combines uncertainties from both the time series
        model and the ML model. If high sentiment risk is detected (sentiment < -0.6),
        the interval is widened by 50% to account for increased market volatility.
        
        The combined uncertainty is computed as:
        combined_uncertainty = sqrt(ts_uncertainty^2 + ml_uncertainty^2)
        
        This assumes the uncertainties are independent, which is reasonable since
        they come from different model types capturing different aspects of the data.
        
        Args:
            prediction: Final predicted price
            ts_uncertainty: Uncertainty from time series model (half-width of CI)
            ml_uncertainty: Uncertainty from ML model (estimated)
            sentiment_risk: Whether high-risk sentiment is detected
            confidence_level: Confidence level (default: 0.95 for 95% CI)
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        
        Example:
            >>> predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
            >>> lower, upper = predictor.calculate_confidence_interval(
            ...     prediction=3000.0,
            ...     ts_uncertainty=100.0,
            ...     ml_uncertainty=50.0,
            ...     sentiment_risk=True,
            ...     confidence_level=0.95
            ... )
            >>> print(f"CI: [{lower:.2f}, {upper:.2f}]")
            CI: [2832.29, 3167.71]  # Widened by 50% due to sentiment risk
        """
        # Combine uncertainties (assuming independence)
        combined_uncertainty = np.sqrt(ts_uncertainty**2 + ml_uncertainty**2)
        
        # Widen interval if sentiment risk is detected
        if sentiment_risk:
            combined_uncertainty *= 1.5  # 50% wider
            logger.debug(
                f"Widening confidence interval by 50% due to sentiment risk. "
                f"New uncertainty: {combined_uncertainty:.2f}"
            )
        
        # For 95% confidence level, use 1.96 standard deviations
        # For other confidence levels, use appropriate z-score
        if confidence_level == 0.95:
            z_score = 1.96
        elif confidence_level == 0.99:
            z_score = 2.576
        elif confidence_level == 0.90:
            z_score = 1.645
        else:
            # Approximate z-score for other confidence levels
            from scipy import stats
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Calculate bounds
        margin = z_score * combined_uncertainty
        lower_bound = prediction - margin
        upper_bound = prediction + margin
        
        logger.debug(
            f"Confidence interval: [{lower_bound:.2f}, {upper_bound:.2f}] "
            f"(width: {upper_bound - lower_bound:.2f})"
        )
        
        return (lower_bound, upper_bound)
    
    def validate_prediction(
        self,
        prediction: float,
        historical_range: Tuple[float, float]
    ) -> bool:
        """
        Validate prediction is within realistic bounds.
        
        Checks if the prediction falls within the historical price range.
        This helps catch unrealistic predictions that may result from model
        errors or extreme input values.
        
        Args:
            prediction: Predicted price to validate
            historical_range: Tuple of (min_price, max_price) from historical data
        
        Returns:
            True if prediction is valid (within historical range), False otherwise
        
        Example:
            >>> predictor = PricePredictor(ts_model, ml_model, nlp_analyzer)
            >>> is_valid = predictor.validate_prediction(3000.0, (1000.0, 10000.0))
            >>> print(is_valid)
            True
            >>> is_valid = predictor.validate_prediction(15000.0, (1000.0, 10000.0))
            >>> print(is_valid)
            False
        """
        min_price, max_price = historical_range
        
        is_valid = min_price <= prediction <= max_price
        
        if not is_valid:
            logger.warning(
                f"Prediction {prediction:.2f} is outside historical range "
                f"[{min_price:.2f}, {max_price:.2f}]"
            )
        
        return is_valid
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the models used in the predictor.
        
        Returns:
            Dictionary with model information including:
                - model_version: Version identifier
                - sentiment_weight: Sentiment adjustment weight
                - ts_model_params: Time series model hyperparameters
                - ml_model_params: ML model hyperparameters
        """
        return {
            "model_version": self.model_version,
            "sentiment_weight": self.sentiment_weight,
            "ts_model_params": self.ts_model.get_hyperparameters(),
            "ml_model_params": self.ml_model.get_hyperparameters()
        }
    
    def __repr__(self) -> str:
        """String representation of the predictor."""
        return (
            f"PricePredictor(model_version='{self.model_version}', "
            f"sentiment_weight={self.sentiment_weight}, "
            f"ts_model={self.ts_model}, "
            f"ml_model={self.ml_model})"
        )
