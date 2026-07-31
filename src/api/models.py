"""
Pydantic models for API request/response validation.

This module defines all request and response models for the FastAPI endpoints,
ensuring proper validation and documentation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request model for price prediction endpoint.
    
    Attributes:
        horizons: List of prediction horizons in days (e.g., [1, 7, 30])
        market: Market identifier (e.g., "ICE_London", "ICE_NY")
        include_sentiment: Whether to include sentiment analysis in predictions
    """
    
    horizons: List[int] = Field(
        ...,
        description="List of prediction horizons in days",
        min_length=1,
        example=[1, 7, 30]
    )
    market: str = Field(
        ...,
        description="Market identifier",
        example="ICE_London"
    )
    include_sentiment: bool = Field(
        default=True,
        description="Whether to include sentiment analysis"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "horizons": [1, 7, 30],
                "market": "ICE_London",
                "include_sentiment": True
            }
        }


class PredictionItem(BaseModel):
    """
    Individual prediction item in the response.
    
    Attributes:
        horizon: Prediction horizon in days
        price: Predicted price in USD/MT
        confidence_interval: Tuple of (lower_bound, upper_bound)
        confidence_level: Confidence level (e.g., 0.95)
        timestamp: When prediction was made
    """
    
    horizon: int = Field(..., description="Prediction horizon in days")
    price: float = Field(..., description="Predicted price in USD/MT")
    confidence_interval: List[float] = Field(
        ...,
        description="Confidence interval [lower, upper]",
        min_length=2,
        max_length=2
    )
    confidence_level: float = Field(..., description="Confidence level")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    components: Optional[dict] = Field(None, description="Individual engine predictions (baseline, nhits, prophet, etc.)")


class PredictionResponse(BaseModel):
    """
    Response model for price prediction endpoint.
    
    Attributes:
        predictions: List of predictions for each horizon
        model_version: Version of the model used
        sentiment_score: Aggregated sentiment score (if included)
        market: Market identifier
    """
    
    predictions: List[PredictionItem] = Field(
        ...,
        description="List of predictions for each horizon"
    )
    model_version: str = Field(..., description="Model version identifier")
    sentiment_score: Optional[float] = Field(
        None,
        description="Aggregated sentiment score (-1 to +1)"
    )
    market: str = Field(..., description="Market identifier")
    current_price: Optional[float] = Field(
        None,
        description="Current spot price"
    )
    current_date: Optional[str] = Field(
        None,
        description="Date of the current price"
    )
    historical_prices: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of recent historical prices (e.g. last 14 days)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "horizon": 1,
                        "price": 3250.50,
                        "confidence_interval": [3100.00, 3400.00],
                        "confidence_level": 0.95,
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                ],
                "model_version": "v1.2.3",
                "sentiment_score": -0.15,
                "market": "ICE_London"
            }
        }


class PerformanceMetricsItem(BaseModel):
    """
    Individual performance metrics item.
    
    Attributes:
        timestamp: When metrics were computed
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        mape: Mean Absolute Percentage Error
        directional_accuracy: Percentage of correct up/down predictions
        coverage_rate: Percentage of actual values within CI
        mean_interval_width: Average width of confidence intervals
    """
    
    timestamp: datetime = Field(..., description="Metrics computation timestamp")
    rmse: float = Field(..., description="Root Mean Squared Error")
    mae: float = Field(..., description="Mean Absolute Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    directional_accuracy: float = Field(
        ...,
        description="Directional accuracy (0-1)"
    )
    coverage_rate: float = Field(..., description="Coverage rate (0-1)")
    mean_interval_width: float = Field(
        ...,
        description="Mean confidence interval width"
    )


class PerformanceResponse(BaseModel):
    """
    Response model for performance metrics endpoint.
    
    Attributes:
        model_version: Version of the model
        metrics: List of performance metrics over time
        start_date: Start date of the metrics period
        end_date: End date of the metrics period
    """
    
    model_version: str = Field(..., description="Model version identifier")
    metrics: List[PerformanceMetricsItem] = Field(
        ...,
        description="List of performance metrics"
    )
    start_date: datetime = Field(..., description="Start date of metrics period")
    end_date: datetime = Field(..., description="End date of metrics period")


class HorizonValidationMetrics(BaseModel):
    """Walk-forward metrics for one horizon."""

    horizon: int
    mape: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    directional_accuracy: Optional[float] = None
    n_predictions: Optional[int] = None


class ValidationMetricsResponse(BaseModel):
    """Honest walk-forward validation metrics from latest backtest report."""

    report_timestamp: Optional[str] = None
    report_path: Optional[str] = None
    validation_type: str = "walk_forward_multi_horizon"
    n_origins: Optional[int] = None
    horizons: List[int] = Field(default_factory=list)
    xgb_metrics: List[HorizonValidationMetrics] = Field(default_factory=list)
    legacy_holdout_mape_1step: Optional[float] = None
    ensemble_calibration: Optional[Dict[str, Any]] = None
    conformal_intervals: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    """
    Information about a model version.
    
    Attributes:
        name: Model name
        version: Model version
        stage: Model stage (Staging, Production, Archived)
        created_at: When model was created
        metrics: Latest performance metrics
    """
    
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: str = Field(..., description="Model stage")
    created_at: datetime = Field(..., description="Model creation timestamp")
    metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Latest performance metrics"
    )


class ModelsResponse(BaseModel):
    """
    Response model for listing available models.
    
    Attributes:
        models: List of available model versions
        current_production_version: Version currently in production
    """
    
    models: List[ModelInfo] = Field(..., description="List of available models")
    current_production_version: Optional[str] = Field(
        None,
        description="Version currently in production"
    )


class RetrainingRequest(BaseModel):
    """
    Request model for triggering model retraining.
    
    Attributes:
        model_type: Type of model to retrain (all, prophet, xgboost, finbert)
        reason: Reason for retraining
    """
    
    model_type: str = Field(
        default="all",
        description="Type of model to retrain",
        pattern="^(all|prophet|xgboost|finbert)$"
    )
    reason: str = Field(
        ...,
        description="Reason for triggering retraining",
        min_length=1
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "all",
                "reason": "Performance degradation detected"
            }
        }


class RetrainingResponse(BaseModel):
    """
    Response model for retraining trigger endpoint.
    
    Attributes:
        status: Status of the retraining request
        message: Human-readable message
        job_id: Identifier for the retraining job
        estimated_completion: Estimated completion time
    """
    
    status: str = Field(..., description="Status of retraining request")
    message: str = Field(..., description="Human-readable message")
    job_id: str = Field(..., description="Retraining job identifier")
    estimated_completion: Optional[datetime] = Field(
        None,
        description="Estimated completion time"
    )


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    
    Attributes:
        error: Error type
        message: Error message
        detail: Additional error details
    """
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


class BriefRequest(BaseModel):
    """Requete de brief marche (Claude)."""

    market: str = Field(..., description="ICE_NY, COFFEE_ROBUSTA, etc.")
    mode: str = Field(
        default="standard",
        description="standard (Sonnet) ou advanced (Opus, quota 3/jour)",
    )
    question: Optional[str] = Field(
        None,
        description="Question optionnelle pour le mode advanced",
    )
    force_refresh: bool = Field(default=False, description="Ignorer le cache brief")


class MarketIntelligenceResponse(BaseModel):
    """Brief structure + contexte marche."""

    market: str
    market_display_name: str
    unit: str
    tradingview_symbol: Optional[str] = None
    current_price: Optional[float] = None
    current_date: Optional[str] = None
    model_version: Optional[str] = None
    sentiment_score: Optional[float] = None
    predictions: Optional[List[Dict[str, Any]]] = None
    brief: Dict[str, Any]
    mode: str
    opus_remaining: int = 3
    generated_at: str
    cached: bool = False


class TradingViewAlert(BaseModel):
    """Payload envoye par une alerte Pine Script TradingView (lecture de marche enrichie)."""

    secret: str = Field(..., description="Secret partage pour authentifier le webhook")
    market: str = Field(
        ...,
        description="Marche cible: ICE_NY, COFFEE_ROBUSTA (ou alias TradingView)",
    )
    signal_type: str = Field(
        ...,
        description="Type de signal: buy | sell | support_break | resistance_break | trend_change | custom",
    )
    price: Optional[float] = Field(None, description="Prix au moment de l'alerte")
    tf: Optional[str] = Field(None, description="Timeframe TradingView (1D, 4H, etc.)")
    ticker: Optional[str] = Field(None, description="Ticker TradingView (ex: PEPPERSTONE:COCOA)")
    message: Optional[str] = Field(None, description="Message libre de l'alerte")
    indicator: Optional[str] = Field(None, description="Nom de l'indicateur/strategie declencheur")
    timestamp: Optional[str] = Field(None, description="Horodatage TradingView")
    # Lecture de marche (calculee cote Pine, interpretee par Claude)
    trend: Optional[str] = Field(
        None, description="Tendance: bullish | bearish | neutral"
    )
    momentum: Optional[str] = Field(
        None,
        description="Momentum: strong_buy | buy | neutral | sell | strong_sell",
    )
    change_pct: Optional[float] = Field(None, description="Variation session en %")
    rsi: Optional[float] = Field(None, description="RSI 14 (0-100), traduit en langage marche pour Claude")
    price_vs_ma: Optional[str] = Field(
        None, description="Position vs moyennes: above | below | mixed"
    )
    support: Optional[float] = Field(None, description="Support proche")
    resistance: Optional[float] = Field(None, description="Resistance proche")
    volume_ratio: Optional[float] = Field(
        None, description="Volume relatif vs moyenne (1.0 = normal)"
    )
    mode: str = Field(
        default="standard",
        description="standard (Sonnet, cache 24h) ou advanced (Opus, quota jour)",
    )
    force_refresh: bool = Field(default=True, description="Regenerer le brief (defaut True pour un webhook)")


class TradingViewAlertResponse(BaseModel):
    """Reponse renvoyee a TradingView + payload brief inclus."""

    received: bool = True
    alert_id: Optional[str] = None
    market: str
    signal_type: str
    intelligence: MarketIntelligenceResponse


class LatestTradingViewAlert(BaseModel):
    """Derniere alerte TradingView (pour polling dashboard)."""

    id: str
    market: str
    signal_type: str
    price: Optional[float] = None
    tf: Optional[str] = None
    ticker: Optional[str] = None
    message: Optional[str] = None
    trend: Optional[str] = None
    momentum: Optional[str] = None
    support: Optional[float] = None
    resistance: Optional[float] = None
    change_pct: Optional[float] = None
    received_at: str
    brief_signal: Optional[str] = None
    brief_summary: Optional[str] = None


class RecentTradingViewAlertsResponse(BaseModel):
    """Liste des alertes TradingView recentes."""

    market: str
    alerts: List[LatestTradingViewAlert]


class DashboardNotification(BaseModel):
    """Notification persistée pour le dashboard."""

    id: str
    market: str
    source: str = "tradingview"
    kind: str
    title: str
    body: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    is_read: bool = False
    created_at: str


class NotificationsListResponse(BaseModel):
    market: str
    notifications: List[DashboardNotification]
    unread_count: int
