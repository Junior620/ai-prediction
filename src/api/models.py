"""
Pydantic models for API request/response validation.

This module defines all request and response models for the FastAPI endpoints,
ensuring proper validation and documentation.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request model for price prediction endpoint.
    
    Attributes:
        horizons: List of prediction horizons in days (e.g., [1, 7, 14, 30])
        market: Market identifier (e.g., "ICE_London", "ICE_NY")
        include_sentiment: Whether to include sentiment analysis in predictions
    """
    
    horizons: List[int] = Field(
        ...,
        description="List of prediction horizons in days",
        min_length=1,
        example=[1, 7, 14, 30]
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
                "horizons": [1, 7, 14, 30],
                "market": "ICE_London",
                "include_sentiment": True
            }
        }


class ForecastView(BaseModel):
    """One forecast: the kept initial, the latest same-target estimate, or today's calculation."""

    price: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    target_date: Optional[str] = None
    issued_date: Optional[str] = None
    model_version: Optional[str] = None
    status: Optional[Literal["experimental", "validated", "unavailable"]] = None


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
    price: Optional[float] = Field(None, description="Predicted price in USD/MT")
    confidence_interval: Optional[List[float]] = Field(
        None,
        description="Confidence interval [lower, upper]",
        min_length=2,
        max_length=2,
    )
    confidence_level: Optional[float] = Field(None, description="Confidence level")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    status: Literal["experimental", "validated", "unavailable"] = Field(
        "experimental",
        description="experimental and unavailable in experimental mode; validated only for an active validated horizon",
    )
    components: Optional[dict] = Field(None, description="Individual engine predictions (baseline, nhits, prophet, etc.)")
    target_date: Optional[str] = Field(None, description="Échéance de l'engagement affiché")
    reference_price: Optional[float] = Field(None, description="Cours du jour d'ouverture de l'engagement")
    initial_horizon: Optional[int] = Field(None, description="Horizon du jour d'ouverture")
    sessions_remaining: Optional[int] = Field(None, description="Séances de bourse restantes jusqu'à l'échéance")
    latest_price: Optional[float] = Field(None, description="Dernière estimation publiée de la même échéance")
    latest_date: Optional[str] = Field(None, description="Date de calcul de la dernière estimation")
    latest_gap: Optional[float] = Field(None, description="Écart de la dernière estimation avec le prix initial")
    latest_gap_pct: Optional[float] = Field(None, description="Écart en pourcentage du prix initial")
    revision_notice: Optional[Literal["inversion", "revision"]] = Field(
        None,
        description="inversion de sens, ou révision au-delà de la bande neutre",
    )
    has_revision: Optional[bool] = Field(None, description="Une estimation plus récente vise la même échéance")
    latest_sessions_remaining: Optional[int] = Field(
        None,
        description="Séances restantes au moment du calcul de la dernière révision",
    )
    revision_matches_today: Optional[bool] = Field(
        None,
        description="La dernière révision a été calculée aujourd'hui pour le délai encore ouvert",
    )
    latest_unavailable: Optional[bool] = Field(
        None,
        description="Un calcul de la même échéance a échoué",
    )
    initial: Optional[ForecastView] = None
    latest: Optional[ForecastView] = None
    fresh: Optional[ForecastView] = None


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
    service_calendar: Optional[str] = Field(
        None,
        description="Préparation des séances appliquée par le lot servi",
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
    market: Optional[str] = Field(None, description="Market of this row")
    horizon: Optional[int] = Field(None, description="Horizon of this row")


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
    market: Optional[str] = Field(None, description="Market filter echoed in the response")
    horizon: Optional[int] = Field(None, description="Horizon filter echoed in the response")


class HorizonValidationMetrics(BaseModel):
    """Walk-forward metrics for one horizon."""

    horizon: int
    mape: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    directional_accuracy: Optional[float] = None
    n_predictions: Optional[int] = None
    validated: bool = False
    relative_gain: Optional[float] = None
    gap_ci_low: Optional[float] = None
    gap_ci_high: Optional[float] = None
    n_effective: Optional[float] = None
    fallback_rate: Optional[float] = None
    measurement: Optional[str] = None
    reason: Optional[str] = None
    gap_reading: Optional[str] = None
    available_mape: Optional[float] = None
    naive_mape: Optional[float] = None
    abstention_count: Optional[int] = None
    abstention_rate: Optional[float] = None
    coverage: Optional[float] = None
    coverage_note: Optional[str] = None
    warning_count: Optional[int] = None


class ValidationMetricsResponse(BaseModel):
    """Honest walk-forward validation metrics from latest backtest report."""

    report_timestamp: Optional[str] = None
    report_path: Optional[str] = None
    validation_type: str = "walk_forward_multi_horizon"
    n_origins: Optional[int] = None
    origin_start: Optional[str] = None
    origin_end: Optional[str] = None
    target_start: Optional[str] = None
    target_end: Optional[str] = None
    evaluated_component: Optional[str] = None
    horizons: List[int] = Field(default_factory=list)
    xgb_metrics: List[HorizonValidationMetrics] = Field(default_factory=list)
    legacy_holdout_mape_1step: Optional[float] = None
    ensemble_calibration: Optional[Dict[str, Any]] = None
    conformal_intervals: Optional[Dict[str, Any]] = None
    release_version: Optional[str] = None
    release_matches_report: bool = False
    report_id: Optional[str] = None
    measurement_status: Optional[str] = None
    partition: Optional[str] = None


class HistoricalEvaluationItem(BaseModel):
    report_id: str
    timestamp: Optional[str] = None
    kind: Optional[str] = None
    period: Optional[str] = None
    protocol: Optional[str] = None
    counts: Optional[str] = None
    compatible_with_current_lot: bool = False
    note: str


class HistoricalEvaluationsResponse(BaseModel):
    market: str
    release_version: Optional[str] = None
    reports: List[HistoricalEvaluationItem] = Field(default_factory=list)


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


class FuturesHorizonPrediction(BaseModel):
    horizon: int
    price: float
    method: str = "xgboost"
    change_pct: Optional[float] = None


class FuturesContractItem(BaseModel):
    contract: str
    symbol: str
    yahoo_symbol: Optional[str] = None
    price_usd: float
    change: Optional[float] = None
    volume: Optional[float] = None
    predictions: List[FuturesHorizonPrediction] = Field(default_factory=list)


class FuturesCurveResponse(BaseModel):
    contracts: List[FuturesContractItem]
    collected_at: Optional[str] = None
    source: Optional[str] = None
    model_version: Optional[str] = None
    spot_pct_by_horizon: Optional[Dict[str, float]] = None
    unit: str = "USD/MT"
    currency: str = "USD"
    # Courbe ICE London continue C.v.0..3 (GBP) — secondaire
    london_contracts: List[FuturesContractItem] = Field(default_factory=list)
    london_collected_at: Optional[str] = None
    london_source: Optional[str] = None
    london_unit: str = "GBP/MT"
    london_currency: str = "GBP"


class LondonHistoryPoint(BaseModel):
    date: str
    price: float
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    source: Optional[str] = None


class LondonTermContract(BaseModel):
    contract_rank: int
    symbol: str
    close: float
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    label: str


class LondonMarketResponse(BaseModel):
    """Microstructure ICE London (Databento) : OHLCV + OI + courbe d'echeances."""
    latest: Optional[LondonHistoryPoint] = None
    history: List[LondonHistoryPoint] = Field(default_factory=list)
    term_structure: List[LondonTermContract] = Field(default_factory=list)
    term_date: Optional[str] = None
    unit: str = "GBP/MT"
    source: Optional[str] = "databento"


class ModelComparisonMetric(BaseModel):
    model: str
    label: str
    horizon: int
    mae: float
    rmse: float
    mape: float
    n: int


class ModelComparisonResponse(BaseModel):
    generated_at: Optional[str] = None
    split_date: Optional[str] = None
    n_train: Optional[int] = None
    n_test: Optional[int] = None
    metrics: List[ModelComparisonMetric] = Field(default_factory=list)
    note: Optional[str] = None


class CocoaWeatherLocation(BaseModel):
    location: str
    country: str
    temperature_c: Optional[float] = None
    precipitation_mm: Optional[float] = None
    humidity: Optional[int] = None
    condition: Optional[str] = None
    collected_at: Optional[str] = None


class CocoaWeatherResponse(BaseModel):
    """Instantané WeatherAPI. Il n'entre pas dans le prix publié."""

    locations: List[CocoaWeatherLocation] = Field(default_factory=list)
    collected_at: Optional[str] = None
    note: str = "Contexte météo, non utilisé par la prévision."


class ForecastTrackingPoint(BaseModel):
    forecast_id: Optional[str] = None
    origin_date: Optional[str] = None
    origin_price: Optional[float] = None
    target_date: Optional[str] = None
    horizon: Optional[int] = None
    price: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    status: Optional[str] = None
    model_version: Optional[str] = None
    reference_price: Optional[float] = None
    trace: Optional[str] = None
    revises: List[str] = Field(default_factory=list)
    error: Optional[float] = None


class ForecastChartPoint(BaseModel):
    date: str
    price: Optional[float] = None
    model_version: Optional[str] = None
    gap: bool = False


class ModelChangeDiagnostic(BaseModel):
    available: bool = False
    comparison: str = "indisponible"
    previous_model_version: Optional[str] = None
    previous_price: Optional[float] = None
    new_price: Optional[float] = None
    note: Optional[str] = None


class RevisionDelayStep(BaseModel):
    """Gain at one remaining horizon, or between two successive revisions."""

    remaining: int
    comparison: Literal["initial_vs_remaining", "successive"]
    expected: int = 0
    available: int = 0
    abstentions: int = 0
    missing_emissions: int = 0
    missing_realizations: int = 0
    session_jumps: int = 0
    availability_rate: Optional[float] = None
    mean_initial_error: Optional[float] = None
    mean_revised_error: Optional[float] = None
    mean_spot_error: Optional[float] = None
    mean_gain: Optional[float] = None
    improved_share: Optional[float] = None
    gain_low: Optional[float] = None
    gain_high: Optional[float] = None
    verdict: Literal["etabli", "defavorable", "incertain"] = "incertain"
    interval_level: float = 0.90
    usable_blocks: int = 0
    block: Optional[int] = None


class RevisionValueHorizon(BaseModel):
    horizon: int
    comparison: str = "initiale contre dernière estimation avant échéance"
    verdict: Literal["etabli", "defavorable", "incertain"] = "incertain"
    expiries: int = 0
    mean_initial_error: Optional[float] = None
    mean_latest_error: Optional[float] = None
    mean_gain: Optional[float] = None
    improved_share: Optional[float] = None
    worsened_share: Optional[float] = None
    improvement_sum: Optional[float] = None
    degradation_sum: Optional[float] = None
    gain_low: Optional[float] = None
    gain_high: Optional[float] = None
    block: Optional[int] = None
    interval_level: float = 0.90
    usable_blocks: Optional[int] = None
    steps: List[RevisionDelayStep] = Field(default_factory=list)


class RevisionValueResponse(BaseModel):
    market: str
    as_of: Optional[str] = None
    note: Optional[str] = None
    service_calendar: Optional[str] = None
    study_calendar: Optional[str] = None
    price_fingerprint: Optional[str] = None
    report: Optional[str] = None
    procedure: Optional[str] = None
    live_price_differs: Optional[bool] = None
    horizons: List[RevisionValueHorizon] = Field(default_factory=list)


class ForecastTrackingResponse(BaseModel):
    market: str
    found: Optional[bool] = None
    initial_id: Optional[str] = None
    initials: List[ForecastTrackingPoint] = Field(default_factory=list)
    total_initials: Optional[int] = None
    offset: Optional[int] = None
    initial: Optional[ForecastTrackingPoint] = None
    reference_price: Optional[float] = None
    revisions: List[ForecastTrackingPoint] = Field(default_factory=list)
    latest: Optional[ForecastTrackingPoint] = None
    gap: Optional[float] = None
    gap_pct: Optional[float] = None
    direction: Optional[str] = None
    direction_vs_emission_spot: Optional[str] = None
    inversions: Optional[int] = None
    points: List[ForecastChartPoint] = Field(default_factory=list)
    remaining_horizon: Optional[int] = None
    mature: Optional[bool] = None
    realized_price: Optional[float] = None
    initial_error: Optional[float] = None
    latest_error: Optional[float] = None
    model_comparison: Optional[ModelChangeDiagnostic] = None
