"""
FastAPI application for the Cocoa Price Prediction System.

This module implements the REST API with the following endpoints:
- POST /api/v1/predict: Generate price predictions
- GET /api/v1/performance: Retrieve performance metrics
- GET /api/v1/models: List available models
- POST /api/v1/retrain: Trigger model retraining (admin only)

Implements Requirements 10.1-10.5, 12.3, 12.4, 13.1, 13.4, 13.5
"""

from datetime import datetime, timedelta
from typing import List, Optional
import uuid
import sys
from fastapi import FastAPI, Depends, HTTPException, status, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import pandas as pd
from supabase import create_client, Client

from src.api.models import (
    PredictionRequest,
    PredictionResponse,
    PredictionItem,
    PerformanceResponse,
    PerformanceMetricsItem,
    ValidationMetricsResponse,
    HorizonValidationMetrics,
    ModelsResponse,
    ModelInfo,
    RetrainingRequest,
    RetrainingResponse,
    ErrorResponse,
    BriefRequest,
    MarketIntelligenceResponse,
    TradingViewAlert,
    TradingViewAlertResponse,
    LatestTradingViewAlert,
    RecentTradingViewAlertsResponse,
    DashboardNotification,
    NotificationsListResponse,
    FuturesCurveResponse,
    FuturesContractItem,
    FuturesHorizonPrediction,
    LondonMarketResponse,
    LondonHistoryPoint,
    LondonTermContract,
    ModelComparisonResponse,
    ModelComparisonMetric,
)
from src.api.auth import verify_token, verify_admin_token, decode_token
from src.api.cache import RedisCache
from src.api.ws_hub import notification_hub
from src.api.notification_service import (
    create_tv_notification_record,
    persist_notification,
    publish_notification,
    list_notifications,
    mark_notification_read,
    mark_all_notifications_read,
)
from src.models.price_predictor import PricePredictor
from src.models.improved_price_predictor import ImprovedPricePredictor
from src.models.direct_horizon_trainer import DirectHorizonTrainer
from src.models.time_series_model import TimeSeriesModel
from src.models.ml_model import MLModel
from src.models.futures_curve_predictor import FuturesCurvePredictor
from src.nlp.nlp_analyzer import NLPAnalyzer
from src.models.model_manager import ModelManager
from src.monitoring.performance_monitor import PerformanceMonitor
from src.monitoring.alert_system import get_alert_system, AlertSeverity, AlertType
from src.models.data_models import NewsArticle
from src.models.market_registry import (
    MarketConfig,
    get_market_config,
    list_api_markets,
    load_all_markets,
    resolve_api_market,
)
from src.intelligence.brief_service import BriefService
from config.settings import get_settings

# Configure structured logging (Requirement 12.3)
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)
logger.add(
    "logs/api_{time:YYYY-MM-DD}.log",
    rotation="00:00",  # Rotate at midnight
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    serialize=True  # JSON format for structured logging
)
logger.add(
    "logs/api_errors_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="90 days",
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    serialize=True
)

# Initialize FastAPI app
app = FastAPI(
    title="Cocoa Price Prediction API",
    description="REST API for hybrid cocoa price prediction system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS: browser origins only (HTTP API traffic should go through the Next.js BFF)
_settings = get_settings()
_cors_origins = [
    o.strip() for o in (_settings.cors_origins or "").split(",") if o.strip()
]
if not _cors_origins:
    _cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (will be initialized on startup)
redis_cache: Optional[RedisCache] = None
supabase_client: Optional[Client] = None
model_manager: Optional[ModelManager] = None
performance_monitor: Optional[PerformanceMonitor] = None
price_predictor: Optional[PricePredictor] = None  # alias for the cocoa predictor
predictors: dict = {}  # market_id -> ImprovedPricePredictor
brief_service: Optional[BriefService] = None
futures_curve_predictor: Optional[FuturesCurvePredictor] = None
futures_named_predictor: Optional[FuturesCurvePredictor] = None
alert_system = get_alert_system()


# Default values for econometric features (used as fallback when Supabase data is unavailable)
_DEFAULT_EXOG = {
    'temperature': 25.0,       # Average temperature in cocoa regions (°C)
    'rainfall': 120.0,         # Average monthly rainfall (mm)
    'stock_level': 50000.0,    # Estimated global stock level (metric tons)
    'production': 4000000.0,   # Annual production estimate (metric tons)
    'fx_rate_xaf_usd': 0.0016, # XAF/USD exchange rate
    'fx_rate_gbp_usd': 1.27,   # GBP/USD exchange rate
    'fx_rate_eur_usd': 1.09,   # EUR/USD exchange rate
}


def _fetch_exog_features(client: Client, n_rows: int) -> pd.DataFrame:
    """
    Fetch the latest econometric features from Supabase.
    
    Tries to load real data from the econometric_data table. If the table
    is empty or the query fails, falls back to sensible default values.
    
    Args:
        client: Supabase client instance.
        n_rows: Number of rows to return (one per prediction horizon).
        
    Returns:
        DataFrame with econometric feature columns.
    """
    try:
        response = (
            client
            .table("econometric_data")
            .select("temperature, rainfall, stock_level, production, "
                    "fx_rate_xaf_usd, fx_rate_gbp_usd, fx_rate_eur_usd")
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
        
        if response.data:
            row = response.data[0]
            values = {
                col: float(row.get(col) or _DEFAULT_EXOG[col])
                for col in _DEFAULT_EXOG
            }
            logger.info(
                f"Loaded real econometric features from Supabase "
                f"(temperature={values['temperature']}, "
                f"fx_eur_usd={values['fx_rate_eur_usd']})"
            )
        else:
            values = _DEFAULT_EXOG.copy()
            logger.info("No econometric data in Supabase, using defaults")
    except Exception as e:
        values = _DEFAULT_EXOG.copy()
        logger.warning(f"Failed to fetch econometric data: {e} — using defaults")
    
    return pd.DataFrame({col: [val] * n_rows for col, val in values.items()})


# Access logging middleware (Requirement 13.4)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all API requests with user identification and timestamp.
    
    Implements Requirement 13.4: Log all access attempts with user_id and timestamp
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    # Extract user_id from authorization header if present
    user_id = "anonymous"
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            from src.api.auth import decode_token
            token = auth_header.split(" ")[1]
            payload = decode_token(token)
            user_id = payload.get("sub", "unknown")
        except Exception:
            user_id = "invalid_token"
    
    # Log request
    logger.info(
        f"Request started",
        extra={
            "request_id": request_id,
            "user_id": user_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown",
            "timestamp": start_time.isoformat()
        }
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Log response
        logger.info(
            f"Request completed",
            extra={
                "request_id": request_id,
                "user_id": user_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return response
        
    except Exception as e:
        # Log error
        logger.error(
            f"Request failed",
            extra={
                "request_id": request_id,
                "user_id": user_id,
                "method": request.method,
                "path": request.url.path,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        raise


@app.on_event("startup")
async def startup_event():
    """
    Initialize services on application startup.
    
    Implements structured error handling with CRITICAL alerts (Requirement 12.4).
    """
    global redis_cache, supabase_client, model_manager, performance_monitor, price_predictor, brief_service, futures_curve_predictor, futures_named_predictor
    
    logger.info("Starting up Cocoa Price Prediction API...")
    
    # Load settings
    settings = get_settings()
    
    # Initialize Redis cache
    try:
        redis_cache = RedisCache()
        if redis_cache.health_check():
            logger.info("Redis cache initialized successfully")
        else:
            logger.warning("Redis cache health check failed")
        # Live notifications: Redis pub/sub -> WebSocket fan-out
        if redis_cache is not None:
            notification_hub.start_redis_subscriber(
                settings.redis_host,
                settings.redis_port,
                settings.redis_password if settings.redis_password else None,
                settings.redis_db,
            )
    except Exception as e:
        logger.error(f"Failed to initialize Redis cache: {e}")
        redis_cache = None
        # Non-critical error, continue without cache
    
    # Initialize Supabase client
    try:
        supabase_client = create_client(
            settings.supabase_url,
            settings.supabase_key
        )
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize Supabase client: {e}")
        alert_system.send_alert(
            severity=AlertSeverity.CRITICAL,
            alert_type=AlertType.SYSTEM_ERROR,
            message="Failed to initialize Supabase database connection",
            details={"error": str(e)}
        )
        raise
    
    # Initialize Model Manager
    try:
        model_manager = ModelManager(
            tracking_uri=settings.mlflow_tracking_uri,
            registry_uri=settings.mlflow_registry_uri
        )
        logger.info("Model Manager initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize Model Manager: {e}")
        alert_system.send_alert(
            severity=AlertSeverity.CRITICAL,
            alert_type=AlertType.SYSTEM_ERROR,
            message="Failed to initialize MLflow Model Manager",
            details={"error": str(e)}
        )
        raise
    
    # Initialize Performance Monitor
    try:
        performance_monitor = PerformanceMonitor(
            supabase_client=supabase_client
        )
        logger.info("Performance Monitor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Performance Monitor: {e}")
        alert_system.send_alert(
            severity=AlertSeverity.ERROR,
            alert_type=AlertType.SYSTEM_ERROR,
            message="Failed to initialize Performance Monitor",
            details={"error": str(e)}
        )
        # Non-critical, continue without performance monitoring
        performance_monitor = None
    
    # Load production models: one predictor per market declared in config.yaml
    try:
        nlp_analyzer = NLPAnalyzer()
        logger.info("✅ NLP Analyzer initialized")
    except Exception as e:
        logger.error(f"Failed to initialize NLP Analyzer: {e}")
        nlp_analyzer = None

    pred_cfg: dict = {}
    try:
        import yaml
        with open("config/config.yaml", encoding="utf-8") as f:
            pred_cfg = (yaml.safe_load(f) or {}).get("prediction", {})
    except Exception:
        pass

    for market_id, market_cfg in load_all_markets().items():
        try:
            predictor = _load_market_predictor(market_cfg, settings, nlp_analyzer, pred_cfg)
            if predictor is not None:
                predictors[market_id] = predictor
        except Exception as e:
            logger.warning(f"Failed to load models for market '{market_id}': {e}")

    price_predictor = predictors.get("cocoa")

    if not predictors:
        logger.warning("No market predictor loaded - API will start but predictions unavailable")
        alert_system.send_alert(
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.MODEL_FAILURE,
            message="Failed to load production models on startup",
            details={"markets": list(load_all_markets())}
        )
    else:
        logger.info(f"✅ Predictors loaded for markets: {sorted(predictors)}")

    try:
        futures_curve_predictor = FuturesCurvePredictor(source="london")
        n_sym = len(getattr(futures_curve_predictor, "_models", {}) or {})
        if n_sym == 0:
            logger.warning("London futures models empty — fallback Investing/Yahoo dir")
            futures_curve_predictor = FuturesCurvePredictor(source="investing")
            n_sym = len(getattr(futures_curve_predictor, "_models", {}) or {})
        logger.info(
            "FuturesCurvePredictor ready (%s, %d symbols)",
            getattr(futures_curve_predictor, "source", "?"),
            n_sym,
        )
    except Exception as e:
        futures_curve_predictor = None
        logger.warning(f"FuturesCurvePredictor not loaded: {e}")

    try:
        futures_named_predictor = FuturesCurvePredictor(source="london_named")
        n_named = len(getattr(futures_named_predictor, "_models", {}) or {})
        logger.info("Futures named London predictor ready (%d contracts)", n_named)
        if n_named == 0:
            futures_named_predictor = None
    except Exception as e:
        futures_named_predictor = None
        logger.warning(f"Futures named predictor not loaded: {e}")

    brief_service = BriefService(redis_cache=redis_cache)
    logger.info("BriefService (Claude) initialized")
    app.state.journal_schema_ok = _journal_schema_ok()
    if not app.state.journal_schema_ok:
        logger.critical("Le journal des prévisions n'est pas enregistré.")

    logger.info("Cocoa Price Prediction API started successfully")


def _latest_price_date(price_table: str) -> Optional[str]:
    """Newest session date in the market table, or None if it cannot be read."""
    if supabase_client is None:
        return None
    try:
        response = (
            supabase_client.table(price_table)
            .select("date")
            .order("date", desc=True)
            .limit(1)
            .execute()
        )
        if response.data:
            return str(response.data[0]["date"])[:10]
    except Exception as exc:
        logger.warning(f"Lecture de la dernière séance impossible: {exc}")
    return None


def _journal_schema_ok() -> bool:
    """True when the prediction journal has the columns the evaluator needs."""
    if supabase_client is None:
        return False
    try:
        supabase_client.table("predictions").select(
            "market,origin_date,origin_price,target_date,feature_failure,currency,candidate_price"
        ).limit(1).execute()
        return True
    except Exception as exc:
        logger.critical(f"Journal des prévisions incomplet: {exc}")
        try:
            alert_system.send_alert(
                severity=AlertSeverity.CRITICAL,
                alert_type=AlertType.SYSTEM_ERROR,
                message="Le journal des prévisions n'est pas enregistré",
                details={"error": str(exc)},
            )
        except Exception:
            logger.error("Alerte journal non envoyée")
        return False


def _mark_journal_unavailable(exc: Exception) -> None:
    """A failed journal write is visible on /health until the process restarts."""
    app.state.journal_schema_ok = False
    logger.error(f"Journal des prévisions indisponible: {exc}")
    try:
        alert_system.send_alert(
            severity=AlertSeverity.CRITICAL,
            alert_type=AlertType.SYSTEM_ERROR,
            message="Le journal des prévisions n'est pas enregistré",
            details={"error": str(exc)},
        )
    except Exception:
        logger.error("Alerte journal non envoyée")


def _load_market_predictor(
    market_cfg: MarketConfig,
    settings,
    nlp_analyzer,
    pred_cfg: dict,
) -> Optional[ImprovedPricePredictor]:
    """Load the artifacts named by the active release. Never the newest file."""
    import json as _json
    import pickle
    from pathlib import Path

    from src.models.release_manifest import artifact_path, load_active_release

    try:
        release = load_active_release(market_cfg.market_id)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"[{market_cfg.market_id}] {exc}")
        return None

    prophet_path = artifact_path(release, "prophet")
    xgboost_path = artifact_path(release, "xgboost")
    info_path = artifact_path(release, "improved_info")
    if not prophet_path.exists() or not xgboost_path.exists() or not info_path.exists():
        logger.error(f"[{market_cfg.market_id}] Artefact du manifeste introuvable.")
        return None

    meta = _json.loads(info_path.read_text(encoding="utf-8"))
    if meta.get("target") != "next_session":
        logger.error(
            f"[{market_cfg.market_id}] Métadonnées sans cible next_session : non chargé."
        )
        return None

    model_version = release.get("version") or prophet_path.stem.replace("prophet_", "")
    feature_cols = meta.get("feature_cols")
    feature_set = meta.get("feature_set", "baseline")
    logger.info(
        f"[{market_cfg.market_id}] Manifeste {model_version}, validé={release.get('validated')}"
    )

    with open(prophet_path, "rb") as f:
        prophet_model = pickle.load(f)
    with open(xgboost_path, "rb") as f:
        xgboost_model = pickle.load(f)

    nhits_model = None
    nhits_path = artifact_path(release, "nhits")
    try:
        if nhits_path.exists():
            from neuralforecast import NeuralForecast
            nhits_model = NeuralForecast.load(path=str(nhits_path))
            logger.info(f"[{market_cfg.market_id}] N-HiTS chargé depuis {nhits_path.name}")
    except Exception as e:
        logger.warning(f"[{market_cfg.market_id}] N-HiTS du manifeste illisible: {e}")
        nhits_model = None

    direct_models = DirectHorizonTrainer.load_from_info(
        str(artifact_path(release, "direct_info"))
    )
    model_dir = prophet_path.parent
    sentiment_w = 0.0 if release.get("scored_variant") == "no_sentiment" else float(
        pred_cfg.get("sentiment_weight_production", pred_cfg.get("sentiment_weight", 0.05))
    )

    predictor = ImprovedPricePredictor(
        prophet_model=prophet_model,
        xgboost_model=xgboost_model,
        nlp_analyzer=nlp_analyzer,
        sentiment_weight=sentiment_w,
        model_version=model_version,
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_key,
        nhits_model=nhits_model,
        ensemble_weights_file=str(artifact_path(release, "ensemble_weights")),
        ensemble_fallback=pred_cfg.get("ensemble_fallback"),
        multi_step_mode=pred_cfg.get("multi_step_mode", "recursive"),
        direct_horizon_models=direct_models,
        conformal_intervals_file=str(artifact_path(release, "conformal_intervals")),
        confidence_level=pred_cfg.get("confidence_level", 0.90),
        price_bounds=market_cfg.price_bounds,
        price_table=market_cfg.price_table,
        nhits_unique_id=market_cfg.nhits_unique_id,
        garch_enabled=market_cfg.garch_enabled,
        models_dir=str(model_dir),
        max_abs_change_pct=pred_cfg.get("max_abs_change_pct"),
        recent_range_days=int(pred_cfg.get("recent_range_days", 252)),
        recent_range_padding_pct=float(pred_cfg.get("recent_range_padding_pct", 15.0)),
        feature_cols=feature_cols,
        feature_set=feature_set,
    )
    predictor.active_release = release
    n_engines = 3 if nhits_model else 2
    logger.info(
        f"[{market_cfg.market_id}] ✅ Predictor ready: {n_engines} engines, "
        f"version={model_version}, garch={'on' if market_cfg.garch_enabled else 'off'}"
    )
    return predictor


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on application shutdown.
    """
    logger.info("Shutting down Cocoa Price Prediction API...")

    try:
        await notification_hub.stop()
    except Exception as e:
        logger.error(f"Error stopping notification hub: {e}")
    
    # Close Redis connection
    if redis_cache and redis_cache.redis_client:
        try:
            redis_cache.redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
    
    logger.info("Cocoa Price Prediction API shut down successfully")


@app.get("/")
async def root():
    """
    Root endpoint providing API information.
    """
    return {
        "name": "Cocoa Price Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/api/v1/predict",
            "performance": "/api/v1/performance",
            "models": "/api/v1/models",
            "retrain": "/api/v1/retrain"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "redis": redis_cache.health_check() if redis_cache else False,
            "supabase": supabase_client is not None,
            "model_manager": model_manager is not None,
            "price_predictor": price_predictor is not None
        },
        "markets_loaded": sorted(predictors.keys()),
        "journal_schema_ok": bool(getattr(app.state, "journal_schema_ok", False)),
    }
    
    # Determine overall health
    all_healthy = all(health_status["services"].values()) and health_status["journal_schema_ok"]
    health_status["status"] = "healthy" if all_healthy else "degraded"
    
    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(content=health_status, status_code=status_code)


@app.get("/api/v1/markets")
async def list_markets():
    """
    List all configured markets with their availability status.
    """
    markets = []
    for market_id, cfg in load_all_markets().items():
        predictor = predictors.get(market_id)
        markets.append({
            "market_id": market_id,
            "display_name": cfg.display_name,
            "api_markets": cfg.api_markets,
            "unit": cfg.unit,
            "source": cfg.source,
            "contract_symbol": cfg.contract_symbol,
            "garch_enabled": cfg.garch_enabled,
            "tradingview_symbol": cfg.tradingview_symbol,
            "tradingview_embed_symbol": cfg.tradingview_embed_symbol,
            "tradingview_embed_label": cfg.tradingview_embed_label,
            "tradingview_alert_symbol": cfg.tradingview_alert_symbol,
            "available": predictor is not None,
            "model_version": predictor.model_version if predictor else None,
        })
    return {"markets": markets}


def _resolve_predictor_for_api_market(api_market: str):
    """Return (market_cfg, predictor) or raise HTTPException."""
    market_cfg = resolve_api_market(api_market)
    if market_cfg is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown market '{api_market}'. Valid: {list_api_markets()}",
        )
    predictor = predictors.get(market_cfg.market_id)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Predictor unavailable for '{api_market}'",
        )
    return market_cfg, predictor


@app.get(
    "/api/v1/market-intelligence",
    response_model=MarketIntelligenceResponse,
    responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def get_market_intelligence(
    market: str = "ICE_NY",
    mode: str = "standard",
    force_refresh: bool = False,
    user: str = Depends(verify_token),
) -> MarketIntelligenceResponse:
    """
    Brief marche genere par Claude a partir des predictions ML existantes.
    mode=standard (Sonnet, cache 24h) | mode=advanced (Opus, 3 requetes/jour/utilisateur).
    """
    if brief_service is None:
        raise HTTPException(status_code=503, detail="BriefService not initialized")

    advanced = mode.lower() == "advanced"
    _, predictor = _resolve_predictor_for_api_market(market)

    try:
        result = brief_service.generate(
            api_market=market,
            predictor=predictor,
            supabase=supabase_client,
            user_id=user,
            advanced=advanced,
            force_refresh=force_refresh,
        )
        return MarketIntelligenceResponse(**result)
    except PermissionError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Market intelligence failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/brief",
    response_model=MarketIntelligenceResponse,
    responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def post_market_brief(
    request: BriefRequest,
    user: str = Depends(verify_token),
) -> MarketIntelligenceResponse:
    """Alias POST pour brief marche (supporte question en mode advanced)."""
    if brief_service is None:
        raise HTTPException(status_code=503, detail="BriefService not initialized")

    advanced = request.mode.lower() == "advanced"
    _, predictor = _resolve_predictor_for_api_market(request.market)

    try:
        result = brief_service.generate(
            api_market=request.market,
            predictor=predictor,
            supabase=supabase_client,
            user_id=user,
            advanced=advanced,
            user_question=request.question,
            force_refresh=request.force_refresh,
        )
        return MarketIntelligenceResponse(**result)
    except PermissionError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Brief generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _resolve_market_from_alert(payload: TradingViewAlert) -> str:
    """
    Accept either an API market identifier (ICE_NY) or a TradingView ticker
    (ICEEUR:C1!, PEPPERSTONE:COCOA, ROBCOFFEE, ICEEUR:RC1!).
    Returns the canonical API market.
    """
    if resolve_api_market(payload.market) is not None:
        return payload.market

    candidates = [payload.market, payload.ticker]
    for cfg in load_all_markets().values():
        tv_symbols = {
            (cfg.tradingview_symbol or "").upper(),
            (cfg.tradingview_embed_symbol or "").upper(),
            (cfg.tradingview_alert_symbol or "").upper(),
        }
        # Alias courts (ex: C1! pour ICEEUR:C1!)
        for sym in list(tv_symbols):
            if ":" in sym:
                tv_symbols.add(sym.split(":", 1)[1])
        tv_symbols.discard("")
        for cand in candidates:
            if not cand:
                continue
            c = cand.upper()
            if c in tv_symbols or c.split(":")[-1] in tv_symbols:
                if cfg.api_markets:
                    return cfg.api_markets[0]
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=(
            f"Marche non reconnu: '{payload.market}' (ticker='{payload.ticker}'). "
            f"Utilisez un identifiant API ({list_api_markets()}) ou un ticker TradingView configure."
        ),
    )


def _persist_tradingview_alert(
    payload: TradingViewAlert, api_market: str
) -> Optional[str]:
    """Persist alert to Supabase (best-effort). Returns alert_id or None."""
    if supabase_client is None:
        return None
    try:
        record = {
            "market": api_market,
            "signal_type": payload.signal_type,
            "price": payload.price,
            "tf": payload.tf,
            "ticker": payload.ticker,
            "indicator": payload.indicator,
            "message": payload.message,
            "tv_timestamp": payload.timestamp,
            "trend": payload.trend,
            "momentum": payload.momentum,
            "change_pct": payload.change_pct,
            "rsi": payload.rsi,
            "price_vs_ma": payload.price_vs_ma,
            "support": payload.support,
            "resistance": payload.resistance,
            "volume_ratio": payload.volume_ratio,
            "mode": payload.mode,
            "received_at": datetime.utcnow().isoformat(),
        }
        resp = (
            supabase_client.table("tradingview_alerts")
            .insert(record)
            .execute()
        )
        if resp.data:
            return str(resp.data[0].get("id"))
    except Exception as e:
        logger.warning(f"TradingView alert not persisted (table missing?): {e}")
    return None


@app.post(
    "/api/v1/webhooks/tradingview",
    response_model=TradingViewAlertResponse,
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def tradingview_webhook(payload: TradingViewAlert) -> TradingViewAlertResponse:
    """
    Receoit une alerte Pine Script TradingView, declenche un brief Claude
    contextualise et retourne la reponse d'intelligence marche.

    Authentification: champ 'secret' du body compare a TRADINGVIEW_WEBHOOK_SECRET.
    """
    import hmac

    configured_secret = get_settings().tradingview_webhook_secret
    if not configured_secret:
        raise HTTPException(
            status_code=503,
            detail="Webhook non configure: definissez TRADINGVIEW_WEBHOOK_SECRET dans .env",
        )
    if not hmac.compare_digest(payload.secret, configured_secret):
        logger.warning(f"Webhook TradingView rejete (secret invalide, market={payload.market})")
        raise HTTPException(status_code=401, detail="Secret webhook invalide")

    if brief_service is None:
        raise HTTPException(status_code=503, detail="BriefService not initialized")

    api_market = _resolve_market_from_alert(payload)
    _, predictor = _resolve_predictor_for_api_market(api_market)

    alert_context = {
        "signal_type": payload.signal_type,
        "price": payload.price,
        "tf": payload.tf,
        "ticker": payload.ticker,
        "indicator": payload.indicator,
        "message": payload.message,
        "timestamp": payload.timestamp,
        "trend": payload.trend,
        "momentum": payload.momentum,
        "change_pct": payload.change_pct,
        "rsi": payload.rsi,
        "price_vs_ma": payload.price_vs_ma,
        "support": payload.support,
        "resistance": payload.resistance,
        "volume_ratio": payload.volume_ratio,
    }

    alert_id = _persist_tradingview_alert(payload, api_market)

    advanced = payload.mode.lower() == "advanced"

    try:
        result = brief_service.generate(
            api_market=api_market,
            predictor=predictor,
            supabase=supabase_client,
            user_id=f"tradingview:{payload.indicator or 'pine'}",
            advanced=advanced,
            force_refresh=payload.force_refresh,
            alert_context=alert_context,
        )
        logger.info(
            f"Webhook TradingView traite: market={api_market} signal={payload.signal_type} "
            f"mode={payload.mode} alert_id={alert_id}"
        )
        brief = result.get("brief") or {}
        summary = brief.get("summary") or ""
        snapshot = {
            "id": alert_id or f"tv-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "market": api_market,
            "signal_type": payload.signal_type,
            "price": payload.price,
            "tf": payload.tf,
            "ticker": payload.ticker,
            "message": payload.message,
            "trend": payload.trend,
            "momentum": payload.momentum,
            "support": payload.support,
            "resistance": payload.resistance,
            "change_pct": payload.change_pct,
            "received_at": datetime.utcnow().isoformat() + "Z",
            "brief_signal": brief.get("signal"),
            "brief_summary": summary[:280] if isinstance(summary, str) else None,
        }
        if redis_cache:
            redis_cache.set_latest_tv_alert(api_market, snapshot)

        # Persist + push live notification (WebSocket via Redis pub/sub)
        notif_record = create_tv_notification_record(snapshot)
        saved = persist_notification(supabase_client, notif_record) or {
            "id": snapshot["id"],
            **notif_record,
        }
        publish_notification(redis_cache, saved)

        return TradingViewAlertResponse(
            received=True,
            alert_id=alert_id,
            market=api_market,
            signal_type=payload.signal_type,
            intelligence=MarketIntelligenceResponse(**result),
        )
    except PermissionError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Webhook TradingView failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/tradingview/alerts/latest",
    response_model=Optional[LatestTradingViewAlert],
    responses={401: {"model": ErrorResponse}},
)
async def get_latest_tradingview_alert(
    market: str = "ICE_NY",
    user: str = Depends(verify_token),
) -> Optional[LatestTradingViewAlert]:
    """
    Derniere alerte TradingView pour un marche (polling dashboard).
    Source: Redis (ecrit a chaque webhook). Fallback Supabase si Redis vide.
    """
    resolved = _resolve_market_key_for_alerts(market)

    if redis_cache:
        cached = redis_cache.get_latest_tv_alert(resolved)
        if cached:
            return LatestTradingViewAlert(**cached)

    if supabase_client is not None:
        try:
            resp = (
                supabase_client.table("tradingview_alerts")
                .select("*")
                .eq("market", resolved)
                .order("received_at", desc=True)
                .limit(1)
                .execute()
            )
            if resp.data:
                row = resp.data[0]
                return LatestTradingViewAlert(
                    id=str(row.get("id")),
                    market=row.get("market") or resolved,
                    signal_type=row.get("signal_type") or "custom",
                    price=row.get("price"),
                    tf=row.get("tf"),
                    ticker=row.get("ticker"),
                    message=row.get("message"),
                    trend=row.get("trend"),
                    momentum=row.get("momentum"),
                    support=row.get("support"),
                    resistance=row.get("resistance"),
                    change_pct=row.get("change_pct"),
                    received_at=str(row.get("received_at") or datetime.utcnow().isoformat()),
                )
        except Exception as e:
            logger.warning(f"Lecture derniere alerte Supabase echouee: {e}")

    return None


@app.get(
    "/api/v1/tradingview/alerts/recent",
    response_model=RecentTradingViewAlertsResponse,
    responses={401: {"model": ErrorResponse}},
)
async def get_recent_tradingview_alerts(
    market: str = "ICE_NY",
    limit: int = 5,
    user: str = Depends(verify_token),
) -> RecentTradingViewAlertsResponse:
    """Historique court des alertes TradingView (polling / panneau dashboard)."""
    resolved = _resolve_market_key_for_alerts(market)
    limit = max(1, min(int(limit), 20))
    alerts: List[LatestTradingViewAlert] = []

    if redis_cache:
        for row in redis_cache.get_recent_tv_alerts(resolved, limit):
            try:
                alerts.append(LatestTradingViewAlert(**row))
            except Exception:
                continue

    if not alerts and supabase_client is not None:
        try:
            resp = (
                supabase_client.table("tradingview_alerts")
                .select("*")
                .eq("market", resolved)
                .order("received_at", desc=True)
                .limit(limit)
                .execute()
            )
            for row in resp.data or []:
                alerts.append(
                    LatestTradingViewAlert(
                        id=str(row.get("id")),
                        market=row.get("market") or resolved,
                        signal_type=row.get("signal_type") or "custom",
                        price=row.get("price"),
                        tf=row.get("tf"),
                        ticker=row.get("ticker"),
                        message=row.get("message"),
                        trend=row.get("trend"),
                        momentum=row.get("momentum"),
                        support=row.get("support"),
                        resistance=row.get("resistance"),
                        change_pct=row.get("change_pct"),
                        received_at=str(
                            row.get("received_at") or datetime.utcnow().isoformat()
                        ),
                    )
                )
        except Exception as e:
            logger.warning(f"Lecture alertes recentes Supabase echouee: {e}")

    return RecentTradingViewAlertsResponse(market=resolved, alerts=alerts)


@app.websocket("/api/v1/ws/notifications")
async def notifications_ws(
    websocket: WebSocket,
    token: str = Query(...),
    market: str = Query("ICE_NY"),
):
    """
    WebSocket live notifications.
    Connect: wss://api.../api/v1/ws/notifications?token=JWT&market=ICE_NY
    """
    try:
        payload = decode_token(token)
        _ = payload.get("sub")
    except HTTPException:
        await websocket.close(code=4401)
        return
    except Exception:
        await websocket.close(code=4401)
        return

    resolved = _resolve_market_key_for_alerts(market)
    await notification_hub.connect(websocket, resolved)
    try:
        await websocket.send_json(
            {"type": "connected", "market": resolved, "ts": datetime.utcnow().isoformat() + "Z"}
        )
        while True:
            # Keepalive / ignore client pings
            msg = await websocket.receive_text()
            if msg in ("ping", '{"type":"ping"}'):
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        await notification_hub.disconnect(websocket, resolved)
    except Exception:
        await notification_hub.disconnect(websocket, resolved)


@app.get(
    "/api/v1/notifications",
    response_model=NotificationsListResponse,
    responses={401: {"model": ErrorResponse}},
)
async def get_notifications(
    market: str = "ICE_NY",
    limit: int = 30,
    unread_only: bool = False,
    user: str = Depends(verify_token),
) -> NotificationsListResponse:
    resolved = _resolve_market_key_for_alerts(market)
    limit = max(1, min(int(limit), 100))
    rows = list_notifications(supabase_client, resolved, limit=limit, unread_only=unread_only)
    unread = sum(1 for r in rows if not r.get("is_read"))
    if not unread_only:
        # unread_count should reflect full unread, not just page
        all_unread = list_notifications(supabase_client, resolved, limit=100, unread_only=True)
        unread = len(all_unread)
    return NotificationsListResponse(
        market=resolved,
        notifications=[DashboardNotification(**r) for r in rows],
        unread_count=unread,
    )


@app.post(
    "/api/v1/notifications/{notification_id}/read",
    responses={401: {"model": ErrorResponse}},
)
async def read_notification(
    notification_id: str,
    user: str = Depends(verify_token),
):
    ok = mark_notification_read(supabase_client, notification_id)
    return {"ok": ok, "id": notification_id}


@app.post(
    "/api/v1/notifications/read-all",
    responses={401: {"model": ErrorResponse}},
)
async def read_all_notifications(
    market: str = "ICE_NY",
    user: str = Depends(verify_token),
):
    resolved = _resolve_market_key_for_alerts(market)
    updated = mark_all_notifications_read(supabase_client, resolved)
    return {"ok": True, "market": resolved, "updated": updated}


def _resolve_market_key_for_alerts(market: str) -> str:
    """Map ticker / alias to API market id used when persisting alerts."""
    try:
        return _resolve_market_from_alert(
            TradingViewAlert(
                secret="x",
                market=market,
                signal_type="custom",
            )
        )
    except HTTPException:
        return market.upper()


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict_price(
    request: PredictionRequest,
    user: str = Depends(verify_token)
) -> PredictionResponse:
    """
    Generate price predictions for specified horizons.
    
    This endpoint:
    1. Checks Redis cache for existing predictions
    2. If cache miss, generates new predictions using the hybrid model
    3. Stores predictions in cache with 1-hour TTL
    4. Returns predictions with confidence intervals
    
    Requirements: 10.1, 10.2, 10.3, 10.4
    """
    logger.info(
        f"Prediction request received: market={request.market}, "
        f"horizons={request.horizons}, include_sentiment={request.include_sentiment}"
    )
    
    # Resolve the requested market to its predictor
    market_cfg = resolve_api_market(request.market or "ICE_NY")
    if market_cfg is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown market '{request.market}'. Valid markets: {list_api_markets()}"
        )
    
    market_predictor = predictors.get(market_cfg.market_id)
    if market_predictor is None:
        logger.error(f"Predictor not initialized for market '{market_cfg.market_id}'")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction service is not available for market '{request.market}'"
        )
    
    # Check cache first
    if redis_cache:
        cached_prediction = redis_cache.get_prediction(
            market=request.market,
            horizons=request.horizons,
            include_sentiment=request.include_sentiment
        )
        
        if cached_prediction:
            latest_session = _latest_price_date(market_cfg.price_table)
            cached_day = str(cached_prediction.current_date or "")[:10]
            if latest_session and cached_day and cached_day != latest_session:
                logger.info("Cache ignoré: la dernière séance a changé.")
            else:
                logger.info("Returning cached prediction")
                from src.models.served_forecast import publish_or_close

                publish_or_close(
                    cached_prediction.predictions,
                    getattr(market_predictor, "active_release", None),
                    cached_prediction.current_price,
                )
                return cached_prediction
    
    try:
        # Fetch recent news for sentiment analysis (cocoa-specific news pipeline)
        recent_news = []
        if request.include_sentiment and market_cfg.market_id == "cocoa":
            try:
                # Query news from last 7 days (instead of 24 hours for better coverage)
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                response = (
                    supabase_client
                    .table("news_articles")
                    .select("*")
                    .gte("published_at", cutoff_time.isoformat())
                    .order("published_at", desc=True)
                    .limit(50)
                    .execute()
                )
                
                # Convert to NewsArticle objects
                for row in response.data:
                    article = NewsArticle(
                        id=row["id"],
                        source=row["source"],
                        title=row["title"],
                        content=row["content"],
                        published_at=datetime.fromisoformat(row["published_at"]),
                        url=row["url"],
                        keywords=row.get("keywords", []),
                        sentiment_score=row.get("sentiment_score"),
                        is_high_risk=row.get("is_high_risk")
                    )
                    recent_news.append(article)
                
                logger.info(f"Fetched {len(recent_news)} recent news articles")
            except Exception as e:
                logger.warning(f"Failed to fetch news articles: {e}")
                recent_news = []
        
        # Fetch real econometric features from Supabase (with fallback to defaults)
        exog_features = _fetch_exog_features(supabase_client, len(request.horizons))
        
        # Generate predictions
        predictions = market_predictor.predict(
            horizons=request.horizons,
            exog_features=exog_features,
            recent_news=recent_news
        )
        
        # Convert to response format
        prediction_items = []
        for pred in predictions:
            item = PredictionItem(
                horizon=pred.horizon,
                price=pred.price,
                confidence_interval=[
                    pred.confidence_interval[0],
                    pred.confidence_interval[1]
                ],
                confidence_level=pred.confidence_level,
                timestamp=pred.timestamp,
                components=pred.components if hasattr(pred, 'components') and pred.components else None
            )
            prediction_items.append(item)
        
        # Calculate aggregated sentiment score
        sentiment_score = None
        if request.include_sentiment and recent_news:
            try:
                sentiment_score = market_predictor.nlp_analyzer.aggregate_sentiment(
                    recent_news
                )
            except Exception as e:
                logger.warning(f"Failed to aggregate sentiment: {e}")
        
        # Fetch current price and historical data
        current_price = None
        current_date = None
        historical_prices = None
        try:
            hist_response = (
                supabase_client.table(market_cfg.price_table)
                .select("date,price")
                .order("date", desc=True)
                .limit(60)
                .execute()
            )
            if hist_response.data:
                current_price = hist_response.data[0]["price"]
                current_date = hist_response.data[0]["date"]
                historical_prices = [
                    {"date": row["date"], "price": row["price"]}
                    for row in reversed(hist_response.data)
                ]
        except Exception as e:
            logger.warning(f"Failed to fetch historical prices: {e}")

        snapshot = getattr(market_predictor, "last_snapshot", None) or {}
        if snapshot.get("origin_date"):
            current_date = snapshot["origin_date"]
            current_price = snapshot.get("origin_price", current_price)

        from src.models.served_forecast import publish_or_close

        publish_or_close(
            prediction_items,
            getattr(market_predictor, "active_release", None),
            current_price,
        )

        # Create response
        response = PredictionResponse(
            predictions=prediction_items,
            model_version=market_predictor.model_version,
            sentiment_score=sentiment_score,
            market=request.market,
            current_price=current_price,
            current_date=current_date,
            historical_prices=historical_prices
        )
        
        # Cache the response
        if redis_cache:
            redis_cache.set_prediction(
                market=request.market,
                horizons=request.horizons,
                include_sentiment=request.include_sentiment,
                prediction_response=response
            )
        
        # Log prediction to database
        try:
            from src.models.served_forecast import snapshot_journal_fields

            currency = "GBP" if "GBP" in (market_cfg.unit or "") else "USD"
            for pred in prediction_items:
                parts = pred.components or {}
                journal = snapshot_journal_fields(parts, {"price": current_price, "date": current_date})
                origin_date = journal.get("origin_date")
                origin_price = journal.get("origin_price")
                target_date = journal.get("target_date")
                candidate_price = parts.get("candidate_price", pred.price)
                supabase_client.table("predictions").insert({
                    "horizon": pred.horizon,
                    "predicted_price": pred.price,
                    "lower_bound": pred.confidence_interval[0],
                    "upper_bound": pred.confidence_interval[1],
                    "confidence_level": pred.confidence_level,
                    "model_version": market_predictor.model_version,
                    "baseline_component": parts.get("baseline"),
                    "residual_component": parts.get("residual"),
                    "sentiment_component": parts.get("sentiment"),
                    "created_at": pred.timestamp.isoformat(),
                    "market": request.market,
                    "origin_date": origin_date,
                    "origin_price": float(origin_price) if origin_price else None,
                    "target_date": target_date,
                    "feature_failure": bool(parts.get("feature_failure")),
                    "currency": currency,
                    "candidate_price": float(candidate_price) if candidate_price is not None else None,
                }).execute()
        except Exception as e:
            _mark_journal_unavailable(e)
        
        logger.info(f"Successfully generated {len(predictions)} predictions")
        return response
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate predictions: {str(e)}"
        )


@app.get(
    "/api/v1/performance",
    response_model=PerformanceResponse,
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_performance_metrics(
    start_date: datetime,
    end_date: datetime,
    model_version: Optional[str] = None,
    market: Optional[str] = None,
    horizon: Optional[int] = None,
    user: str = Depends(verify_token)
) -> PerformanceResponse:
    """
    Retrieve model performance metrics for a date range.
    
    Requirements: 10.2
    """
    logger.info(
        f"Performance metrics request: start_date={start_date}, "
        f"end_date={end_date}, model_version={model_version}"
    )
    
    # Validate date range
    if start_date >= end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date must be before end_date"
        )
    
    try:
        # Query metrics from database
        query = (
            supabase_client
            .table("model_metrics")
            .select("*")
            .gte("created_at", start_date.isoformat())
            .lte("created_at", end_date.isoformat())
            .order("created_at", desc=True)
        )
        
        # Filter by model version if specified
        if model_version:
            query = query.eq("model_version", model_version)
        if market:
            query = query.eq("market", market)
        if horizon is not None:
            query = query.eq("horizon", horizon)
        
        response = query.execute()
        
        if not response.data:
            logger.warning("No performance metrics found for the specified period")
            return PerformanceResponse(
                model_version=model_version or "unknown",
                metrics=[],
                start_date=start_date,
                end_date=end_date,
                market=market,
                horizon=horizon,
            )
        
        # Convert to response format
        metrics_items = []
        for row in response.data:
            item = PerformanceMetricsItem(
                timestamp=datetime.fromisoformat(row["created_at"]),
                rmse=float(row["rmse"]),
                mae=float(row["mae"]),
                mape=float(row["mape"]),
                directional_accuracy=float(row["directional_accuracy"]),
                coverage_rate=float(row["coverage_rate"]),
                mean_interval_width=float(row["mean_interval_width"]),
                market=row.get("market"),
                horizon=row.get("horizon"),
            )
            metrics_items.append(item)
        
        # Get model version from first result if not specified
        if not model_version:
            model_version = response.data[0]["model_version"]
        
        logger.info(f"Retrieved {len(metrics_items)} performance metrics")
        
        return PerformanceResponse(
            model_version=model_version,
            metrics=metrics_items,
            start_date=start_date,
            end_date=end_date,
            market=market,
            horizon=horizon,
        )
        
    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance metrics: {str(e)}"
        )


@app.get(
    "/api/v1/validation/metrics",
    response_model=ValidationMetricsResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def get_validation_metrics(
    market: Optional[str] = None,
    user: str = Depends(verify_token),
) -> ValidationMetricsResponse:
    """Return walk-forward metrics for the report named by the active release."""
    from src.validation.report_loader import load_release_summary

    if market:
        market_cfg = resolve_api_market(market)
        if market_cfg is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown market '{market}'. Valid markets: {list_api_markets()}",
            )
        reports_dir = (
            "reports/walk_forward"
            if market_cfg.market_id == "cocoa"
            else f"reports/walk_forward/{market_cfg.market_id}"
        )
    else:
        reports_dir = "reports/walk_forward"

    from src.models.release_manifest import horizon_is_validated, load_active_release

    release = None
    release_version = None
    market_id = market_cfg.market_id if market else "cocoa"
    try:
        release = load_active_release(market_id)
        release_version = release.get("version")
    except (FileNotFoundError, ValueError):
        release = None

    summary = load_release_summary(reports_dir, release)
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Le manifeste ne nomme aucun rapport de validation présent.",
        )

    wf = summary.get("walk_forward", {})
    components = wf.get("summary_by_component", {})
    component_name = "published_pred" if "published_pred" in components else "xgb_pred"
    block = components.get(component_name, {})
    period = wf.get("evaluated_period") or {}

    report_version = (summary.get("promotion") or {}).get("candidate_version")
    release_matches = bool(
        release
        and report_version
        and report_version == release_version
    )
    decisions = ((summary.get("promotion") or {}).get("decisions")) or {}

    metrics = []
    for h in wf.get("horizons", []):
        h_data = block.get(str(h), block.get(h, {}))
        decision = decisions.get(str(h)) or {}
        gap = decision.get("gap_ci") or [None, None]
        validated = bool(
            release
            and horizon_is_validated(release, int(h))
            and release_matches
            and decision.get("validated")
        )
        metrics.append(
            HorizonValidationMetrics(
                horizon=int(h),
                mape=h_data.get("mape"),
                rmse=h_data.get("rmse"),
                mae=h_data.get("mae"),
                directional_accuracy=h_data.get("directional_accuracy"),
                n_predictions=h_data.get("n_predictions"),
                validated=validated,
                relative_gain=decision.get("relative_gain"),
                gap_ci_low=gap[0] if gap else None,
                gap_ci_high=gap[1] if len(gap) > 1 else None,
                n_effective=decision.get("n_effective"),
                fallback_rate=decision.get("fallback_rate"),
                measurement="procedure" if decision else None,
                reason=(
                    decision.get("reason")
                    if release_matches
                    else "Le rapport ne correspond pas au manifeste chargé."
                ),
            )
        )

    legacy = summary.get("legacy_holdout_baseline", {}) or {}

    return ValidationMetricsResponse(
        report_timestamp=summary.get("timestamp"),
        report_path=summary.get("_report_path"),
        validation_type=summary.get("validation_type", "walk_forward_multi_horizon"),
        n_origins=wf.get("n_origins"),
        origin_start=period.get("origin_start"),
        origin_end=period.get("origin_end"),
        target_start=period.get("target_start"),
        target_end=period.get("target_end"),
        evaluated_component=component_name,
        horizons=wf.get("horizons", []),
        xgb_metrics=metrics,
        legacy_holdout_mape_1step=legacy.get("mape_1step_holdout"),
        ensemble_calibration=summary.get("ensemble_calibration"),
        conformal_intervals=summary.get("conformal_intervals"),
        release_version=release_version,
        release_matches_report=release_matches,
    )


@app.get("/api/v1/prediction-history")
async def get_prediction_history(
    limit: int = 20,
    horizon: Optional[int] = None,
    market: Optional[str] = None,
    token_payload: dict = Depends(verify_token)
):
    """Return recent prediction history for one market."""
    try:
        query = supabase_client.table("predictions").select(
            "created_at,horizon,predicted_price,lower_bound,upper_bound,model_version,"
            "market,origin_date,origin_price,target_date,feature_failure,currency"
        ).order("created_at", desc=True)

        if market:
            query = query.eq("market", market)
        if horizon:
            query = query.eq("horizon", horizon)

        query = query.limit(limit)
        result = query.execute()

        return {"predictions": result.data, "count": len(result.data)}
    except Exception as e:
        logger.error(f"Failed to retrieve prediction history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve prediction history: {str(e)}"
        )


def _enrich_named_futures(
    raw_contracts: List[dict],
    spot_pct: dict,
    include_predictions: bool,
) -> List[dict]:
    """Mois nommes + predictions (XGBoost dedie si dispo, sinon spot_shift)."""
    if include_predictions:
        # Londres DEC26… : modeles par echeance
        is_london_named = any(
            str(c.get("currency") or "").upper() == "GBP"
            or "london_named" in str(c.get("source") or "")
            or "FM" in str(c.get("symbol") or "")
            for c in raw_contracts
        )
        if is_london_named and futures_named_predictor is not None:
            try:
                return futures_named_predictor.predict_london_named_curve(
                    raw_contracts, spot_pct_by_horizon=spot_pct or None
                )
            except Exception as exc:
                logger.warning(f"london named predict failed: {exc}")

        if futures_curve_predictor is not None:
            try:
                # Predictor Londres C.v.* ignore les symboles NY -> spot_shift via investing
                if getattr(futures_curve_predictor, "source", "") != "london":
                    return futures_curve_predictor.predict_curve(
                        raw_contracts, spot_pct_by_horizon=spot_pct or None
                    )
            except Exception as exc:
                logger.warning(f"named futures predict failed: {exc}")

    enriched: List[dict] = []
    for c in raw_contracts:
        price = float(c.get("price_usd") or c.get("price") or 0)
        preds = []
        if include_predictions and price > 0:
            for h, pct in spot_pct.items():
                p_price = round(price * (1.0 + pct), 2)
                preds.append(
                    {
                        "horizon": h,
                        "price": p_price,
                        "method": "spot_shift",
                        "change_pct": round(pct * 100, 2),
                    }
                )
        enriched.append(
            {
                "contract": c.get("contract") or c.get("symbol"),
                "symbol": c.get("symbol"),
                "yahoo_symbol": c.get("yahoo_symbol"),
                "price_usd": price,
                "change": c.get("change"),
                "volume": c.get("volume"),
                "predictions": preds,
            }
        )
    return enriched


def _to_futures_items(enriched: List[dict]) -> List[FuturesContractItem]:
    return [
        FuturesContractItem(
            contract=str(c.get("contract") or c.get("symbol") or ""),
            symbol=str(c.get("symbol") or ""),
            yahoo_symbol=c.get("yahoo_symbol"),
            price_usd=float(c.get("price_usd") or 0),
            change=c.get("change"),
            volume=c.get("volume"),
            predictions=[
                FuturesHorizonPrediction(**p) for p in (c.get("predictions") or [])
            ],
        )
        for c in enriched
    ]


@app.get("/api/v1/futures", response_model=FuturesCurveResponse)
async def get_futures(
    include_predictions: bool = True,
    token_payload: dict = Depends(verify_token),
):
    """Courbe Londres £ (C.v.*) + courbe Investing mois nommes (Dec26, Mar27...)."""
    try:
        spot_pct: dict = {}
        model_version = None
        if include_predictions and price_predictor is not None:
            try:
                current = None
                _cocoa_cfg = get_market_config("cocoa")
                hist = (
                    supabase_client.table(_cocoa_cfg.price_table)
                    .select("price")
                    .order("date", desc=True)
                    .limit(1)
                    .execute()
                )
                if hist.data:
                    current = float(hist.data[0]["price"])
                preds = price_predictor.predict(
                    horizons=[1, 7, 14, 30],
                    recent_news=[],
                )
                model_version = getattr(price_predictor, "model_version", None)
                if current and current > 0:
                    for p in preds:
                        spot_pct[int(p.horizon)] = (float(p.price) / current) - 1.0
            except Exception as e:
                logger.warning(f"Spot pct for futures curve unavailable: {e}")

        # --- Mois nommes : preferer ICE London GBP (DEC26...), sinon Investing/Yahoo NY ---
        named_items: List[FuturesContractItem] = []
        named_collected_at = None
        named_source = None
        named_currency = "USD"
        named_unit = "USD/MT"
        try:
            inv = (
                supabase_client.table("cocoa_futures")
                .select("data,collected_at,source")
                .order("collected_at", desc=True)
                .limit(20)
                .execute()
            )
            chosen = None
            # 1) Snapshot Londres nomme le plus recent
            for row in inv.data or []:
                src = str(row.get("source") or "")
                if "london_named" in src or src in (
                    "databento_london_named",
                    "ice_london_named",
                ):
                    chosen = row
                    break
            # 2) Sinon dernier snapshot (Investing / Yahoo NY)
            if chosen is None and inv.data:
                chosen = inv.data[0]

            if chosen:
                named_collected_at = chosen.get("collected_at")
                named_source = chosen.get("source", "investing_com")
                raw = chosen.get("data") or []
                raw = [
                    c for c in raw
                    if str(c.get("symbol") or "").upper() not in ("CCY00", "CASH")
                    and "cash" not in str(c.get("contract") or "").lower()
                ]
                is_gbp = (
                    "london_named" in str(named_source)
                    or any(str(c.get("currency") or "").upper() == "GBP" for c in raw)
                )
                if is_gbp:
                    named_currency = "GBP"
                    named_unit = "GBP/MT"
                named_items = _to_futures_items(
                    _enrich_named_futures(raw, spot_pct, include_predictions)
                )
        except Exception as exc:
            logger.warning(f"named futures snapshot unavailable: {exc}")

        # --- ICE London continuous C.v.0..3 (GBP) ---
        london_contracts: List[dict] = []
        term_date = None
        try:
            latest_term = (
                supabase_client.table("cocoa_london_contracts")
                .select("date")
                .order("date", desc=True)
                .limit(1)
                .execute()
            )
            if latest_term.data:
                term_date = str(latest_term.data[0]["date"])[:10]
                contracts_resp = (
                    supabase_client.table("cocoa_london_contracts")
                    .select("date, contract_rank, symbol, close, volume, open_interest")
                    .eq("date", term_date)
                    .order("contract_rank")
                    .execute()
                )
                for c in contracts_resp.data or []:
                    rank = int(c["contract_rank"])
                    london_contracts.append(
                        {
                            "contract_rank": rank,
                            "symbol": str(c.get("symbol") or f"C.v.{rank}"),
                            "close": float(c["close"]),
                            "volume": c.get("volume"),
                            "label": _RANK_LABELS.get(rank, f"Rank {rank}"),
                        }
                    )
        except Exception as exc:
            logger.warning(f"london futures snapshot unavailable: {exc}")

        london_items: List[FuturesContractItem] = []
        if london_contracts:
            history_by_rank: dict = {}
            if include_predictions and futures_curve_predictor is not None:
                try:
                    from src.models.futures_curve_predictor import fetch_london_rank_history

                    for c in london_contracts:
                        rank = int(c["contract_rank"])
                        history_by_rank[rank] = fetch_london_rank_history(
                            supabase_client, rank
                        )
                    enriched = futures_curve_predictor.predict_london_curve(
                        london_contracts,
                        spot_pct_by_horizon=spot_pct or None,
                        history_by_rank=history_by_rank,
                    )
                except Exception as exc:
                    logger.warning(f"London futures predict failed: {exc}")
                    enriched = _enrich_named_futures(
                        [
                            {
                                "contract": c["label"],
                                "symbol": c["symbol"],
                                "price_usd": c["close"],
                                "volume": c.get("volume"),
                            }
                            for c in london_contracts
                        ],
                        spot_pct,
                        include_predictions,
                    )
            else:
                enriched = _enrich_named_futures(
                    [
                        {
                            "contract": c["label"],
                            "symbol": c["symbol"],
                            "price_usd": c["close"],
                            "volume": c.get("volume"),
                        }
                        for c in london_contracts
                    ],
                    spot_pct,
                    include_predictions,
                )
            london_items = _to_futures_items(enriched)

        # contracts = mois nommes (Dec26...) pour le panneau principal ;
        # london_contracts = C.v.0..3 GBP.
        if named_items:
            return FuturesCurveResponse(
                contracts=named_items,
                collected_at=named_collected_at,
                source=named_source or "investing_com",
                unit=named_unit,
                currency=named_currency,
                model_version=model_version
                or (
                    futures_curve_predictor._meta.get("trained_at")
                    if futures_curve_predictor
                    else None
                ),
                spot_pct_by_horizon={str(k): round(v, 6) for k, v in spot_pct.items()} or None,
                london_contracts=london_items,
                london_collected_at=term_date,
                london_source="databento_london" if london_items else None,
                london_unit="GBP/MT",
                london_currency="GBP",
            )

        if london_items:
            return FuturesCurveResponse(
                contracts=london_items,
                collected_at=term_date,
                source="databento_london",
                unit="GBP/MT",
                currency="GBP",
                model_version=model_version
                or (
                    futures_curve_predictor._meta.get("trained_at")
                    if futures_curve_predictor
                    else None
                ),
                spot_pct_by_horizon={str(k): round(v, 6) for k, v in spot_pct.items()} or None,
            )

        return FuturesCurveResponse(contracts=[], collected_at=None)
    except Exception as e:
        logger.error(f"Failed to retrieve futures: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve futures: {str(e)}"
        )


_RANK_LABELS = {
    0: "Front (C.v.0)",
    1: "2e echeance",
    2: "3e echeance",
    3: "4e echeance",
}

_MODEL_LABELS = {
    "M1_Prophet_Close": "M1 Prophet (Close)",
    "M2_XGB_OHLCV": "M2 XGB (OHLCV)",
    "M3_XGB_OHLCV_OI": "M3 XGB (+ OI)",
    "M4_XGB_Full": "M4 Complet",
}


@app.get("/api/v1/london-market", response_model=LondonMarketResponse)
async def get_london_market(
    history_days: int = Query(60, ge=7, le=365),
    token_payload: dict = Depends(verify_token),
):
    """Microstructure cacao ICE London (Databento) : volume, OI, courbe d'echeances."""
    try:
        market = get_market_config("cocoa")
        hist_resp = (
            supabase_client.table(market.price_table)
            .select("date, price, volume, open_interest, source")
            .order("date", desc=True)
            .limit(history_days)
            .execute()
        )
        rows = list(reversed(hist_resp.data or []))
        history = [
            LondonHistoryPoint(
                date=str(r["date"])[:10],
                price=float(r["price"]),
                volume=float(r["volume"]) if r.get("volume") is not None else None,
                open_interest=float(r["open_interest"]) if r.get("open_interest") is not None else None,
                source=r.get("source"),
            )
            for r in rows
        ]
        # Preferer la derniere barre avec microstructure (Databento) pour les KPI volume/OI
        latest = history[-1] if history else None
        for point in reversed(history):
            if point.volume is not None or point.open_interest is not None:
                latest = point
                break

        term: List[LondonTermContract] = []
        term_date = None
        try:
            latest_term = (
                supabase_client.table("cocoa_london_contracts")
                .select("date")
                .order("date", desc=True)
                .limit(1)
                .execute()
            )
            if latest_term.data:
                term_date = str(latest_term.data[0]["date"])[:10]
                contracts = (
                    supabase_client.table("cocoa_london_contracts")
                    .select("date, contract_rank, symbol, close, volume, open_interest")
                    .eq("date", term_date)
                    .order("contract_rank")
                    .execute()
                )
                for c in contracts.data or []:
                    rank = int(c["contract_rank"])
                    term.append(
                        LondonTermContract(
                            contract_rank=rank,
                            symbol=str(c.get("symbol") or f"C.v.{rank}"),
                            close=float(c["close"]),
                            volume=float(c["volume"]) if c.get("volume") is not None else None,
                            open_interest=(
                                float(c["open_interest"])
                                if c.get("open_interest") is not None
                                else None
                            ),
                            label=_RANK_LABELS.get(rank, f"Rank {rank}"),
                        )
                    )
        except Exception as exc:
            logger.warning(f"london contracts unavailable: {exc}")

        return LondonMarketResponse(
            latest=latest,
            history=history,
            term_structure=term,
            term_date=term_date,
            unit=market.unit,
            source=(latest.source if latest else "databento"),
        )
    except Exception as e:
        logger.error(f"Failed to retrieve london market: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve london market: {str(e)}",
        )


@app.get("/api/v1/model-comparison", response_model=ModelComparisonResponse)
async def get_model_comparison(
    token_payload: dict = Depends(verify_token),
):
    """Resultats de l'etude comparative M1-M4 (memoire) — fichier config/reports."""
    try:
        from pathlib import Path
        import json as _json

        candidates = [
            Path("config/model_comparison_latest.json"),
            Path(__file__).resolve().parents[2] / "config" / "model_comparison_latest.json",
        ]
        # Aussi le plus recent dans reports/
        reports_dir = Path(__file__).resolve().parents[2] / "reports"
        if reports_dir.is_dir():
            candidates.extend(
                sorted(reports_dir.glob("model_comparison_*.json"), reverse=True)[:1]
            )

        payload = None
        for path in candidates:
            if path.is_file():
                payload = _json.loads(path.read_text(encoding="utf-8"))
                break

        if not payload:
            return ModelComparisonResponse(
                metrics=[],
                note="Aucun rapport de comparaison trouve. Lancer run_model_comparison.py",
            )

        metrics: List[ModelComparisonMetric] = []
        for model_key, by_h in (payload.get("models") or {}).items():
            for h, m in by_h.items():
                metrics.append(
                    ModelComparisonMetric(
                        model=model_key,
                        label=_MODEL_LABELS.get(model_key, model_key),
                        horizon=int(h),
                        mae=float(m.get("mae") or 0),
                        rmse=float(m.get("rmse") or 0),
                        mape=float(m.get("mape") or 0),
                        n=int(m.get("n") or 0),
                    )
                )

        return ModelComparisonResponse(
            generated_at=payload.get("generated_at"),
            split_date=payload.get("split_date"),
            n_train=payload.get("n_train"),
            n_test=payload.get("n_test"),
            metrics=metrics,
            note="M1=Prophet(Close) · M2=XGB(OHLCV) · M3=XGB(+OI) · M4=Complet",
        )
    except Exception as e:
        logger.error(f"Failed to retrieve model comparison: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model comparison: {str(e)}",
        )


@app.get(
    "/api/v1/models",
    response_model=ModelsResponse,
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def list_models(
    user: str = Depends(verify_token)
) -> ModelsResponse:
    """
    List available model versions and their status.
    
    Requirements: 10.3
    """
    logger.info("Models list request received")
    
    try:
        # Get model versions from MLflow
        model_names = ["cocoa_prophet", "cocoa_xgboost", "cocoa_finbert"]
        all_models = []
        current_production_version = None
        
        for model_name in model_names:
            try:
                versions = model_manager.list_model_versions(
                    model_name=model_name,
                    max_results=5
                )
                
                for version in versions:
                    # Get model info
                    info = model_manager.get_model_info(
                        model_name=model_name,
                        version=version.version
                    )
                    
                    model_info = ModelInfo(
                        name=model_name,
                        version=version.version,
                        stage=version.current_stage,
                        created_at=datetime.fromtimestamp(
                            version.creation_timestamp / 1000
                        ),
                        metrics=info.get("metrics")
                    )
                    
                    all_models.append(model_info)
                    
                    # Track production version
                    if version.current_stage == "Production":
                        current_production_version = f"{model_name}:{version.version}"
                
            except Exception as e:
                logger.warning(f"Failed to get versions for {model_name}: {e}")
                continue
        
        logger.info(f"Retrieved {len(all_models)} model versions")
        
        return ModelsResponse(
            models=all_models,
            current_production_version=current_production_version
        )
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@app.post(
    "/api/v1/retrain",
    response_model=RetrainingResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def trigger_retraining(
    request: RetrainingRequest,
    admin_user: str = Depends(verify_admin_token)
) -> RetrainingResponse:
    """
    Trigger manual model retraining (admin only).
    
    This endpoint creates a retraining job that will:
    1. Fetch latest training data
    2. Retrain specified models
    3. Validate new models
    4. Promote to staging if validation passes
    
    Requirements: 10.5, 13.1
    """
    logger.info(
        f"Retraining request received: model_type={request.model_type}, "
        f"reason={request.reason}, admin_user={admin_user}"
    )
    
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # In production, this would trigger an async retraining job
        # For now, we'll just log the request and return a response
        logger.info(
            f"Retraining job {job_id} created for model_type={request.model_type}"
        )
        
        # Estimate completion time (e.g., 2 hours from now)
        estimated_completion = datetime.utcnow() + timedelta(hours=2)
        
        # Invalidate prediction cache since models will be updated
        if redis_cache:
            invalidated = redis_cache.invalidate_all_predictions()
            logger.info(f"Invalidated {invalidated} cached predictions")
        
        return RetrainingResponse(
            status="accepted",
            message=f"Retraining job created for {request.model_type} model(s)",
            job_id=job_id,
            estimated_completion=estimated_completion
        )
        
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger retraining: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """
    Handle HTTP exceptions with consistent error response format.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.status_code,
            "message": exc.detail,
            "detail": None
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """
    Handle unexpected exceptions.
    """
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "detail": str(exc) if app.debug else None
        }
    )
