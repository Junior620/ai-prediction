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
from fastapi import FastAPI, Depends, HTTPException, status, Request
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
)
from src.api.auth import verify_token, verify_admin_token
from src.api.cache import RedisCache
from src.models.price_predictor import PricePredictor
from src.models.improved_price_predictor import ImprovedPricePredictor
from src.models.direct_horizon_trainer import DirectHorizonTrainer
from src.models.time_series_model import TimeSeriesModel
from src.models.ml_model import MLModel
from src.nlp.nlp_analyzer import NLPAnalyzer
from src.models.model_manager import ModelManager
from src.monitoring.performance_monitor import PerformanceMonitor
from src.monitoring.alert_system import get_alert_system, AlertSeverity, AlertType
from src.models.data_models import NewsArticle
from src.models.market_registry import (
    MarketConfig,
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
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
    global redis_cache, supabase_client, model_manager, performance_monitor, price_predictor, brief_service
    
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

    brief_service = BriefService(redis_cache=redis_cache)
    logger.info("BriefService (Claude) initialized")

    logger.info("Cocoa Price Prediction API started successfully")


def _load_market_predictor(
    market_cfg: MarketConfig,
    settings,
    nlp_analyzer,
    pred_cfg: dict,
) -> Optional[ImprovedPricePredictor]:
    """Auto-discover and load the latest models for one market."""
    import pickle
    from pathlib import Path

    model_dir = Path(market_cfg.models_dir)
    prophet_files = sorted(model_dir.glob("prophet_improved_*.pkl"))
    xgboost_files = sorted(model_dir.glob("xgboost_improved_*.pkl"))

    if not prophet_files or not xgboost_files:
        logger.info(
            f"ℹ️ No trained models for market '{market_cfg.market_id}' in {model_dir}/ — skipping"
        )
        return None

    prophet_path = prophet_files[-1]  # Most recent by filename (timestamp-sorted)
    xgboost_path = xgboost_files[-1]
    model_version = prophet_path.stem.replace("prophet_", "")

    logger.info(
        f"[{market_cfg.market_id}] Models: Prophet={prophet_path.name}, XGBoost={xgboost_path.name}"
    )

    with open(prophet_path, "rb") as f:
        prophet_model = pickle.load(f)
    with open(xgboost_path, "rb") as f:
        xgboost_model = pickle.load(f)

    # Optional 3rd engine: N-HiTS
    nhits_model = None
    try:
        nhits_dirs = [d for d in sorted(model_dir.glob("nhits_*")) if d.is_dir()]
        if nhits_dirs:
            from neuralforecast import NeuralForecast
            nhits_model = NeuralForecast.load(path=str(nhits_dirs[-1]))
            logger.info(f"[{market_cfg.market_id}] ✅ N-HiTS loaded from {nhits_dirs[-1]}")
        else:
            logger.info(f"[{market_cfg.market_id}] No N-HiTS model — 2 engines")
    except Exception as e:
        logger.warning(f"[{market_cfg.market_id}] Failed to load N-HiTS: {e} — 2 engines")
        nhits_model = None

    direct_models = DirectHorizonTrainer.load_latest(str(model_dir))

    predictor = ImprovedPricePredictor(
        prophet_model=prophet_model,
        xgboost_model=xgboost_model,
        nlp_analyzer=nlp_analyzer,
        sentiment_weight=pred_cfg.get("sentiment_weight", 0.05),
        model_version=model_version,
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_key,
        nhits_model=nhits_model,
        ensemble_weights_file=market_cfg.ensemble_weights_file,
        ensemble_fallback=pred_cfg.get("ensemble_fallback"),
        multi_step_mode=pred_cfg.get("multi_step_mode", "recursive"),
        direct_horizon_models=direct_models,
        conformal_intervals_file=market_cfg.conformal_intervals_file,
        confidence_level=pred_cfg.get("confidence_level", 0.90),
        price_bounds=market_cfg.price_bounds,
        price_table=market_cfg.price_table,
        nhits_unique_id=market_cfg.nhits_unique_id,
        garch_enabled=market_cfg.garch_enabled,
        models_dir=str(model_dir),
    )
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
        "markets_loaded": sorted(predictors.keys())
    }
    
    # Determine overall health
    all_healthy = all(health_status["services"].values())
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
            logger.info("Returning cached prediction")
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
                .limit(14)
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
            for pred in predictions:
                supabase_client.table("predictions").insert({
                    "horizon": pred.horizon,
                    "predicted_price": pred.price,
                    "lower_bound": pred.confidence_interval[0],
                    "upper_bound": pred.confidence_interval[1],
                    "confidence_level": pred.confidence_level,
                    "model_version": pred.model_version,
                    "baseline_component": pred.components.get("baseline"),
                    "residual_component": pred.components.get("residual"),
                    "sentiment_component": pred.components.get("sentiment"),
                    "created_at": pred.timestamp.isoformat()
                }).execute()
        except Exception as e:
            logger.error(f"Failed to log prediction to database: {e}")
        
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
        
        response = query.execute()
        
        if not response.data:
            logger.warning("No performance metrics found for the specified period")
            return PerformanceResponse(
                model_version=model_version or "unknown",
                metrics=[],
                start_date=start_date,
                end_date=end_date
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
                mean_interval_width=float(row["mean_interval_width"])
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
            end_date=end_date
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
async def get_validation_metrics(user: str = Depends(verify_token)) -> ValidationMetricsResponse:
    """Return honest walk-forward validation metrics from the latest backtest report."""
    from src.validation.report_loader import load_latest_summary, extract_walk_forward_reference

    summary = load_latest_summary("reports/walk_forward")
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No walk-forward validation report found. Run scripts/run_walk_forward_validation.py",
        )

    ref = extract_walk_forward_reference(summary) or {}
    wf = summary.get("walk_forward", {})
    xgb = wf.get("summary_by_component", {}).get("xgb_pred", {})

    metrics = []
    for h in wf.get("horizons", []):
        h_data = xgb.get(str(h), xgb.get(h, {}))
        metrics.append(
            HorizonValidationMetrics(
                horizon=int(h),
                mape=h_data.get("mape"),
                rmse=h_data.get("rmse"),
                mae=h_data.get("mae"),
                directional_accuracy=h_data.get("directional_accuracy"),
                n_predictions=h_data.get("n_predictions"),
            )
        )

    legacy = summary.get("legacy_holdout_baseline", {})

    return ValidationMetricsResponse(
        report_timestamp=summary.get("timestamp"),
        report_path=summary.get("_report_path"),
        validation_type=summary.get("validation_type", "walk_forward_multi_horizon"),
        n_origins=wf.get("n_origins"),
        horizons=wf.get("horizons", []),
        xgb_metrics=metrics,
        legacy_holdout_mape_1step=legacy.get("mape_1step_holdout"),
        ensemble_calibration=summary.get("ensemble_calibration"),
        conformal_intervals=summary.get("conformal_intervals"),
    )


@app.get("/api/v1/prediction-history")
async def get_prediction_history(
    limit: int = 20,
    horizon: Optional[int] = None,
    token_payload: dict = Depends(verify_token)
):
    """Return recent prediction history from the database."""
    try:
        query = supabase_client.table("predictions").select(
            "created_at,horizon,predicted_price,lower_bound,upper_bound,model_version"
        ).order("created_at", desc=True)

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


@app.get("/api/v1/futures")
async def get_futures(
    token_payload: dict = Depends(verify_token)
):
    """Return the latest cocoa futures contracts from the database."""
    try:
        result = supabase_client.table("cocoa_futures").select(
            "data,collected_at,source"
        ).order("collected_at", desc=True).limit(1).execute()

        if not result.data:
            return {"contracts": [], "collected_at": None}

        row = result.data[0]
        return {
            "contracts": row["data"],
            "collected_at": row["collected_at"],
            "source": row.get("source", "investing.com")
        }
    except Exception as e:
        logger.error(f"Failed to retrieve futures: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve futures: {str(e)}"
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
