"""
Improved Price Predictor for hybrid cocoa price forecasting.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datetime import datetime
from loguru import logger
from supabase import create_client
import os

from src.models.data_models import Prediction, NewsArticle
from src.models.direct_horizon_trainer import DirectHorizonTrainer
from src.models.ensemble_weights import (
    DEFAULT_FALLBACK,
    load_ensemble_weights,
)
from src.models.horizon_schedule import scheduled_cap, scheduled_weights
from src.models.hybrid_features import (
    FEATURE_COLS,
    build_prediction_row,
    future_business_date,
    prepare_training_frame,
    resolve_feature_cols,
)
from src.models.conformal_intervals import load_conformal_margins
from src.models.served_forecast import direct_level, publish_level


def historical_cache_is_fresh(
    age_seconds: float,
    cached_fingerprint: Optional[str],
    remote_fingerprint: Optional[str],
) -> bool:
    """A same-day price correction is a different snapshot, even inside the hour."""
    if age_seconds >= 3600:
        return False
    if not cached_fingerprint or not remote_fingerprint:
        return False
    return cached_fingerprint == remote_fingerprint


class ImprovedPricePredictor:
    """Hybrid predictor: XGBoost primary, Prophet features, optional N-HiTS ensemble."""

    def __init__(
        self,
        prophet_model,
        xgboost_model,
        nlp_analyzer,
        sentiment_weight: float = 0.05,
        model_version: str = "improved_1.0.0",
        supabase_url: str = None,
        supabase_key: str = None,
        nhits_model=None,
        ensemble_weights_file: str = "config/ensemble_weights.json",
        ensemble_fallback: Optional[Dict[str, float]] = None,
        multi_step_mode: str = "recursive",
        direct_horizon_models: Optional[Dict[int, object]] = None,
        conformal_intervals_file: str = "config/conformal_intervals.json",
        confidence_level: float = 0.90,
        price_bounds: Optional[Tuple[float, float]] = None,
        price_table: str = "cocoa_prices",
        nhits_unique_id: str = "cocoa_ice_ny",
        garch_enabled: bool = False,
        models_dir: str = "models",
        max_abs_change_pct: Optional[Dict[str, float]] = None,
        recent_range_days: int = 252,
        recent_range_padding_pct: float = 15.0,
        feature_cols: Optional[List[str]] = None,
        feature_set: str = "baseline",
        direct_info_path: Optional[str] = None,
    ):
        self.prophet_model = prophet_model
        self.xgboost_model = xgboost_model
        self.nlp_analyzer = nlp_analyzer
        self.sentiment_weight = sentiment_weight
        self.model_version = model_version
        self.nhits_model = nhits_model
        self.ensemble_fallback = ensemble_fallback or dict(DEFAULT_FALLBACK)
        self.multi_step_mode = multi_step_mode
        self.direct_horizon_models = direct_horizon_models or {}
        self.confidence_level = confidence_level
        self.price_bounds = price_bounds or (1000.0, 10000.0)
        self.price_table = price_table
        self.nhits_unique_id = nhits_unique_id
        self.garch_enabled = garch_enabled
        self.models_dir = models_dir
        self.max_abs_change_pct = max_abs_change_pct or {
            "1": 6.0,
            "7": 14.0,
            "14": 18.0,
            "30": 22.0,
        }
        self.recent_range_days = int(recent_range_days)
        self.recent_range_padding_pct = float(recent_range_padding_pct)
        self.feature_set = feature_set
        self.feature_cols = list(feature_cols) if feature_cols else list(FEATURE_COLS)
        self._garch_forecast = None
        self._garch_fit_time = None
        self._ensemble_weights = load_ensemble_weights(
            ensemble_weights_file, self.ensemble_fallback
        )
        self._conformal_margins = load_conformal_margins(conformal_intervals_file)

        if supabase_url and supabase_key:
            self.supabase_client = create_client(supabase_url, supabase_key)
        else:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")
            if supabase_url and supabase_key:
                self.supabase_client = create_client(supabase_url, supabase_key)
            else:
                self.supabase_client = None
                logger.warning("Supabase client not initialized - predictions may fail")

        self._historical_data = None
        self._last_data_fetch = None
        self._data_fingerprint = None
        self._remote_fingerprint = None
        self._frozen_frame = None
        self._nhits_levels = None

        self.direct_target = "price"
        self.direct_info_path = direct_info_path
        try:
            if direct_info_path:
                meta_path = Path(direct_info_path)
                meta = {}
                if meta_path.exists():
                    import json
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                meta = DirectHorizonTrainer.load_latest_meta(self.models_dir)
            self.direct_target = str((meta or {}).get("target") or "price")
        except Exception as e:
            logger.debug(f"No direct horizon metadata: {e}")

        if not self.direct_horizon_models:
            try:
                if direct_info_path:
                    self.direct_horizon_models = DirectHorizonTrainer.load_from_info(direct_info_path)
                else:
                    self.direct_horizon_models = DirectHorizonTrainer.load_latest(self.models_dir)
            except Exception as e:
                logger.debug(f"No direct horizon models loaded: {e}")

        logger.info(
            f"ImprovedPricePredictor initialized: version={model_version}, "
            f"feature_set={self.feature_set}, n_features={len(self.feature_cols)}, "
            f"nhits={'yes' if nhits_model else 'no'}, "
            f"multi_step={multi_step_mode}, "
            f"direct_h={list(self.direct_horizon_models)}, "
            f"conformal={'yes' if self._conformal_margins else 'no'}"
        )

    def _recent_price_range(self, df_clean: pd.DataFrame) -> Tuple[float, float]:
        """Min/max recents avec marge, bornes par price_bounds marche."""
        from src.models.served_forecast import recent_price_range

        return recent_price_range(
            df_clean["price"],
            self.recent_range_days,
            self.recent_range_padding_pct,
            self.price_bounds,
        )

    def _dampen_vs_spot(
        self, price: float, spot: float, horizon: int
    ) -> Tuple[float, float, bool]:
        """Garde-fou vs spot. Seul le plafond en pourcentage ramène au cours."""
        central, change_pct, feature_failure = publish_level(
            price,
            spot,
            horizon,
            self.max_abs_change_pct,
            self._conformal_margins,
        )
        max_pct = float(self.max_abs_change_pct.get(str(horizon), 20.0))
        if feature_failure:
            raw_pct = (price / spot - 1.0) * 100.0 if spot > 0 else float("nan")
            logger.warning(
                "h=%sd feature failure: raw %.2f is %+.2f%% vs spot %.2f "
                "(cap %.1f%%). Central scenario stays at the spot, not the cap.",
                horizon,
                price,
                raw_pct,
                spot,
                max_pct,
            )
        return central, change_pct, feature_failure

    def _market_id(self) -> str:
        if self.price_table == "cocoa_london_prices":
            return "cocoa"
        if "robusta" in (self.price_table or ""):
            return "coffee_robusta"
        return "other"

    def _remember_identities(self, raw: pd.DataFrame, filled: pd.DataFrame) -> None:
        from src.models.hybrid_features import consumed_snapshot_rows
        from src.models.release_manifest import data_fingerprint

        self._data_fingerprint = data_fingerprint(consumed_snapshot_rows(raw))
        self._served_fingerprint = data_fingerprint(consumed_snapshot_rows(filled))

    def _fetch_historical_data(self, force_refresh: bool = False) -> pd.DataFrame:
        from src.models.hybrid_features import fill_missing_ohlcv

        if self._frozen_frame is not None:
            raw = self._frozen_frame
            filled = fill_missing_ohlcv(raw)
            self._remember_identities(raw, filled)
            return filled
        if not force_refresh and self._historical_data is not None and self._last_data_fetch is not None:
            age = (datetime.now() - self._last_data_fetch).total_seconds()
            remote = self._remote_fingerprint or self._remote_snapshot_fingerprint()
            if historical_cache_is_fresh(age, self._data_fingerprint, remote):
                return self._historical_data

        if self.supabase_client is None:
            raise RuntimeError("Supabase client not initialized")

        select_candidates = [
            "date, price, open, high, low, volume, open_interest",
            "date, price, open, high, low, volume",
            "date, price",
        ]
        all_data = []
        for select_clause in select_candidates:
            all_data = []
            page_size = 1000
            offset = 0
            try:
                while True:
                    response = (
                        self.supabase_client.table(self.price_table)
                        .select(select_clause)
                        .order("date")
                        .range(offset, offset + page_size - 1)
                        .execute()
                    )
                    if not response.data:
                        break
                    all_data.extend(response.data)
                    offset += page_size
                    if len(response.data) < page_size:
                        break
                break
            except Exception:
                continue

        from src.models.hybrid_features import fill_missing_ohlcv, prepare_market_frame

        raw = prepare_market_frame(pd.DataFrame(all_data), market_id=self._market_id())
        filled = fill_missing_ohlcv(raw)
        self._remember_identities(raw, filled)
        self._historical_data = filled
        self._nhits_levels = None
        self._last_data_fetch = datetime.now()
        return filled

    def _remote_snapshot_fingerprint(self) -> Optional[str]:
        """Hash the stored columns. A same-day correction does not match the cache."""
        if self.supabase_client is None:
            return None
        from src.models.hybrid_features import consumed_snapshot_rows, prepare_market_frame
        from src.models.release_manifest import data_fingerprint

        select_candidates = (
            "date,price,open,high,low,volume,open_interest",
            "date,price,open,high,low,volume",
            "date,price",
        )
        for clause in select_candidates:
            rows = []
            offset = 0
            page = 1000
            try:
                while True:
                    response = (
                        self.supabase_client.table(self.price_table)
                        .select(clause)
                        .order("date")
                        .range(offset, offset + page - 1)
                        .execute()
                    )
                    batch = list(response.data or [])
                    rows.extend(batch)
                    if len(batch) < page:
                        break
                    offset += page
                prepared = prepare_market_frame(pd.DataFrame(rows), market_id=self._market_id())
                return data_fingerprint(consumed_snapshot_rows(prepared))
            except Exception:
                continue
        return None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_features, _ = prepare_training_frame(df, prophet_model=self.prophet_model)
        for col in self.feature_cols:
            if col not in df_features.columns:
                df_features[col] = 0.0
        micro = [c for c in self.feature_cols if c not in FEATURE_COLS]
        if micro:
            present = [c for c in micro if c in df_features.columns]
            if present:
                df_features[present] = df_features[present].ffill().fillna(0.0)
        return df_features

    def _get_garch_forecast(self, df_clean: pd.DataFrame, horizons: List[int]):
        """Fit (avec cache 1 h) le GARCH(1,1)-t sur l'historique de prix."""
        if not self.garch_enabled:
            return None
        if self._garch_forecast is not None and self._garch_fit_time is not None:
            age = (datetime.now() - self._garch_fit_time).total_seconds()
            if age < 3600:
                return self._garch_forecast
        try:
            from src.models.garch_engine import fit_garch_forecast

            self._garch_forecast = fit_garch_forecast(
                df_clean["price"], horizons=tuple(sorted(set(horizons) | {1, 7, 14, 30}))
            )
            self._garch_fit_time = datetime.now()
        except Exception as e:
            logger.warning(f"GARCH fit failed: {e}")
            self._garch_forecast = None
        return self._garch_forecast

    def bind_frame(self, frame: pd.DataFrame) -> None:
        """Use one frozen table for training and for the forecasts of that run."""
        self._frozen_frame = frame.copy()
        self._historical_data = None
        self._nhits_levels = None

    def _predict_xgb_price(
        self,
        horizon: int,
        df_clean: pd.DataFrame,
        last_row: pd.Series,
        current_price: float,
        current_date,
    ) -> float:
        """Direct model of this horizon. A missing model is not replaced."""
        del df_clean, current_date
        if horizon not in self.direct_horizon_models:
            return float("nan")
        return direct_level(
            self.direct_horizon_models[horizon],
            last_row,
            self.feature_cols,
            current_price,
            self.direct_target,
        )

    def _nhits_price_at(self, df_clean: pd.DataFrame, horizon: int, weights: dict) -> Optional[float]:
        """One 30-step N-HiTS forecast per snapshot, reread at the requested horizon."""
        if self.nhits_model is None or float(weights.get("nhits", 0.0) or 0.0) <= 0.0:
            return None
        if self._nhits_levels is None:
            levels: Dict[int, float] = {}
            try:
                nhits_input = df_clean[["date", "price"]].copy()
                nhits_input.columns = ["ds", "y"]
                nhits_input["unique_id"] = self.nhits_unique_id
                forecast = self.nhits_model.predict(df=nhits_input.sort_values("ds"))
                for index in range(min(30, len(forecast))):
                    levels[index + 1] = float(forecast["NHITS"].iloc[index])
            except Exception as exc:
                logger.warning(f"N-HiTS prediction failed: {exc}")
            self._nhits_levels = levels
        return self._nhits_levels.get(int(horizon))

    def predict(
        self,
        horizons: List[int],
        exog_features: pd.DataFrame = None,
        recent_news: List[NewsArticle] = None,
        historical_range: Optional[Tuple[float, float]] = None,
    ) -> List[Prediction]:
        logger.info(f"Generating improved predictions for horizons: {horizons}")

        df = self._fetch_historical_data()
        df_with_features = self._prepare_features(df)
        published_price = float(df["price"].iloc[-1])
        published_date = pd.to_datetime(df["date"].iloc[-1])
        for c in self.feature_cols:
            if c not in df_with_features.columns:
                df_with_features[c] = 0.0
        # Garder la dernière séance : un NaN micro (OI vide) ne doit pas
        # faire reculer le spot utilisé par le modèle.
        df_with_features[self.feature_cols] = df_with_features[self.feature_cols].ffill()
        df_clean = df_with_features.dropna(subset=["price", "price_lag_30"]).copy()
        df_clean[self.feature_cols] = df_clean[self.feature_cols].fillna(0.0)

        if df_clean.empty:
            raise RuntimeError("No valid data after feature preparation")

        current_price = float(df_clean["price"].iloc[-1])
        current_date = df_clean["date"].iloc[-1]
        last_row = df_clean.iloc[-1]
        origin_day = pd.to_datetime(current_date).date().isoformat()
        self.last_snapshot = {
            "origin_date": origin_day,
            "origin_price": current_price,
            "snapshot_id": f"{origin_day}:{current_price:.4f}",
        }
        model_date = pd.to_datetime(current_date)
        if (
            model_date.normalize() != published_date.normalize()
            or abs(current_price - published_price) > 0.5
        ):
            logger.warning(
                "Spot modele %.2f (%s) != cours publie %.2f (%s)",
                current_price,
                model_date.date(),
                published_price,
                published_date.date(),
            )

        sentiment_score = 0.0
        if recent_news:
            try:
                sentiment_score = self.nlp_analyzer.aggregate_sentiment(recent_news)
            except Exception as e:
                logger.warning(f"Failed to aggregate sentiment: {e}")

        if historical_range is None:
            historical_range = self._recent_price_range(df_clean)

        predictions = []
        current_time = datetime.now()
        self._nhits_levels = None
        garch_forecast = self._get_garch_forecast(df_clean, horizons)

        for horizon in horizons:
            try:
                self._append_horizon(
                    predictions,
                    horizon=horizon,
                    df_clean=df_clean,
                    last_row=last_row,
                    current_price=current_price,
                    current_date=current_date,
                    origin_day=origin_day,
                    sentiment_score=sentiment_score,
                    historical_range=historical_range,
                    garch_forecast=garch_forecast,
                    current_time=current_time,
                    published_price=published_price,
                )
            except Exception as exc:
                logger.warning(f"Horizon {horizon} indisponible: {exc}")
                placeholder = max(float(current_price), 1000.0)
                failed_target = pd.to_datetime(future_business_date(current_date, horizon)).date().isoformat()
                predictions.append(
                    Prediction(
                        horizon=horizon,
                        price=placeholder,
                        confidence_interval=(placeholder, placeholder + 1.0),
                        confidence_level=self.confidence_level,
                        timestamp=current_time,
                        model_version=self.model_version,
                        components={
                            "baseline": placeholder,
                            "residual": 0.0,
                            "sentiment": 0.0,
                            "calculation_failed": True,
                            "origin_date": origin_day,
                            "origin_price": float(current_price),
                            "target_date": failed_target,
                            "ensemble_weights": {},
                        },
                    )
                )
        return predictions

    def _append_horizon(
        self,
        predictions,
        *,
        horizon,
        df_clean,
        last_row,
        current_price,
        current_date,
        origin_day,
        sentiment_score,
        historical_range,
        garch_forecast,
        current_time,
        published_price,
    ):
        future_date = future_business_date(current_date, horizon)
        features_future = build_prediction_row(
                last_row,
                current_price,
                future_date,
                self.prophet_model,
                feature_cols=self.feature_cols,
            )
        prophet_yhat_future = float(features_future["prophet_yhat"].iloc[0])
        xgb_price = self._predict_xgb_price(
            horizon, df_clean, last_row, current_price, current_date
        )

        weights = scheduled_weights(
            horizon, self._ensemble_weights, self.ensemble_fallback
        )
        caps = dict(self.max_abs_change_pct)
        caps[str(horizon)] = scheduled_cap(horizon, self.max_abs_change_pct)
        nhits_price = self._nhits_price_at(df_clean, horizon, weights)

        from src.models.served_forecast import compose_served_price

        spot_anchor = published_price if published_price > 0 else current_price
        served = compose_served_price(
            xgb_price=xgb_price,
            prophet_price=prophet_yhat_future,
            nhits_price=nhits_price,
            weights=weights,
            spot=spot_anchor,
            horizon=horizon,
            max_abs_change_pct=caps,
            conformal_margins=self._conformal_margins,
            price_bounds=historical_range,
            garch_forecast=garch_forecast,
            confidence_level=self.confidence_level,
            price_volatility=float(df_clean["price"].std()),
        )
        final_price = served["price"]
        lower_bound = served["lower"]
        upper_bound = served["upper"]
        feature_failure = served["feature_failure"]
        if served.get("large_move_warning"):
            logger.warning(
                "h={} mouvement au-delà du seuil: ensemble {:.2f} vs spot {:.2f}. Prix conservé.",
                horizon,
                served["ensemble"],
                spot_anchor,
            )
        if served.get("engine_missing") or not math.isfinite(float(final_price)) or final_price <= 0:
            placeholder = max(float(current_price), 1000.0)
            predictions.append(
                Prediction(
                    horizon=horizon,
                    price=placeholder,
                    confidence_interval=(placeholder, placeholder + 1.0),
                    confidence_level=self.confidence_level,
                    timestamp=current_time,
                    model_version=self.model_version,
                    components={
                        "baseline": placeholder,
                        "residual": 0.0,
                        "sentiment": 0.0,
                        "calculation_failed": True,
                        "engine_missing": bool(served.get("engine_missing")),
                        "origin_date": origin_day,
                        "origin_price": float(current_price),
                        "target_date": pd.to_datetime(future_date).date().isoformat(),
                        "ensemble_weights": weights,
                        "nhits": None,
                    },
                )
            )
            return
        band_unavailable = lower_bound is None or upper_bound is None or not (lower_bound < upper_bound)
        interval = (
            (max(float(final_price), 500.0), max(float(final_price), 500.0) + 1.0)
            if band_unavailable
            else (float(lower_bound), float(upper_bound))
        )

        predictions.append(
            Prediction(
                horizon=horizon,
                price=float(final_price),
                confidence_interval=interval,
                confidence_level=self.confidence_level,
                timestamp=current_time,
                model_version=self.model_version,
                components={
                    "baseline": float(xgb_price),
                    "nhits": float(nhits_price) if nhits_price is not None else None,
                    "prophet": float(prophet_yhat_future),
                    "ensemble": float(served["ensemble"]),
                    "residual": 0.0,
                    "sentiment": 0.0,
                    "sentiment_score": float(sentiment_score),
                    "dampened_change_pct": float(served["dampened_change_pct"]),
                    "raw_change_pct": float(served["raw_change_pct"]),
                    "feature_failure": False,
                    "large_move_warning": bool(served.get("large_move_warning")),
                    "engine_missing": False,
                    "band_unavailable": bool(band_unavailable),
                    "origin_date": origin_day,
                    "origin_price": float(current_price),
                    "target_date": pd.to_datetime(future_date).date().isoformat(),
                    "snapshot_id": self.last_snapshot["snapshot_id"],
                    "scored_variant": "no_sentiment",
                    "model_spot": float(current_price),
                    "published_spot": float(published_price),
                    "ensemble_weights": weights,
                    "garch_annualized_volatility": served["garch_annualized_volatility"],
                    "high_volatility_regime": bool(served["high_volatility_regime"]),
                    "xgb_mode": (
                        "direct_hstep" if horizon in self.direct_horizon_models else "missing"
                    ),
                },
            )
        )

    def get_model_info(self) -> Dict[str, object]:
        return {
            "model_version": self.model_version,
            "model_type": "improved_hybrid",
            "engines": 3 if self.nhits_model else 2,
            "sentiment_weight": self.sentiment_weight,
            "multi_step_mode": self.multi_step_mode,
            "direct_horizons": list(self.direct_horizon_models.keys()),
            "ensemble_weights_loaded": bool(self._ensemble_weights),
            "conformal_intervals_loaded": bool(self._conformal_margins),
            "confidence_level": self.confidence_level,
            "price_table": self.price_table,
            "garch_enabled": self.garch_enabled,
        }
