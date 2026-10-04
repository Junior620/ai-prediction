"""
Improved Price Predictor for hybrid cocoa price forecasting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
from supabase import create_client
import os

from src.models.data_models import Prediction, NewsArticle
from src.models.direct_horizon_trainer import DirectHorizonTrainer
from src.models.ensemble_weights import (
    DEFAULT_FALLBACK,
    combine_ensemble,
    get_weights_for_horizon,
    load_ensemble_weights,
)
from src.models.hybrid_features import (
    FEATURE_COLS,
    build_prediction_row,
    clean_price_dataframe,
    direct_feature_row,
    future_business_date,
    guard_forecast,
    prepare_training_frame,
    resolve_feature_cols,
)
from src.models.conformal_intervals import (
    apply_interval,
    get_margins_for_horizon,
    heuristic_interval,
    load_conformal_margins,
)
from src.models.multi_step_predictor import predict_frozen, predict_recursive


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

        self.direct_target = "price"
        try:
            meta = DirectHorizonTrainer.load_latest_meta(self.models_dir)
            self.direct_target = str(meta.get("target") or "price")
        except Exception as e:
            logger.debug(f"No direct horizon metadata: {e}")

        if not self.direct_horizon_models:
            try:
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
        tail = df_clean.tail(max(30, self.recent_range_days))
        lo = float(tail["price"].min())
        hi = float(tail["price"].max())
        pad = self.recent_range_padding_pct / 100.0
        mid = (lo + hi) / 2.0 if hi > lo else lo
        span = max(hi - lo, mid * 0.05)
        lo2 = lo - span * pad
        hi2 = hi + span * pad
        b_lo, b_hi = self.price_bounds
        return (max(b_lo, lo2), min(b_hi, hi2))

    def _dampen_vs_spot(
        self, price: float, spot: float, horizon: int
    ) -> Tuple[float, float, bool]:
        """Garde-fou vs spot.

        Un écart au-delà de ``max_abs_change_pct`` est un échec de features :
        le scénario central reste le cours, le plafond n'est pas publié.
        """
        max_pct = float(self.max_abs_change_pct.get(str(horizon), 20.0))
        margin = get_margins_for_horizon(horizon, self._conformal_margins or {})
        central, change_pct, feature_failure = guard_forecast(price, spot, max_pct, margin)
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

    def _fetch_historical_data(self, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self._historical_data is not None:
            if self._last_data_fetch is not None:
                age = (datetime.now() - self._last_data_fetch).total_seconds()
                if age < 3600:
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

        df = clean_price_dataframe(pd.DataFrame(all_data))
        for col in ("open", "high", "low", "volume", "open_interest"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").ffill()
        self._historical_data = df
        self._last_data_fetch = datetime.now()
        return df

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

    def _predict_xgb_price(
        self,
        horizon: int,
        df_clean: pd.DataFrame,
        last_row: pd.Series,
        current_price: float,
        current_date,
    ) -> float:
        """XGBoost price using direct h-step, recursive, or frozen strategy."""
        cols = self.feature_cols
        if horizon in self.direct_horizon_models:
            # Same row the direct model was trained on (features known at t).
            features = direct_feature_row(last_row, feature_cols=cols)
            raw = float(self.direct_horizon_models[horizon].predict(features[cols])[0])
            return DirectHorizonTrainer.level_from_prediction(
                raw, current_price, self.direct_target
            )

        if horizon == 1 or self.multi_step_mode == "frozen":
            return predict_frozen(
                last_row,
                current_price,
                current_date,
                horizon,
                self.prophet_model,
                self.xgboost_model,
                feature_cols=cols,
            )

        keep = [
            c
            for c in ("date", "price", "open", "high", "low", "volume", "open_interest")
            if c in df_clean.columns
        ]
        df_history = df_clean[keep].copy()
        return predict_recursive(
            df_history,
            self.prophet_model,
            self.xgboost_model,
            horizon,
            feature_cols=cols,
        )

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
        garch_forecast = self._get_garch_forecast(df_clean, horizons)

        for horizon in horizons:
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

            nhits_price = None
            if self.nhits_model is not None:
                try:
                    nhits_input = df_clean[["date", "price"]].copy()
                    nhits_input.columns = ["ds", "y"]
                    nhits_input["unique_id"] = self.nhits_unique_id
                    nhits_forecast = self.nhits_model.predict(df=nhits_input.sort_values("ds"))
                    if horizon <= len(nhits_forecast):
                        nhits_price = float(nhits_forecast["NHITS"].iloc[horizon - 1])
                except Exception as e:
                    logger.warning(f"N-HiTS prediction failed for horizon {horizon}d: {e}")

            weights = get_weights_for_horizon(
                horizon, self._ensemble_weights, self.ensemble_fallback
            )
            # Toujours passer par combine_ensemble : si N-HiTS est absent,
            # les poids xgb/prophet sont renormalises (ne pas retomber sur xgb seul).
            ensemble_price = combine_ensemble(
                xgb_price, prophet_yhat_future, nhits_price, weights
            )

            sentiment_adjustment = sentiment_score * self.sentiment_weight * ensemble_price
            raw_price = ensemble_price + sentiment_adjustment
            # Plafond mesuré sur le cours publié, pas sur une séance plus ancienne.
            spot_anchor = published_price if published_price > 0 else current_price
            raw_change_pct = (
                (raw_price / spot_anchor - 1.0) * 100.0 if spot_anchor > 0 else 0.0
            )
            final_price, dampened_pct, feature_failure = self._dampen_vs_spot(
                raw_price, spot_anchor, horizon
            )
            final_price = float(np.clip(final_price, historical_range[0], historical_range[1]))

            sentiment_factor = 1.3 if abs(sentiment_score) > 0.6 else 1.0
            if self._conformal_margins:
                lower_bound, upper_bound = apply_interval(
                    final_price,
                    horizon,
                    self._conformal_margins,
                    (historical_range[0], historical_range[1]),
                )
                if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
                    lower_bound, upper_bound = heuristic_interval(
                        final_price,
                        horizon,
                        float(df_clean["price"].std()),
                        (historical_range[0], historical_range[1]),
                        self.confidence_level,
                        sentiment_factor,
                    )
            else:
                lower_bound, upper_bound = heuristic_interval(
                    final_price,
                    horizon,
                    float(df_clean["price"].std()),
                    (historical_range[0], historical_range[1]),
                    self.confidence_level,
                    sentiment_factor,
                )

            # Couche GARCH : par borne, on garde la plus large entre conforme et GARCH.
            # Le conforme garantit la couverture en régime normal ; le GARCH élargit
            # l'intervalle en période de forte volatilité.
            garch_ann_vol = None
            high_vol_regime = False
            if garch_forecast is not None:
                garch_interval = garch_forecast.interval_around(
                    final_price, horizon, self.confidence_level
                )
                if garch_interval is not None:
                    g_lo, g_hi = garch_interval
                    lower_bound = min(lower_bound, g_lo)
                    upper_bound = max(upper_bound, g_hi)
                garch_ann_vol = garch_forecast.annualized_volatility(horizon)
                high_vol_regime = garch_forecast.high_volatility_regime(horizon)
                lower_bound = float(np.clip(lower_bound, historical_range[0], historical_range[1]))
                upper_bound = float(np.clip(upper_bound, historical_range[0], historical_range[1]))

            predictions.append(
                Prediction(
                    horizon=horizon,
                    price=float(final_price),
                    confidence_interval=(float(lower_bound), float(upper_bound)),
                    confidence_level=self.confidence_level,
                    timestamp=current_time,
                    model_version=self.model_version,
                    components={
                        "baseline": float(xgb_price),
                        "nhits": float(nhits_price) if nhits_price is not None else None,
                        "prophet": float(prophet_yhat_future),
                        "ensemble": float(ensemble_price),
                        "residual": 0.0,
                        "sentiment": float(sentiment_adjustment),
                        "dampened_change_pct": float(dampened_pct),
                        "raw_change_pct": float(raw_change_pct),
                        "feature_failure": bool(feature_failure),
                        "model_spot": float(current_price),
                        "published_spot": float(published_price),
                        "ensemble_weights": weights,
                        "garch_annualized_volatility": (
                            float(garch_ann_vol) if garch_ann_vol is not None else None
                        ),
                        "high_volatility_regime": bool(high_vol_regime),
                        "xgb_mode": (
                            "direct_hstep" if horizon in self.direct_horizon_models
                            else ("frozen" if horizon == 1 or self.multi_step_mode == "frozen"
                                  else "recursive")
                        ),
                    },
                )
            )

        return predictions

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
