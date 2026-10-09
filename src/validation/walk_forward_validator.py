"""
Walk-forward expanding-window validation for Prophet + XGBoost hybrid model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.models.hybrid_features import (
    DEFAULT_PROPHET_PARAMS,
    DEFAULT_XGB_PARAMS,
    FEATURE_COLS,
    build_prediction_row,
    build_price_lookup,
    future_business_date,
    prepare_training_frame,
)
from src.models.hybrid_trainer import HybridModelTrainer
from src.models.direct_horizon_trainer import DirectHorizonTrainer
from src.models.multi_step_predictor import predict_frozen, predict_recursive
from src.models.ensemble_weights import (
    get_weights_for_horizon,
    load_ensemble_weights,
)
from src.models.served_forecast import compose_served_price, direct_level
from src.validation.metrics import aggregate_by_horizon


@dataclass
class WalkForwardConfig:
    horizons: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    min_train_days: int = 252
    step_size: int = 5
    max_origins: Optional[int] = None
    origin_window: str = "recent"
    include_recursive: bool = True
    include_direct_hstep: bool = False
    ensemble_weights_file: Optional[str] = None
    replay_nhits: bool = False
    nhits_unique_id: str = "cocoa_ice_london"
    feature_cols: Optional[List[str]] = None
    max_abs_change_pct: Dict[str, float] = field(
        default_factory=lambda: {"1": 6.0, "7": 14.0, "14": 18.0, "30": 22.0}
    )
    conformal_margins: Optional[Dict[str, Any]] = None
    price_bounds: tuple = (1000.0, 15000.0)
    recent_range_days: int = 252
    recent_range_padding_pct: float = 15.0
    garch_enabled: bool = False
    defer_bands: bool = False
    confidence_level: float = 0.90
    prophet_params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PROPHET_PARAMS))
    xgb_params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_XGB_PARAMS))


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    summary: Dict[str, Dict[int, Dict[str, float]]]
    n_origins: int
    config: WalkForwardConfig


def drift_price(prices: pd.Series, horizon: int, lookback: int = 20) -> float:
    """Project the origin close by the mean daily return known up to that date."""
    series = pd.Series(prices).astype(float).dropna()
    if series.empty:
        return float("nan")
    origin = float(series.iloc[-1])
    returns = series.pct_change().dropna().tail(lookback)
    mean_return = float(returns.mean()) if len(returns) else 0.0
    return origin * ((1.0 + mean_return) ** int(horizon))


class WalkForwardValidator:
    """Expanding-window walk-forward backtest with multi-horizon evaluation."""

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()
        self.trainer = HybridModelTrainer(
            prophet_params=self.config.prophet_params,
            xgb_params=self.config.xgb_params,
        )

    def _origin_indices(self, n_rows: int, max_horizon: int) -> List[int]:
        """Return row indices used as prediction origins."""
        start = self.config.min_train_days - 1
        end = n_rows - max_horizon - 1
        if start > end:
            return []

        indices = list(range(start, end + 1, self.config.step_size))
        limit = self.config.max_origins
        if limit is not None and len(indices) > limit:
            if self.config.origin_window == "historical":
                indices = indices[:limit]
            else:
                indices = indices[-limit:]
        return indices

    def _direct_last_row(self, df_train: pd.DataFrame, prophet_model) -> Optional[pd.Series]:
        """Last training session with the same feature columns as the direct model."""
        frame, _ = prepare_training_frame(df_train, prophet_model=prophet_model)
        cols = list(self.config.feature_cols or FEATURE_COLS)
        for col in cols:
            if col not in frame.columns:
                frame[col] = 0.0
        frame[cols] = frame[cols].ffill()
        clean = frame.dropna(subset=["price", "price_lag_30"])
        if clean.empty:
            return None
        clean = clean.copy()
        clean[cols] = clean[cols].fillna(0.0)
        return clean.iloc[-1]

    def run(self, df: pd.DataFrame) -> WalkForwardResult:
        """
        Run walk-forward validation on price data (date, price columns).

        At each origin t, trains on df[:t+1] and predicts each configured horizon.
        """
        df = df.sort_values("date").reset_index(drop=True)
        price_lookup = build_price_lookup(df)
        max_horizon = max(self.config.horizons)

        origin_indices = self._origin_indices(len(df), max_horizon)
        logger.info(
            f"Walk-forward: {len(origin_indices)} origins, horizons={self.config.horizons}"
        )

        records: List[Dict[str, Any]] = []

        for origin_idx in origin_indices:
            df_train = df.iloc[: origin_idx + 1].copy()
            origin_date = df_train["date"].iloc[-1]
            origin_price = float(df_train["price"].iloc[-1])

            prophet_model, xgb_model, df_train_features = self.trainer.fit(df_train)
            observed = [
                column
                for column in df_train_features.columns
                if df_train_features[column].notna().any()
            ]
            train_clean = df_train_features.dropna(subset=observed)
            if train_clean.empty:
                continue

            last_row = train_clean.iloc[-1]
            direct_models = None
            direct_last = None
            direct_target = "close_delta"
            nhits_levels: Dict[int, float] = {}
            if self.config.replay_nhits:
                try:
                    from src.models.nhits_replay import forecast_nhits

                    nhits_levels = forecast_nhits(df_train, self.config.nhits_unique_id)
                except Exception as exc:
                    logger.warning(f"N-HiTS replay failed at {origin_date}: {exc}")
            garch_forecast = None
            if self.config.garch_enabled:
                try:
                    from src.models.garch_engine import fit_garch_forecast

                    garch_forecast = fit_garch_forecast(
                        df_train["price"],
                        horizons=tuple(self.config.horizons),
                    )
                except Exception as exc:
                    logger.warning(f"GARCH replay failed at {origin_date}: {exc}")
            if self.config.include_direct_hstep:
                try:
                    direct_trainer = DirectHorizonTrainer(
                        horizons=list(self.config.horizons),
                        feature_cols=self.config.feature_cols,
                    )
                    direct_models, _ = direct_trainer.fit(
                        df_train, prophet_model=prophet_model
                    )
                    direct_target = direct_trainer.target
                    direct_last = self._direct_last_row(df_train, prophet_model)
                except Exception as exc:
                    logger.warning(f"Direct h-step fit failed at {origin_date}: {exc}")
                    direct_models = None

            for horizon in self.config.horizons:
                target_date = future_business_date(origin_date, horizon)
                target_ts = pd.Timestamp(target_date).normalize()

                if target_ts not in price_lookup.index:
                    continue

                actual = float(price_lookup.loc[target_ts])
                features_future = build_prediction_row(
                    last_row, origin_price, target_date, prophet_model
                )
                xgb_pred = predict_frozen(
                    last_row, origin_price, origin_date, horizon, prophet_model, xgb_model
                )
                prophet_pred = float(features_future["prophet_yhat"].iloc[0])

                naive_pred = float(origin_price)
                drift_pred = drift_price(df_train["price"], horizon)
                record: Dict[str, Any] = {
                    "origin_date": origin_date,
                    "origin_price": origin_price,
                    "target_date": target_ts,
                    "horizon": horizon,
                    "actual": actual,
                    "xgb_pred": xgb_pred,
                    "prophet_pred": prophet_pred,
                    "naive_pred": naive_pred,
                    "drift_pred": drift_pred,
                    "xgb_error": xgb_pred - actual,
                    "xgb_error_pct": abs(xgb_pred - actual) / actual * 100,
                    "prophet_error": prophet_pred - actual,
                    "prophet_error_pct": abs(prophet_pred - actual) / actual * 100,
                    "origin_idx": origin_idx,
                    "train_size": len(df_train),
                }

                if self.config.include_recursive and horizon > 1:
                    try:
                        rec_pred = predict_recursive(
                            df_train, prophet_model, xgb_model, horizon
                        )
                        record["xgb_pred_recursive"] = rec_pred
                        record["xgb_recursive_error_pct"] = abs(rec_pred - actual) / actual * 100
                    except Exception:
                        record["xgb_pred_recursive"] = float("nan")

                if (
                    direct_models is not None
                    and direct_last is not None
                    and horizon in direct_models
                ):
                    try:
                        cols = self.config.feature_cols or list(FEATURE_COLS)
                        raw_level = direct_level(
                            direct_models[horizon],
                            direct_last,
                            cols,
                            origin_price,
                            direct_target,
                        )
                        weights = {"xgb": 1.0, "prophet": 0.0, "nhits": 0.0}
                        if self.config.ensemble_weights_file:
                            weights = get_weights_for_horizon(
                                horizon,
                                load_ensemble_weights(self.config.ensemble_weights_file),
                                weights,
                            )
                        nhits_weight = float(weights.get("nhits", 0.0))
                        nhits_price = nhits_levels.get(horizon) if nhits_weight > 0 else None
                        engine_missing = nhits_weight > 0 and nhits_price is None
                        served = compose_served_price(
                            xgb_price=raw_level,
                            prophet_price=prophet_pred,
                            nhits_price=nhits_price,
                            weights=weights,
                            spot=origin_price,
                            horizon=horizon,
                            max_abs_change_pct=self.config.max_abs_change_pct,
                            conformal_margins=(
                                None if self.config.defer_bands else self.config.conformal_margins
                            ),
                            price_bounds=self.config.price_bounds,
                            recent_prices=df_train["price"],
                            recent_range_days=self.config.recent_range_days,
                            recent_range_padding_pct=self.config.recent_range_padding_pct,
                            garch_forecast=garch_forecast,
                            confidence_level=self.config.confidence_level,
                            price_volatility=float(df_train["price"].std()),
                        )
                        record["xgb_pred_direct"] = raw_level
                        record["published_pred"] = served["price"]
                        record["feature_failure"] = bool(served["feature_failure"])
                        record["engine_missing"] = bool(engine_missing)
                        record["engines_present"] = ",".join(
                            name
                            for name, present in (
                                ("xgb", np.isfinite(raw_level)),
                                ("prophet", np.isfinite(prophet_pred)),
                                ("nhits", nhits_price is not None),
                            )
                            if present
                        )
                        record["weights_used"] = json.dumps(weights, sort_keys=True)
                        if not self.config.defer_bands:
                            record["published_lower"] = served["lower"]
                            record["published_upper"] = served["upper"]
                        record["xgb_direct_error_pct"] = abs(raw_level - actual) / actual * 100
                    except Exception:
                        record["xgb_pred_direct"] = float("nan")
                        record["published_pred"] = float("nan")

                records.append(record)

        predictions_df = pd.DataFrame(records)
        pred_columns = ["xgb_pred", "prophet_pred", "naive_pred", "drift_pred"]
        for extra in ("xgb_pred_recursive", "xgb_pred_direct", "published_pred"):
            if extra in predictions_df.columns:
                pred_columns.append(extra)
        summary = aggregate_by_horizon(
            predictions_df,
            pred_columns=pred_columns,
        )

        return WalkForwardResult(
            predictions=predictions_df,
            summary=summary,
            n_origins=len(origin_indices),
            config=self.config,
        )
