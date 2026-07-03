"""
Walk-forward expanding-window validation for Prophet + XGBoost hybrid model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from src.models.hybrid_features import (
    DEFAULT_PROPHET_PARAMS,
    DEFAULT_XGB_PARAMS,
    FEATURE_COLS,
    build_prediction_row,
    build_price_lookup,
    future_business_date,
)
from src.models.hybrid_trainer import HybridModelTrainer
from src.models.direct_horizon_trainer import DirectHorizonTrainer
from src.models.multi_step_predictor import predict_recursive
from src.validation.metrics import aggregate_by_horizon


@dataclass
class WalkForwardConfig:
    horizons: List[int] = field(default_factory=lambda: [1, 7, 30])
    min_train_days: int = 252
    step_size: int = 5
    max_origins: Optional[int] = None
    include_recursive: bool = True
    include_direct_hstep: bool = False
    prophet_params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PROPHET_PARAMS))
    xgb_params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_XGB_PARAMS))


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    summary: Dict[str, Dict[int, Dict[str, float]]]
    n_origins: int
    config: WalkForwardConfig


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
        if self.config.max_origins is not None:
            indices = indices[: self.config.max_origins]
        return indices

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
            train_clean = df_train_features.dropna()
            if train_clean.empty:
                continue

            last_row = train_clean.iloc[-1]

            for horizon in self.config.horizons:
                target_date = future_business_date(origin_date, horizon)
                target_ts = pd.Timestamp(target_date).normalize()

                if target_ts not in price_lookup.index:
                    continue

                actual = float(price_lookup.loc[target_ts])
                features_future = build_prediction_row(
                    last_row, origin_price, target_date, prophet_model
                )
                xgb_pred = float(xgb_model.predict(features_future[FEATURE_COLS])[0])
                prophet_pred = float(features_future["prophet_yhat"].iloc[0])

                record: Dict[str, Any] = {
                    "origin_date": origin_date,
                    "origin_price": origin_price,
                    "target_date": target_ts,
                    "horizon": horizon,
                    "actual": actual,
                    "xgb_pred": xgb_pred,
                    "prophet_pred": prophet_pred,
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

                if self.config.include_direct_hstep and horizon in (7, 30):
                    try:
                        direct_trainer = DirectHorizonTrainer(horizons=[horizon])
                        direct_models, _ = direct_trainer.fit(df_train, prophet_model=prophet_model)
                        direct_model = direct_models[horizon]
                        train_clean = df_train_features.dropna()
                        if not train_clean.empty:
                            last = train_clean.iloc[-1]
                            feat_row = build_prediction_row(
                                last,
                                origin_price,
                                target_date,
                                prophet_model,
                            )
                            direct_pred = float(
                                direct_model.predict(feat_row[FEATURE_COLS])[0]
                            )
                            record["xgb_pred_direct"] = direct_pred
                            record["xgb_direct_error_pct"] = abs(direct_pred - actual) / actual * 100
                    except Exception:
                        record["xgb_pred_direct"] = float("nan")

                records.append(record)

        predictions_df = pd.DataFrame(records)
        pred_columns = ["xgb_pred", "prophet_pred"]
        if "xgb_pred_recursive" in predictions_df.columns:
            pred_columns.append("xgb_pred_recursive")
        if "xgb_pred_direct" in predictions_df.columns:
            pred_columns.append("xgb_pred_direct")
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
