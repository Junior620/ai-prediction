"""
Training pipeline for hybrid Prophet + XGBoost cocoa price models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import xgboost as xgb
from prophet import Prophet

from src.models.hybrid_features import (
    DEFAULT_PROPHET_PARAMS,
    DEFAULT_XGB_PARAMS,
    FEATURE_COLS,
    prepare_training_frame,
)


def next_session_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose next close is known. Features stay those of date t."""
    out = frame.copy()
    out["target_price"] = out["price"].shift(-1)
    return out.dropna(subset=["target_price"]).reset_index(drop=True)


class HybridModelTrainer:
    """Fit Prophet and XGBoost on a training slice without data leakage."""

    def __init__(
        self,
        prophet_params: Optional[Dict[str, Any]] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = 20,
        val_fraction: float = 0.1,
        feature_cols: Optional[Sequence[str]] = None,
    ):
        self.prophet_params = {**DEFAULT_PROPHET_PARAMS, **(prophet_params or {})}
        self.xgb_params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.feature_cols: List[str] = list(feature_cols) if feature_cols else list(FEATURE_COLS)

    def fit(
        self,
        df_train: pd.DataFrame,
    ) -> Tuple[Prophet, xgb.XGBRegressor, pd.DataFrame]:
        """
        Train Prophet on df_train, build features, then fit XGBoost.

        Returns:
            prophet_model, xgb_model, df_with_features (including NaN rows from lags)
        """
        df_features, prophet_model = prepare_training_frame(
            df_train,
            prophet_params=self.prophet_params,
        )
        # Forward-fill optional microstructure then fill remaining gaps
        for col in self.feature_cols:
            if col not in df_features.columns:
                df_features[col] = 0.0
        micro = [c for c in self.feature_cols if c not in FEATURE_COLS]
        if micro:
            df_features[micro] = df_features[micro].ffill().fillna(0.0)

        df_clean = df_features.dropna(subset=self.feature_cols + ["price"]).reset_index(drop=True)
        # Features at t include that close. The target is the next close, not price[t].
        df_clean = next_session_frame(df_clean)

        if len(df_clean) < 2:
            raise ValueError("Not enough training rows after feature preparation")

        X = df_clean[self.feature_cols]
        y = df_clean["target_price"]

        eval_set = None
        if len(df_clean) >= 20 and self.val_fraction > 0:
            split_idx = max(1, int(len(df_clean) * (1 - self.val_fraction)))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            eval_set = [(X_val, y_val)]
        else:
            X_train, y_train = X, y

        xgb_model = xgb.XGBRegressor(
            **self.xgb_params,
            early_stopping_rounds=self.early_stopping_rounds if eval_set is not None else None,
        )
        fit_kwargs: Dict[str, Any] = {"verbose": False}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set

        xgb_model.fit(X_train, y_train, **fit_kwargs)

        return prophet_model, xgb_model, df_features
