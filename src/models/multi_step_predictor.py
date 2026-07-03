"""Recursive multi-step prediction for multi-horizon XGBoost forecasting."""

from __future__ import annotations

from typing import Union

import pandas as pd
from prophet import Prophet

from src.models.hybrid_features import (
    FEATURE_COLS,
    add_prophet_features,
    build_prediction_row,
    build_technical_features,
    future_business_date,
)


def predict_frozen(
    last_row: pd.Series,
    current_price: float,
    current_date: Union[pd.Timestamp, pd.Timestamp],
    horizon: int,
    prophet_model: Prophet,
    xgb_model,
) -> float:
    """Single-shot prediction with frozen lags (legacy production behavior)."""
    future_date = future_business_date(current_date, horizon)
    features = build_prediction_row(last_row, current_price, future_date, prophet_model)
    return float(xgb_model.predict(features[FEATURE_COLS])[0])


def predict_recursive(
    df_history: pd.DataFrame,
    prophet_model: Prophet,
    xgb_model,
    horizon: int,
) -> float:
    """
    Recursive multi-step: predict J+1 repeatedly, inject synthetic prices, update lags.

    Args:
        df_history: Historical ``date`` and ``price`` up to origin (inclusive).
        prophet_model: Fitted Prophet (not refit during recursion).
        xgb_model: Fitted XGBoost 1-step model.
        horizon: Target horizon in business days.

    Returns:
        Price prediction at the target horizon.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if horizon == 1:
        work = build_technical_features(df_history.copy())
        work = add_prophet_features(work, prophet_model)
        clean = work.dropna()
        last = clean.iloc[-1]
        return predict_frozen(
            last, float(last["price"]), last["date"], 1, prophet_model, xgb_model
        )

    work = df_history[["date", "price"]].copy().sort_values("date").reset_index(drop=True)
    prediction = None

    for _ in range(horizon):
        feat = build_technical_features(work)
        feat = add_prophet_features(feat, prophet_model)
        clean = feat.dropna()
        if clean.empty:
            raise ValueError("Not enough history for recursive prediction")

        last = clean.iloc[-1]
        current_price = float(last["price"])
        current_date = last["date"]
        next_date = future_business_date(current_date, 1)
        row = build_prediction_row(last, current_price, next_date, prophet_model)
        prediction = float(xgb_model.predict(row[FEATURE_COLS])[0])

        work = pd.concat(
            [
                work,
                pd.DataFrame({"date": [pd.Timestamp(next_date).normalize()], "price": [prediction]}),
            ],
            ignore_index=True,
        )

    return float(prediction)
