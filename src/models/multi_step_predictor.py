"""Recursive multi-step prediction for multi-horizon XGBoost forecasting."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

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
    feature_cols: Optional[Sequence[str]] = None,
) -> float:
    """One-step model: the last session's features forecast the next close.

    Horizons above one still shift the calendar. That row is not the training
    convention; the recursive path is the multi-session fallback.
    """
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    if int(horizon) == 1:
        row = pd.DataFrame([{c: last_row.get(c, 0.0) for c in cols}])
        return float(xgb_model.predict(row[cols])[0])
    future_date = future_business_date(current_date, horizon)
    features = build_prediction_row(
        last_row, current_price, future_date, prophet_model, feature_cols=cols
    )
    return float(xgb_model.predict(features[cols])[0])


def predict_recursive(
    df_history: pd.DataFrame,
    prophet_model: Prophet,
    xgb_model,
    horizon: int,
    feature_cols: Optional[Sequence[str]] = None,
) -> float:
    """
    Recursive multi-step: predict J+1 repeatedly, inject synthetic prices, update lags.
    """
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if horizon == 1:
        work = build_technical_features(df_history.copy())
        work = add_prophet_features(work, prophet_model)
        for c in cols:
            if c not in work.columns:
                work[c] = 0.0
        clean = work.dropna(subset=[c for c in FEATURE_COLS if c in work.columns])
        last = clean.iloc[-1]
        return predict_frozen(
            last,
            float(last["price"]),
            last["date"],
            1,
            prophet_model,
            xgb_model,
            feature_cols=cols,
        )

    # Keep extra columns if present for OHLCV/OI continuity
    keep = [c for c in ("date", "price", "open", "high", "low", "volume", "open_interest") if c in df_history.columns]
    work = df_history[keep].copy().sort_values("date").reset_index(drop=True)
    prediction = None

    for _ in range(horizon):
        feat = build_technical_features(work)
        feat = add_prophet_features(feat, prophet_model)
        for c in cols:
            if c not in feat.columns:
                feat[c] = 0.0
        micro = [c for c in cols if c not in FEATURE_COLS and c in feat.columns]
        if micro:
            feat[micro] = feat[micro].ffill().fillna(0.0)
        clean = feat.dropna(subset=[c for c in FEATURE_COLS if c in feat.columns])
        if clean.empty:
            raise ValueError("Not enough history for recursive prediction")

        last = clean.iloc[-1]
        current_price = float(last["price"])
        current_date = last["date"]
        next_date = future_business_date(current_date, 1)
        row = pd.DataFrame([{c: last.get(c, 0.0) for c in cols}])
        prediction = float(xgb_model.predict(row[cols])[0])

        new_row = {"date": pd.Timestamp(next_date).normalize(), "price": prediction}
        for c in ("open", "high", "low", "volume", "open_interest"):
            if c in work.columns:
                new_row[c] = last.get(c)
        work = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)

    return float(prediction)
