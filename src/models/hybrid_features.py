"""
Shared feature engineering for the hybrid Prophet + XGBoost cocoa price model.

Used by training, inference, and walk-forward validation to guarantee identical logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay
from prophet import Prophet

FEATURE_COLS = [
    "price_lag_1",
    "price_lag_3",
    "price_lag_7",
    "price_lag_14",
    "price_lag_30",
    "price_ma_7",
    "price_ma_14",
    "price_ma_30",
    "price_ma_60",
    "price_std_7",
    "price_std_30",
    "price_change_1d",
    "price_change_7d",
    "price_change_30d",
    "prophet_trend",
    "prophet_yearly",
    "prophet_yhat",
    "year",
    "month",
    "day_of_week",
    "day_of_year",
    "quarter",
]

DEFAULT_PROPHET_PARAMS: Dict[str, Any] = {
    "changepoint_prior_scale": 0.1,
    "seasonality_prior_scale": 5.0,
    "yearly_seasonality": True,
    "weekly_seasonality": False,
    "daily_seasonality": False,
    "changepoint_range": 0.8,
}

DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def load_price_data_from_supabase(
    supabase_client,
    min_date: str = "2020-01-01",
    table_name: str = "cocoa_prices",
) -> pd.DataFrame:
    """Load and clean price data from Supabase (any market price table)."""
    all_data = []
    page_size = 1000
    offset = 0

    while True:
        response = (
            supabase_client.table(table_name)
            .select("date, price")
            .order("date")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        if not response.data:
            break
        all_data.extend(response.data)
        offset += page_size

    df = pd.DataFrame(all_data)
    return clean_price_dataframe(df, min_date=min_date)


def clean_price_dataframe(df: pd.DataFrame, min_date: str = "2020-01-01") -> pd.DataFrame:
    """Normalize, filter, and remove extreme outliers from a price DataFrame."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out = out[out["date"] >= min_date].copy()

    mean_price = out["price"].mean()
    std_price = out["price"].std()
    out = out[
        (out["price"] >= mean_price - 3 * std_price)
        & (out["price"] <= mean_price + 3 * std_price)
    ].copy()

    return out.reset_index(drop=True)


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, momentum, and temporal features."""
    out = df.copy()

    out["price_lag_1"] = out["price"].shift(1)
    out["price_lag_3"] = out["price"].shift(3)
    out["price_lag_7"] = out["price"].shift(7)
    out["price_lag_14"] = out["price"].shift(14)
    out["price_lag_30"] = out["price"].shift(30)

    out["price_ma_7"] = out["price"].rolling(window=7).mean()
    out["price_ma_14"] = out["price"].rolling(window=14).mean()
    out["price_ma_30"] = out["price"].rolling(window=30).mean()
    out["price_ma_60"] = out["price"].rolling(window=60).mean()

    out["price_std_7"] = out["price"].rolling(window=7).std()
    out["price_std_30"] = out["price"].rolling(window=30).std()

    out["price_change_1d"] = out["price"].pct_change(1)
    out["price_change_7d"] = out["price"].pct_change(7)
    out["price_change_30d"] = out["price"].pct_change(30)

    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_year"] = out["date"].dt.dayofyear
    out["quarter"] = out["date"].dt.quarter

    return out


def fit_prophet(
    df_train: pd.DataFrame,
    prophet_params: Optional[Dict[str, Any]] = None,
) -> Prophet:
    """Fit Prophet on training data only."""
    params = {**DEFAULT_PROPHET_PARAMS, **(prophet_params or {})}
    df_prophet = df_train[["date", "price"]].rename(columns={"date": "ds", "price": "y"})

    model = Prophet(**params)
    model.fit(df_prophet)
    return model


def add_prophet_features(df: pd.DataFrame, prophet_model: Prophet) -> pd.DataFrame:
    """Add in-sample Prophet components to a DataFrame."""
    out = df.copy()
    df_prophet = out[["date", "price"]].rename(columns={"date": "ds", "price": "y"})
    forecast = prophet_model.predict(df_prophet[["ds"]])

    out["prophet_trend"] = forecast["trend"].values
    out["prophet_yearly"] = forecast["yearly"].values if "yearly" in forecast.columns else 0
    out["prophet_yhat"] = forecast["yhat"].values
    return out


def prepare_training_frame(
    df: pd.DataFrame,
    prophet_model: Optional[Prophet] = None,
    prophet_params: Optional[Dict[str, Any]] = None,
) -> tuple[pd.DataFrame, Prophet]:
    """
    Build full feature matrix: technical features + Prophet components.

    If prophet_model is None, fits Prophet on df (caller must pass train-only data).
    """
    if prophet_model is None:
        prophet_model = fit_prophet(df, prophet_params)

    with_technical = build_technical_features(df)
    with_prophet = add_prophet_features(with_technical, prophet_model)
    return with_prophet, prophet_model


def build_prediction_row(
    last_row: pd.Series,
    current_price: float,
    future_date: Union[datetime, pd.Timestamp],
    prophet_model: Prophet,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame for multi-horizon inference."""
    if isinstance(future_date, pd.Timestamp):
        future_dt = future_date.to_pydatetime()
    else:
        future_dt = future_date

    future_prophet = prophet_model.predict(pd.DataFrame({"ds": [future_dt]}))
    prophet_trend_future = future_prophet["trend"].values[0]
    prophet_yearly_future = (
        future_prophet["yearly"].values[0] if "yearly" in future_prophet.columns else 0
    )
    prophet_yhat_future = future_prophet["yhat"].values[0]

    return pd.DataFrame(
        {
            "price_lag_1": [current_price],
            "price_lag_3": [last_row["price_lag_3"]],
            "price_lag_7": [last_row["price_lag_7"]],
            "price_lag_14": [last_row["price_lag_14"]],
            "price_lag_30": [last_row["price_lag_30"]],
            "price_ma_7": [last_row["price_ma_7"]],
            "price_ma_14": [last_row["price_ma_14"]],
            "price_ma_30": [last_row["price_ma_30"]],
            "price_ma_60": [last_row["price_ma_60"]],
            "price_std_7": [last_row["price_std_7"]],
            "price_std_30": [last_row["price_std_30"]],
            "price_change_1d": [last_row["price_change_1d"]],
            "price_change_7d": [last_row["price_change_7d"]],
            "price_change_30d": [last_row["price_change_30d"]],
            "prophet_trend": [prophet_trend_future],
            "prophet_yearly": [prophet_yearly_future],
            "prophet_yhat": [prophet_yhat_future],
            "year": [future_dt.year],
            "month": [future_dt.month],
            "day_of_week": [future_dt.weekday()],
            "day_of_year": [future_dt.timetuple().tm_yday],
            "quarter": [(future_dt.month - 1) // 3 + 1],
        }
    )


def future_business_date(origin_date: Union[datetime, pd.Timestamp], horizon: int) -> datetime:
    """Return origin + horizon business days (matches production inference)."""
    return (pd.to_datetime(origin_date) + BusinessDay(horizon)).to_pydatetime()


def price_at_date(df: pd.DataFrame, target_date: Union[datetime, pd.Timestamp]) -> Optional[float]:
    """Look up actual price on target_date; None if missing."""
    target = pd.Timestamp(target_date).normalize()
    matches = df[df["date"].dt.normalize() == target]
    if matches.empty:
        return None
    return float(matches["price"].iloc[-1])


def build_price_lookup(df: pd.DataFrame) -> pd.Series:
    """Build normalized-date -> price lookup for fast backtesting."""
    series = df.copy()
    series["date"] = pd.to_datetime(series["date"]).dt.normalize()
    return series.set_index("date")["price"]
