"""Lagged cocoa-belt weather. Every column at date t uses observations through t."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

FEATURE_COLS_WEATHER = [
    "rain_sum_7",
    "rain_sum_30",
    "rain_sum_90",
    "temp_ma_7",
    "temp_ma_30",
    "rain_lag_1",
    "rain_lag_7",
    "rain_lag_14",
    "rain_lag_30",
    "temp_lag_1",
    "temp_lag_7",
    "temp_lag_14",
    "temp_lag_30",
    "rain_anomaly",
    "temp_anomaly",
    "drought_index",
]


def build_weather_features(
    weather: pd.DataFrame,
    climatology_end: Union[str, pd.Timestamp],
    availability_lag_days: int = 3,
) -> pd.DataFrame:
    """Rolling rain, temperature, lags, anomalies and a 90-day drought index.

    The day-of-year normals are estimated only on dates strictly before
    ``climatology_end``. A later rain reading cannot move an earlier row.
    """
    frame = weather.copy()
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    frame["precip_mm"] = pd.to_numeric(frame["precip_mm"], errors="coerce")
    frame["temp_c"] = pd.to_numeric(frame["temp_c"], errors="coerce")
    frame["doy"] = frame["date"].dt.dayofyear.clip(upper=365)

    frame["rain_sum_7"] = frame["precip_mm"].rolling(7, min_periods=7).sum()
    frame["rain_sum_30"] = frame["precip_mm"].rolling(30, min_periods=30).sum()
    frame["rain_sum_90"] = frame["precip_mm"].rolling(90, min_periods=90).sum()
    frame["temp_ma_7"] = frame["temp_c"].rolling(7, min_periods=7).mean()
    frame["temp_ma_30"] = frame["temp_c"].rolling(30, min_periods=30).mean()
    for lag in (1, 7, 14, 30):
        frame[f"rain_lag_{lag}"] = frame["precip_mm"].shift(lag)
        frame[f"temp_lag_{lag}"] = frame["temp_c"].shift(lag)

    cutoff = pd.Timestamp(climatology_end).normalize()
    history = frame[frame["date"] < cutoff]
    rain_norm = history.groupby("doy")["precip_mm"].mean()
    temp_norm = history.groupby("doy")["temp_c"].mean()
    rain90_mean = history.groupby("doy")["rain_sum_90"].mean()
    rain90_std = history.groupby("doy")["rain_sum_90"].std(ddof=0).replace(0, np.nan)

    frame["rain_anomaly"] = frame["precip_mm"] - frame["doy"].map(rain_norm)
    frame["temp_anomaly"] = frame["temp_c"] - frame["doy"].map(temp_norm)
    frame["drought_index"] = (
        frame["rain_sum_90"] - frame["doy"].map(rain90_mean)
    ) / frame["doy"].map(rain90_std)
    lag = max(0, int(availability_lag_days))
    if lag:
        frame["date"] = frame["date"] + pd.Timedelta(days=lag)
    return frame.reset_index(drop=True)


def attach_weather(
    prices: pd.DataFrame,
    weather_features: pd.DataFrame,
) -> pd.DataFrame:
    """Join the latest weather on or before each price date. Never the next day."""
    left = prices.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.normalize()
    left = left.sort_values("date")
    if weather_features.empty:
        for column in FEATURE_COLS_WEATHER:
            left[column] = np.nan
        return left
    right = weather_features.copy()
    right["date"] = pd.to_datetime(right["date"]).dt.normalize()
    keep = ["date", *[column for column in FEATURE_COLS_WEATHER if column in right.columns]]
    right = right[keep].sort_values("date")
    merged = pd.merge_asof(left, right, on="date", direction="backward")
    return merged
