"""
Shared feature engineering for the hybrid Prophet + XGBoost cocoa price model.

Used by training, inference, and walk-forward validation to guarantee identical logic.

Feature sets (pour etude comparative / production) :
- FEATURE_COLS              : prix + Prophet (baseline)
- FEATURE_COLS_OHLCV        : + RSI, high-low, volume, momentum
- FEATURE_COLS_OI           : + open interest
- FEATURE_COLS_TERM         : + spreads d'echeances
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

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

FEATURE_COLS_OHLCV = [
    "rsi_14",
    "hl_range_pct",
    "return_1d",
    "realized_vol_7",
    "realized_vol_30",
    "volume",
    "volume_change_1d",
    "volume_ratio_30",
]

FEATURE_COLS_OI = [
    "open_interest",
    "oi_change_1d",
    "oi_change_7d",
    "return_oi_interaction",
]

FEATURE_COLS_TERM = [
    "spread_1_0",
    "spread_2_0",
    "spread_3_0",
    "curve_slope",
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


def resolve_feature_cols(
    include_ohlcv: bool = False,
    include_oi: bool = False,
    include_term: bool = False,
) -> List[str]:
    cols = list(FEATURE_COLS)
    if include_ohlcv:
        cols.extend(FEATURE_COLS_OHLCV)
    if include_oi:
        cols.extend(FEATURE_COLS_OI)
    if include_term:
        cols.extend(FEATURE_COLS_TERM)
    return cols


def load_price_data_from_supabase(
    supabase_client,
    min_date: str = "2020-01-01",
    table_name: str = "cocoa_prices",
    extra_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load and clean price data from Supabase (any market price table)."""
    preferred = list(extra_columns) if extra_columns is not None else [
        "open",
        "high",
        "low",
        "volume",
        "open_interest",
        "source",
    ]
    # Essayer d'abord le select complet, puis sans open_interest, puis prix seul
    select_candidates = [
        ", ".join(["date", "price"] + preferred),
        ", ".join(["date", "price"] + [c for c in preferred if c != "open_interest"]),
        "date, price",
    ]

    all_data = []
    page_size = 1000
    selected = None

    for select_clause in select_candidates:
        all_data = []
        offset = 0
        try:
            while True:
                response = (
                    supabase_client.table(table_name)
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
            selected = select_clause
            break
        except Exception:
            continue

    if selected is None:
        return clean_price_dataframe(pd.DataFrame(columns=["date", "price"]), min_date=min_date)

    df = pd.DataFrame(all_data)
    return clean_price_dataframe(df, min_date=min_date)


def load_term_structure_from_supabase(
    supabase_client,
    table_name: str = "cocoa_london_contracts",
    min_date: str = "2019-01-01",
) -> pd.DataFrame:
    """Charge les echeances et pivot en colonnes close_0..close_3."""
    all_data = []
    offset = 0
    page_size = 1000
    while True:
        try:
            response = (
                supabase_client.table(table_name)
                .select("date, contract_rank, close, volume, open_interest")
                .gte("date", min_date)
                .order("date")
                .range(offset, offset + page_size - 1)
                .execute()
            )
        except Exception:
            return pd.DataFrame()
        if not response.data:
            break
        all_data.extend(response.data)
        offset += page_size
        if len(response.data) < page_size:
            break

    if not all_data:
        return pd.DataFrame()

    raw = pd.DataFrame(all_data)
    raw["date"] = pd.to_datetime(raw["date"])
    pivot = raw.pivot_table(
        index="date", columns="contract_rank", values="close", aggfunc="last"
    )
    pivot = pivot.rename(columns={i: f"close_{i}" for i in pivot.columns})
    pivot = pivot.reset_index()
    return pivot


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


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, momentum, temporal, and optional OHLCV/OI/term features."""
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

    # --- OHLCV (uniquement si colonnes source presentes) ---
    has_ohlc = (
        all(c in out.columns for c in ("open", "high", "low"))
        and out[["open", "high", "low"]].notna().any().any()
    )
    has_volume = "volume" in out.columns and out["volume"].notna().any()
    has_oi = "open_interest" in out.columns and out["open_interest"].notna().any()
    has_term = any(f"close_{r}" in out.columns for r in (1, 2, 3))

    if has_ohlc or has_volume or has_oi or has_term:
        out["return_1d"] = out["price"].pct_change(1, fill_method=None)
        out["realized_vol_7"] = out["return_1d"].rolling(7).std()
        out["realized_vol_30"] = out["return_1d"].rolling(30).std()
        out["rsi_14"] = _compute_rsi(out["price"], 14)

    if has_ohlc:
        out["hl_range_pct"] = (out["high"] - out["low"]) / out["price"].replace(0, np.nan)

    if has_volume:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
        out["volume_change_1d"] = out["volume"].pct_change(1, fill_method=None)
        vol_ma = out["volume"].rolling(30).mean()
        out["volume_ratio_30"] = out["volume"] / vol_ma.replace(0, np.nan)

    if has_oi:
        out["open_interest"] = pd.to_numeric(out["open_interest"], errors="coerce")
        out["oi_change_1d"] = out["open_interest"].pct_change(1, fill_method=None)
        out["oi_change_7d"] = out["open_interest"].pct_change(7, fill_method=None)
        ret = out["return_1d"] if "return_1d" in out.columns else out["price"].pct_change(1, fill_method=None)
        out["return_oi_interaction"] = np.sign(ret.fillna(0)) * out["oi_change_1d"]

    if has_term:
        for rank in (1, 2, 3):
            col = f"close_{rank}"
            spread = f"spread_{rank}_0"
            if col in out.columns and "close_0" in out.columns:
                out[spread] = out[col] - out["close_0"]
            elif col in out.columns:
                out[spread] = out[col] - out["price"]
        if "spread_3_0" in out.columns:
            out["curve_slope"] = out["spread_3_0"] / 3.0

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
    feature_cols: Optional[Sequence[str]] = None,
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

    row: Dict[str, Any] = {
        "price_lag_1": current_price,
        "price_lag_3": last_row.get("price_lag_3"),
        "price_lag_7": last_row.get("price_lag_7"),
        "price_lag_14": last_row.get("price_lag_14"),
        "price_lag_30": last_row.get("price_lag_30"),
        "price_ma_7": last_row.get("price_ma_7"),
        "price_ma_14": last_row.get("price_ma_14"),
        "price_ma_30": last_row.get("price_ma_30"),
        "price_ma_60": last_row.get("price_ma_60"),
        "price_std_7": last_row.get("price_std_7"),
        "price_std_30": last_row.get("price_std_30"),
        "price_change_1d": last_row.get("price_change_1d"),
        "price_change_7d": last_row.get("price_change_7d"),
        "price_change_30d": last_row.get("price_change_30d"),
        "prophet_trend": prophet_trend_future,
        "prophet_yearly": prophet_yearly_future,
        "prophet_yhat": prophet_yhat_future,
        "year": future_dt.year,
        "month": future_dt.month,
        "day_of_week": future_dt.weekday(),
        "day_of_year": future_dt.timetuple().tm_yday,
        "quarter": (future_dt.month - 1) // 3 + 1,
    }

    optional = (
        FEATURE_COLS_OHLCV + FEATURE_COLS_OI + FEATURE_COLS_TERM
    )
    for col in optional:
        if col in last_row.index:
            row[col] = last_row.get(col)

    cols = list(feature_cols) if feature_cols is not None else FEATURE_COLS
    for col in cols:
        row.setdefault(col, last_row.get(col, 0.0))

    return pd.DataFrame({k: [row.get(k)] for k in cols})


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
