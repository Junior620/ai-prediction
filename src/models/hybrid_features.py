"""
Shared feature engineering for the hybrid Prophet + XGBoost cocoa price model.

Used by training, inference, and walk-forward validation to guarantee identical logic.

Feature sets (pour etude comparative / production) :
- FEATURE_COLS              : prix + Prophet (baseline)
- FEATURE_COLS_OHLCV        : + RSI, high-low, volume, momentum
- FEATURE_COLS_OI           : + open interest
- FEATURE_COLS_TERM         : + spreads d'echeances
- FEATURE_COLS_WEATHER      : cumuls, retards, anomalies, secheresse (candidat seul)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay as BusinessDay
from pandas.tseries.offsets import CustomBusinessDay
from prophet import Prophet

from src.models.weather_features import FEATURE_COLS_WEATHER

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

# Variables sans échelle pour les modèles directs : un niveau en £ ou une date
# laisse les arbres mémoriser un épisode (2024 à 4 000 £, « juillet »).
FEATURE_COLS_RELATIVE = [
    "price_lag_1_rel",
    "price_lag_3_rel",
    "price_lag_7_rel",
    "price_lag_14_rel",
    "price_lag_30_rel",
    "price_ma_7_rel",
    "price_ma_14_rel",
    "price_ma_30_rel",
    "price_ma_60_rel",
    "price_std_7_rel",
    "price_std_30_rel",
    "price_change_1d",
    "price_change_7d",
    "price_change_30d",
    "rsi_14",
    "hl_range_pct",
    "realized_vol_7",
    "realized_vol_30",
    "volume_ratio_30",
    "volume_change_1d",
    "oi_change_1d",
    "oi_change_7d",
    "month",
    "day_of_week",
]

# Arbres peu profonds et régularisés : avec six ans de données, une
# profondeur 8 apprend le bruit d'une seule année.
SHALLOW_XGB_PARAMS: Dict[str, Any] = {
    **DEFAULT_XGB_PARAMS,
    "n_estimators": 150,
    "max_depth": 3,
    "min_child_weight": 20,
    "reg_lambda": 10.0,
}

# Recette des modèles directs par marché. Un marché absent garde l'ancienne
# recette (variation en £, variables du lot, paramètres par défaut).
DIRECT_RECIPES: Dict[str, Dict[str, Any]] = {
    "cocoa": {
        "target": "log_return",
        "feature_cols": FEATURE_COLS_RELATIVE,
        "xgb_params": SHALLOW_XGB_PARAMS,
    },
}


def direct_recipe(market_id: str, feature_cols: Sequence[str]) -> Dict[str, Any]:
    """Cible, variables et paramètres des modèles directs de ce marché."""
    recipe = DIRECT_RECIPES.get(market_id)
    if recipe is None:
        return {
            "target": "close_delta",
            "feature_cols": list(feature_cols),
            "xgb_params": dict(DEFAULT_XGB_PARAMS),
        }
    return {
        "target": recipe["target"],
        "feature_cols": list(recipe["feature_cols"]),
        "xgb_params": dict(recipe["xgb_params"]),
    }


def resolve_feature_cols(
    include_ohlcv: bool = False,
    include_oi: bool = False,
    include_term: bool = False,
    include_weather: bool = False,
) -> List[str]:
    cols = list(FEATURE_COLS)
    if include_ohlcv:
        cols.extend(FEATURE_COLS_OHLCV)
    if include_oi:
        cols.extend(FEATURE_COLS_OI)
    if include_term:
        cols.extend(FEATURE_COLS_TERM)
    if include_weather:
        cols.extend(FEATURE_COLS_WEATHER)
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
    last_error: Optional[Exception] = None

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
        except Exception as exc:
            last_error = exc
            continue

    if selected is None:
        if last_error is not None:
            print(
                f"[WARN] Chargement {table_name} echoue (colonnes/API): {last_error}"
            )
        return clean_price_dataframe(pd.DataFrame(columns=["date", "price"]), min_date=min_date)

    if not all_data:
        print(f"[WARN] Table {table_name}: 0 lignes retournees (select={selected})")

    df = pd.DataFrame(all_data)
    cleaned = clean_price_dataframe(df, min_date=min_date)
    if table_name == "cocoa_london_prices":
        cleaned = apply_oi_receipts(cleaned)
    return cleaned


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


def apply_oi_receipts(frame: pd.DataFrame, ledger_path: Optional[Path] = None) -> pd.DataFrame:
    """Keep open interest only when a receipt for that contract was already in hand.

    The maximum open interest of another contract is never used. A session
    without ``ts_recv`` loses its open interest.
    """
    out = frame.copy()
    if "open_interest" not in out.columns:
        out["open_interest"] = np.nan
    path = ledger_path or (Path(__file__).resolve().parents[2] / "data" / "cocoa_oi_receipts.csv")
    if not path.exists():
        out["open_interest"] = np.nan
        return out
    ledger = pd.read_csv(path)
    if ledger.empty or "ts_recv" not in ledger.columns or "quantity" not in ledger.columns or "symbol" not in ledger.columns:
        out["open_interest"] = np.nan
        return out
    ledger = ledger[ledger["symbol"].astype(str) == "C.v.0"].copy()
    if ledger.empty:
        out["open_interest"] = np.nan
        return out
    ledger["ts_recv"] = pd.to_datetime(ledger["ts_recv"], utc=True, errors="coerce")
    ledger = ledger.dropna(subset=["ts_recv", "quantity"]).sort_values("ts_recv")
    ledger["available_on"] = ledger["ts_recv"].dt.tz_convert(None).dt.normalize()
    right = ledger.groupby("available_on", as_index=False)["quantity"].last()
    right = right.rename(columns={"available_on": "date", "quantity": "open_interest_received"})
    left = out.drop(columns=["open_interest"], errors="ignore")
    left["date"] = pd.to_datetime(left["date"]).dt.tz_localize(None).dt.normalize()
    merged = pd.merge_asof(
        left.sort_values("date"),
        right.sort_values("date"),
        on="date",
        direction="backward",
    )
    merged["open_interest"] = merged["open_interest_received"]
    return merged.drop(columns=["open_interest_received"])


def clean_price_dataframe(df: pd.DataFrame, min_date: str = "2020-01-01") -> pd.DataFrame:
    """Sort, drop duplicate dates, apply the minimum date, and drop impossible OHLC bars."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "price"])

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out = out[out["date"] >= min_date].copy()
    if len(out) == 0:
        return out.reset_index(drop=True)

    out = _drop_impossible_ohlc(out)
    return out.reset_index(drop=True)


def _drop_impossible_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop bars that cannot be a real session. No statistic uses future rows."""
    if not {"high", "low"}.issubset(frame.columns):
        return frame
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    price = pd.to_numeric(frame["price"], errors="coerce") if "price" in frame.columns else None
    impossible = high.notna() & low.notna() & (high < low)
    if price is not None:
        priced = price.notna() & high.notna() & low.notna()
        impossible = impossible | (priced & ((price > high) | (price < low)))
    return frame.loc[~impossible].copy()


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    # A window of only gains has a zero average loss. RSI is then 100, not missing.
    rs = avg_gain / avg_loss
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

    price = out["price"].replace(0, np.nan)
    for col in ("price_lag_1", "price_lag_3", "price_lag_7", "price_lag_14", "price_lag_30",
                "price_ma_7", "price_ma_14", "price_ma_30", "price_ma_60"):
        out[f"{col}_rel"] = out[col] / price - 1.0
    out["price_std_7_rel"] = out["price_std_7"] / price
    out["price_std_30_rel"] = out["price_std_30"] / price

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


def _finite_or_zero(value: Any) -> float:
    """Coerce a feature value to a finite float; missing or invalid becomes 0."""
    try:
        if value is None or pd.isna(value):
            return 0.0
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(out):
        return 0.0
    return out


def asof_feature_row(
    last_row: pd.Series,
    feature_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Feature row at the last observed session, identical to direct-horizon training.

    At date t the training row has ``price_lag_1`` = close[t-1], rolling stats
    that include close[t], and Prophet components on date t. Serving that row
    unchanged is the J+h forecast. Overwriting ``price_lag_1`` with today's
    close, or shifting Prophet to t+h, builds a vector the model never saw.
    """
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    return pd.DataFrame({col: [_finite_or_zero(last_row.get(col, 0.0))] for col in cols})


# Prophet's level sits far from the London close (changepoint_range locks the
# last 20% of the trend). Direct h-step models must not see that level, or a
# 30-day forecast mean-reverts toward it.
PROPHET_LEVEL_COLS = ("prophet_trend", "prophet_yearly", "prophet_yhat")


def direct_feature_row(
    last_row: pd.Series,
    feature_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """As-of feature row for direct h-step models, without the Prophet level."""
    row = asof_feature_row(last_row, feature_cols)
    for col in PROPHET_LEVEL_COLS:
        if col in row.columns:
            row[col] = 0.0
    return row


def guard_forecast(
    price: float,
    spot: float,
    max_pct: float,
    margin: Optional[tuple] = None,
) -> tuple[float, float, bool]:
    """Central scenario after the percentage cap.

    A move past ``max_pct`` stays visible and is flagged. The conformal band
    is not a reason to discard the price.
    ``margin`` is accepted so older callers keep working and is ignored.
    """
    del margin
    return resolve_capped_forecast(price, spot, max_pct)


def resolve_capped_forecast(
    price: float,
    spot: float,
    max_pct: float,
) -> tuple[float, float, bool]:
    """Return ``(central_price, change_pct, large_move_warning)``.

    A finite positive price past ``max_pct`` stays visible. The third value
    is a warning, not a reason to replace the forecast with the spot.
    """
    if spot <= 0 or not np.isfinite(price) or not np.isfinite(spot):
        return float(price), 0.0, False
    if not (np.isfinite(price) and float(price) > 0):
        return float(price), 0.0, False
    raw_pct = (float(price) / float(spot) - 1.0) * 100.0
    warning = abs(raw_pct) > float(max_pct) + 1e-9
    return float(price), float(raw_pct), bool(warning)


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


CONSUMED_COLUMNS = ("open", "high", "low", "volume", "open_interest")
# The October 2026 cocoa snapshot keeps closed dates the study drops.
# Removing them changes price_lag_1 and the served J+7. Lots already trained
# keep those rows. The calendar name is part of the procedure fingerprint.
SERVICE_CALENDAR = "keep_closed_rows"
OHLCV_FILL = "last_observation_before_features"
_OHLCV_COLUMNS = ("open", "high", "low", "volume")


def is_closed_collection_date(day, holidays: Optional[Sequence] = None) -> bool:
    """A weekend or a date the project calendar lists as closed."""
    stamp = pd.Timestamp(day).normalize()
    if stamp.weekday() >= 5:
        return True
    closed = {pd.Timestamp(item).normalize() for item in exchange_holidays(holidays)}
    return stamp in closed


def session_anomalies(frame: pd.DataFrame) -> list[str]:
    """Dates present in a snapshot that are not trading sessions. Rows are kept."""
    if frame is None or len(frame) == 0 or "date" not in frame.columns:
        return []
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna().dt.normalize()
    return sorted(
        {stamp.strftime("%Y-%m-%d") for stamp in dates if is_closed_collection_date(stamp)}
    )


def session_inventory(frame: pd.DataFrame, start: str, end: str) -> dict:
    """Closed dates and missing sessions. The frame itself is not changed."""
    present = set()
    if frame is not None and len(frame) and "date" in frame.columns:
        present = {
            pd.Timestamp(day).strftime("%Y-%m-%d")
            for day in pd.to_datetime(frame["date"], errors="coerce").dropna()
        }
    window_start = pd.Timestamp(start).normalize()
    window_end = pd.Timestamp(end).normalize()
    expected: list[str] = []
    cursor = window_start
    while cursor <= window_end:
        if not is_closed_collection_date(cursor):
            expected.append(cursor.strftime("%Y-%m-%d"))
        cursor += pd.Timedelta(days=1)
    closed = session_anomalies(frame)
    return {
        "rows": 0 if frame is None else int(len(frame)),
        "closed_dates": closed,
        "window_start": window_start.strftime("%Y-%m-%d"),
        "window_end": window_end.strftime("%Y-%m-%d"),
        "expected_sessions": expected,
        "missing_sessions": [day for day in expected if day not in present],
        "closed_dates_in_window": [
            day for day in closed if window_start.strftime("%Y-%m-%d") <= day <= window_end.strftime("%Y-%m-%d")
        ],
    }


def trading_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    """Cotation sessions only. Closed collection rows stay in the source file."""
    if frame is None or len(frame) == 0 or "date" not in frame.columns:
        return pd.DataFrame(columns=["date", "price"]) if frame is None else frame.iloc[0:0].copy()
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    closed = out["date"].map(is_closed_collection_date)
    return out.loc[~closed].reset_index(drop=True)


def fill_missing_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing open, high, low and volume from the previous observation.

    This runs before features. Open interest stays as collected. The raw
    snapshot identity is taken before this fill.
    """
    filled = frame.copy()
    for column in _OHLCV_COLUMNS:
        if column in filled.columns:
            filled[column] = pd.to_numeric(filled[column], errors="coerce").ffill()
    return filled


def prepare_market_frame(
    frame: pd.DataFrame,
    *,
    market_id: str,
    ledger_path: Optional[Path] = None,
    roll_series_path: Optional[Path] = None,
) -> pd.DataFrame:
    """One cleaning path for training, the journal command and the API.

    Closed dates stay. ``trading_sessions`` is the separate study calendar.
    Cocoa prices come from the series without contract jumps
    (``src/data_collection/cocoa_roll.py``).
    """
    cleaned = clean_price_dataframe(frame)
    if market_id == "cocoa":
        cleaned = apply_oi_receipts(cleaned, ledger_path)
        cleaned = _apply_cocoa_roll(cleaned, roll_series_path)
    return cleaned


def _apply_cocoa_roll(frame: pd.DataFrame, roll_series_path: Optional[Path]) -> pd.DataFrame:
    from loguru import logger

    from src.data_collection.cocoa_roll import SERIES_PATH, apply_roll_series, load_series

    path = Path(roll_series_path) if roll_series_path is not None else SERIES_PATH
    series = load_series(path)
    if series is None:
        # Sans la série, les sauts entre échéances de C.v.0 restent dans les prix.
        logger.error(f"Série cacao sans saut absente ({path}) : prix C.v.0 bruts utilisés")
        return frame
    return apply_roll_series(frame, series, is_closed=is_closed_collection_date)


def model_inputs(
    frame: pd.DataFrame,
    *,
    market_id: str,
    ledger_path: Optional[Path] = None,
    roll_series_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw prepared snapshot, then the filled frame the model actually reads."""
    raw = prepare_market_frame(
        frame, market_id=market_id, ledger_path=ledger_path, roll_series_path=roll_series_path
    )
    return raw, fill_missing_ohlcv(raw)


def consumed_snapshot_rows(frame: pd.DataFrame) -> list[dict]:
    """Normalized columns the models actually read. A value change alters the identity."""
    if frame is None or len(frame) == 0 or "date" not in frame.columns:
        return []
    ordered = frame.sort_values("date")
    rows = []
    for _, row in ordered.iterrows():
        price = row.get("price")
        item = {
            "date": str(pd.to_datetime(row["date"]).date()),
            "price": None if price is None or pd.isna(price) else float(price),
        }
        for column in CONSUMED_COLUMNS:
            if column not in ordered.columns or pd.isna(row.get(column)):
                item[column] = None
            else:
                item[column] = float(row[column])
        rows.append(item)
    return rows


_HOLIDAY_CACHE: Optional[tuple] = None
_RULE_CACHE: Dict[tuple, Any] = {}


def exchange_holidays(extra: Optional[Sequence] = None) -> list:
    """Venue closures. A test can pass extra dates without editing the file."""
    global _HOLIDAY_CACHE
    path = Path(__file__).resolve().parents[2] / "config" / "exchange_holidays.json"
    stamp = path.stat().st_mtime_ns if path.exists() else None
    if extra is None and _HOLIDAY_CACHE is not None and _HOLIDAY_CACHE[0] == stamp:
        return list(_HOLIDAY_CACHE[1])
    closed: list = []
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        closed.extend(payload.get("dates") or [])
    if extra:
        closed.extend(extra)
    result = [pd.Timestamp(day).normalize() for day in closed]
    if extra is None:
        _HOLIDAY_CACHE = (stamp, result)
    return list(result)


def _business_rule(closed: Sequence):
    key = tuple(pd.Timestamp(day).value for day in closed)
    rule = _RULE_CACHE.get(key)
    if rule is None:
        rule = CustomBusinessDay(holidays=list(closed)) if closed else BusinessDay()
        _RULE_CACHE[key] = rule
    return rule


def future_business_date(
    origin_date: Union[datetime, pd.Timestamp],
    horizon: int,
    holidays: Optional[Sequence] = None,
) -> datetime:
    """Origin plus horizon sessions, skipping weekends and listed closures."""
    closed = exchange_holidays(holidays)
    rule = _business_rule(closed)
    return (pd.to_datetime(origin_date) + int(horizon) * rule).to_pydatetime()


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
