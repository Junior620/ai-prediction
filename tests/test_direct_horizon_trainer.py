"""Tests for direct h-step horizon training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.direct_horizon_trainer import DirectHorizonTrainer
from src.models.hybrid_features import (
    asof_feature_row,
    future_business_date,
    guard_forecast,
    resolve_capped_forecast,
)


def test_direct_h1_target_is_next_session():
    dates = pd.bdate_range("2020-01-01", periods=80)
    prices = np.linspace(3000, 3200, 80)
    df = pd.DataFrame({"date": dates, "price": prices})

    trainer = DirectHorizonTrainer(horizons=[1])
    X_df, y = trainer._build_direct_dataset(df, horizon=1)

    idx = X_df.index[10]
    origin = pd.Timestamp(df.loc[idx, "date"])
    target_date = pd.Timestamp(future_business_date(origin, 1)).normalize()
    expected = float(df[df["date"].dt.normalize() == target_date]["price"].iloc[0])
    origin_price = float(df.loc[idx, "price"])
    assert y.loc[idx] == expected - origin_price
    assert float(X_df.loc[idx, "price_lag_1"]) == float(df.loc[idx - 1, "price"])


def test_default_horizons_include_one_day():
    assert DirectHorizonTrainer().horizons == [1, 7, 14, 30]


def test_direct_dataset_target_alignment():
    dates = pd.bdate_range("2020-01-01", periods=80)
    prices = np.linspace(3000, 3200, 80)
    df = pd.DataFrame({"date": dates, "price": prices})

    trainer = DirectHorizonTrainer(horizons=[7])
    X_df, y = trainer._build_direct_dataset(df, horizon=7)

    assert len(X_df) == len(y)
    assert len(X_df) > 0

    idx = X_df.index[0]
    origin = df.loc[idx, "date"]
    target_date = pd.Timestamp(future_business_date(origin, 7)).normalize()
    expected = float(df[df["date"].dt.normalize() == target_date]["price"].iloc[0])
    origin_price = float(df.loc[idx, "price"])
    assert y.loc[idx] == expected - origin_price


def test_fit_and_save_roundtrip(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=100)
    prices = np.linspace(3000, 3300, 100)
    df = pd.DataFrame({"date": dates, "price": prices})

    trainer = DirectHorizonTrainer(horizons=[7])
    models, meta = trainer.fit(df)
    paths = trainer.save(models, meta, models_dir=str(tmp_path))

    loaded = DirectHorizonTrainer.load_latest(str(tmp_path))
    assert 7 in loaded


def test_asof_feature_row_keeps_training_lag():
    last = pd.Series(
        {
            "price_lag_1": 4000.0,
            "price_ma_7": 4100.0,
            "prophet_yhat": 4050.0,
            "open_interest": None,
        }
    )
    row = asof_feature_row(last, feature_cols=["price_lag_1", "price_ma_7", "prophet_yhat", "open_interest"])
    assert float(row["price_lag_1"].iloc[0]) == 4000.0
    assert float(row["price_ma_7"].iloc[0]) == 4100.0
    assert float(row["prophet_yhat"].iloc[0]) == 4050.0
    assert float(row["open_interest"].iloc[0]) == 0.0


def test_cap_breach_keeps_spot_as_central_scenario():
    central, change, failed = resolve_capped_forecast(3350.0, 4149.0, 14.0)
    assert failed is True
    assert central == 4149.0
    assert change == 0.0

    central, change, failed = resolve_capped_forecast(4100.0, 4149.0, 6.0)
    assert failed is False
    assert central == 4100.0
    assert change < 0


def test_conformal_margin_does_not_zero_a_small_j1_move():
    # +110 £ on a 4364 close is outside the ±71 J+1 band, but inside 2× that band.
    central, change, failed = guard_forecast(4474.0, 4364.0, 6.0, (71.0, 71.0))
    assert failed is False
    assert central == 4474.0
    assert change > 0


def test_j30_inside_the_cap_is_not_zeroed_by_the_band():
    # −807 £ is inside the 22% cap. The band no longer discards it.
    central, change, failed = guard_forecast(3557.0, 4364.0, 22.0, (196.0, 196.0))
    assert failed is False
    assert central == 3557.0
    assert change < 0


def test_cap_breach_still_keeps_the_close():
    central, change, failed = guard_forecast(3458.0, 4390.0, 14.0, (196.0, 196.0))
    assert failed is True
    assert central == 4390.0
    assert change == 0.0


def test_direct_dataset_zeros_prophet_level():
    dates = pd.bdate_range("2020-01-01", periods=80)
    prices = np.linspace(3000, 3200, 80)
    df = pd.DataFrame({"date": dates, "price": prices})
    X_df, _ = DirectHorizonTrainer(horizons=[1])._build_direct_dataset(df, horizon=1)
    assert float(X_df["prophet_yhat"].iloc[-1]) == 0.0
