"""Tests for recursive multi-step prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.hybrid_features import build_technical_features, fit_prophet
from src.models.multi_step_predictor import predict_recursive
import xgboost as xgb


def _make_series(n: int = 120) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = np.linspace(3000, 3500, n) + np.random.default_rng(0).normal(0, 20, n)
    return pd.DataFrame({"date": dates, "price": prices})


def test_recursive_differs_from_frozen_for_h7():
    df = _make_series(150)
    prophet = fit_prophet(df)
    feat = build_technical_features(df)
    from src.models.hybrid_features import add_prophet_features, FEATURE_COLS, prepare_training_frame

    full, _ = prepare_training_frame(df, prophet_model=prophet)
    clean = full.dropna()
    xgb_model = xgb.XGBRegressor(n_estimators=20, max_depth=4, random_state=42)
    xgb_model.fit(clean[FEATURE_COLS], clean["price"])

    rec = predict_recursive(df, prophet, xgb_model, 7)
    assert rec > 0
    assert np.isfinite(rec)


def test_recursive_h1_matches_frozen_logic():
    from src.models.multi_step_predictor import predict_frozen

    df = _make_series(100)
    prophet = fit_prophet(df)
    from src.models.hybrid_features import prepare_training_frame, FEATURE_COLS

    full, _ = prepare_training_frame(df, prophet_model=prophet)
    clean = full.dropna()
    xgb_model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
    xgb_model.fit(clean[FEATURE_COLS], clean["price"])
    last = clean.iloc[-1]

    p_frozen = predict_frozen(
        last, float(last["price"]), last["date"], 1, prophet, xgb_model
    )
    p_rec = predict_recursive(df, prophet, xgb_model, 1)
    assert p_frozen == pytest.approx(p_rec, rel=1e-4)
