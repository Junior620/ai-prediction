"""Checks for the measurement-reliability corrections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.hybrid_features import clean_price_dataframe
from src.models.hybrid_trainer import next_session_frame
from src.monitoring.performance_monitor import PerformanceMonitor
from src.validation.served_backtest import (
    beats_naive_on_all_horizons,
    split_chronological,
)
from src.validation.walk_forward_validator import WalkForwardConfig, WalkForwardValidator, drift_price
from tests.test_walk_forward_validator import make_synthetic_prices


def test_recent_origins_are_the_last_ones():
    df = make_synthetic_prices(400)
    base = dict(horizons=[1], min_train_days=100, step_size=10)
    all_indices = WalkForwardValidator(WalkForwardConfig(**base))._origin_indices(len(df), 1)
    recent = WalkForwardValidator(
        WalkForwardConfig(**base, max_origins=5, origin_window="recent")
    )._origin_indices(len(df), 1)
    historical = WalkForwardValidator(
        WalkForwardConfig(**base, max_origins=5, origin_window="historical")
    )._origin_indices(len(df), 1)
    assert recent == all_indices[-5:]
    assert historical == all_indices[:5]
    assert recent[0] > historical[-1]


def test_drift_uses_only_the_past():
    prices = pd.Series([100.0, 110.0, 90.0])
    # Last known close is 90. Mean of the two past returns is not taken from a future bar.
    projected = drift_price(prices.iloc[:2], horizon=1, lookback=20)
    assert projected == pytest.approx(121.0)
    assert drift_price(prices, horizon=1, lookback=20) != pytest.approx(projected)


def test_next_session_target_is_the_following_close():
    frame = pd.DataFrame({"price": [10.0, 12.0, 11.0], "price_change_1d": [0.0, 0.2, -1 / 12]})
    out = next_session_frame(frame)
    assert list(out["price"]) == [10.0, 12.0]
    assert list(out["target_price"]) == [12.0, 11.0]
    assert out["target_price"].iloc[0] != out["price"].iloc[0]


def test_clean_price_keeps_extremes_and_drops_impossible_bars():
    dates = pd.bdate_range("2024-01-02", periods=6)
    prices = [100.0, 101.0, 102.0, 100.0, 5000.0, 103.0]
    frame = pd.DataFrame(
        {
            "date": dates,
            "price": prices,
            "high": [101, 102, 103, 101, 5100, 90],
            "low": [99, 100, 101, 99, 4900, 104],
        }
    )
    cleaned = clean_price_dataframe(frame, min_date="2020-01-01")
    assert 5000.0 in set(cleaned["price"])
    assert 103.0 not in set(cleaned["price"])
    assert len(cleaned) == 5


def test_direction_uses_the_origin_price():
    monitor = PerformanceMonitor(supabase_client=object())
    y_true = np.array([110.0, 105.0])
    y_pred = np.array([101.0, 108.0])
    lower = np.array([90.0, 90.0])
    upper = np.array([130.0, 130.0])
    versus_origin = monitor.compute_metrics(
        y_true, y_pred, lower, upper, origin_price=np.array([100.0, 100.0])
    )
    successive = monitor.compute_metrics(y_true, y_pred, lower, upper)
    assert versus_origin["directional_accuracy"] == pytest.approx(1.0)
    assert successive["directional_accuracy"] == pytest.approx(0.0)


def test_held_out_slice_must_beat_the_unchanged_close():
    frame = pd.DataFrame(
        {
            "origin_date": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"],
            "target_date": ["2024-01-02", "2024-01-10", "2024-02-02", "2024-02-12"],
            "horizon": [1, 7, 1, 7],
            "origin_price": [100.0, 100.0, 100.0, 100.0],
            "actual": [110.0, 120.0, 90.0, 80.0],
            "published_pred": [108.0, 115.0, 92.0, 85.0],
            "naive_pred": [100.0, 100.0, 100.0, 100.0],
        }
    )
    _cal, test = split_chronological(frame)
    # Horizons 14 and 30 are absent, so the slice is not accepted.
    assert beats_naive_on_all_horizons(test, "published_pred") is False
    worse = test.copy()
    worse["published_pred"] = 1000.0
    assert beats_naive_on_all_horizons(worse, "published_pred") is False


def test_evaluation_follows_the_candidate_price_not_the_published_close():
    from evaluate_predictions import scored_forecast

    price, source = scored_forecast({"predicted_price": 100.0, "candidate_price": 110.0})
    assert price == 110.0
    assert source == "candidat"
    published, source = scored_forecast({"predicted_price": 100.0, "candidate_price": None})
    assert published == 100.0
    assert source == "publie"


def test_unavailable_card_does_not_present_hold():
    source = Path("frontend/src/components/dashboard/PredictionHorizonCard.tsx").read_text(
        encoding="utf-8"
    )
    start = source.index("if (unavailable)")
    unavailable = source[start:].split("return (")[1].split(");")[0]
    assert "Prévision indisponible" in unavailable
    assert "HOLD" not in unavailable
    assert "Confiance" not in source
    assert "Probabilité" not in source
