"""Checks for the measurement-reliability corrections."""

from __future__ import annotations

import json
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
    all_indices = WalkForwardValidator(WalkForwardConfig(**base))._origin_indices(df["date"], 1)
    recent = WalkForwardValidator(
        WalkForwardConfig(**base, max_origins=5, origin_window="recent")
    )._origin_indices(df["date"], 1)
    historical = WalkForwardValidator(
        WalkForwardConfig(**base, max_origins=5, origin_window="historical")
    )._origin_indices(df["date"], 1)
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
    assert "Indisponible" in unavailable
    assert "HOLD" not in unavailable
    assert "Confiance" not in source
    assert "Probabilité" not in source


def test_volume_change_changes_the_snapshot_fingerprint():
    from src.models.hybrid_features import consumed_snapshot_rows
    from src.models.release_manifest import data_fingerprint

    frame = pd.DataFrame(
        {"date": pd.to_datetime(["2026-01-02"]), "price": [100.0], "volume": [10.0], "open_interest": [1.0]}
    )
    changed = frame.copy()
    changed["volume"] = 11.0
    assert data_fingerprint(consumed_snapshot_rows(frame)) != data_fingerprint(consumed_snapshot_rows(changed))


def test_same_day_price_correction_invalidates_the_predictor_cache():
    from src.models.improved_price_predictor import historical_cache_is_fresh

    assert historical_cache_is_fresh(10, "aaa", "aaa") is True
    assert historical_cache_is_fresh(10, "aaa", "bbb") is False
    assert historical_cache_is_fresh(10, "aaa", None) is False
    assert historical_cache_is_fresh(3600, "aaa", "aaa") is False


def test_cocoa_training_and_service_share_open_interest(tmp_path: Path):
    from src.models.hybrid_features import apply_oi_receipts, prepare_market_frame

    ledger = tmp_path / "cocoa_oi_receipts.csv"
    ledger.write_text(
        "ts_recv,quantity,symbol\n2026-01-02T12:00:00Z,10,C.v.0\n2026-01-06T12:00:00Z,12,C.v.0\n",
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "price": [100.0, 101.0, 102.0],
            "open_interest": [1.0, 2.0, 3.0],
        }
    )
    trained = apply_oi_receipts(frame, ledger)
    served = prepare_market_frame(frame, market_id="cocoa", ledger_path=ledger)
    assert list(trained["open_interest"]) == list(served["open_interest"])
    assert int(trained["open_interest"].notna().sum()) == int(served["open_interest"].notna().sum())


def test_closed_saturday_is_flagged_and_is_not_a_session():
    from src.models.hybrid_features import session_anomalies
    from src.validation.revision_stability import study_window

    dates = list(pd.bdate_range("2026-08-03", periods=5))
    dates.append(pd.Timestamp("2026-08-29"))
    frame = pd.DataFrame({"date": dates, "price": range(len(dates))})
    assert "2026-08-29" in session_anomalies(frame)
    _origins, sessions, _missing = study_window(frame, as_of="2026-10-06", n_origins=60, horizon=1)
    assert pd.Timestamp("2026-08-29") not in set(sessions)


def test_missing_target_does_not_replace_the_origin():
    from src.models.hybrid_features import future_business_date
    from src.validation.revision_stability import study_window

    dates = pd.bdate_range("2024-01-02", periods=80)
    frame = pd.DataFrame({"date": dates, "price": range(len(dates))})
    origins, _sessions, _missing = study_window(
        frame, as_of=str(dates[-1].date()), n_origins=10, horizon=5
    )
    target = pd.Timestamp(future_business_date(origins[-1], 5)).normalize()
    cut = frame[pd.to_datetime(frame["date"]).dt.normalize() != target].reset_index(drop=True)
    origins_after, _sessions_after, missing = study_window(
        cut, as_of=str(dates[-1].date()), n_origins=10, horizon=5
    )
    assert origins_after[-1] == origins[-1]
    assert missing


def test_procedure_fingerprint_covers_a_referenced_direct_model(tmp_path: Path):
    from src.models.release_manifest import procedure_fingerprint

    code = tmp_path / "src" / "models"
    code.mkdir(parents=True)
    for name in (
        "hybrid_features.py",
        "weather_features.py",
        "served_forecast.py",
        "nhits_replay.py",
        "direct_horizon_trainer.py",
    ):
        (code / name).write_text("x = 1\n", encoding="utf-8")
    validation = tmp_path / "src" / "validation"
    validation.mkdir(parents=True)
    (validation / "walk_forward_validator.py").write_text("x = 1\n", encoding="utf-8")
    (validation / "acceptance.py").write_text("x = 1\n", encoding="utf-8")
    (validation / "revision_stability.py").write_text("x = 1\n", encoding="utf-8")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_revision_stability.py").write_text("x = 1\n", encoding="utf-8")
    config = tmp_path / "config"
    config.mkdir()
    (config / "acceptance.json").write_text("{}\n", encoding="utf-8")
    (config / "exchange_holidays.json").write_text("{}\n", encoding="utf-8")
    model = tmp_path / "direct.pkl"
    model.write_bytes(b"model-a")
    info = tmp_path / "direct_info.json"
    info.write_text(json.dumps({"horizons": {"1": {"model_path": "direct.pkl"}}}), encoding="utf-8")
    first = procedure_fingerprint(root=tmp_path, artifact_paths={"direct_info": "direct_info.json"})
    model.write_bytes(b"model-b")
    second = procedure_fingerprint(root=tmp_path, artifact_paths={"direct_info": "direct_info.json"})
    assert first != second


def test_resume_matches_a_continuous_run():
    from src.models.prediction_journal import count_inversions
    from src.validation.revision_stability import replay_policies

    dates = pd.bdate_range("2026-01-05", periods=8)
    frame = pd.DataFrame({"date": dates, "price": [100 + index for index in range(8)]})

    def history_for(session):
        return frame[pd.to_datetime(frame["date"]) <= pd.Timestamp(session)].copy()

    def train_fn(history):
        return {"stamp": float(history["price"].iloc[-1]), "rows": len(history)}

    def price_fn(bundle, history, origin_price, session, horizon):
        current = float(history["price"].iloc[-1])
        return bundle["stamp"] * 0.1 + current + horizon

    def direction_fn(price, reference):
        if price is None or not reference:
            return None
        change = (float(price) / float(reference) - 1.0) * 100.0
        if change > 1.5:
            return "hausse"
        if change < -1.5:
            return "baisse"
        return "neutre"

    continuous = replay_policies(
        dates,
        history_for,
        train_fn,
        price_fn,
        official_horizons=(1, 7),
        direction_fn=direction_fn,
    )
    interrupted = replay_policies(
        dates,
        history_for,
        train_fn,
        price_fn,
        official_horizons=(1, 7),
        direction_fn=direction_fn,
        done_sessions={batch["session"] for batch in continuous[:-1]},
        prior_records=[row for batch in continuous[:-1] for row in batch["rows"]],
    )
    assert interrupted[0]["rows"] == continuous[-1]["rows"]
    assert count_inversions(["hausse", "neutre", "baisse"]) == 1


def test_custom_business_day_is_replaced_before_nhits_save(tmp_path):
    from pandas.tseries.offsets import CustomBusinessDay

    from src.models.nhits_replay import restore_exchange_calendar, strip_custom_business_day

    class Holder:
        def __init__(self):
            self.freq = CustomBusinessDay(holidays=["2026-08-29"])
            self.nested = {"freq": CustomBusinessDay(holidays=["2026-09-12"])}

    holder = Holder()
    strip_custom_business_day(holder)
    assert holder.freq == "B"
    assert holder.nested["freq"] == "B"

    class Loaded:
        def __init__(self):
            self.freq = "B"
            self.models = []

    sidecar = tmp_path / "freq_holidays.json"
    sidecar.write_text(json.dumps({"holidays": ["2026-08-29"]}), encoding="utf-8")
    loaded = Loaded()
    restore_exchange_calendar(loaded, tmp_path)
    assert type(loaded.freq).__name__ == "CustomBusinessDay"
