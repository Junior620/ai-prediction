"""Acceptance rules and the audit reproductions they close."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.hybrid_features import (
    build_technical_features,
    future_business_date,
    _compute_rsi,
)
from src.models.release_manifest import (
    artifact_path,
    load_active_release,
    promote_accepted_horizons,
)
from src.models.served_forecast import compose_served_price, publish_or_close, snapshot_journal_fields
from src.validation.report_loader import find_release_summary
from src.validation.acceptance import (
    effective_sample_size,
    evaluate_horizon,
    load_acceptance,
    moving_block_ci,
    temporal_overlap,
)
from src.validation.served_backtest import (
    beats_naive_on_all_horizons,
    calibration_respects_cutoff,
    split_chronological,
)


def _frame(rows):
    return pd.DataFrame(rows)


def test_a_missing_prediction_does_not_beat_the_close_by_disappearing():
    rules = load_acceptance()
    frame = _frame(
        {
            "origin_date": ["2024-01-02", "2024-01-03"],
            "target_date": ["2024-01-03", "2024-01-04"],
            "horizon": [1, 1],
            "origin_price": [100.0, 100.0],
            "actual": [101.0, 200.0],
            "naive_pred": [100.0, 100.0],
            "published_pred": [102.0, np.nan],
            "feature_failure": [False, True],
        }
    )
    decision = evaluate_horizon(frame, 1, rules, step=5)
    # Both rows count. The second published price is the close, so the model
    # cannot look accurate by dropping the hard case.
    assert decision["n"] == 2
    assert decision["validated"] is False
    assert beats_naive_on_all_horizons(frame, "published_pred") is False


def test_a_two_hundredths_gain_is_not_enough():
    rules = dict(load_acceptance())
    rules["bootstrap_replicates"] = 50
    # 2.41 versus 2.43 is a real gap and still below the 10% relative minimum.
    actual = np.full(30, 100.0)
    naive = np.full(30, 97.57)
    published = np.full(30, 97.59)
    frame = _frame(
        {
            "origin_date": pd.date_range("2024-01-02", periods=30, freq="B"),
            "target_date": pd.date_range("2024-01-03", periods=30, freq="B"),
            "horizon": 1,
            "origin_price": naive,
            "actual": actual,
            "naive_pred": naive,
            "published_pred": published,
            "feature_failure": False,
        }
    )
    decision = evaluate_horizon(frame, 1, rules, step=5)
    assert decision["relative_gain"] < 0.10
    assert decision["validated"] is False


def test_block_interval_and_effective_sample_size():
    assert temporal_overlap(1, 5) == 0
    assert temporal_overlap(7, 5) == 2
    assert temporal_overlap(14, 5) == 3
    assert temporal_overlap(30, 5) == 6
    assert effective_sample_size(40, 30, 5) < 10
    low, high = moving_block_ci([2.0] * 20, length=4, replicates=40, seed=1)
    assert low > 0 and high > 0


def test_calibration_targets_stay_before_the_test_origin():
    origins = pd.bdate_range("2021-01-04", periods=8, freq="B")
    rows = []
    for origin in origins:
        for horizon, shift in ((1, 1), (30, 30)):
            rows.append(
                {
                    "origin_date": origin,
                    "target_date": origin + pd.offsets.BDay(shift),
                    "horizon": horizon,
                    "origin_price": 100.0,
                    "actual": 101.0,
                    "naive_pred": 100.0,
                    "published_pred": 100.0,
                }
            )
    frame = pd.DataFrame(rows)
    calibration, test = split_chronological(frame)
    cutoff = pd.to_datetime(test["origin_date"]).min()
    assert pd.to_datetime(calibration["target_date"]).max() < cutoff
    assert calibration_respects_cutoff(frame)
    assert (calibration["horizon"] == 30).sum() < (frame["horizon"] == 30).sum() / 2


def test_width_cap_rejects_a_band_wider_than_the_close():
    rules = dict(load_acceptance())
    rules["bootstrap_replicates"] = 40
    rules["min_relative_mape_gain"] = 0.0
    rng = np.random.default_rng(0)
    actual = 1000 + rng.normal(0, 5, 40)
    published = actual + rng.normal(0, 80, 40)
    naive = np.full(40, 1000.0)
    frame = _frame(
        {
            "origin_date": pd.bdate_range("2024-01-02", periods=40, freq="B"),
            "target_date": pd.bdate_range("2024-01-03", periods=40, freq="B"),
            "horizon": 1,
            "origin_price": naive,
            "actual": actual,
            "naive_pred": naive,
            "published_pred": published,
            "published_lower": published - 200.0,
            "published_upper": published + 200.0,
            "naive_lower": naive - 5.0,
            "naive_upper": naive + 5.0,
            "feature_failure": False,
        }
    )
    decision = evaluate_horizon(
        frame,
        1,
        rules,
        step=5,
    )
    assert decision["model_width"] > decision["naive_width"]
    assert "étroite" in decision["reason"]
    assert decision["validated"] is False


def test_coverage_uses_the_frozen_margin_not_a_refit_on_the_test():
    rules = dict(load_acceptance())
    rules["bootstrap_replicates"] = 40
    actual = np.full(24, 100.0)
    published = np.full(24, 99.0)
    naive = np.full(24, 80.0)
    frame = _frame(
        {
            "origin_date": pd.bdate_range("2024-01-02", periods=24, freq="B"),
            "target_date": pd.bdate_range("2024-01-03", periods=24, freq="B"),
            "horizon": 1,
            "origin_price": naive,
            "actual": actual,
            "naive_pred": naive,
            "published_pred": published,
            "published_lower": published,
            "published_upper": published,
            "naive_lower": naive - 50.0,
            "naive_upper": naive + 50.0,
            "feature_failure": False,
        }
    )
    decision = evaluate_horizon(frame, 1, rules, step=5)
    assert decision["coverage"] == 0.0
    assert "hors échantillon" in decision["reason"]
    assert decision["validated"] is False
    bare = frame.drop(columns=["published_lower", "published_upper", "naive_lower", "naive_upper"])
    missing = evaluate_horizon(bare, 1, rules, step=5)
    assert "Bornes finales absentes" in missing["reason"]
    assert missing["validated"] is False


def test_active_release_ignores_a_newer_file_name(tmp_path):
    market = tmp_path / "models"
    market.mkdir()
    old = market / "prophet_improved_20261006_213603.pkl"
    newer = market / "prophet_improved_20261006_235959.pkl"
    old.write_bytes(b"old")
    newer.write_bytes(b"new")
    manifest = {
        "market": "cocoa",
        "version": "improved_20261006_213603",
        "validated": False,
        "artifacts": {"prophet": str(old)},
        "horizons": {"1": {"validated": False}},
    }
    path = tmp_path / "config"
    path.mkdir()
    (path / "active_release_cocoa.json").write_text(json.dumps(manifest), encoding="utf-8")
    release = load_active_release("cocoa", root=tmp_path)
    assert artifact_path(release, "prophet", root=tmp_path) == old
    assert release["validated"] is False


def test_a_refused_horizon_does_not_rewrite_the_manifest(tmp_path):
    manifest = {
        "market": "cocoa",
        "version": "improved_old",
        "validated": False,
        "artifacts": {"conformal_intervals": "config/conformal_intervals.json"},
        "horizons": {"1": {"validated": False}, "7": {"validated": False}},
    }
    config = tmp_path / "config"
    config.mkdir()
    path = config / "active_release_cocoa.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    assert promote_accepted_horizons("cocoa", [], "20261007_000000", {"by_horizon": {}}, tmp_path) is False
    assert path.read_text(encoding="utf-8") == before


def test_an_accepted_horizon_is_the_only_one_marked(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "active_release_cocoa.json").write_text(
        json.dumps(
            {
                "market": "cocoa",
                "version": "improved_old",
                "validated": False,
                "artifacts": {},
                "horizons": {
                    "1": {"validated": False},
                    "7": {"validated": False},
                    "14": {"validated": False},
                    "30": {"validated": False},
                },
            }
        ),
        encoding="utf-8",
    )
    assert promote_accepted_horizons("cocoa", ["1"], "20261008_120000", None, tmp_path)
    release = load_active_release("cocoa", root=tmp_path)
    assert release["validated"] is True
    assert release["horizons"]["1"]["validated"] is True
    assert release["horizons"]["7"]["validated"] is False
    assert release["horizons"]["30"]["validated"] is False
    assert release["source_report"] == "20261008_120000"


def test_missing_nhits_renormalizes_instead_of_keeping_xgboost_alone():
    served = compose_served_price(
        xgb_price=110.0,
        prophet_price=90.0,
        nhits_price=None,
        weights={"xgb": 0.4, "prophet": 0.2, "nhits": 0.4},
        spot=100.0,
        horizon=1,
        max_abs_change_pct={"1": 20.0},
        price_bounds=(50.0, 200.0),
    )
    assert abs(served["price"] - (0.4 * 110.0 + 0.2 * 90.0) / 0.6) < 1e-6
    assert served["sentiment"] == 0.0
    assert served["scored_variant"] == "no_sentiment"


def test_an_unvalidated_horizon_publishes_the_close_and_keeps_the_candidate():
    class _Item:
        def __init__(self):
            self.horizon = 1
            self.price = 110.0
            self.confidence_interval = [100.0, 120.0]
            self.components = {}

    item = _Item()
    publish_or_close(
        [item],
        {"validated": False, "horizons": {"1": {"validated": False}}, "release_mode": "experimental"},
        100.0,
        release_mode="experimental",
    )
    assert item.price == 110.0
    assert item.confidence_interval == [100.0, 120.0]
    assert item.components["candidate_price"] == 110.0
    assert item.components["candidate_lower"] == 100.0
    assert item.components["status"] == "experimental"
    assert item.status == "experimental"

    kept = _Item()
    publish_or_close(
        [kept],
        {"validated": True, "horizons": {"1": {"validated": True}}},
        100.0,
    )
    assert kept.price == 110.0
    assert kept.components["candidate_price"] == 110.0
    assert kept.components["served_as_close"] is False
    assert kept.status == "validated"

    blocked = _Item()
    publish_or_close(
        [blocked],
        {"validated": True, "horizons": {"1": {"validated": True}}},
        100.0,
        release_mode="experimental",
    )
    assert blocked.status == "experimental"
    assert blocked.price == 110.0
    assert blocked.components["status"] == "experimental"


def test_metrics_follow_the_report_named_by_the_manifest(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "20260101_000000_summary.json").write_text("{}", encoding="utf-8")
    (reports / "20261006_024613_summary.json").write_text('{"named": true}', encoding="utf-8")
    release = {"margins_provenance": {"source_report": "20261006_024613"}}
    found = find_release_summary(str(reports), release)
    assert found is not None
    assert found.name == "20261006_024613_summary.json"
    assert find_release_summary(str(reports), {"source_report": "missing"}) is None


def test_journal_keeps_the_model_snapshot_not_a_later_read():
    fields = snapshot_journal_fields(
        {
            "origin_date": "2026-10-02",
            "origin_price": 4390.0,
            "target_date": "2026-10-05",
            "snapshot_id": "2026-10-02:4390.0000",
        },
        {"origin_date": "2026-10-05", "origin_price": 4500.0},
    )
    assert fields["origin_date"] == "2026-10-02"
    assert fields["origin_price"] == 4390.0
    assert fields["target_date"] == "2026-10-05"


def test_rsi_is_100_when_every_move_is_a_gain():
    prices = pd.Series(np.arange(1, 30, dtype=float))
    rsi = _compute_rsi(prices, window=14)
    assert rsi.iloc[-1] == 100.0


def test_future_business_date_skips_a_listed_holiday():
    # Thursday 30 April. Friday 1 May is passed in, and Monday 4 May is on the ICE list.
    landed = future_business_date("2026-04-30", 1, holidays=["2026-05-01"])
    assert landed.date().isoformat() == "2026-05-05"
    cocoa = future_business_date("2026-04-02", 1)
    assert cocoa.date().isoformat() == "2026-04-07"


def test_a_future_open_interest_does_not_change_a_past_session():
    dates = pd.bdate_range("2024-01-02", periods=40)
    frame = pd.DataFrame(
        {
            "date": dates,
            "price": np.linspace(1000, 1100, 40),
            "open_interest": np.linspace(100, 140, 40),
        }
    )
    before = build_technical_features(frame)
    changed = frame.copy()
    changed.loc[changed.index[-1], "open_interest"] = 9999.0
    after = build_technical_features(changed)
    past = -5
    assert before["oi_change_1d"].iloc[past] == after["oi_change_1d"].iloc[past]
    assert before["open_interest"].iloc[past] == after["open_interest"].iloc[past]


def test_unvalidated_horizon_card_shows_the_close_label():
    source = Path("frontend/src/components/dashboard/PredictionHorizonCard.tsx").read_text(
        encoding="utf-8"
    )
    assert "Prévision expérimentale — non validée" in source
    assert "Indisponible" in source
    panel = Path("frontend/src/components/dashboard/AnalysisPanel.tsx").read_text(encoding="utf-8")
    assert "Probabilité" not in panel
    scenarios = Path("frontend/src/lib/marketAnalytics.ts").read_text(encoding="utf-8")
    assert "probability" not in scenarios
