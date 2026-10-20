"""The journal and the cache name the rows the model actually read."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.improved_price_predictor import ImprovedPricePredictor
from src.models.prediction_journal import append_forecast, collapse_fingerprint_twins
from src.models.revision_command import input_fingerprints


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-10-08", "2026-10-09", "2026-10-13"]),
            "price": [4200.0, 4241.0, 4277.0],
            "open": [4200.0, 4241.0, None],
            "high": [4210.0, 4250.0, None],
            "low": [4190.0, 4230.0, None],
            "volume": [10.0, 12.0, None],
        }
    )


def test_command_and_request_use_the_consumed_fingerprint_and_a_correction_is_new():
    frame = _frame()
    predictor = ImprovedPricePredictor.__new__(ImprovedPricePredictor)
    predictor.price_table = "cocoa_london_prices"
    predictor.active_release = {}
    predictor.bind_frame(frame)
    predictor._fetch_historical_data()
    tokens = input_fingerprints(frame, {})
    assert predictor._data_fingerprint == tokens["source"]
    assert predictor._served_fingerprint == tokens["consumed"]
    assert tokens["source"] != tokens["consumed"]
    changed = frame.copy()
    changed.loc[changed.index[-1], "price"] = 9999.0
    assert input_fingerprints(changed, {})["consumed"] != tokens["consumed"]


def test_the_same_consumed_fingerprint_is_one_emission(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    common = dict(
        market="ICE_NY",
        model_version="improved_a",
        release_mode="experimental",
        horizon=1,
        price=4229.78,
        lower=4100.0,
        upper=4300.0,
        origin_date="2026-10-13",
        origin_price=4277.0,
        target_date="2026-10-14",
        status="experimental",
        procedure_fingerprint="procedure",
        data_fingerprint="consumed",
        source_fingerprint="source",
        path=journal,
    )
    assert append_forecast(**common) != "duplicate"
    assert append_forecast(**common) == "duplicate"


def test_reading_collapses_a_fingerprint_twin_and_keeps_a_different_price():
    first = {
        "forecast_id": "early",
        "recorded_at": "2026-10-14T10:00:00+00:00",
        "market": "ICE_NY",
        "origin_date": "2026-10-13",
        "target_date": "2026-10-14",
        "horizon": 1,
        "model_version": "improved_a",
        "procedure_fingerprint": "procedure",
        "price": 4229.78,
        "data_fingerprint": "before-fill",
        "diagnostic": False,
        "opens_initial_id": "early",
        "revises": [],
    }
    twin = dict(first)
    twin["forecast_id"] = "late"
    twin["recorded_at"] = "2026-10-14T11:00:00+00:00"
    twin["data_fingerprint"] = "after-fill"
    twin["opens_initial_id"] = "late"
    other = dict(first)
    other["forecast_id"] = "other"
    other["price"] = 4300.0
    other["data_fingerprint"] = "corrected"
    other["opens_initial_id"] = "other"
    collapsed = collapse_fingerprint_twins([first, twin])
    assert [row["forecast_id"] for row in collapsed] == ["early"]
    kept = collapse_fingerprint_twins([first, twin, other])
    assert {row["forecast_id"] for row in kept} == {"early", "other"}
