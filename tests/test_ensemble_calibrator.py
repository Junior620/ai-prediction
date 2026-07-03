"""Tests for ensemble weight calibration."""

from __future__ import annotations

import pandas as pd
import pytest

from src.validation.ensemble_calibrator import EnsembleCalibrator, merge_backtest_predictions


def test_merge_and_calibrate_xgb_only(tmp_path):
    wf = pd.DataFrame(
        {
            "origin_date": pd.to_datetime(["2021-01-01", "2021-01-08", "2021-01-01", "2021-01-08"]),
            "horizon": [1, 1, 7, 7],
            "actual": [100.0, 110.0, 105.0, 95.0],
            "xgb_pred": [98.0, 108.0, 102.0, 97.0],
            "prophet_pred": [99.0, 109.0, 103.0, 96.0],
            "origin_price": [100.0, 110.0, 100.0, 110.0],
        }
    )
    wf_path = tmp_path / "wf.csv"
    wf.to_csv(wf_path, index=False)

    result = EnsembleCalibrator().run(str(wf_path), xgb_only=True)
    assert "1" in result.weights_payload["by_horizon"]
    assert "7" in result.weights_payload["by_horizon"]
    w = result.weights_payload["by_horizon"]["1"]
    assert abs(w["xgb"] + w["prophet"] - 1.0) < 0.01


def test_merge_with_nhits(tmp_path):
    wf = pd.DataFrame(
        {
            "origin_date": pd.to_datetime(["2021-01-01", "2021-01-01"]),
            "horizon": [1, 1],
            "actual": [100.0, 110.0],
            "xgb_pred": [98.0, 108.0],
            "prophet_pred": [99.0, 109.0],
            "origin_price": [100.0, 110.0],
        }
    )
    nh = pd.DataFrame(
        {
            "cutoff": pd.to_datetime(["2021-01-01", "2021-01-01"]),
            "horizon": [1, 1],
            "nhits_pred": [97.0, 107.0],
        }
    )
    wf_path = tmp_path / "wf.csv"
    nh_path = tmp_path / "nh.csv"
    wf.to_csv(wf_path, index=False)
    nh.to_csv(nh_path, index=False)

    merged = merge_backtest_predictions(str(wf_path), str(nh_path))
    assert merged["nhits_pred"].notna().all()
