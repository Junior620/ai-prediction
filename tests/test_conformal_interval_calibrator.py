"""Tests for conformal prediction interval calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.conformal_intervals import (
    apply_interval,
    heuristic_interval,
    load_conformal_margins,
)
from src.validation.conformal_interval_calibrator import (
    ConformalIntervalCalibrator,
    ConformalIntervalCalibratorConfig,
    conformal_quantile,
)


def test_conformal_quantile_finite_sample():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    q = conformal_quantile(scores, coverage_level=0.90)
    assert q >= 9.0


def test_empirical_coverage_on_synthetic(tmp_path):
    rng = np.random.default_rng(42)
    n = 120
    rows = []
    for i in range(n):
        actual = 3000.0 + rng.normal(0, 200)
        error = rng.normal(0, 150)
        pred = actual + error
        rows.append(
            {
                "origin_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=i),
                "horizon": 1,
                "actual": actual,
                "xgb_pred": pred,
                "prophet_pred": pred + rng.normal(0, 20),
                "origin_price": 3000.0,
            }
        )
    wf_path = tmp_path / "wf.csv"
    pd.DataFrame(rows).to_csv(wf_path, index=False)

    weights_path = tmp_path / "weights.json"
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump({"by_horizon": {"1": {"xgb": 0.5, "prophet": 0.5, "nhits": 0.0}}}, f)

    result = ConformalIntervalCalibrator(
        ConformalIntervalCalibratorConfig(coverage_level=0.90)
    ).run(str(wf_path), ensemble_weights_file=str(weights_path))

    coverage = result.intervals_payload["by_horizon"]["1"]["empirical_coverage"]
    assert coverage >= 0.88


def test_margins_increase_with_horizon(tmp_path):
    wf = pd.DataFrame(
        {
            "origin_date": pd.to_datetime(["2021-01-01"] * 9),
            "horizon": [1, 1, 1, 7, 7, 7, 30, 30, 30],
            "actual": [3000, 3100, 2900, 3000, 3200, 2800, 3000, 3300, 2700],
            "xgb_pred": [3050, 3150, 2950, 3100, 3300, 2900, 3200, 3500, 2900],
            "prophet_pred": [3040, 3140, 2940, 3080, 3280, 2880, 3180, 3480, 2880],
            "origin_price": [3000.0] * 9,
        }
    )
    wf_path = tmp_path / "wf.csv"
    wf.to_csv(wf_path, index=False)

    weights_path = tmp_path / "weights.json"
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "by_horizon": {
                    "1": {"xgb": 0.5, "prophet": 0.5, "nhits": 0.0},
                    "7": {"xgb": 0.5, "prophet": 0.5, "nhits": 0.0},
                    "30": {"xgb": 0.5, "prophet": 0.5, "nhits": 0.0},
                }
            },
            f,
        )

    result = ConformalIntervalCalibrator().run(
        str(wf_path), ensemble_weights_file=str(weights_path)
    )
    m1 = result.intervals_payload["by_horizon"]["1"]["margin_lower"]
    m7 = result.intervals_payload["by_horizon"]["7"]["margin_lower"]
    m30 = result.intervals_payload["by_horizon"]["30"]["margin_lower"]
    assert m30 >= m7 >= m1


def test_apply_interval_from_config(tmp_path):
    cfg = {
        "by_horizon": {
            "1": {"margin_lower": 100.0, "margin_upper": 100.0},
        }
    }
    cfg_path = tmp_path / "conformal.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)

    margins = load_conformal_margins(str(cfg_path))
    lower, upper = apply_interval(3000.0, 1, margins, (1000.0, 10000.0))
    assert lower == 2900.0
    assert upper == 3100.0


def test_heuristic_fallback_when_no_margins():
    lower, upper = heuristic_interval(
        price=3000.0,
        horizon=7,
        price_volatility=500.0,
        price_bounds=(1000.0, 10000.0),
        confidence_level=0.90,
    )
    assert lower < 3000.0 < upper
    assert upper - lower > 0


def test_load_missing_file_returns_empty():
    assert load_conformal_margins("/nonexistent/conformal_intervals.json") == {}
