"""Tests for walk-forward multi-horizon validation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.models.hybrid_features import future_business_date, price_at_date
from src.validation.metrics import aggregate_by_horizon, compute_horizon_metrics
from src.validation.report import build_summary_payload, save_report
from src.validation.walk_forward_validator import WalkForwardConfig, WalkForwardValidator


def make_synthetic_prices(n_days: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate business-day price series with trend + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    trend = np.linspace(3000, 4500, n_days)
    noise = rng.normal(0, 50, n_days)
    prices = trend + noise
    return pd.DataFrame({"date": dates, "price": prices})


class TestWalkForwardValidator:
    def test_origin_count_with_step_size(self):
        df = make_synthetic_prices(400)
        config = WalkForwardConfig(
            horizons=[1, 7],
            min_train_days=100,
            step_size=10,
            max_origins=None,
        )
        validator = WalkForwardValidator(config)
        max_horizon = max(config.horizons)
        n_rows = len(df)
        expected = len(
            list(range(config.min_train_days - 1, n_rows - max_horizon, config.step_size))
        )
        indices = validator._origin_indices(n_rows, max_horizon)
        assert len(indices) == expected

    def test_no_data_leakage_prophet_fit_on_train_only(self):
        df = make_synthetic_prices(350)
        fit_lengths = []

        original_fit = __import__("prophet", fromlist=["Prophet"]).Prophet.fit

        def tracking_fit(self, train_df, *args, **kwargs):
            fit_lengths.append(len(train_df))
            return original_fit(self, train_df, *args, **kwargs)

        config = WalkForwardConfig(
            horizons=[1],
            min_train_days=120,
            step_size=30,
            max_origins=3,
        )

        with patch("src.models.hybrid_features.Prophet.fit", tracking_fit):
            result = WalkForwardValidator(config).run(df)

        assert len(fit_lengths) == config.max_origins
        for length in fit_lengths:
            assert length <= len(df)
        assert not result.predictions.empty
        assert result.predictions["train_size"].max() <= len(df)

    def test_multi_horizon_actual_matches_target_date(self):
        df = make_synthetic_prices(300)
        config = WalkForwardConfig(
            horizons=[1, 7],
            min_train_days=100,
            step_size=20,
            max_origins=2,
        )
        result = WalkForwardValidator(config).run(df)

        for _, row in result.predictions.iterrows():
            expected_date = pd.Timestamp(
                future_business_date(row["origin_date"], int(row["horizon"]))
            ).normalize()
            assert pd.Timestamp(row["target_date"]).normalize() == expected_date
            expected_price = price_at_date(df, expected_date)
            assert expected_price is not None
            assert row["actual"] == pytest.approx(expected_price)

    def test_summary_has_metrics_per_horizon(self):
        df = make_synthetic_prices(350)
        config = WalkForwardConfig(
            horizons=[1, 7],
            min_train_days=100,
            step_size=25,
            max_origins=4,
            include_recursive=True,
        )
        result = WalkForwardValidator(config).run(df)

        for horizon in config.horizons:
            metrics = result.summary["xgb_pred"][horizon]
            assert metrics["n_predictions"] > 0
            assert metrics["mape"] >= 0
            assert metrics["rmse"] >= 0

        if 7 in config.horizons:
            assert "xgb_pred_recursive" in result.predictions.columns


class TestMetrics:
    def test_compute_horizon_metrics(self):
        preds = pd.DataFrame(
            {
                "origin_price": [100.0, 100.0],
                "actual": [110.0, 90.0],
                "xgb_pred": [108.0, 92.0],
            }
        )
        metrics = compute_horizon_metrics(preds, "xgb_pred")
        assert metrics["n_predictions"] == 2
        assert metrics["mae"] == pytest.approx(2.0)
        assert metrics["directional_accuracy"] == pytest.approx(1.0)

    def test_aggregate_by_horizon(self):
        preds = pd.DataFrame(
            {
                "horizon": [1, 1, 7],
                "origin_price": [100.0, 100.0, 100.0],
                "actual": [105.0, 95.0, 110.0],
                "xgb_pred": [104.0, 96.0, 108.0],
            }
        )
        agg = aggregate_by_horizon(preds, ["xgb_pred"])
        assert 1 in agg["xgb_pred"]
        assert 7 in agg["xgb_pred"]
        assert agg["xgb_pred"][1]["n_predictions"] == 2


class TestReport:
    def test_save_report_creates_files(self, tmp_path):
        df = make_synthetic_prices(200)
        config = WalkForwardConfig(
            horizons=[1],
            min_train_days=80,
            step_size=20,
            max_origins=2,
        )
        wf_result = WalkForwardValidator(config).run(df)

        paths = save_report(str(tmp_path), wf_result)
        assert Path(paths["summary_json"]).exists()
        assert Path(paths["walk_forward_csv"]).exists()

        with open(paths["summary_json"], encoding="utf-8") as f:
            payload = json.load(f)
        assert "walk_forward" in payload
        assert "summary_by_component" in payload["walk_forward"]

        csv_df = pd.read_csv(paths["walk_forward_csv"])
        expected_cols = {
            "origin_date",
            "horizon",
            "actual",
            "xgb_pred",
            "prophet_pred",
        }
        assert expected_cols.issubset(set(csv_df.columns))

    def test_build_summary_payload_structure(self):
        df = make_synthetic_prices(200)
        config = WalkForwardConfig(
            horizons=[1],
            min_train_days=80,
            step_size=30,
            max_origins=1,
        )
        wf_result = WalkForwardValidator(config).run(df)
        payload = build_summary_payload(wf_result, holdout_baseline={"mape_1step_holdout": 5.0})
        assert payload["validation_type"] == "walk_forward_multi_horizon"
        assert "legacy_holdout_baseline" in payload
