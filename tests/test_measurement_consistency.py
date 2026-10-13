"""Service and backtest share one price and one interval."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.ensemble_weights import combine_ensemble
from src.models.served_forecast import build_served_interval
from src.validation.served_backtest import apply_frozen_bands
from src.validation.variant_scores import (
    SANS_PROPHET_NOTE,
    score_variants,
    variant_prices,
    weights_without_prophet,
)
from src.validation.walk_forward_validator import WalkForwardConfig, WalkForwardValidator


def test_service_interval_matches_the_backtest_row():
    margins = {"1": {"margin_lower": 10.0, "margin_upper": 12.0}}
    frame = pd.DataFrame(
        [
            {
                "horizon": 1,
                "published_pred": 100.0,
                "recent_lower": 90.0,
                "recent_upper": 130.0,
                "garch_log_half": 0.02,
                "garch_required": True,
            }
        ]
    )
    direct = build_served_interval(
        100.0, 1, margins=margins, recent_bounds=(90.0, 130.0),
        garch_log_half=0.02, garch_required=True,
    )
    rebuilt = apply_frozen_bands(
        frame, "published_pred", margins, "published_lower", "published_upper", (50.0, 200.0)
    )
    assert direct == (float(rebuilt.loc[0, "published_lower"]), float(rebuilt.loc[0, "published_upper"]))


def test_missing_margin_does_not_reuse_an_older_band():
    lower, upper = build_served_interval(100.0, 1, margins=None, garch_required=False)
    assert lower is None and upper is None


def test_blend_without_prophet_is_renormalized_before_the_mix():
    weights = weights_without_prophet({"xgb": 0.45, "prophet": 0.10, "nhits": 0.45})
    assert weights["prophet"] == 0.0
    assert weights["xgb"] == pytest.approx(0.5)
    assert weights["nhits"] == pytest.approx(0.5)
    mixed = combine_ensemble(110.0, 999.0, 90.0, weights)
    assert mixed == pytest.approx(100.0)
    assert combine_ensemble(110.0, 80.0, None, {"xgb": 0.5, "prophet": 0.0, "nhits": 0.5}) != mixed


def test_variant_names_the_prophet_limit_and_does_not_promote(tmp_path):
    rows = []
    for index, origin in enumerate(pd.bdate_range("2024-01-02", periods=8)):
        for horizon in (1, 7):
            rows.append(
                {
                    "origin_date": origin,
                    "origin_price": 100.0 + index,
                    "target_date": origin + pd.Timedelta(days=horizon),
                    "horizon": horizon,
                    "actual": 101.0 + index,
                    "xgb_pred_direct": 100.5 + index,
                    "prophet_pred": 102.0 + index,
                    "nhits_pred": 99.5 + index,
                    "naive_pred": 100.0 + index,
                    "weights_used": json.dumps({"xgb": 0.45, "prophet": 0.10, "nhits": 0.45}),
                    "recent_lower": 80.0,
                    "recent_upper": 140.0,
                    "garch_required": False,
                }
            )
    frame = pd.DataFrame(rows)
    rules = {
        "horizons": [1, 7],
        "step_size": 1,
        "min_relative_mape_gain": 0.10,
        "max_fallback_rate": 0.20,
        "min_observations": 30,
        "min_temporal_blocks": 6,
        "coverage_level": 0.90,
        "bootstrap_replicates": 20,
        "bootstrap_alpha": 0.05,
    }
    payload = score_variants(
        frame,
        {"1": {"xgb": 0.45, "prophet": 0.10, "nhits": 0.45}},
        acceptance=rules,
        test_origins=3,
        step_size=1,
        price_bounds=(50.0, 200.0),
    )
    assert SANS_PROPHET_NOTE in payload["note"]
    assert payload["promoted"] is False
    assert set(payload["variants"]) == {
        "cours_inchange",
        "ensemble_actuel",
        "xgboost_direct",
        "nhits_seul",
        "sans_prophet_dans_le_melange",
    }
    prices = variant_prices(frame, {"1": {"xgb": 0.45, "prophet": 0.10, "nhits": 0.45}})
    assert prices["sans_prophet_dans_le_melange"].iloc[0] == pytest.approx(
        combine_ensemble(100.5, 102.0, 99.5, weights_without_prophet(
            {"xgb": 0.45, "prophet": 0.10, "nhits": 0.45}
        ))
    )


def test_named_origin_stays_when_the_longest_horizon_is_not_realized_yet():
    dates = pd.Series(pd.bdate_range("2024-01-02", periods=12))
    config = WalkForwardConfig(
        horizons=[1, 30],
        min_train_days=5,
        origin_dates=[str(dates.iloc[8].date())],
        include_direct_hstep=False,
    )
    indices = WalkForwardValidator(config)._origin_indices(dates, 30)
    assert indices == [8]


def test_checkpoint_with_another_identity_is_refused(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    pd.DataFrame([{"origin_date": "2024-01-02", "horizon": 1}]).to_json(
        checkpoint, orient="records", lines=True
    )
    checkpoint.with_suffix(checkpoint.suffix + ".identity").write_text("ancien", encoding="utf-8")
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=40),
            "price": [100.0 + index for index in range(40)],
        }
    )
    config = WalkForwardConfig(
        horizons=[1],
        min_train_days=5,
        step_size=5,
        max_origins=1,
        checkpoint_path=str(checkpoint),
        checkpoint_identity="nouveau",
        include_direct_hstep=False,
    )
    with pytest.raises(RuntimeError, match="reprise"):
        WalkForwardValidator(config).run(frame)
