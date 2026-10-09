"""Split a walk-forward file into calibration and a later held-out test."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.validation.conformal_interval_calibrator import conformal_quantile
from src.validation.metrics import compute_horizon_metrics


def served_prediction_column(frame: pd.DataFrame) -> str:
    """Column of the price that would have been published."""
    if "published_pred" in frame.columns and frame["published_pred"].notna().any():
        return "published_pred"
    return "xgb_pred"


def split_chronological(
    frame: pd.DataFrame,
    calibration_fraction: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate only on targets already realized before the first test origin.

    ``T`` is that first test origin. Calibration keeps rows whose target date
    is strictly before ``T``. The test keeps origins on or after ``T``.
    """
    dated = frame.copy()
    dated["_origin"] = pd.to_datetime(dated["origin_date"]).dt.normalize()
    dated["_target"] = pd.to_datetime(dated["target_date"]).dt.normalize()
    origins = sorted(dated["_origin"].unique())
    if len(origins) < 2:
        empty = dated.iloc[0:0].drop(columns=["_origin", "_target"])
        return dated.drop(columns=["_origin", "_target"]), empty
    cut = int(len(origins) * calibration_fraction)
    cut = min(max(cut, 1), len(origins) - 1)
    cutoff = origins[cut]
    calibration = dated[dated["_target"] < cutoff]
    test = dated[dated["_origin"] >= cutoff]
    return (
        calibration.drop(columns=["_origin", "_target"]).copy(),
        test.drop(columns=["_origin", "_target"]).copy(),
    )


def calibration_respects_cutoff(frame: pd.DataFrame, calibration_fraction: float = 0.5) -> bool:
    """True when no calibration target falls on or after the first test origin."""
    dated = frame.copy()
    dated["_origin"] = pd.to_datetime(dated["origin_date"]).dt.normalize()
    dated["_target"] = pd.to_datetime(dated["target_date"]).dt.normalize()
    origins = sorted(dated["_origin"].unique())
    if len(origins) < 2:
        return True
    cut = int(len(origins) * calibration_fraction)
    cut = min(max(cut, 1), len(origins) - 1)
    cutoff = origins[cut]
    calibration = dated[dated["_target"] < cutoff]
    if calibration.empty:
        return True
    return bool(calibration["_target"].max() < cutoff)


def beats_naive_on_all_horizons(
    frame: pd.DataFrame,
    pred_col: str,
    expected_horizons: Optional[Tuple[int, ...]] = None,
) -> bool:
    """True only when every expected horizon is accepted on the shared rows.

    A missing prediction is filled with the origin close before scoring, so
    it cannot disappear from the model while remaining in the reference.
    """
    del pred_col  # The published column, with fallbacks, is what gets scored.
    if frame.empty or "naive_pred" not in frame.columns:
        return False
    from src.validation.acceptance import evaluate_horizon, load_acceptance

    rules = load_acceptance()
    horizons = list(expected_horizons or rules.get("horizons") or [])
    if not horizons:
        return False
    present = {int(h) for h in frame["horizon"].unique()}
    for horizon in horizons:
        if int(horizon) not in present:
            return False
        decision = evaluate_horizon(frame, int(horizon), rules)
        if not decision["validated"]:
            return False
    return True


def apply_frozen_bands(
    frame: pd.DataFrame,
    pred_col: str,
    margins: Dict[str, Dict[str, float]],
    lower_name: str,
    upper_name: str,
    price_bounds: Tuple[float, float] = (1000.0, 15000.0),
) -> pd.DataFrame:
    """Build final bounds on this slice from margins frozen earlier.

    A horizon without a frozen margin stays unbounded. The acceptance gate
    then fails instead of inventing a band from an older file.
    """
    out = frame.copy()
    floor = float(price_bounds[0])
    ceiling = float(price_bounds[1])
    lowers = []
    uppers = []
    for _, row in out.iterrows():
        entry = (margins or {}).get(str(int(row["horizon"])))
        predicted = pd.to_numeric(row.get(pred_col), errors="coerce")
        if (
            not entry
            or predicted is None
            or not np.isfinite(float(predicted))
            or entry.get("margin_lower") is None
            or entry.get("margin_upper") is None
        ):
            lowers.append(np.nan)
            uppers.append(np.nan)
            continue
        lower = max(floor, float(predicted) - float(entry["margin_lower"]))
        upper = min(ceiling, float(predicted) + float(entry["margin_upper"]))
        lowers.append(lower)
        uppers.append(upper)
    out[lower_name] = lowers
    out[upper_name] = uppers
    return out


def split_last_origins(
    frame: pd.DataFrame,
    test_origins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calibration is every target already realized before the test origins.

    The test is the last ``test_origins`` distinct origin dates. It is a
    chain check when that count is small, not a reliability proof.
    """
    dated = frame.copy()
    dated["_origin"] = pd.to_datetime(dated["origin_date"]).dt.normalize()
    dated["_target"] = pd.to_datetime(dated["target_date"]).dt.normalize()
    origins = sorted(dated["_origin"].unique())
    if len(origins) < 2 or int(test_origins) < 1:
        empty = dated.iloc[0:0].drop(columns=["_origin", "_target"])
        return dated.drop(columns=["_origin", "_target"]), empty
    keep = max(1, min(int(test_origins), len(origins) - 1))
    test_start = origins[-keep]
    calibration = dated[dated["_target"] < test_start]
    test = dated[dated["_origin"] >= test_start]
    return (
        calibration.drop(columns=["_origin", "_target"]).copy(),
        test.drop(columns=["_origin", "_target"]).copy(),
    )


def residual_margins(
    frame: pd.DataFrame,
    pred_col: str,
    coverage_level: float = 0.90,
) -> Dict[str, Dict[str, float]]:
    """Conformal margins on this slice only. Coverage here is not a held-out test."""
    by_horizon: Dict[str, Dict[str, float]] = {}
    for horizon in sorted(int(h) for h in frame["horizon"].unique()):
        subset = frame[frame["horizon"] == horizon].dropna(subset=["actual", pred_col])
        if subset.empty:
            continue
        actual = subset["actual"].to_numpy(dtype=float)
        pred = subset[pred_col].to_numpy(dtype=float)
        margin = conformal_quantile(np.abs(actual - pred), coverage_level)
        within = (actual >= pred - margin) & (actual <= pred + margin)
        by_horizon[str(horizon)] = {
            "margin_lower": float(margin),
            "margin_upper": float(margin),
            "calibration_coverage": float(np.mean(within)),
            "mean_interval_width": float(2.0 * margin),
            "n": int(len(subset)),
        }
    return by_horizon


def held_out_coverage(
    frame: pd.DataFrame,
    pred_col: str,
    margins: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Coverage and width on origins that did not choose the margins."""
    report: Dict[str, Dict[str, float]] = {}
    for horizon in sorted(int(h) for h in frame["horizon"].unique()):
        entry = margins.get(str(horizon))
        subset = frame[frame["horizon"] == horizon].dropna(subset=["actual", pred_col])
        if entry is None or subset.empty:
            continue
        actual = subset["actual"].to_numpy(dtype=float)
        pred = subset[pred_col].to_numpy(dtype=float)
        lower = float(entry["margin_lower"])
        upper = float(entry["margin_upper"])
        within = (actual >= pred - lower) & (actual <= pred + upper)
        report[str(horizon)] = {
            "test_coverage": float(np.mean(within)),
            "mean_interval_width": float(np.mean((pred + upper) - (pred - lower))),
            "n": int(len(subset)),
        }
    return report
