"""Calibrate conformal prediction intervals from walk-forward backtest residuals."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.models.ensemble_weights import combine_ensemble, get_weights_for_horizon, load_ensemble_weights
from src.validation.ensemble_calibrator import merge_backtest_predictions


@dataclass
class ConformalIntervalCalibratorConfig:
    coverage_level: float = 0.90
    asymmetric: bool = False


@dataclass
class ConformalIntervalCalibrationResult:
    intervals_payload: Dict[str, Any]
    merged_predictions: pd.DataFrame


def conformal_quantile(scores: np.ndarray, coverage_level: float) -> float:
    """Finite-sample conformal quantile for split conformal prediction."""
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(scores[0])
    q_level = min(math.ceil((n + 1) * coverage_level) / n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def _compute_ensemble_series(
    subset: pd.DataFrame,
    weights: Dict[str, float],
) -> np.ndarray:
    preds = []
    for _, row in subset.iterrows():
        nhits_val = row.get("nhits_pred")
        nhits_price = float(nhits_val) if pd.notna(nhits_val) else None
        preds.append(
            combine_ensemble(
                float(row["xgb_pred"]),
                float(row["prophet_pred"]),
                nhits_price,
                weights,
            )
        )
    return np.asarray(preds, dtype=float)


def _horizon_margins(
    actual: np.ndarray,
    ensemble_pred: np.ndarray,
    coverage_level: float,
    asymmetric: bool,
) -> tuple[float, float, float, float, int]:
    """Return lower margin, upper margin, empirical coverage, mean width, n."""
    mask = np.isfinite(actual) & np.isfinite(ensemble_pred)
    actual = actual[mask]
    ensemble_pred = ensemble_pred[mask]
    n = len(actual)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0

    if asymmetric:
        lower_scores = np.maximum(0.0, ensemble_pred - actual)
        upper_scores = np.maximum(0.0, actual - ensemble_pred)
        margin_lower = conformal_quantile(lower_scores, coverage_level)
        margin_upper = conformal_quantile(upper_scores, coverage_level)
    else:
        abs_scores = np.abs(actual - ensemble_pred)
        margin = conformal_quantile(abs_scores, coverage_level)
        margin_lower = margin_upper = margin

    lower_bounds = ensemble_pred - margin_lower
    upper_bounds = ensemble_pred + margin_upper
    within = (actual >= lower_bounds) & (actual <= upper_bounds)
    empirical_coverage = float(np.mean(within))
    mean_width = float(np.mean(upper_bounds - lower_bounds))

    return margin_lower, margin_upper, empirical_coverage, mean_width, n


class ConformalIntervalCalibrator:
    """Compute per-horizon conformal margins from walk-forward OOS predictions."""

    def __init__(self, config: Optional[ConformalIntervalCalibratorConfig] = None):
        self.config = config or ConformalIntervalCalibratorConfig()

    def run(
        self,
        walk_forward_csv: str,
        nhits_csv: Optional[str] = None,
        ensemble_weights_file: Optional[str] = None,
        ensemble_weights: Optional[Dict[str, Dict[str, float]]] = None,
        source_report: Optional[str] = None,
    ) -> ConformalIntervalCalibrationResult:
        merged = merge_backtest_predictions(walk_forward_csv, nhits_csv)

        if ensemble_weights is None:
            weights_path = ensemble_weights_file or "config/ensemble_weights.json"
            ensemble_weights = load_ensemble_weights(weights_path)

        horizons = sorted(merged["horizon"].unique())
        by_horizon: Dict[str, Dict[str, float]] = {}

        for h in horizons:
            subset = merged[merged["horizon"] == h].dropna(
                subset=["actual", "xgb_pred", "prophet_pred"]
            )
            if subset.empty:
                continue

            weights = get_weights_for_horizon(int(h), ensemble_weights)
            ensemble_pred = _compute_ensemble_series(subset, weights)
            subset = subset.copy()
            subset["ensemble_pred"] = ensemble_pred

            margin_lower, margin_upper, coverage, mean_width, n = _horizon_margins(
                subset["actual"].values,
                ensemble_pred,
                self.config.coverage_level,
                self.config.asymmetric,
            )
            by_horizon[str(h)] = {
                "margin_lower": margin_lower,
                "margin_upper": margin_upper,
                "empirical_coverage": coverage,
                "mean_interval_width": mean_width,
                "n": n,
            }

        if source_report is None:
            source_report = Path(walk_forward_csv).stem.replace("_walk_forward_predictions", "")

        payload = {
            "source_report": source_report,
            "calibrated_at": datetime.now().isoformat(),
            "coverage_level": self.config.coverage_level,
            "asymmetric": self.config.asymmetric,
            "by_horizon": by_horizon,
        }

        return ConformalIntervalCalibrationResult(
            intervals_payload=payload,
            merged_predictions=merged,
        )

    def save(self, result: ConformalIntervalCalibrationResult, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.intervals_payload, f, indent=2)
        return str(path)
