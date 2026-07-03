"""Calibrate ensemble weights from walk-forward and N-HiTS backtest CSVs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.ensemble_weights import DEFAULT_FALLBACK
from src.validation.metrics import compute_horizon_metrics


@dataclass
class EnsembleCalibratorConfig:
    grid_step: float = 0.05
    fixed_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FALLBACK))


@dataclass
class EnsembleCalibrationResult:
    weights_payload: Dict[str, Any]
    merged_predictions: pd.DataFrame
    comparison: Dict[str, Dict[str, float]]


def _normalize_origin(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col]).dt.normalize()


def merge_backtest_predictions(
    walk_forward_csv: str,
    nhits_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Merge walk-forward and optional N-HiTS predictions on (origin, horizon)."""
    wf = pd.read_csv(walk_forward_csv)
    wf["origin_date"] = _normalize_origin(wf, "origin_date")
    wf["horizon"] = wf["horizon"].astype(int)

    if nhits_csv and Path(nhits_csv).exists():
        nh = pd.read_csv(nhits_csv)
        nh["origin_date"] = _normalize_origin(nh, "cutoff")
        nh["horizon"] = nh["horizon"].astype(int)
        merged = wf.merge(
            nh[["origin_date", "horizon", "nhits_pred"]],
            on=["origin_date", "horizon"],
            how="left",
        )
    else:
        merged = wf.copy()
        merged["nhits_pred"] = np.nan

    return merged


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred) & (actual != 0)
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def _grid_weights_2d(step: float) -> List[tuple[float, float]]:
    vals = np.arange(0.0, 1.0 + step / 2, step)
    return [(float(a), float(1.0 - a)) for a in vals]


def _grid_weights_3d(step: float) -> List[tuple[float, float, float]]:
    vals = np.arange(0.0, 1.0 + step / 2, step)
    combos = []
    for w_xgb, w_nhits, w_prophet in product(vals, repeat=3):
        if abs(w_xgb + w_nhits + w_prophet - 1.0) < step / 2:
            combos.append((float(w_xgb), float(w_nhits), float(w_prophet)))
    return combos


def _optimize_horizon(
    subset: pd.DataFrame,
    step: float,
    xgb_only: bool,
) -> tuple[Dict[str, float], float, float]:
    """Return best weights, fixed mape, calibrated mape for one horizon."""
    actual = subset["actual"].values
    xgb = subset["xgb_pred"].values
    prophet = subset["prophet_pred"].values
    nhits = subset["nhits_pred"].values if "nhits_pred" in subset.columns else np.full(len(subset), np.nan)
    has_nhits = np.isfinite(nhits).sum() >= max(3, len(subset) // 4)

    fixed = DEFAULT_FALLBACK
    if has_nhits and not xgb_only:
        fixed_pred = (
            fixed["xgb"] * xgb + fixed["nhits"] * nhits + fixed["prophet"] * prophet
        )
    else:
        total = fixed["xgb"] + fixed["prophet"]
        fixed_pred = (fixed["xgb"] * xgb + fixed["prophet"] * prophet) / total

    fixed_mape = _mape(actual, fixed_pred)
    best_mape = fixed_mape
    best_weights = {"xgb": fixed["xgb"], "prophet": fixed["prophet"]}
    if has_nhits and not xgb_only:
        best_weights["nhits"] = fixed["nhits"]

    if has_nhits and not xgb_only:
        for w_xgb, w_nhits, w_prophet in _grid_weights_3d(step):
            pred = w_xgb * xgb + w_nhits * nhits + w_prophet * prophet
            mape = _mape(actual, pred)
            if mape < best_mape:
                best_mape = mape
                best_weights = {"xgb": w_xgb, "nhits": w_nhits, "prophet": w_prophet}
    else:
        for w_xgb, w_prophet in _grid_weights_2d(step):
            pred = w_xgb * xgb + w_prophet * prophet
            mape = _mape(actual, pred)
            if mape < best_mape:
                best_mape = mape
                best_weights = {"xgb": w_xgb, "prophet": w_prophet, "nhits": 0.0}

    return best_weights, fixed_mape, best_mape


class EnsembleCalibrator:
    """Optimize ensemble weights per horizon on backtest predictions."""

    def __init__(self, config: Optional[EnsembleCalibratorConfig] = None):
        self.config = config or EnsembleCalibratorConfig()

    def run(
        self,
        walk_forward_csv: str,
        nhits_csv: Optional[str] = None,
        xgb_only: bool = False,
        source_report: Optional[str] = None,
    ) -> EnsembleCalibrationResult:
        merged = merge_backtest_predictions(walk_forward_csv, nhits_csv)
        horizons = sorted(merged["horizon"].unique())

        by_horizon: Dict[str, Dict[str, float]] = {}
        fixed_mape: Dict[str, float] = {}
        calibrated_mape: Dict[str, float] = {}

        for h in horizons:
            subset = merged[merged["horizon"] == h].dropna(subset=["actual", "xgb_pred", "prophet_pred"])
            if subset.empty:
                continue
            weights, f_mape, c_mape = _optimize_horizon(
                subset, self.config.grid_step, xgb_only=xgb_only or subset["nhits_pred"].notna().sum() < 3
            )
            by_horizon[str(h)] = weights
            fixed_mape[str(h)] = f_mape
            calibrated_mape[str(h)] = c_mape

        if source_report is None:
            source_report = Path(walk_forward_csv).stem.replace("_walk_forward_predictions", "")

        payload = {
            "source_report": source_report,
            "calibrated_at": datetime.now().isoformat(),
            "by_horizon": by_horizon,
            "fixed_baseline_mape": fixed_mape,
            "calibrated_mape": calibrated_mape,
        }

        comparison = {
            "fixed": fixed_mape,
            "calibrated": calibrated_mape,
        }

        return EnsembleCalibrationResult(
            weights_payload=payload,
            merged_predictions=merged,
            comparison=comparison,
        )

    def save(self, result: EnsembleCalibrationResult, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.weights_payload, f, indent=2)
        return str(path)
