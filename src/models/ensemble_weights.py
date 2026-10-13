"""Load and apply calibrated ensemble weights per horizon."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

DEFAULT_FALLBACK = {"xgb": 0.4, "nhits": 0.4, "prophet": 0.2}


def load_ensemble_weights(
    weights_file: str = "config/ensemble_weights.json",
    fallback: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Load per-horizon weights. Returns dict keyed by horizon string.

    Falls back to ``fallback`` or DEFAULT_FALLBACK for each horizon.
    """
    fb = fallback or DEFAULT_FALLBACK
    path = Path(weights_file)
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    by_horizon = data.get("by_horizon", {})
    return {str(h): w for h, w in by_horizon.items()}


def get_weights_for_horizon(
    horizon: int,
    weights_by_horizon: Dict[str, Dict[str, float]],
    fallback: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Return weights for a horizon, with fallback."""
    fb = fallback or DEFAULT_FALLBACK
    key = str(horizon)
    if key in weights_by_horizon:
        return weights_by_horizon[key]
    return fb.copy()


def _finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def combine_ensemble(
    xgb_price: float,
    prophet_price: float,
    nhits_price: Optional[float],
    weights: Dict[str, float],
) -> float:
    """Weighted combination. A required engine that is missing is not renormalized."""
    terms = []
    if float(weights.get("xgb", 0.0) or 0.0) > 0.0:
        if not _finite_number(xgb_price):
            return float("nan")
        terms.append((float(weights["xgb"]), float(xgb_price)))
    if float(weights.get("prophet", 0.0) or 0.0) > 0.0:
        if not _finite_number(prophet_price):
            return float("nan")
        terms.append((float(weights["prophet"]), float(prophet_price)))
    if float(weights.get("nhits", 0.0) or 0.0) > 0.0:
        if not _finite_number(nhits_price):
            return float("nan")
        terms.append((float(weights["nhits"]), float(nhits_price)))
    if not terms:
        return float("nan")
    total = sum(weight for weight, _ in terms)
    if total <= 0:
        return float("nan")
    return sum(weight * price for weight, price in terms) / total
