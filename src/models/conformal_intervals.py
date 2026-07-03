"""Load and apply conformal prediction interval margins."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def load_conformal_margins(
    intervals_file: str = "config/conformal_intervals.json",
) -> Dict[str, Dict[str, float]]:
    """Load per-horizon margins. Returns empty dict if file absent."""
    path = Path(intervals_file)
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return {str(h): m for h, m in data.get("by_horizon", {}).items()}


def get_margins_for_horizon(
    horizon: int,
    margins_by_horizon: Dict[str, Dict[str, float]],
) -> Optional[Tuple[float, float]]:
    """Return (margin_lower, margin_upper) for horizon, or None if unavailable."""
    entry = margins_by_horizon.get(str(horizon))
    if not entry:
        return None
    lower = entry.get("margin_lower")
    upper = entry.get("margin_upper")
    if lower is None or upper is None or not np.isfinite(lower) or not np.isfinite(upper):
        return None
    return float(lower), float(upper)


def apply_interval(
    price: float,
    horizon: int,
    margins_by_horizon: Dict[str, Dict[str, float]],
    price_bounds: Tuple[float, float],
) -> Tuple[float, float]:
    """Apply conformal margins to a point prediction, clipped to price bounds."""
    margins = get_margins_for_horizon(horizon, margins_by_horizon)
    if margins is None:
        return float("nan"), float("nan")

    margin_lower, margin_upper = margins
    lower = max(price - margin_lower, price_bounds[0])
    upper = min(price + margin_upper, price_bounds[1])
    return float(lower), float(upper)


def heuristic_interval(
    price: float,
    horizon: int,
    price_volatility: float,
    price_bounds: Tuple[float, float],
    confidence_level: float = 0.90,
    sentiment_factor: float = 1.0,
) -> Tuple[float, float]:
    """Fallback heuristic interval when conformal margins are unavailable."""
    z = 1.645 if confidence_level <= 0.90 else 1.96
    uncertainty = price_volatility * 0.3 * np.sqrt(horizon / 30) * sentiment_factor
    lower = max(price - z * uncertainty, price_bounds[0])
    upper = min(price + z * uncertainty, price_bounds[1])
    return float(lower), float(upper)
