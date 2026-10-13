"""Weights and caps for horizons 1 to 30.

Anchors 1, 7, 14 and 30 keep their stored values. Every other horizon is the
linear mix of the two surrounding anchors, then the weights are normalised.
Conformal margins are not interpolated.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

ANCHORS: tuple[int, ...] = (1, 7, 14, 30)
OFFICIAL_HORIZONS: tuple[int, ...] = (1, 7, 14, 30)
FROZEN_HORIZONS: tuple[int, ...] = (7, 14, 30)
TRACKED_HORIZONS: tuple[int, ...] = tuple(range(1, 31))


def _as_float_map(values: Mapping) -> Dict[str, float]:
    return {str(key): float(value or 0.0) for key, value in values.items()}


def _normalize(weights: Mapping[str, float]) -> Dict[str, float]:
    cleaned = {str(key): max(0.0, float(value or 0.0)) for key, value in weights.items()}
    total = sum(cleaned.values())
    if total <= 0.0:
        return {key: 0.0 for key in cleaned}
    return {key: value / total for key, value in cleaned.items()}


def _anchors_present(keys: Sequence[int]) -> tuple[int, ...]:
    present = [int(key) for key in ANCHORS if int(key) in set(int(item) for item in keys)]
    return tuple(present)


def _bracket(horizon: int, anchors: Sequence[int]) -> tuple[int, int, float]:
    ordered = tuple(sorted(int(item) for item in anchors))
    if not ordered:
        raise ValueError("Aucun ancrage")
    h = int(horizon)
    if h <= ordered[0]:
        return ordered[0], ordered[0], 0.0
    if h >= ordered[-1]:
        return ordered[-1], ordered[-1], 0.0
    for left, right in zip(ordered, ordered[1:]):
        if left <= h <= right:
            span = right - left
            share = 0.0 if span == 0 else (h - left) / span
            return left, right, share
    return ordered[-1], ordered[-1], 0.0


def scheduled_weights(
    horizon: int,
    weights_by_horizon: Optional[Mapping] = None,
    fallback: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Stored anchor weights, or the normalised interpolation between them."""
    stored = {
        str(key): _as_float_map(value)
        for key, value in (weights_by_horizon or {}).items()
        if isinstance(value, Mapping)
    }
    key = str(int(horizon))
    if key in stored:
        return _normalize(stored[key])
    anchors = _anchors_present([int(name) for name in stored])
    if len(anchors) >= 2:
        left, right, share = _bracket(int(horizon), anchors)
        left_weights = stored[str(left)]
        right_weights = stored[str(right)]
        names = set(left_weights) | set(right_weights)
        mixed = {
            name: (1.0 - share) * float(left_weights.get(name, 0.0))
            + share * float(right_weights.get(name, 0.0))
            for name in names
        }
        return _normalize(mixed)
    if anchors:
        return _normalize(stored[str(anchors[0])])
    return _normalize(dict(fallback or {}))


def scheduled_cap(horizon: int, caps: Optional[Mapping] = None) -> float:
    """Stored cap, or the linear interpolation of the surrounding anchors."""
    stored = {int(key): float(value) for key, value in (caps or {}).items()}
    if int(horizon) in stored:
        return stored[int(horizon)]
    anchors = _anchors_present(list(stored)) or tuple(sorted(stored))
    if not anchors:
        return 20.0
    if len(anchors) == 1:
        return stored[anchors[0]]
    left, right, share = _bracket(int(horizon), anchors)
    return (1.0 - share) * stored[left] + share * stored[right]


def schedule_table(
    weights_by_horizon: Optional[Mapping] = None,
    caps: Optional[Mapping] = None,
) -> dict:
    """The procedure actually applied, so it can be hashed with the lot."""
    rows = []
    for horizon in TRACKED_HORIZONS:
        rows.append(
            {
                "horizon": horizon,
                "weights": scheduled_weights(horizon, weights_by_horizon),
                "max_abs_change_pct": scheduled_cap(horizon, caps),
                "margin": "exact_horizon_only",
            }
        )
    return {
        "rule": "linear_between_anchors_then_normalize",
        "anchors": list(ANCHORS),
        "official_horizons": list(OFFICIAL_HORIZONS),
        "margins": "not_interpolated",
        "rows": rows,
    }
