"""Shared direct-horizon forecast used by the API and the walk-forward."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.models.conformal_intervals import apply_interval, get_margins_for_horizon, heuristic_interval
from src.models.direct_horizon_trainer import DirectHorizonTrainer
from src.models.ensemble_weights import combine_ensemble
from src.models.hybrid_features import direct_feature_row, guard_forecast


def direct_level(
    model,
    last_row: pd.Series,
    feature_cols: Sequence[str],
    origin_price: float,
    target: str,
) -> float:
    """Price level from a direct h-step model, same row as training."""
    cols = list(feature_cols)
    row = direct_feature_row(last_row, feature_cols=cols)
    raw = float(model.predict(row[cols])[0])
    return DirectHorizonTrainer.level_from_prediction(raw, origin_price, target)


def publish_or_close(items, release: Optional[dict], spot: Optional[float]) -> None:
    """Unvalidated horizons publish the close. The candidate price stays beside it."""
    from src.models.release_manifest import horizon_is_validated

    if spot is None:
        return
    spot_value = float(spot)
    for item in items:
        components = dict(getattr(item, "components", None) or {})
        candidate = components.get("candidate_price", item.price)
        components["candidate_price"] = float(candidate)
        if release and horizon_is_validated(release, int(item.horizon)):
            components["served_as_close"] = False
        else:
            item.price = spot_value
            item.confidence_interval = [spot_value, spot_value]
            components["served_as_close"] = True
        item.components = components


def snapshot_journal_fields(components: dict, later_read: dict | None = None) -> dict:
    """Journal the session the model used. A later database read is ignored."""
    del later_read
    return {
        "origin_date": components.get("origin_date"),
        "origin_price": components.get("origin_price"),
        "target_date": components.get("target_date"),
        "snapshot_id": components.get("snapshot_id"),
    }


def recent_price_range(
    prices: pd.Series,
    days: int = 252,
    padding_pct: float = 15.0,
    price_bounds: Sequence[float] = (1000.0, 15000.0),
) -> tuple[float, float]:
    """Recent min and max, padded, then clipped to the market bounds."""
    tail = pd.to_numeric(prices, errors="coerce").dropna().tail(max(30, int(days)))
    if tail.empty:
        return float(price_bounds[0]), float(price_bounds[1])
    lo = float(tail.min())
    hi = float(tail.max())
    pad = float(padding_pct) / 100.0
    mid = (lo + hi) / 2.0 if hi > lo else lo
    span = max(hi - lo, mid * 0.05)
    lo2 = lo - span * pad
    hi2 = hi + span * pad
    return (max(float(price_bounds[0]), lo2), min(float(price_bounds[1]), hi2))


def compose_served_price(
    *,
    xgb_price: float,
    prophet_price: float,
    nhits_price: Optional[float],
    weights: Dict[str, float],
    spot: float,
    horizon: int,
    max_abs_change_pct: Optional[Dict[str, float]] = None,
    conformal_margins: Optional[Dict] = None,
    price_bounds: Sequence[float] = (1000.0, 15000.0),
    recent_prices: Optional[pd.Series] = None,
    recent_range_days: int = 252,
    recent_range_padding_pct: float = 15.0,
    garch_forecast: Any = None,
    confidence_level: float = 0.90,
    price_volatility: Optional[float] = None,
) -> Dict[str, Any]:
    """Published price shared by the API and the acceptance replay.

    Sentiment stays at zero: it cannot be rebuilt on past sessions, so it
    is not allowed to move the price that the backtest scores.
    A missing N-HiTS price renormalizes the remaining weights.
    """
    ensemble_price = combine_ensemble(xgb_price, prophet_price, nhits_price, weights)
    final_price, dampened_pct, feature_failure = publish_level(
        ensemble_price,
        spot,
        horizon,
        max_abs_change_pct,
        conformal_margins,
    )
    bounds = (float(price_bounds[0]), float(price_bounds[1]))
    if recent_prices is not None:
        bounds = recent_price_range(
            recent_prices,
            recent_range_days,
            recent_range_padding_pct,
            price_bounds,
        )
    final_price = float(np.clip(final_price, bounds[0], bounds[1]))
    raw_change_pct = (ensemble_price / spot - 1.0) * 100.0 if spot > 0 else 0.0

    if conformal_margins:
        lower_bound, upper_bound = apply_interval(
            final_price, horizon, conformal_margins, bounds
        )
        if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
            vol = float(price_volatility) if price_volatility is not None else 0.0
            lower_bound, upper_bound = heuristic_interval(
                final_price, horizon, vol, bounds, confidence_level, 1.0
            )
    else:
        vol = float(price_volatility) if price_volatility is not None else 0.0
        lower_bound, upper_bound = heuristic_interval(
            final_price, horizon, vol, bounds, confidence_level, 1.0
        )

    garch_ann_vol = None
    high_vol_regime = False
    if garch_forecast is not None:
        garch_interval = garch_forecast.interval_around(final_price, horizon, confidence_level)
        if garch_interval is not None:
            g_lo, g_hi = garch_interval
            lower_bound = min(lower_bound, g_lo)
            upper_bound = max(upper_bound, g_hi)
        garch_ann_vol = garch_forecast.annualized_volatility(horizon)
        high_vol_regime = garch_forecast.high_volatility_regime(horizon)
        lower_bound = float(np.clip(lower_bound, bounds[0], bounds[1]))
        upper_bound = float(np.clip(upper_bound, bounds[0], bounds[1]))

    return {
        "price": float(final_price),
        "lower": float(lower_bound),
        "upper": float(upper_bound),
        "feature_failure": bool(feature_failure),
        "ensemble": float(ensemble_price),
        "sentiment": 0.0,
        "dampened_change_pct": float(dampened_pct),
        "raw_change_pct": float(raw_change_pct),
        "weights": dict(weights),
        "garch_annualized_volatility": (
            float(garch_ann_vol) if garch_ann_vol is not None else None
        ),
        "high_volatility_regime": bool(high_vol_regime),
        "scored_variant": "no_sentiment",
    }


def publish_level(
    price: float,
    spot: float,
    horizon: int,
    max_abs_change_pct: Optional[Dict[str, float]] = None,
    conformal_margins: Optional[Dict] = None,
) -> tuple[float, float, bool]:
    """Central price after the cap and the far-from-spot guard."""
    caps = max_abs_change_pct or {}
    max_pct = float(caps.get(str(horizon), caps.get(horizon, 20.0)))
    margin = get_margins_for_horizon(horizon, conformal_margins or {})
    return guard_forecast(price, spot, max_pct, margin)
