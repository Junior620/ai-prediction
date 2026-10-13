"""Shared direct-horizon forecast used by the API and the walk-forward."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.models.conformal_intervals import apply_interval, get_margins_for_horizon
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


def publish_or_close(
    items,
    release: Optional[dict],
    spot: Optional[float],
    *,
    release_mode: str = "active",
) -> None:
    """Keep the model price. An unvalidated horizon is experimental, never the close.

    Status is one of experimental, validated, unavailable. Experimental mode
    cannot emit validated. A failed calculation clears the price and the band.
    """
    from src.models.release_manifest import horizon_is_validated

    del spot
    experimental = release_mode == "experimental" or (
        isinstance(release, dict) and release.get("release_mode") == "experimental"
    )
    for item in items:
        components = dict(getattr(item, "components", None) or {})
        if components.get("display_final"):
            if release_mode == "experimental" and getattr(item, "status", None) == "validated":
                item.status = "experimental"
            continue
        raw_price = getattr(item, "price", None)
        try:
            finite_price = raw_price is not None and bool(np.isfinite(float(raw_price)))
        except (TypeError, ValueError):
            finite_price = False
        interval = list(getattr(item, "confidence_interval", None) or [])
        finite_band = False
        if len(interval) == 2:
            try:
                finite_band = all(v is not None and bool(np.isfinite(float(v))) for v in interval)
            except (TypeError, ValueError):
                finite_band = False
        if finite_price:
            components["candidate_price"] = float(raw_price)
        if finite_band:
            components["candidate_lower"] = float(interval[0])
            components["candidate_upper"] = float(interval[1])
        weights = components.get("ensemble_weights") or {}
        try:
            nhits_required = float(weights.get("nhits", 0.0) or 0.0) > 0.0
        except (TypeError, ValueError):
            nhits_required = False
        nhits_missing = nhits_required and components.get("nhits") is None
        positive_price = finite_price and float(raw_price) > 0
        if finite_band and positive_price:
            band = coherent_band(float(raw_price), float(interval[0]), float(interval[1]))
            if band is None or bool(components.get("band_unavailable")):
                finite_band = False
                item.confidence_interval = None
        elif bool(components.get("band_unavailable")):
            finite_band = False
            item.confidence_interval = None
        failed = (
            bool(components.get("calculation_failed"))
            or bool(components.get("engine_missing"))
            or not positive_price
            or nhits_missing
        )
        release_ok = bool(release) and horizon_is_validated(release, int(item.horizon))
        if failed:
            status = "unavailable"
        elif experimental or not release_ok:
            status = "experimental"
        else:
            status = "validated"
        if experimental and status == "validated":
            status = "experimental"
        components["status"] = status
        components["served_as_close"] = False
        if status == "unavailable":
            item.price = None
            item.confidence_interval = None
        elif status == "experimental":
            components["label"] = "Prévision expérimentale — non validée"
            if bool(components.get("large_move_warning")):
                components["warning"] = "Mouvement au-delà du seuil — prix conservé"
            if finite_band:
                components["band_label"] = "Bande expérimentale — couverture non démontrée"
            else:
                item.confidence_interval = None
        try:
            item.status = status
        except Exception:
            pass
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


def coherent_band(
    price: float,
    lower: float,
    upper: float,
) -> Optional[tuple[float, float]]:
    """A published band contains its own central price and is ordered."""
    try:
        center = float(price)
        lo = float(lower)
        hi = float(upper)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(center) and np.isfinite(lo) and np.isfinite(hi)):
        return None
    if lo > hi or center < lo or center > hi:
        return None
    if lo == hi:
        return None
    return lo, hi


def build_served_interval(
    price: float,
    horizon: int,
    *,
    margins: Optional[Dict] = None,
    price_bounds: Sequence[float] = (1000.0, 15000.0),
    recent_bounds: Optional[tuple] = None,
    garch_log_half: Optional[float] = None,
    garch_required: bool = False,
) -> tuple[Optional[float], Optional[float]]:
    """Final interval shared by the API and the historical replay.

    A missing margin, or a required GARCH width, leaves the band unpublished.
    An older margin is never substituted.
    """
    if not np.isfinite(price) or float(price) <= 0:
        return None, None
    if not margins:
        return None, None
    if garch_required and (garch_log_half is None or not np.isfinite(float(garch_log_half))):
        return None, None
    bounds = (
        (float(recent_bounds[0]), float(recent_bounds[1]))
        if recent_bounds is not None
        else (float(price_bounds[0]), float(price_bounds[1]))
    )
    try:
        lower, upper = apply_interval(float(price), horizon, margins, bounds)
    except Exception:
        return None, None
    if garch_log_half is not None and np.isfinite(float(garch_log_half)):
        half = float(garch_log_half)
        lower = min(float(lower), float(price) * float(np.exp(-half)))
        upper = max(float(upper), float(price) * float(np.exp(half)))
        lower = float(np.clip(lower, bounds[0], bounds[1]))
        upper = float(np.clip(upper, bounds[0], bounds[1]))
    band = coherent_band(float(price), lower, upper)
    if band is None:
        return None, None
    return band


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

    Sentiment stays at zero. A required engine that is missing is not
    replaced by a renormalized blend. A large move stays visible.
    """
    del price_volatility
    ensemble_price = combine_ensemble(xgb_price, prophet_price, nhits_price, weights)
    engine_missing = not np.isfinite(ensemble_price)
    if engine_missing:
        return {
            "price": float("nan"),
            "lower": None,
            "upper": None,
            "feature_failure": False,
            "large_move_warning": False,
            "engine_missing": True,
            "ensemble": float("nan"),
            "sentiment": 0.0,
            "dampened_change_pct": 0.0,
            "raw_change_pct": 0.0,
            "weights": dict(weights),
            "recent_lower": None,
            "recent_upper": None,
            "garch_log_half": None,
            "garch_annualized_volatility": None,
            "high_volatility_regime": False,
            "scored_variant": "no_sentiment",
        }
    final_price, dampened_pct, large_move = publish_level(
        ensemble_price,
        spot,
        horizon,
        max_abs_change_pct,
        conformal_margins,
    )
    recent_bounds = None
    if recent_prices is not None:
        recent_bounds = recent_price_range(
            recent_prices,
            recent_range_days,
            recent_range_padding_pct,
            price_bounds,
        )
    garch_log_half = None
    garch_ann_vol = None
    high_vol_regime = False
    garch_required = garch_forecast is not None
    if garch_forecast is not None and final_price > 0:
        garch_interval = garch_forecast.interval_around(final_price, horizon, confidence_level)
        if garch_interval is not None and float(garch_interval[0]) > 0:
            garch_log_half = float(np.log(float(garch_interval[1]) / float(final_price)))
        garch_ann_vol = garch_forecast.annualized_volatility(horizon)
        high_vol_regime = garch_forecast.high_volatility_regime(horizon)
    lower_bound, upper_bound = build_served_interval(
        final_price,
        horizon,
        margins=conformal_margins,
        price_bounds=price_bounds,
        recent_bounds=recent_bounds,
        garch_log_half=garch_log_half,
        garch_required=garch_required,
    )
    raw_change_pct = (ensemble_price / spot - 1.0) * 100.0 if spot > 0 else 0.0
    return {
        "price": float(final_price),
        "lower": lower_bound,
        "upper": upper_bound,
        "feature_failure": False,
        "large_move_warning": bool(large_move),
        "engine_missing": False,
        "ensemble": float(ensemble_price),
        "sentiment": 0.0,
        "dampened_change_pct": float(dampened_pct),
        "raw_change_pct": float(raw_change_pct),
        "weights": dict(weights),
        "recent_lower": None if recent_bounds is None else float(recent_bounds[0]),
        "recent_upper": None if recent_bounds is None else float(recent_bounds[1]),
        "garch_log_half": garch_log_half,
        "garch_required": bool(garch_required),
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
