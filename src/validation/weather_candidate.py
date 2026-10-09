"""Weather candidate versus the current cocoa model.

A report can be written from any chronological split. The active manifest is
left untouched unless every horizon beats the current model by 10 percent MAE,
MAPE and fallback do not degrade, J+30 passes the frozen close test, and the
scored targets all fall after 2026-10-02.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

WEATHER_HORIZONS = (30, 60, 90)
SPLIT_DATE = "2025-01-01"
UNUSED_TARGET_AFTER = "2026-10-02"
MAE_GAIN_MIN = 0.10
MAPE_SLACK_PP = 0.5
FALLBACK_SLACK = 0.02


def score_predictions(
    actual: Sequence[float],
    predicted: Sequence[float],
    naive: Sequence[float],
    target_dates: Sequence[Any],
    unused_after: str = UNUSED_TARGET_AFTER,
) -> Dict[str, Any]:
    """MAE and MAPE. A missing prediction is scored as the close and counted as a fallback."""
    y = np.asarray(list(actual), dtype=float)
    pred = np.asarray(list(predicted), dtype=float)
    base = np.asarray(list(naive), dtype=float)
    dates = pd.to_datetime(pd.Series(list(target_dates)))
    usable = np.isfinite(y) & np.isfinite(base) & (y != 0)
    if not usable.any():
        return {
            "n": 0,
            "mae": None,
            "mape": None,
            "fallback_rate": None,
            "n_after_cutoff": 0,
        }
    scored = np.where(np.isfinite(pred), pred, base)[usable]
    truth = y[usable]
    kept_dates = dates[usable]
    fallback = ~np.isfinite(pred[usable])
    cutoff = pd.Timestamp(unused_after)
    return {
        "n": int(len(truth)),
        "mae": float(np.mean(np.abs(truth - scored))),
        "mape": float(np.mean(np.abs((truth - scored) / truth)) * 100.0),
        "fallback_rate": float(np.mean(fallback)),
        "n_after_cutoff": int((kept_dates > cutoff).sum()),
    }


def decide_weather_promotion(
    per_horizon: Iterable[Mapping[str, Any]],
    acceptance_j30: bool,
) -> Dict[str, Any]:
    """Both the MAE rule and the frozen J+30 close test are required."""
    reasons: List[str] = []
    rows = list(per_horizon)
    if not rows:
        reasons.append("Aucune comparaison d'horizon.")
    for row in rows:
        horizon = int(row["horizon"])
        if int(row.get("n_after_cutoff") or 0) < 1:
            reasons.append(
                f"J+{horizon} : aucune cible réalisée après le {UNUSED_TARGET_AFTER}."
            )
        current_mae = row.get("current_mae")
        candidate_mae = row.get("candidate_mae")
        if current_mae is None or candidate_mae is None or float(current_mae) <= 0:
            reasons.append(f"J+{horizon} : MAE indisponible.")
        else:
            gain = (float(current_mae) - float(candidate_mae)) / float(current_mae)
            if gain < MAE_GAIN_MIN:
                reasons.append(
                    f"J+{horizon} : baisse de MAE de {gain:.1%}, sous les 10 %."
                )
        current_mape = row.get("current_mape")
        candidate_mape = row.get("candidate_mape")
        if (
            current_mape is None
            or candidate_mape is None
            or float(candidate_mape) > float(current_mape) + MAPE_SLACK_PP
        ):
            reasons.append(f"J+{horizon} : le MAPE se dégrade.")
        current_fallback = row.get("current_fallback")
        candidate_fallback = row.get("candidate_fallback")
        if (
            current_fallback is None
            or candidate_fallback is None
            or float(candidate_fallback) > float(current_fallback) + FALLBACK_SLACK
        ):
            reasons.append(f"J+{horizon} : le repli vers la clôture se dégrade.")
    if not acceptance_j30:
        reasons.append(
            "J+30 ne tient pas le critère figé contre la clôture inchangée."
        )
    return {"promote": len(reasons) == 0, "reasons": reasons}
