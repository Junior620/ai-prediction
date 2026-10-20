"""Plan a cocoa comparison that only trains on prices already known.

The served lot is not replaced. A chain without the next session is not
evidence that revisions are stable.
"""

from __future__ import annotations

from typing import Any, Sequence

from src.models.hybrid_features import future_business_date

OUT_OF_SAMPLE_NOTE = (
    "Évaluation hors entraînement : à chaque origine, les exemples ont une cible "
    "déjà connue. k et la règle de révision sont choisis avant la période réservée. "
    "Aucune variante ne remplace le prix servi."
)


def _day(value) -> str:
    return str(value)[:10]


def _shift(day: str, sessions: int) -> str:
    return future_business_date(day, sessions).date().isoformat()


def known_targets(dates: Sequence[str], as_of: str, horizon: int) -> list[str]:
    """Target dates of rows whose outcome is already known on as_of."""
    cutoff = _day(as_of)
    found = []
    for day in dates:
        if _day(day) > cutoff:
            continue
        target = _shift(_day(day), horizon)
        if target <= cutoff:
            found.append(target)
    return found


def matured_origins(dates: Sequence[str], lookup: dict[str, float], horizon: int) -> list[str]:
    origins = []
    for day in dates:
        if _shift(_day(day), horizon) in lookup:
            origins.append(_day(day))
    return origins


def split_windows(
    dates: Sequence[str],
    lookup: dict[str, float],
    *,
    horizon: int = 30,
    calibration: int = 8,
    reserved: int = 8,
    gap: int = 30,
) -> dict[str, list[str]]:
    """Reserve the last origins. Calibration ends before a gap of `gap` sessions."""
    matured = matured_origins(dates, lookup, horizon)
    if len(matured) < reserved + calibration + gap:
        raise ValueError("Pas assez de séances mûres pour séparer calibration et période réservée")
    test = matured[-reserved:]
    first_test = test[0]
    eligible = [
        day
        for day in matured
        if _shift(_shift(day, horizon), gap) <= first_test and day < first_test
    ]
    if len(eligible) < calibration:
        raise ValueError("La séparation de 30 séances ne laisse pas de calibration")
    chosen = eligible[-calibration:]
    last_target = _shift(chosen[-1], horizon)
    if _shift(last_target, gap) > first_test:
        raise ValueError("Une cible de calibration entre dans la période réservée")
    return {"calibration": chosen, "reserved": test, "gap_sessions": [str(gap)]}


def walk_days(origin: str, horizon: int) -> list[tuple[str, int, str]]:
    """Each later session keeps the same target and the remaining horizon."""
    target = _shift(origin, horizon)
    rows = []
    day = origin
    for remaining in range(horizon, 0, -1):
        rows.append((day, remaining, target))
        if remaining == 1:
            break
        day = _shift(day, 1)
    return rows


def next_session_present(rows: Sequence[dict[str, Any]]) -> bool:
    """True when the session after the opening has its own estimate."""
    if len(rows) < 2:
        return False
    ordered = sorted(rows, key=lambda item: (str(item.get("session")), -int(item.get("horizon") or 0)))
    opening = ordered[0]
    expected = _shift(str(opening["session"]), 1)
    return any(str(item.get("session"))[:10] == expected and item is not opening for item in ordered)
