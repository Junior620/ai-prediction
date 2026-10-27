"""The calendar named by a lot must travel with every descriptor of that lot."""

from __future__ import annotations

import json
from typing import Any

from src.models.hybrid_features import SERVICE_CALENDAR
from src.models.release_manifest import artifact_path

# Calendrier des nouveaux lots. Un lot déjà entraîné garde celui qu'il déclare (require_calendar).
# Robusta : trading_sessions depuis le lot B (série Databento sans saut, plus de lignes de week-end).
MARKET_CALENDAR = {
    "cocoa": "trading_sessions",
    "coffee_robusta": "trading_sessions",
}


class CalendarOmitted(ValueError):
    """A lot that names its calendar was asked to run without that field."""


def require_calendar(release: dict[str, Any] | None, meta: dict[str, Any] | None = None) -> str:
    """Return the calendar to apply. A declared calendar cannot be dropped."""
    release = release or {}
    if meta is None:
        meta = _metadata(release)
    declared = str(meta["calendar"]) if meta is not None and "calendar" in meta else None
    carried = release.get("calendar")
    if declared is not None and not carried:
        raise CalendarOmitted(f"Calendrier déclaré omis: {declared}")
    if declared is not None and str(carried) != declared:
        raise CalendarOmitted(
            f"Calendrier du descripteur {carried}, métadonnées {declared}"
        )
    if carried:
        return str(carried)
    return SERVICE_CALENDAR


def _metadata(release: dict[str, Any]) -> dict[str, Any] | None:
    artifacts = release.get("artifacts") or {}
    if not artifacts.get("improved_info"):
        return None
    try:
        path = artifact_path(release, "improved_info")
    except KeyError:
        return None
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
