"""Index of stored reports. A report is history, not a validation of the loaded lot."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

_STAMP = re.compile(r"(20\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")

PATH_MARKETS = ("cocoa", "coffee_robusta")
_META_MARKETS = {
    "cocoa": "cocoa",
    "ice_ny": "cocoa",
    "coffee": "coffee_robusta",
    "coffee_robusta": "coffee_robusta",
}


def _read(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _period(payload: dict) -> Optional[str]:
    origins = payload.get("origins") or []
    if origins:
        return f"{origins[0]} – {origins[-1]}"
    start = payload.get("origin_start")
    end = payload.get("origin_end")
    if start or end:
        return f"{start or '?'} – {end or '?'}"
    return None


def _counts(payload: dict) -> Optional[str]:
    policies = payload.get("policies")
    if isinstance(policies, list) and policies:
        parts = []
        for row in policies:
            if not isinstance(row, dict):
                continue
            horizon = row.get("initial_horizon")
            mature = row.get("mature_initials")
            policy = row.get("policy")
            if horizon is None or mature is None:
                continue
            parts.append(f"{policy} J+{horizon}: {mature}")
        if parts:
            return ", ".join(parts[:4])
    origins = payload.get("n_origins")
    if origins is not None:
        return f"origines {origins}"
    return None


def _timestamp(path: Path, payload: dict) -> Optional[str]:
    """Prefer the time the report was written. A study cutoff is not that time."""
    raw = payload.get("timestamp")
    if raw and "T" in str(raw):
        return str(raw)
    match = _STAMP.search(path.name)
    if match:
        year, month, day, hour, minute, second = match.groups()
        return f"{year}-{month}-{day}T{hour}:{minute}:{second}"
    if raw or payload.get("as_of"):
        return str(raw or payload.get("as_of"))
    return None


def _item(path: Path, root: Path, kind: str) -> dict:
    payload = _read(path)
    protocol = payload.get("validation_type") or payload.get("note") or kind
    if kind == "revision_stability":
        protocol = "rejeu des révisions, aucune cadence retenue"
    return {
        "report_id": str(path.relative_to(root / "reports")).replace("\\", "/"),
        "timestamp": _timestamp(path, payload),
        "kind": kind,
        "period": _period(payload),
        "protocol": str(protocol)[:180],
        "counts": _counts(payload),
        "compatible_with_current_lot": False,
        "note": "Rapport historique. Il ne valide pas le lot chargé.",
    }


def assigned_market(path: Path, payload: dict, reports: Path) -> Optional[str]:
    """Market from the report metadata, then from an explicit path segment.

    A contradiction, or a path that names both markets, is not assigned.
    A walk-forward file with neither clue stays with cocoa: that tree was
    written before the market segment existed.
    """
    try:
        parts = path.relative_to(reports).parts
    except ValueError:
        return None
    named = [part for part in parts if part in PATH_MARKETS]
    if len(set(named)) > 1:
        return None
    path_market = named[0] if named else None
    raw = payload.get("market") or payload.get("market_id")
    meta_market = None
    if raw is not None:
        meta_market = _META_MARKETS.get(str(raw).strip().lower())
        if meta_market is None:
            return None
    if meta_market and path_market and meta_market != path_market:
        return None
    if meta_market or path_market:
        return meta_market or path_market
    if parts and parts[0] == "walk_forward" and "coffee_robusta" not in parts:
        return "cocoa"
    return None


def list_market_reports(market_id: str, root: Path) -> list[dict]:
    reports = root / "reports"
    items: list[dict] = []
    for folder_name, kind in (
        ("walk_forward", "walk_forward"),
        ("revision_stability", "revision_stability"),
    ):
        folder = reports / folder_name
        if not folder.exists():
            continue
        for path in folder.rglob("*_summary.json"):
            payload = _read(path)
            if assigned_market(path, payload, reports) != market_id:
                continue
            items.append(_item(path, root, kind))
    items.sort(key=lambda item: (item.get("timestamp") or "", item["report_id"]), reverse=True)
    return items
