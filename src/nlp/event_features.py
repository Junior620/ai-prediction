"""Daily event features visible at a forecast instant. Later rows stay out."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from src.nlp.event_rules import load_rules, parse_utc

WINDOWS = (7, 14, 30, 60, 90)


def visible_versions(versions: Sequence[Dict[str, Any]], origin: Any) -> List[Dict[str, Any]]:
    """Latest version of each URL whose availability and classification are at or before origin."""
    moment = parse_utc(origin)
    if moment is None:
        return []
    chosen: Dict[str, Dict[str, Any]] = {}
    for row in versions:
        available = parse_utc(row.get("available_at"))
        classified = parse_utc(row.get("classified_at"))
        if available is None or classified is None:
            continue
        if available > moment or classified > moment:
            continue
        url = str(row.get("url") or "")
        if not url:
            continue
        current = chosen.get(url)
        if current is None or parse_utc(current["classified_at"]) < classified:
            chosen[url] = row
    return list(chosen.values())


def feature_row(versions: Sequence[Dict[str, Any]], origin: Any) -> Dict[str, float]:
    rules = load_rules()
    categories = list((rules.get("categories") or {}).keys())
    moment = parse_utc(origin)
    rows = visible_versions(versions, origin)
    values: Dict[str, float] = {}
    for window in WINDOWS:
        start_ok = []
        if moment is not None:
            for row in rows:
                seen = parse_utc(row.get("available_at"))
                if seen is None:
                    continue
                age_days = (moment - seen).total_seconds() / 86400.0
                if 0 <= age_days <= window:
                    start_ok.append(row)
        counted = _unique_events(start_ok)
        for category in categories:
            subset = [row for row in counted if row.get("category") == category]
            values[f"{category}_count_{window}"] = float(len(subset))
            values[f"{category}_severity_{window}"] = float(sum(float(row.get("severity") or 0) for row in subset))
            up = sum(1 for row in subset if row.get("economic_direction") == "hausse")
            down = sum(1 for row in subset if row.get("economic_direction") == "baisse")
            values[f"{category}_net_{window}"] = float(up - down)
        values[f"ci_count_{window}"] = float(sum(1 for row in counted if row.get("country") == "CI"))
        values[f"gh_count_{window}"] = float(sum(1 for row in counted if row.get("country") == "GH"))
        values[f"other_count_{window}"] = float(
            sum(1 for row in counted if row.get("country") not in (None, "CI", "GH"))
        )
        values[f"important_count_{window}"] = float(sum(1 for row in counted if row.get("important")))
    return values


def feature_frame(versions: Sequence[Dict[str, Any]], origins: Iterable[Any]) -> pd.DataFrame:
    records = []
    for origin in origins:
        row = feature_row(versions, origin)
        row["origin"] = parse_utc(origin)
        records.append(row)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def redundant_pairs(frame: pd.DataFrame, threshold: float = 0.95) -> List[tuple[str, str, float]]:
    """Pairs of numeric columns that move together. They stay out of a candidate until they add skill."""
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < 2 or len(numeric) < 5:
        return []
    corr = numeric.corr()
    found = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                found.append((left, right, float(value)))
    return found


def _unique_events(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for row in rows:
        key = str(row.get("event_key") or row.get("url"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique
