"""Cocoa-belt weather: live snapshots for display, daily history for a candidate.

The live table ``weather_data`` is a WeatherAPI snapshot. The price model does
not read it. Training uses a separate daily series from NASA POWER.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_MISSING = -900.0
CI_WEIGHT = 2.0 / 3.0
GH_WEIGHT = 1.0 / 3.0

# WeatherAPI cities already known to the live collector.
SNAPSHOT_LOCATIONS = (
    "Abidjan,Ivory Coast",
    "Yamoussoukro,Ivory Coast",
    "Accra,Ghana",
    "Kumasi,Ghana",
)

# Production belts, not only the capitals.
COCOA_WEATHER_POINTS: tuple[dict[str, Any], ...] = (
    {"region": "soubre", "country": "CI", "latitude": 5.79, "longitude": -6.59},
    {"region": "daloa", "country": "CI", "latitude": 6.88, "longitude": -6.45},
    {"region": "abengourou", "country": "CI", "latitude": 6.73, "longitude": -3.50},
    {"region": "kumasi", "country": "GH", "latitude": 6.69, "longitude": -1.62},
    {"region": "sunyani", "country": "GH", "latitude": 7.34, "longitude": -2.33},
    {"region": "kade", "country": "GH", "latitude": 6.09, "longitude": -0.83},
)

_ORIGIN_TOKENS = ("ivory", "ivoire", "ghana")


def is_cocoa_origin(row: Dict[str, Any]) -> bool:
    blob = f"{row.get('country') or ''} {row.get('location') or ''}".lower()
    return any(token in blob for token in _ORIGIN_TOKENS)


def latest_origin_weather(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Latest snapshot per location, Côte d'Ivoire and Ghana only."""
    chosen: Dict[str, Dict[str, Any]] = {}
    ordered = sorted(
        rows,
        key=lambda row: str(row.get("collected_at") or ""),
        reverse=True,
    )
    for row in ordered:
        if not is_cocoa_origin(row):
            continue
        location = str(row.get("location") or row.get("name") or "").strip()
        if not location or location in chosen:
            continue
        country = str(row.get("country") or "")
        chosen[location] = {
            "location": location,
            "country": country,
            "temperature_c": _optional_float(row.get("temperature_c")),
            "precipitation_mm": _optional_float(row.get("precipitation_mm")),
            "humidity": _optional_int(row.get("humidity")),
            "condition": row.get("condition"),
            "collected_at": row.get("collected_at"),
        }
    return list(chosen.values())


def parse_nasa_power_payload(
    payload: Dict[str, Any],
    point: Dict[str, Any],
) -> pd.DataFrame:
    """Map one NASA POWER daily JSON payload to point rows."""
    parameter = ((payload.get("properties") or {}).get("parameter") or {})
    temps = parameter.get("T2M") or {}
    temp_min = parameter.get("T2M_MIN") or {}
    temp_max = parameter.get("T2M_MAX") or {}
    rain = parameter.get("PRECTOTCORR") or {}
    dates = sorted(set(temps) | set(temp_min) | set(temp_max) | set(rain))
    rows = []
    for key in dates:
        if not str(key).isdigit():
            continue
        rows.append(
            {
                "date": datetime.strptime(str(key), "%Y%m%d").date().isoformat(),
                "region": point["region"],
                "country": point["country"],
                "latitude": float(point["latitude"]),
                "longitude": float(point["longitude"]),
                "precip_mm": _nasa_float(rain.get(key)),
                "temp_c": _nasa_float(temps.get(key)),
                "temp_min_c": _nasa_float(temp_min.get(key)),
                "temp_max_c": _nasa_float(temp_max.get(key)),
                "source": "nasa_power",
            }
        )
    return pd.DataFrame(rows)


def fetch_nasa_power_point(
    point: Dict[str, Any],
    start: str,
    end: str,
    session: Any = None,
) -> pd.DataFrame:
    """Download one point. ``start`` and ``end`` are YYYY-MM-DD."""
    import requests

    getter = session.get if session is not None else requests.get
    response = getter(
        NASA_POWER_URL,
        params={
            "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR",
            "community": "AG",
            "longitude": point["longitude"],
            "latitude": point["latitude"],
            "start": start.replace("-", ""),
            "end": end.replace("-", ""),
            "format": "JSON",
        },
        timeout=120,
    )
    response.raise_for_status()
    return parse_nasa_power_payload(response.json(), point)


def blend_country_daily(points: pd.DataFrame) -> pd.DataFrame:
    """Country means, then about two thirds Côte d'Ivoire and one third Ghana.

    A day missing one country keeps the countries that reported, with their
    weights renormalized. No future day is read.
    """
    if points.empty:
        return pd.DataFrame(
            columns=["date", "precip_mm", "temp_c", "temp_min_c", "temp_max_c"]
        )
    frame = points.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    for column in ("precip_mm", "temp_c", "temp_min_c", "temp_max_c"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    country = (
        frame.groupby(["date", "country"], as_index=False)[
            ["precip_mm", "temp_c", "temp_min_c", "temp_max_c"]
        ]
        .mean()
    )
    weights = {"CI": CI_WEIGHT, "GH": GH_WEIGHT}
    blended: List[Dict[str, Any]] = []
    for day, group in country.groupby("date"):
        total = 0.0
        acc = {name: 0.0 for name in ("precip_mm", "temp_c", "temp_min_c", "temp_max_c")}
        for _, row in group.iterrows():
            weight = weights.get(str(row["country"]))
            if weight is None or not np.isfinite(row["precip_mm"]) or not np.isfinite(row["temp_c"]):
                continue
            total += weight
            for name in acc:
                value = row[name]
                acc[name] += weight * (float(value) if np.isfinite(value) else 0.0)
        if total <= 0:
            continue
        blended.append(
            {
                "date": pd.Timestamp(day),
                **{name: acc[name] / total for name in acc},
            }
        )
    out = pd.DataFrame(blended)
    if out.empty:
        return out
    return out.sort_values("date").reset_index(drop=True)


def rows_for_supabase(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    records = []
    for row in frame.to_dict(orient="records"):
        day = row["date"]
        if hasattr(day, "date"):
            day = pd.Timestamp(day).date().isoformat()
        records.append(
            {
                "date": str(day)[:10],
                "region": row["region"],
                "country": row["country"],
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "precip_mm": _optional_float(row.get("precip_mm")),
                "temp_c": _optional_float(row.get("temp_c")),
                "temp_min_c": _optional_float(row.get("temp_min_c")),
                "temp_max_c": _optional_float(row.get("temp_max_c")),
                "source": row.get("source") or "nasa_power",
            }
        )
    return records


def _nasa_float(value: Any) -> Optional[float]:
    number = _optional_float(value)
    if number is None or number <= NASA_MISSING:
        return None
    return number


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _optional_int(value: Any) -> Optional[int]:
    number = _optional_float(value)
    if number is None:
        return None
    return int(number)


def iter_points() -> Iterable[Dict[str, Any]]:
    return COCOA_WEATHER_POINTS
