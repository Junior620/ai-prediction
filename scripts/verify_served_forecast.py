"""Confirm the API is serving the named lot with four finite prices."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.hybrid_features import future_business_date
HORIZONS = (1, 7, 14, 30)
API_MARKETS = {"cocoa": "ICE_NY", "coffee_robusta": "COFFEE_ROBUSTA"}
ALLOWED_STATUS = {"experimental", "validated"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _real_day(value) -> str | None:
    text = str(value or "")[:10]
    if not DATE_RE.match(text):
        return None
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None
    if parsed.strftime("%Y-%m-%d") != text:
        return None
    return text


def expected_target(origin: str, horizon: int) -> str:
    """Target of this horizon on the lot calendar, from the session origin."""
    return future_business_date(origin, int(horizon)).date().isoformat()


def _positive(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _fresh(item: dict) -> dict:
    fresh = item.get("fresh")
    return fresh if isinstance(fresh, dict) else {}


def assess_served(payload: dict, expected_version: str, expected_origin: str) -> tuple[int, list[str]]:
    """Return an exit code and the lines to print.

    0 means today's calculation prices the four horizons. 1 means the lot or
    the origin is wrong. 3 means the lot matches but at least one fresh
    horizon is missing a usable price. A kept initial does not replace it.
    """
    version = payload.get("model_version")
    if version != expected_version:
        return 1, [f"[ERREUR] version servie {version}, lot nommé {expected_version}"]
    origin = str(payload.get("current_date") or "")[:10]
    if origin != expected_origin:
        return 1, [f"[ERREUR] origine servie {origin or 'absente'}, instantané {expected_origin}"]
    by_horizon = {
        int(item.get("horizon")): item
        for item in payload.get("predictions") or []
        if item.get("horizon") is not None
    }
    failures = []
    for horizon in HORIZONS:
        item = by_horizon.get(horizon)
        if item is None:
            failures.append(f"J+{horizon} absent")
            continue
        fresh = _fresh(item)
        status = fresh.get("status")
        target = _real_day(fresh.get("target_date"))
        fresh_origin = _real_day(fresh.get("issued_date"))
        fresh_version = fresh.get("model_version")
        if not fresh.get("issued_date") or not fresh_version:
            failures.append(f"J+{horizon} provenance absente")
        if fresh_version != expected_version:
            failures.append(f"J+{horizon} lot {fresh_version}")
        if fresh_origin != expected_origin:
            failures.append(f"J+{horizon} origine {fresh_origin or 'absente'}")
        if status not in ALLOWED_STATUS:
            failures.append(f"J+{horizon} statut {status}")
        if not _positive(fresh.get("price")):
            failures.append(f"J+{horizon} prix absent")
        if target is None:
            failures.append(f"J+{horizon} échéance absente")
        elif fresh_origin == expected_origin and target != expected_target(expected_origin, horizon):
            failures.append(f"J+{horizon} échéance {target}")
    if failures:
        lines = [f"[PARTIEL] {expected_version} {origin}"]
        lines.extend(f"  {line}" for line in failures)
        return 3, lines
    lines = [f"  origine {origin}"]
    for horizon in HORIZONS:
        fresh = _fresh(by_horizon[horizon])
        lines.append(
            f"  J+{horizon} status={fresh.get('status')} prix={fresh.get('price')} "
            f"échéance={str(fresh.get('target_date') or '')[:10]}"
        )
    lines.append(f"[OK] {expected_version} consultable")
    return 0, lines


def assess_calendars(market: str, release: dict, announced: str | None) -> tuple[int, list[str]]:
    """Fail when the manifest, the lot metadata and the service disagree.

    A disagreement is not a partial success.
    """
    from src.models.release_calendar import MARKET_CALENDAR, CalendarOmitted, require_calendar

    expected = MARKET_CALENDAR[market]
    try:
        carried = require_calendar(release)
    except CalendarOmitted as exc:
        return 1, [f"[ERREUR] {exc}"]
    failures = []
    if carried != expected:
        failures.append(f"calendrier du lot {carried}, attendu {expected}")
    if announced != carried:
        failures.append(f"calendrier annoncé {announced or 'absent'}, lot {carried}")
    if failures:
        return 1, [f"[ERREUR] {release.get('version')}"] + [f"  {line}" for line in failures]
    return 0, [f"[OK] calendrier {carried}"]


def snapshot_origin(market: str) -> str:
    name = "cocoa_prices.csv" if market == "cocoa" else f"{market}_prices.csv"
    path = ROOT / "data" / "freeze" / name
    last = ""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            day = line.split(",", 1)[0].strip()
            if DATE_RE.match(day):
                last = day
    if not last:
        raise RuntimeError(f"Date d'instantané absente: {path.name}")
    return last


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True, choices=tuple(API_MARKETS))
    args = parser.parse_args()
    release_path = ROOT / "config" / f"experimental_release_{args.market}.json"
    if not release_path.exists():
        print(f"[ERREUR] Manifeste expérimental absent: {release_path.name}")
        return 1
    release = json.loads(release_path.read_text(encoding="utf-8"))
    expected = release.get("version")
    try:
        origin = snapshot_origin(args.market)
    except (OSError, RuntimeError) as exc:
        print(f"[ERREUR] {exc}")
        return 1
    token = os.environ.get("API_TOKEN", "")
    if not token:
        print("[AVERTISSEMENT] API_TOKEN absent. Le lot n'est pas confirmé.")
        return 2
    body = json.dumps(
        {
            "market": API_MARKETS[args.market],
            "horizons": list(HORIZONS),
            "include_sentiment": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://localhost:8000/api/v1/predict",
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[AVERTISSEMENT] API illisible: {exc}")
        return 2
    code, lines = assess_calendars(args.market, release, payload.get("service_calendar"))
    for line in lines:
        print(line)
    if code:
        return code
    code, lines = assess_served(payload, str(expected), origin)
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
