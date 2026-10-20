"""Compare keeping the opening forecast with using the latest same-target estimate.

The comparison reads an archived daily replay. It does not retrain. Each estimate
in that replay was built from the sessions known on its own day. A new training
run is not evidence that today's revision is better.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Optional, Sequence

from src.models.horizon_schedule import FROZEN_HORIZONS
from src.models.hybrid_features import SERVICE_CALENDAR, future_business_date
from src.validation.revision_stability import ContradictorySession, unique_session_batches

MEASUREMENT_PROCEDURE = "revision_value_trading_calendar"

ROOT = Path(__file__).resolve().parents[2]
MIN_EXPIRIES = 8
BOOTSTRAP_DRAWS = 2000
INTERVAL = 0.90

_CACHE: dict[tuple, dict[str, Any]] = {}


def load_prices(path: Path) -> dict[str, float]:
    prices: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            day = str(row.get("date") or "")[:10]
            if not day or row.get("price") in (None, ""):
                continue
            prices[day] = float(row["price"])
    return prices


def _row_key(row: dict[str, Any]) -> tuple:
    return (
        row.get("role"),
        row.get("policy"),
        int(row.get("initial_horizon") or 0),
        int(row.get("horizon") or 0),
        str(row.get("session") or "")[:10],
        str(row.get("target") or "")[:10],
    )


def _row_signature(row: dict[str, Any]) -> str:
    return json.dumps(
        {"price": row.get("price"), "spot": row.get("spot")},
        sort_keys=True,
        default=str,
    )


def unique_daily_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one identical emission. Two different prices for the same key are refused."""
    seen: dict[tuple, str] = {}
    kept: list[dict[str, Any]] = []
    for row in records:
        key = _row_key(row)
        signature = _row_signature(row)
        previous = seen.get(key)
        if previous is None:
            seen[key] = signature
            kept.append(row)
            continue
        if previous != signature:
            raise ContradictorySession(
                f"Deux émissions contradictoires pour {key[4]} {key[5]} J+{key[2]}"
            )
    return kept


def load_daily_records(path: Path) -> list[dict[str, Any]]:
    """Published daily initials and revisions, one copy of each session."""
    raw = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw.append(json.loads(line))
    kept: list[dict[str, Any]] = []
    for batch in unique_session_batches(raw, study_id="archived"):
        for row in batch.get("rows") or []:
            if row.get("kind") != "published" or row.get("policy") != "daily":
                continue
            if row.get("role") not in {"initial", "revision"}:
                continue
            horizon = int(row.get("initial_horizon") or 0)
            if horizon not in FROZEN_HORIZONS:
                continue
            kept.append(row)
    return unique_daily_rows(kept)


def _latest_revision(rows: Sequence[dict[str, Any]], opened: str) -> Optional[dict[str, Any]]:
    usable = [
        row
        for row in rows
        if row.get("price") is not None and str(row.get("session") or "") > opened
    ]
    if not usable:
        return None
    return max(usable, key=lambda row: (str(row.get("session") or ""), -int(row.get("horizon") or 0)))


def paired_expiries(
    records: Sequence[dict[str, Any]],
    prices: dict[str, float],
    as_of: str,
) -> dict[int, list[dict[str, Any]]]:
    """One row per opening forecast whose target is reached and revised."""
    initials: dict[tuple[int, str], dict[str, Any]] = {}
    revisions: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in records:
        if row.get("policy") not in (None, "daily"):
            continue
        if row.get("role") not in {"initial", "revision"}:
            continue
        horizon = int(row.get("initial_horizon") or 0)
        target = str(row.get("target") or "")[:10]
        if horizon not in FROZEN_HORIZONS or not target:
            continue
        key = (horizon, target)
        if row.get("role") == "initial":
            current = initials.get(key)
            if current is None or str(row.get("session") or "") < str(current.get("session") or ""):
                initials[key] = row
        else:
            revisions.setdefault(key, []).append(row)
    grouped: dict[int, list[dict[str, Any]]] = {horizon: [] for horizon in FROZEN_HORIZONS}
    cutoff = str(as_of or "")[:10]
    for (horizon, target), initial in initials.items():
        if not cutoff or target > cutoff or initial.get("price") is None:
            continue
        realized = prices.get(target)
        if realized is None:
            continue
        latest = _latest_revision(revisions.get((horizon, target), []), str(initial.get("session") or ""))
        if latest is None or latest.get("price") is None:
            continue
        initial_error = abs(float(initial["price"]) - float(realized))
        latest_error = abs(float(latest["price"]) - float(realized))
        grouped[horizon].append(
            {
                "session": str(initial.get("session") or "")[:10],
                "target": target,
                "realized": float(realized),
                "initial_price": float(initial["price"]),
                "latest_price": float(latest["price"]),
                "latest_session": str(latest.get("session") or "")[:10],
                "initial_error": initial_error,
                "latest_error": latest_error,
                "gain": initial_error - latest_error,
            }
        )
    for horizon in grouped:
        grouped[horizon].sort(key=lambda item: (item["session"], item["target"]))
    return grouped


def block_mean_interval(
    values: Sequence[float],
    block: int,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    level: float = INTERVAL,
    seed: int = 42,
) -> tuple[Optional[float], Optional[float]]:
    """Circular block bootstrap. Neighbouring expiries share one price path.

    The block stays the length of the opening horizon. One block that covers
    the whole sample, or fewer than two non-overlapping blocks, is not enough
    independent information: the interval stays empty and the verdict uncertain.
    """
    sample = [float(value) for value in values]
    count = len(sample)
    width = int(block)
    if count == 0 or width < 1 or count <= width or count // width < 2:
        return None, None
    rng = random.Random(seed)
    means: list[float] = []
    blocks_needed = math.ceil(count / width)
    for _ in range(draws):
        drawn: list[float] = []
        for _block in range(blocks_needed):
            start = rng.randrange(count)
            for step in range(width):
                drawn.append(sample[(start + step) % count])
                if len(drawn) == count:
                    break
            if len(drawn) == count:
                break
        means.append(sum(drawn) / count)
    means.sort()
    return _quantile(means, (1.0 - level) / 2.0), _quantile(means, 1.0 - (1.0 - level) / 2.0)


def _quantile(sorted_values: Sequence[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(sorted_values[upper]) * weight


def _verdict(expiries: int, low: Optional[float], high: Optional[float]) -> str:
    if expiries < MIN_EXPIRIES or low is None or high is None:
        return "incertain"
    if low > 0:
        return "etabli"
    if high < 0:
        return "defavorable"
    return "incertain"


def summarise_horizon(
    rows: Sequence[dict[str, Any]],
    horizon: int,
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda item: (item["session"], item["target"]))
    gains = [float(row["gain"]) for row in ordered]
    count = len(gains)
    improved = sum(1 for gain in gains if gain > 1e-9)
    worsened = sum(1 for gain in gains if gain < -1e-9)
    low, high = block_mean_interval(gains, horizon, draws=draws)
    mean_initial = None if not ordered else sum(float(row["initial_error"]) for row in ordered) / count
    mean_latest = None if not ordered else sum(float(row["latest_error"]) for row in ordered) / count
    mean_gain = None if not gains else sum(gains) / count
    usable = 0 if count == 0 or int(horizon) < 1 else count // int(horizon)
    return {
        "horizon": horizon,
        "comparison": "initiale contre dernière estimation avant échéance",
        "verdict": _verdict(count, low, high),
        "expiries": count,
        "mean_initial_error": mean_initial,
        "mean_latest_error": mean_latest,
        "mean_gain": mean_gain,
        "improved_share": None if not count else improved / count,
        "worsened_share": None if not count else worsened / count,
        "improvement_sum": sum(gain for gain in gains if gain > 0),
        "degradation_sum": sum(-gain for gain in gains if gain < 0),
        "gain_low": low,
        "gain_high": high,
        "block": int(horizon),
        "interval_level": INTERVAL,
        "usable_blocks": usable,
        "steps": [],
    }


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    kept = [float(value) for value in values if value is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)


def _step_summary(
    slot: dict[str, Any],
    *,
    remaining: int,
    comparison: str,
    block: int,
    draws: int,
) -> dict[str, Any]:
    rows = sorted(slot["rows"], key=lambda item: item["session"])
    available = len(rows)
    expected = int(slot["expected"])
    gains = [float(row["gain"]) for row in rows]
    low, high = block_mean_interval(gains, block, draws=draws)
    usable = 0 if available == 0 or int(block) < 1 else available // int(block)
    return {
        "remaining": int(remaining),
        "comparison": comparison,
        "expected": expected,
        "available": available,
        "abstentions": int(slot["abstentions"]),
        "missing_emissions": int(slot.get("missing_emissions") or 0),
        "missing_realizations": int(slot.get("missing_realizations") or 0),
        "session_jumps": int(slot.get("session_jumps") or 0),
        "availability_rate": None if expected == 0 else available / expected,
        "mean_initial_error": _mean([row["initial_error"] for row in rows]),
        "mean_revised_error": _mean([row["revised_error"] for row in rows]),
        "mean_spot_error": _mean([row["spot_error"] for row in rows]),
        "mean_gain": None if not gains else sum(gains) / available,
        "improved_share": None if not available else sum(1 for gain in gains if gain > 1e-9) / available,
        "gain_low": low,
        "gain_high": high,
        "verdict": _verdict(available, low, high),
        "interval_level": INTERVAL,
        "usable_blocks": usable,
        "block": int(block),
    }


def _opening_groups(
    records: Sequence[dict[str, Any]],
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[tuple[int, str], list[dict[str, Any]]]]:
    initials: dict[tuple[int, str], dict[str, Any]] = {}
    revisions: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in records:
        if row.get("policy") not in (None, "daily"):
            continue
        if row.get("role") not in {"initial", "revision"}:
            continue
        horizon = int(row.get("initial_horizon") or 0)
        target = str(row.get("target") or "")[:10]
        if horizon not in FROZEN_HORIZONS or not target:
            continue
        key = (horizon, target)
        if row.get("role") == "initial":
            current = initials.get(key)
            if current is None or str(row.get("session") or "") < str(current.get("session") or ""):
                initials[key] = row
        else:
            revisions.setdefault(key, []).append(row)
    return initials, revisions


def _empty_slot() -> dict[str, Any]:
    return {
        "expected": 0,
        "abstentions": 0,
        "missing_emissions": 0,
        "missing_realizations": 0,
        "session_jumps": 0,
        "rows": [],
    }


def _calendar(records: Sequence[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("session") or "")[:10] for row in records if row.get("session")})


_NEXT_SESSION: dict[str, str] = {}


def _next_trading_day(day: str) -> str:
    found = _NEXT_SESSION.get(day)
    if found is None:
        found = future_business_date(day, 1).date().isoformat()
        _NEXT_SESSION[day] = found
    return found


def _expected_revision_days(opened: str, horizon: int) -> list[str]:
    """Sessions after the opening, one versioned trading day at a time."""
    days: list[str] = []
    cursor = opened
    for _ in range(max(int(horizon) - 1, 0)):
        cursor = _next_trading_day(cursor)
        days.append(cursor)
    return days


def _adjacent(calendar: Sequence[str], earlier: str, later: str) -> bool:
    """Adjacency follows the versioned calendar, not the dates present in an archive."""
    del calendar
    if not earlier or not later or earlier == later:
        return False
    return _next_trading_day(earlier) == later


def _quote(initial: dict[str, Any], row: dict[str, Any], realized: float) -> dict[str, Any]:
    spot = row.get("spot")
    initial_error = abs(float(initial["price"]) - float(realized))
    revised_error = abs(float(row["price"]) - float(realized))
    return {
        "session": str(initial.get("session") or "")[:10],
        "gain": initial_error - revised_error,
        "initial_error": initial_error,
        "revised_error": revised_error,
        "spot_error": None if spot is None else abs(float(spot) - float(realized)),
    }


def delay_steps(
    records: Sequence[dict[str, Any]],
    prices: dict[str, float],
    as_of: str,
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[int, list[dict[str, Any]]]:
    """Score each remaining horizon against the grid of matured initials.

    A missing realized price, a missing emission and a calculation without a
    price stay separate. Successive gains compare two distinct consecutive
    sessions only. A jump of several sessions is counted apart.
    """
    records = unique_daily_rows(records)
    initials, revisions = _opening_groups(records)
    buckets: dict[int, dict[str, dict[int, dict[str, Any]]]] = {
        horizon: {
            kind: {remaining: _empty_slot() for remaining in range(1, horizon)}
            for kind in ("initial_vs_remaining", "successive")
        }
        for horizon in FROZEN_HORIZONS
    }
    cutoff = str(as_of or "")[:10]
    for (horizon, target), initial in initials.items():
        if not cutoff or target > cutoff or initial.get("price") is None:
            continue
        realized = prices.get(target)
        if realized is None:
            for remaining in range(1, horizon):
                buckets[horizon]["initial_vs_remaining"][remaining]["missing_realizations"] += 1
            continue
        opened = str(initial.get("session") or "")[:10]
        rows = [
            row
            for row in revisions.get((horizon, target), [])
            if str(row.get("session") or "")[:10] > opened
        ]
        by_remaining: dict[int, dict[str, Any]] = {}
        by_session: dict[str, dict[str, Any]] = {}
        for row in rows:
            session = str(row.get("session") or "")[:10]
            try:
                remaining = int(row.get("horizon") or 0)
            except (TypeError, ValueError):
                continue
            if remaining < 1 or remaining >= horizon or not session:
                continue
            current = by_remaining.get(remaining)
            if current is not None and str(current.get("session") or "") != session:
                if str(current.get("session") or "") < session:
                    by_remaining[remaining] = row
            elif current is None:
                by_remaining[remaining] = row
            held = by_session.get(session)
            if held is None:
                by_session[session] = row
            elif _row_signature(held) != _row_signature(row):
                raise ContradictorySession(f"Deux révisions contradictoires le {session} vers {target}")
        for remaining in range(1, horizon):
            slot = buckets[horizon]["initial_vs_remaining"][remaining]
            slot["expected"] += 1
            row = by_remaining.get(remaining)
            if row is None:
                slot["missing_emissions"] += 1
                continue
            if row.get("price") is None:
                slot["abstentions"] += 1
                continue
            slot["rows"].append(_quote(initial, row, float(realized)))
        ordered_days = sorted(by_session)
        for earlier, later in zip(ordered_days, ordered_days[1:]):
            if _next_trading_day(earlier) == later:
                continue
            try:
                remaining = int(by_session[later].get("horizon") or 0)
            except (TypeError, ValueError):
                continue
            if 1 <= remaining < horizon:
                buckets[horizon]["successive"][remaining]["session_jumps"] += 1
        expected_days = _expected_revision_days(opened, horizon)
        for index, (earlier, later) in enumerate(zip(expected_days, expected_days[1:])):
            remaining = horizon - (index + 2)
            if remaining < 1 or remaining >= horizon or later > cutoff:
                continue
            slot = buckets[horizon]["successive"][remaining]
            slot["expected"] += 1
            previous = by_session.get(earlier)
            nxt = by_session.get(later)
            if previous is None or nxt is None:
                slot["missing_emissions"] += 1
                continue
            if previous.get("price") is None or nxt.get("price") is None:
                slot["abstentions"] += 1
                continue
            previous_error = abs(float(previous["price"]) - float(realized))
            next_error = abs(float(nxt["price"]) - float(realized))
            spot = nxt.get("spot")
            slot["rows"].append(
                {
                    "session": later,
                    "gain": previous_error - next_error,
                    "initial_error": previous_error,
                    "revised_error": next_error,
                    "spot_error": None if spot is None else abs(float(spot) - float(realized)),
                }
            )
    scored: dict[int, list[dict[str, Any]]] = {}
    for horizon in FROZEN_HORIZONS:
        steps: list[dict[str, Any]] = []
        for comparison in ("initial_vs_remaining", "successive"):
            for remaining in range(horizon - 1, 0, -1):
                slot = buckets[horizon][comparison][remaining]
                if slot["expected"] == 0 and slot["session_jumps"] == 0 and slot["missing_realizations"] == 0:
                    continue
                steps.append(
                    _step_summary(
                        slot,
                        remaining=remaining,
                        comparison=comparison,
                        block=horizon,
                        draws=draws,
                    )
                )
        scored[horizon] = steps
    return scored


def evaluate_records(
    records: Sequence[dict[str, Any]],
    prices: dict[str, float],
    as_of: str,
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> list[dict[str, Any]]:
    records = unique_daily_rows(list(records))
    paired = paired_expiries(records, prices, as_of)
    steps = delay_steps(records, prices, as_of, draws=draws)
    scored = []
    for horizon in FROZEN_HORIZONS:
        summary = summarise_horizon(paired[horizon], horizon, draws=draws)
        summary["steps"] = steps.get(horizon, [])
        scored.append(summary)
    return scored


def _served_calendar(market_id: str) -> str:
    mode = os.getenv("PREDICTION_RELEASE_MODE", "active")
    prefix = "experimental_release" if mode == "experimental" else "active_release"
    path = ROOT / "config" / f"{prefix}_{market_id}.json"
    if path.exists():
        try:
            named = json.loads(path.read_text(encoding="utf-8")).get("calendar")
        except json.JSONDecodeError:
            named = None
        if named:
            return str(named)
    return SERVICE_CALENDAR


def _study_dir(market_id: str) -> Path:
    return ROOT / "reports" / "revision_stability" / market_id / "study_corrected"


def _price_file(market_id: str) -> Path:
    name = "cocoa_prices.csv" if market_id == "cocoa" else f"{market_id}_prices.csv"
    return ROOT / "data" / "freeze" / name


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary_path(study: Path) -> Optional[Path]:
    summaries = sorted(path for path in study.glob("*_summary.json") if path.is_file())
    return summaries[-1] if summaries else None


def _as_of(study: Path) -> str:
    summary = _summary_path(study)
    if summary is None:
        return ""
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return str(payload.get("as_of") or "")[:10]


def bind_realized_prices(study: Path, live: Path) -> tuple[Path, str, bool]:
    """Keep the prices of a score beside that study.

    The live file can change afterwards. The bound copy stays, and editing
    that copy produces another fingerprint.
    """
    study.mkdir(parents=True, exist_ok=True)
    bound = study / "realized_prices.csv"
    stamp_path = study / "realized_prices.sha256"
    if not bound.exists():
        if not live.exists():
            raise FileNotFoundError(live)
        bound.write_bytes(live.read_bytes())
    digest = _digest(bound)
    recorded = stamp_path.read_text(encoding="utf-8").strip() if stamp_path.exists() else ""
    live_differs = live.exists() and _digest(live) != digest
    if recorded and recorded != digest:
        versioned = study / f"realized_prices_{digest[:12]}.csv"
        if not versioned.exists():
            versioned.write_bytes(bound.read_bytes())
        (study / f"realized_prices_{digest[:12]}.sha256").write_text(digest, encoding="utf-8")
        return versioned, digest, live_differs
    if recorded != digest:
        stamp_path.write_text(digest, encoding="utf-8")
    return bound, digest, live_differs


def measurement_identity(
    sessions: Path,
    prices: Path,
    summary: Optional[Path],
    draws: int,
) -> tuple:
    """Cache identity: prices, cutoff, report, and procedure."""
    as_of = ""
    summary_digest = ""
    summary_name = ""
    if summary is not None and summary.exists():
        summary_digest = _digest(summary)
        summary_name = summary.name
        payload = json.loads(summary.read_text(encoding="utf-8"))
        as_of = str(payload.get("as_of") or "")[:10]
    return (
        _digest(sessions),
        _digest(prices),
        summary_name,
        summary_digest,
        as_of,
        MEASUREMENT_PROCEDURE,
        int(draws),
    )


def revision_value_for_market(market_id: str, *, draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    """Score the archived daily replay for one market. The result is cached."""
    study = _study_dir(market_id)
    sessions = study / "sessions.jsonl"
    prices_path = _price_file(market_id)
    note = (
        "Rejeu quotidien archivé. Il ne juge pas le lot chargé. "
        "Les cotations fermées ne sont pas retirées du fichier brut."
    )
    bound_ready = (study / "realized_prices.csv").exists()
    if not sessions.exists() or not (prices_path.exists() or bound_ready):
        return {
            "market": market_id,
            "as_of": None,
            "note": "Rejeu quotidien archivé introuvable.",
            "horizons": [
                summarise_horizon([], horizon, draws=1) for horizon in FROZEN_HORIZONS
            ],
        }
    bound, price_digest, live_differs = bind_realized_prices(study, prices_path)
    summary = _summary_path(study)
    identity = measurement_identity(sessions, bound, summary, draws)
    cached = _CACHE.get((market_id, identity))
    if cached is not None:
        return cached
    as_of = identity[4]
    scored = evaluate_records(
        load_daily_records(sessions),
        load_prices(bound),
        as_of,
        draws=draws,
    )
    payload = {
        "market": market_id,
        "as_of": as_of or None,
        "note": note,
        "service_calendar": _served_calendar(market_id),
        "study_calendar": "trading_sessions",
        "price_fingerprint": price_digest,
        "report": None if summary is None else summary.name,
        "procedure": MEASUREMENT_PROCEDURE,
        "live_price_differs": live_differs,
        "horizons": scored,
    }
    _CACHE[(market_id, identity)] = payload
    return payload
