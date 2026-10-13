"""Compare keeping the opening forecast with using the latest same-target estimate.

The comparison reads an archived daily replay. It does not retrain. Each estimate
in that replay was built from the sessions known on its own day. A new training
run is not evidence that today's revision is better.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Optional, Sequence

from src.models.horizon_schedule import FROZEN_HORIZONS

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


def load_daily_records(path: Path) -> list[dict[str, Any]]:
    """Published daily initials and revisions for the frozen horizons."""
    kept: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            batch = json.loads(line)
            for row in batch.get("rows") or []:
                if row.get("kind") != "published" or row.get("policy") != "daily":
                    continue
                if row.get("role") not in {"initial", "revision"}:
                    continue
                horizon = int(row.get("initial_horizon") or 0)
                if horizon not in FROZEN_HORIZONS:
                    continue
                kept.append(row)
    return kept


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


def delay_steps(
    records: Sequence[dict[str, Any]],
    prices: dict[str, float],
    as_of: str,
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[int, list[dict[str, Any]]]:
    """Score each remaining horizon, and each step from one revision to the next.

    The block length is the opening horizon. A missing revision price is an
    abstention. The spot is the close already known on the revision day.
    """
    initials, revisions = _opening_groups(records)
    buckets: dict[int, dict[str, dict[int, dict[str, Any]]]] = {
        horizon: {
            kind: {
                remaining: {"expected": 0, "abstentions": 0, "rows": []}
                for remaining in range(1, horizon)
            }
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
            continue
        opened = str(initial.get("session") or "")
        rows = [
            row
            for row in revisions.get((horizon, target), [])
            if str(row.get("session") or "") > opened
        ]
        by_remaining: dict[int, dict[str, Any]] = {}
        for row in rows:
            try:
                remaining = int(row.get("horizon") or 0)
            except (TypeError, ValueError):
                continue
            if remaining < 1 or remaining >= horizon:
                continue
            current = by_remaining.get(remaining)
            if current is None or str(row.get("session") or "") > str(current.get("session") or ""):
                by_remaining[remaining] = row
        for remaining, row in by_remaining.items():
            slot = buckets[horizon]["initial_vs_remaining"][remaining]
            slot["expected"] += 1
            if row.get("price") is None:
                slot["abstentions"] += 1
                continue
            spot = row.get("spot")
            initial_error = abs(float(initial["price"]) - float(realized))
            revised_error = abs(float(row["price"]) - float(realized))
            slot["rows"].append(
                {
                    "session": opened,
                    "gain": initial_error - revised_error,
                    "initial_error": initial_error,
                    "revised_error": revised_error,
                    "spot_error": None if spot is None else abs(float(spot) - float(realized)),
                }
            )
        ordered = sorted(rows, key=lambda row: str(row.get("session") or ""))
        for previous, nxt in zip(ordered, ordered[1:]):
            try:
                remaining = int(nxt.get("horizon") or 0)
            except (TypeError, ValueError):
                continue
            if remaining < 1 or remaining >= horizon:
                continue
            slot = buckets[horizon]["successive"][remaining]
            slot["expected"] += 1
            if previous.get("price") is None or nxt.get("price") is None:
                slot["abstentions"] += 1
                continue
            previous_error = abs(float(previous["price"]) - float(realized))
            next_error = abs(float(nxt["price"]) - float(realized))
            spot = nxt.get("spot")
            slot["rows"].append(
                {
                    "session": str(nxt.get("session") or ""),
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
                if slot["expected"] == 0:
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
    paired = paired_expiries(records, prices, as_of)
    steps = delay_steps(records, prices, as_of, draws=draws)
    scored = []
    for horizon in FROZEN_HORIZONS:
        summary = summarise_horizon(paired[horizon], horizon, draws=draws)
        summary["steps"] = steps.get(horizon, [])
        scored.append(summary)
    return scored


def _study_dir(market_id: str) -> Path:
    return ROOT / "reports" / "revision_stability" / market_id / "study_corrected"


def _price_file(market_id: str) -> Path:
    name = "cocoa_prices.csv" if market_id == "cocoa" else f"{market_id}_prices.csv"
    return ROOT / "data" / "freeze" / name


def _as_of(study: Path) -> str:
    summaries = sorted(study.glob("*_summary.json"))
    if not summaries:
        return ""
    payload = json.loads(summaries[-1].read_text(encoding="utf-8"))
    return str(payload.get("as_of") or "")[:10]


def revision_value_for_market(market_id: str, *, draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    """Score the archived daily replay for one market. The result is cached."""
    study = _study_dir(market_id)
    sessions = study / "sessions.jsonl"
    prices_path = _price_file(market_id)
    note = (
        "Rejeu quotidien archivé. La ligne du délai restant compare l'initiale "
        "à la révision de ce délai. L'ancienne comparaison, initiale contre "
        "dernière estimation avant échéance, ne justifie pas la révision du "
        "lendemain. Le lot d'aujourd'hui n'est pas jugé par ce rejeu."
    )
    if not sessions.exists() or not prices_path.exists():
        return {
            "market": market_id,
            "as_of": None,
            "note": "Rejeu quotidien archivé introuvable.",
            "horizons": [
                summarise_horizon([], horizon, draws=1) for horizon in FROZEN_HORIZONS
            ],
        }
    stamp = (sessions.stat().st_mtime_ns, prices_path.stat().st_mtime_ns, draws)
    cached = _CACHE.get((market_id, stamp))
    if cached is not None:
        return cached
    as_of = _as_of(study)
    scored = evaluate_records(
        load_daily_records(sessions),
        load_prices(prices_path),
        as_of,
        draws=draws,
    )
    payload = {"market": market_id, "as_of": as_of or None, "note": note, "horizons": scored}
    _CACHE[(market_id, stamp)] = payload
    return payload
