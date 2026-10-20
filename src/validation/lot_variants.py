"""Compare served prices with measured variants. Nothing here replaces the served price."""

from __future__ import annotations

from typing import Any, Sequence

K_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
POLAR = 1.5


def shrinkage(spot: float, model: float, k: float) -> float:
    return float(spot) + float(k) * (float(model) - float(spot))


def _direction(price: float, reference: float) -> str:
    if not reference:
        return "neutre"
    change = (float(price) - float(reference)) / float(reference) * 100.0
    if change > POLAR:
        return "hausse"
    if change < -POLAR:
        return "baisse"
    return "neutre"


def _inversions(directions: Sequence[str]) -> int:
    last = None
    flips = 0
    for item in directions:
        if item == "neutre":
            continue
        if last is not None and item != last:
            flips += 1
        last = item
    return flips


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _split(
    rows: Sequence[dict[str, Any]],
    holdout_fraction: float,
    *,
    key: str = "session",
) -> tuple[list[dict], list[dict]]:
    days = sorted({str(row.get(key) or "")[:10] for row in rows if row.get(key)})
    if len(days) < 4:
        return list(rows), []
    cut = max(1, int(round(len(days) * (1.0 - holdout_fraction))))
    cut = min(cut, len(days) - 1)
    holdout_days = set(days[cut:])
    calibration = [row for row in rows if str(row.get(key) or "")[:10] not in holdout_days]
    holdout = [row for row in rows if str(row.get(key) or "")[:10] in holdout_days]
    return calibration, holdout


def _choose_k(rows: Sequence[dict[str, Any]]) -> float | None:
    usable = [
        row
        for row in rows
        if row.get("spot") is not None and row.get("price") is not None and row.get("realized") is not None
    ]
    best_k = None
    best_error = None
    for k in K_GRID:
        errors = [
            abs(shrinkage(row["spot"], row["price"], k) - float(row["realized"]))
            for row in usable
        ]
        mean = _mean(errors)
        if mean is None:
            continue
        if best_error is None or mean < best_error:
            best_error = mean
            best_k = k
    return best_k


def _chains(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("realized") is None or row.get("price") is None:
            continue
        key = (int(row.get("initial_horizon") or row.get("horizon") or 0), str(row.get("target") or ""))
        grouped.setdefault(key, []).append(row)
    chains = []
    for chain in grouped.values():
        chains.append(sorted(chain, key=lambda item: str(item.get("session") or "")))
    return chains


def _policy_prices(chain: list[dict[str, Any]], policy: str) -> list[float]:
    prices = []
    initial = next((row for row in chain if row.get("role") == "initial"), chain[0])
    previous = float(initial["price"])
    for row in chain:
        latest = float(row["price"])
        if policy == "initial":
            prices.append(float(initial["price"]))
        elif policy == "previous":
            prices.append(previous)
        else:
            prices.append(latest)
        previous = latest
    return prices


def _policy_score(rows: Sequence[dict[str, Any]], policy: str) -> dict[str, Any]:
    errors: list[float] = []
    flips = 0
    for chain in _chains(rows):
        reference = chain[0].get("reference")
        if reference is None:
            reference = chain[0].get("spot")
        published = _policy_prices(chain, policy)
        directions = []
        for row, price in zip(chain, published):
            if row.get("realized") is None:
                continue
            errors.append(abs(price - float(row["realized"])))
            if reference is not None:
                directions.append(_direction(price, float(reference)))
        flips += _inversions(directions)
    return {
        "policy": policy,
        "mean_absolute_error": _mean(errors),
        "inversions": flips,
        "count": len(errors),
    }


def _choose_policy(rows: Sequence[dict[str, Any]]) -> str | None:
    ranked = [_policy_score(rows, name) for name in ("initial", "previous", "latest")]
    ranked = [item for item in ranked if item["mean_absolute_error"] is not None]
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item["mean_absolute_error"], item["inversions"]))
    return str(ranked[0]["policy"])


def delay_balances(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Errors at each remaining delay, on the same cases, against that day's price."""
    from src.validation.revision_value import block_mean_interval

    buckets: dict[int, list[dict[str, float]]] = {}
    expiries: dict[int, set[str]] = {}
    for chain in _chains(rows):
        initial = next((row for row in chain if row.get("role") == "initial"), chain[0])
        previous = float(initial["price"])
        for row in chain:
            if row.get("realized") is None or row.get("price") is None or row.get("spot") is None:
                previous = float(row["price"]) if row.get("price") is not None else previous
                continue
            realized = float(row["realized"])
            delay = int(row.get("horizon") or 0)
            latest = float(row["price"])
            spot = float(row["spot"])
            opening = float(initial["price"])
            buckets.setdefault(delay, []).append(
                {
                    "initial_error": abs(opening - realized),
                    "previous_error": abs(previous - realized),
                    "latest_error": abs(latest - realized),
                    "unchanged_error": abs(spot - realized),
                }
            )
            expiries.setdefault(delay, set()).add(str(row.get("target") or ""))
            previous = latest
    report = {}
    for delay, items in sorted(buckets.items()):
        gains_versus_spot = [item["unchanged_error"] - item["latest_error"] for item in items]
        low, high = block_mean_interval(gains_versus_spot, max(delay, 1), draws=200)
        insufficient = low is None or high is None
        latest_mean = _mean([item["latest_error"] for item in items])
        initial_mean = _mean([item["initial_error"] for item in items])
        unchanged_mean = _mean([item["unchanged_error"] for item in items])
        beats_initial = (
            latest_mean is not None and initial_mean is not None and latest_mean < initial_mean
        )
        beats_spot = (
            latest_mean is not None and unchanged_mean is not None and latest_mean < unchanged_mean
        )
        established = (
            not insufficient
            and beats_initial
            and beats_spot
            and low is not None
            and low > 0
        )
        report[str(delay)] = {
            "count": len(items),
            "distinct_expiries": len(expiries.get(delay) or ()),
            "initial_mean_absolute_error": initial_mean,
            "previous_mean_absolute_error": _mean([item["previous_error"] for item in items]),
            "latest_mean_absolute_error": latest_mean,
            "unchanged_mean_absolute_error": unchanged_mean,
            "sample_status": "échantillon insuffisant" if insufficient else "intervalle disponible",
            "advantage_established": established,
        }
    return report


def _level_errors(rows: Sequence[dict[str, Any]], k: float | None) -> dict[str, Any]:
    model_errors = []
    spot_errors = []
    shrunk_errors = []
    for row in rows:
        if row.get("realized") is None or row.get("price") is None:
            continue
        realized = float(row["realized"])
        model_errors.append(abs(float(row["price"]) - realized))
        if row.get("spot") is not None:
            spot_errors.append(abs(float(row["spot"]) - realized))
            if k is not None:
                shrunk_errors.append(abs(shrinkage(row["spot"], row["price"], k) - realized))
    return {
        "model_mean_absolute_error": _mean(model_errors),
        "unchanged_mean_absolute_error": _mean(spot_errors),
        "shrinkage_mean_absolute_error": _mean(shrunk_errors),
        "count": len(model_errors),
    }


def evaluate_variants(
    rows: Sequence[dict[str, Any]],
    *,
    holdout_fraction: float = 0.25,
    split_key: str = "session",
) -> dict[str, Any]:
    """Choose k and the revision rule on the calibration period only.

    The holdout is reported afterwards. No variant is written into the served price.
    """
    calibration, holdout = _split(rows, holdout_fraction, key=split_key)
    k = _choose_k(calibration)
    policy = _choose_policy(calibration)
    return {
        "served_price": "direct_model",
        "variant_replaces_served_price": False,
        "status": "experimental",
        "k": k,
        "revision_policy": policy,
        "chosen_on": "calibration",
        "calibration": {
            "levels": _level_errors(calibration, k),
            "policies": [_policy_score(calibration, name) for name in ("initial", "previous", "latest")],
        },
        "holdout": {
            "levels": _level_errors(holdout, k),
            "policies": [_policy_score(holdout, name) for name in ("initial", "previous", "latest")],
        },
    }
