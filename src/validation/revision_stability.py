"""Two cadences on the same targets. Neither cadence is selected."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd

from src.models.hybrid_features import future_business_date, trading_sessions

BLOCK_SESSIONS = 5
AS_OF = "2026-10-06"


def lot_index(session_index: int, block: int = BLOCK_SESSIONS) -> int:
    """The five-session policy reuses the lot trained at the start of the block."""
    if session_index < 0:
        raise ValueError("Séance négative")
    return int(session_index) // int(block)


def params_frozen(session_index: int, block: int = BLOCK_SESSIONS) -> bool:
    return session_index % int(block) != 0


def study_window(
    frame: pd.DataFrame,
    *,
    as_of: str = AS_OF,
    n_origins: int = 60,
    horizon: int = 30,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp], list[str]]:
    """Last trading origins whose calendar target does not pass the cutoff.

    A missing target is reported and excluded from the score later. It does not
    pull an older origin into the window. Closed collection dates are not sessions.
    """
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data = data[data["date"] <= pd.Timestamp(as_of)].sort_values("date")
    trading = trading_sessions(data)
    present = set(pd.to_datetime(trading["date"]).dt.normalize())
    origins: list[pd.Timestamp] = []
    missing: list[str] = []
    for day in trading["date"]:
        target = pd.Timestamp(future_business_date(day, horizon)).normalize()
        if target <= pd.Timestamp(as_of):
            origins.append(pd.Timestamp(day))
            if target not in present:
                missing.append(
                    f"{pd.Timestamp(day).date().isoformat()}->{target.date().isoformat()}"
                )
    chosen = origins[-int(n_origins) :]
    if not chosen:
        return [], [], missing
    first = chosen[0]
    sessions = [
        pd.Timestamp(day)
        for day in trading["date"]
        if pd.Timestamp(day) >= first and pd.Timestamp(day) <= pd.Timestamp(as_of)
    ]
    return chosen, sessions, missing


def bundles_for_resume(index: int, block: int = BLOCK_SESSIONS) -> tuple[Optional[int], Optional[int]]:
    """Indexes to rebuild before a session that was not kept in memory.

    The first value is the previous session. The second is the five-session
    anchor, absent when this session itself renews the block.
    """
    previous = None if index <= 0 else index - 1
    anchor = None if index % int(block) == 0 else (index // int(block)) * int(block)
    return previous, anchor


def _track_key(session: str, horizon: int) -> str:
    return f"{session}|{int(horizon)}"


def tracks_from_records(records: Sequence[dict]) -> dict:
    """Replay published events so a resume keeps the last revision, not only the initial."""
    from src.models.prediction_journal import count_inversions

    tracks = {"daily": {}, "five": {}}
    for row in records:
        if row.get("kind") != "published":
            continue
        policy = row.get("policy")
        if policy not in tracks:
            continue
        if row.get("role") == "initial":
            direction = row.get("direction")
            tracks[policy][_track_key(row["session"], row["initial_horizon"])] = {
                "initial_horizon": int(row["initial_horizon"]),
                "reference": row.get("reference"),
                "price": row.get("price"),
                "direction": direction,
                "directions": [direction],
                "target": str(row["target"])[:10],
                "opened": row["session"],
                "open": pd.Timestamp(row["target"]) > pd.Timestamp(row["session"]),
            }
            continue
        if row.get("role") != "revision":
            continue
        for track in tracks[policy].values():
            if str(track["target"])[:10] != str(row.get("target"))[:10]:
                continue
            if int(track["initial_horizon"]) != int(row.get("initial_horizon")):
                continue
            if track.get("opened") == row.get("session"):
                continue
            directions = list(track.get("directions") or [])
            directions.append(row.get("direction"))
            track["directions"] = directions
            track["price"] = row.get("price")
            track["direction"] = row.get("direction")
            track["inversions"] = count_inversions(directions)
            if pd.Timestamp(track["target"]) <= pd.Timestamp(row["session"]):
                track["open"] = False
    return tracks


def annotate_realized(records: Sequence[dict], lookup: dict, as_of: str) -> list[dict]:
    """Score after generation. The realized price is not an input of training."""
    cutoff = pd.Timestamp(as_of).normalize()
    annotated = []
    for row in records:
        copied = dict(row)
        if copied.get("kind") == "published":
            target = pd.Timestamp(copied["target"]).normalize()
            if target <= cutoff and target in lookup:
                copied["realized"] = float(lookup[target])
            else:
                copied["realized"] = None
                copied["realized_absent"] = bool(target <= cutoff)
        annotated.append(copied)
    return annotated


def _mean_pct(pairs: list[tuple[float, float]]) -> Optional[float]:
    errors = [
        abs(predicted - realized) / realized * 100.0
        for predicted, realized in pairs
        if realized
    ]
    if not errors:
        return None
    return float(sum(errors) / len(errors))


def score_policies(records: Sequence[dict]) -> list[dict]:
    """Model error, unchanged-price error, and revision error by remaining horizon."""
    grouped: dict[tuple, list] = {}
    for row in records:
        if row.get("kind") != "published":
            continue
        grouped.setdefault((row["policy"], int(row["initial_horizon"])), []).append(row)
    report = []
    for (policy, horizon), rows in sorted(grouped.items()):
        initials = [row for row in rows if row.get("role") == "initial"]
        revisions = [row for row in rows if row.get("role") == "revision"]
        mature_initials = [
            row for row in initials if row.get("realized") is not None and row.get("price") is not None
        ]
        model_pairs = [(float(row["price"]), float(row["realized"])) for row in mature_initials]
        origin_pairs = [
            (float(row["reference"]), float(row["realized"]))
            for row in mature_initials
            if row.get("reference") is not None
        ]
        by_remaining: dict[int, list[tuple[float, float]]] = {}
        revision_day_pairs = []
        for row in revisions:
            if row.get("realized") is None or row.get("price") is None:
                continue
            by_remaining.setdefault(int(row["horizon"]), []).append(
                (float(row["price"]), float(row["realized"]))
            )
            if row.get("spot") is not None:
                revision_day_pairs.append((float(row["spot"]), float(row["realized"])))
        amplitudes = [abs(row["delta"]) for row in revisions if row.get("delta") is not None]
        absent = sum(1 for row in initials + revisions if row.get("realized_absent"))
        inversions = sum(int(row.get("inversion") or 0) for row in revisions)
        reached = [
            row for row in initials if row.get("realized") is not None or row.get("realized_absent")
        ]
        available = [
            row for row in reached if row.get("price") is not None and row.get("realized") is not None
        ]
        abstained = [
            row for row in reached if row.get("realized") is not None and row.get("price") is None
        ]
        missing_realizations = [
            row for row in initials if row.get("realized") is None and not row.get("realized_absent")
        ]
        availability = None
        if available or abstained:
            availability = len(available) / (len(available) + len(abstained))
        paired_model: dict[int, list[tuple[float, float]]] = {}
        paired_spot: dict[int, list[tuple[float, float]]] = {}
        for row in revisions:
            if row.get("realized") is None or row.get("price") is None or row.get("spot") is None:
                continue
            key = int(row["horizon"])
            pair = (float(row["price"]), float(row["realized"]))
            spot_pair = (float(row["spot"]), float(row["realized"]))
            paired_model.setdefault(key, []).append(pair)
            paired_spot.setdefault(key, []).append(spot_pair)
        gains = []
        for row in available:
            if row.get("reference") is None or not row.get("realized"):
                continue
            model_error = abs(float(row["price"]) - float(row["realized"]))
            spot_error = abs(float(row["reference"]) - float(row["realized"]))
            gains.append(spot_error - model_error)
        gain_low, gain_high = _gain_interval(gains, horizon)
        report.append(
            {
                "policy": policy,
                "initial_horizon": horizon,
                "mature_initials": len(mature_initials),
                "expected_initials": len(reached),
                "available_prices": len(available),
                "abstentions": len(abstained),
                "missing_realizations": len(missing_realizations),
                "availability_rate": availability,
                "initial_mape": _mean_pct(model_pairs),
                "unchanged_origin_mape": _mean_pct(origin_pairs),
                "revision_mape_by_remaining": {
                    str(key): _mean_pct(values) for key, values in sorted(by_remaining.items())
                },
                "unchanged_revision_day_mape": _mean_pct(revision_day_pairs),
                "paired_revision_mape_by_remaining": {
                    str(key): _mean_pct(values) for key, values in sorted(paired_model.items())
                },
                "paired_unchanged_mape_by_remaining": {
                    str(key): _mean_pct(values) for key, values in sorted(paired_spot.items())
                },
                "unchanged_advantage_low": gain_low,
                "unchanged_advantage_high": gain_high,
                "revision_count": len([row for row in revisions if row.get("price") is not None]),
                "median_abs_revision": None if not amplitudes else float(pd.Series(amplitudes).median()),
                "max_abs_revision": None if not amplitudes else float(pd.Series(amplitudes).max()),
                "inversions": int(inversions),
                "absent_targets": int(absent),
            }
        )
    return report


def _gain_interval(gains: Sequence[float], horizon: int) -> tuple[Optional[float], Optional[float]]:
    """Interval on shared targets. A positive value means the unchanged close was closer."""
    if len(gains) < 2:
        return None, None
    from src.validation.revision_value import block_mean_interval

    return block_mean_interval(gains, max(int(horizon), 1))


class ContradictorySession(RuntimeError):
    """Two stored copies of one session do not match."""


class ArchivePreserved(RuntimeError):
    """An archive that already contains identical copies is not rewritten."""


def _session_signature(batch: dict) -> str:
    payload = batch.get("rows")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def unique_session_batches(batches: Sequence[dict], *, study_id: str) -> list[dict]:
    """Keep one identical copy per session. A contradiction is refused."""
    seen: dict[str, str] = {}
    kept: list[dict] = []
    for batch in batches:
        owner = str(batch.get("study_id") or study_id)
        if batch.get("study_id") and owner != study_id:
            raise ContradictorySession(
                f"Séance {batch.get('session')} appartient à l'étude {owner}"
            )
        key = str(batch.get("session"))
        signature = _session_signature(batch)
        previous = seen.get(key)
        if previous is None:
            seen[key] = signature
            kept.append(batch)
            continue
        if previous != signature:
            raise ContradictorySession(f"Deux copies contradictoires pour {key}")
    return kept


def load_unique_sessions(path: Path, *, study_id: str) -> list[dict]:
    """Read a checkpoint without rewriting it."""
    if not path.exists():
        return []
    raw = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw.append(json.loads(line))
    return unique_session_batches(raw, study_id=study_id)


def _acquire_study_lock(lock: Path) -> int:
    lock.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(8):
        try:
            if lock.exists() and time.time() - lock.stat().st_mtime > 30:
                lock.unlink(missing_ok=True)
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("ascii"))
            return fd
        except (FileExistsError, OSError):
            time.sleep(0.05 * (attempt + 1))
    raise TimeoutError("Verrou d'étude indisponible")


def _release_study_lock(fd: int, lock: Path) -> None:
    try:
        os.close(fd)
    finally:
        lock.unlink(missing_ok=True)


def append_unique_session(path: Path, batch: dict, *, study_id: str) -> str:
    """Exclusive atomic write keyed by study and session.

    An identical session returns ``duplicate``. A file that already holds
    identical copies is left untouched.
    """
    lock = path.with_name(path.name + ".lock")
    fd = _acquire_study_lock(lock)
    try:
        raw = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    raw.append(json.loads(line))
        unique = unique_session_batches(raw, study_id=study_id)
        key = str(batch.get("session"))
        signature = _session_signature(batch)
        for item in unique:
            if str(item.get("session")) != key:
                continue
            if _session_signature(item) == signature:
                return "duplicate"
            raise ContradictorySession(key)
        if len(raw) != len(unique):
            raise ArchivePreserved(
                "Le fichier contient des copies identiques et n'est pas réécrit."
            )
        stored = {**batch, "study_id": study_id}
        unique.append(stored)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unique),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return "written"
    finally:
        _release_study_lock(fd, lock)


def replay_policies(
    sessions: Sequence,
    history_for: Callable,
    train_fn: Callable,
    price_fn: Callable,
    *,
    official_horizons: Sequence[int] = (1, 7, 14, 30),
    origin_dates: Optional[set] = None,
    done_sessions: Optional[set] = None,
    prior_records: Optional[Sequence[dict]] = None,
    direction_fn: Optional[Callable] = None,
    on_session: Optional[Callable] = None,
) -> list[dict]:
    """Run both cadences. A resumed call retrains only the bundles the next session needs."""
    from src.models.hybrid_features import future_business_date
    from src.models.prediction_journal import count_inversions, direction_from_reference

    direct = direction_fn or direction_from_reference
    done = set(done_sessions or [])
    records = [dict(row) for row in (prior_records or [])]
    tracks = tracks_from_records(records)
    five_bundle = None
    previous_daily = None
    new_rows: list[dict] = []

    def remaining(origin, target) -> Optional[int]:
        origin_ts = pd.Timestamp(origin).normalize()
        target_ts = pd.Timestamp(target).normalize()
        if target_ts < origin_ts:
            return None
        for horizon in range(0, 31):
            landed = pd.Timestamp(future_business_date(origin_ts, horizon)).normalize()
            if landed == target_ts:
                return horizon
        return None

    for index, session in enumerate(sessions):
        session_key = pd.Timestamp(session).date().isoformat()
        if session_key in done:
            continue
        history = history_for(session)
        previous_index, anchor_index = bundles_for_resume(index)
        if previous_daily is None and previous_index is not None:
            previous_daily = train_fn(history_for(sessions[previous_index]))
        if five_bundle is None and anchor_index is not None:
            five_bundle = train_fn(history_for(sessions[anchor_index]))
        origin_price = float(history["price"].iloc[-1])
        daily_bundle = train_fn(history)
        if index % BLOCK_SESSIONS == 0:
            if five_bundle is None and index >= BLOCK_SESSIONS:
                previous_five = train_fn(history_for(sessions[index - BLOCK_SESSIONS]))
            else:
                previous_five = five_bundle
            five_bundle = daily_bundle
        else:
            previous_five = None
        session_rows = []
        bundles = {"daily": (daily_bundle, previous_daily), "five": (five_bundle, previous_five)}
        for policy, (bundle, previous_bundle) in bundles.items():
            if previous_bundle is not None:
                for track in list(tracks[policy].values()):
                    if not track["open"]:
                        continue
                    left = remaining(session, track["target"])
                    if left is None or left < 1 or left > 30:
                        continue
                    old_price = price_fn(previous_bundle, history, origin_price, session, left)
                    new_price = price_fn(bundle, history, origin_price, session, left)
                    session_rows.append(
                        {
                            "kind": "diagnostic",
                            "policy": policy,
                            "session": session_key,
                            "target": track["target"],
                            "horizon": left,
                            "initial_horizon": track["initial_horizon"],
                            "previous_price": old_price,
                            "price": new_price,
                            "comparison": "indisponible" if old_price is None or new_price is None else "disponible",
                        }
                    )
            is_origin = origin_dates is None or pd.Timestamp(session).normalize() in origin_dates
            if is_origin:
                for horizon in official_horizons:
                    target = pd.Timestamp(future_business_date(session, horizon)).normalize()
                    price = price_fn(bundle, history, origin_price, session, horizon)
                    direction = direct(price, origin_price)
                    key = _track_key(session_key, horizon)
                    tracks[policy][key] = {
                        "initial_horizon": horizon,
                        "reference": origin_price,
                        "price": price,
                        "direction": direction,
                        "directions": [direction],
                        "target": target.date().isoformat(),
                        "opened": session_key,
                        "open": target > pd.Timestamp(session),
                    }
                    session_rows.append(
                        {
                            "kind": "published",
                            "role": "initial",
                            "policy": policy,
                            "session": session_key,
                            "target": target.date().isoformat(),
                            "horizon": horizon,
                            "initial_horizon": horizon,
                            "reference": origin_price,
                            "spot": origin_price,
                            "price": price,
                            "direction": direction,
                            "inversion": 0,
                            "lot_index": lot_index(index) if policy == "five" else index,
                            "delta": None,
                            "delta_pct": None,
                        }
                    )
            for track in list(tracks[policy].values()):
                if not track["open"] or track.get("opened") == session_key:
                    continue
                left = remaining(session, track["target"])
                if left is None:
                    continue
                if left == 0:
                    track["open"] = False
                    continue
                if left > 30:
                    continue
                price = price_fn(bundle, history, origin_price, session, left)
                previous_price = track["price"]
                delta = None if price is None or previous_price is None else price - previous_price
                reference = track["reference"] or 0
                delta_pct = None if delta is None or not reference else delta / reference * 100
                direction = direct(price, track["reference"])
                directions = list(track.get("directions") or [])
                before = count_inversions(directions)
                directions.append(direction)
                inversion = count_inversions(directions) - before
                session_rows.append(
                    {
                        "kind": "published",
                        "role": "revision",
                        "policy": policy,
                        "session": session_key,
                        "target": track["target"],
                        "horizon": left,
                        "initial_horizon": track["initial_horizon"],
                        "reference": track["reference"],
                        "spot": origin_price,
                        "price": price,
                        "direction": direction,
                        "inversion": inversion,
                        "lot_index": lot_index(index) if policy == "five" else index,
                        "delta": delta,
                        "delta_pct": delta_pct,
                    }
                )
                track["price"] = price
                track["direction"] = direction
                track["directions"] = directions
                if pd.Timestamp(track["target"]) <= pd.Timestamp(session):
                    track["open"] = False
        batch = {"session": session_key, "index": index, "rows": session_rows}
        new_rows.append(batch)
        records.extend(session_rows)
        previous_daily = daily_bundle
        if on_session is not None:
            on_session(batch)
    return new_rows
