"""Immutable local forecast journal, shared by Windows and Docker."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from src.models.horizon_schedule import FROZEN_HORIZONS, JOURNAL_ONLY_HORIZONS, OFFICIAL_HORIZONS
from src.models.hybrid_features import future_business_date

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "data" / "prediction_journal.jsonl"
NEUTRAL_BAND_PCT = 1.5


class JournalLockError(RuntimeError):
    """The journal lock could not be taken. The forecast was not dropped."""


def journal_file(path: Optional[Path] = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("PREDICTION_JOURNAL")
    if override:
        return Path(override)
    return JOURNAL


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def forecast_identity(
    *,
    market: str,
    origin_date: Optional[str],
    target_date: Optional[str],
    model_version: str,
    procedure_fingerprint: str,
    data_fingerprint: str,
    kind: str = "published",
) -> str:
    raw = "|".join(
        [
            kind,
            str(market),
            str(origin_date or ""),
            str(target_date or ""),
            str(model_version),
            str(procedure_fingerprint),
            str(data_fingerprint),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _acquire(lock: Path, retries: int = 8, delay: float = 0.25) -> int:
    lock.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            if lock.exists():
                age = time.time() - lock.stat().st_mtime
                if age > 30:
                    lock.unlink(missing_ok=True)
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("ascii"))
            return fd
        except FileExistsError:
            time.sleep(delay * (attempt + 1))
        except OSError:
            time.sleep(delay * (attempt + 1))
    raise JournalLockError(
        "Verrou du journal indisponible après plusieurs tentatives. "
        "Relancer la commande réécrit la même ligne."
    )


def _release(fd: int, lock: Path) -> None:
    try:
        os.close(fd)
    finally:
        lock.unlink(missing_ok=True)


def _read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    return rows


def _logical_key(row: dict[str, Any]) -> tuple:
    price = row.get("price")
    return (
        str(row.get("market") or ""),
        _day(row.get("origin_date")),
        _day(row.get("target_date")),
        int(row.get("horizon") or -1),
        str(row.get("model_version") or ""),
        str(row.get("procedure_fingerprint") or ""),
        None if price is None else float(price),
        bool(row.get("diagnostic")),
    )


def collapse_fingerprint_twins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One emission when only the data fingerprint differs. The file stays untouched."""
    order = sorted(range(len(rows)), key=lambda index: str(rows[index].get("recorded_at") or ""))
    kept_id: dict[tuple, str] = {}
    redirect: dict[str, str] = {}
    chosen: list[dict[str, Any]] = []
    for index in order:
        row = rows[index]
        key = _logical_key(row)
        forecast_id = str(row.get("forecast_id") or "")
        if key in kept_id:
            if forecast_id:
                redirect[forecast_id] = kept_id[key]
            continue
        if forecast_id:
            kept_id[key] = forecast_id
        chosen.append(dict(row))
    for row in chosen:
        opens = row.get("opens_initial_id")
        if opens is not None and str(opens) in redirect:
            row["opens_initial_id"] = redirect[str(opens)]
        row["revises"] = [redirect.get(str(item), str(item)) for item in (row.get("revises") or [])]
    return chosen


def load_forecasts(path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Read every line. Older lines stay readable and are not rewritten."""
    rows = []
    for row in _read_lines(journal_file(path)):
        if "forecast_id" not in row:
            row = {**row, "trace": "incomplete"}
        rows.append(row)
    return rows


def _initials(rows: list[dict[str, Any]], market: str, target_date: Optional[str]) -> list[str]:
    found = []
    for row in rows:
        if row.get("diagnostic") or row.get("published") is False:
            continue
        if row.get("market") != market or row.get("target_date") != target_date:
            continue
        initial_id = row.get("opens_initial_id")
        forecast_id = row.get("forecast_id")
        if initial_id and forecast_id and initial_id == forecast_id:
            found.append(str(forecast_id))
    return found


def append_record(record: dict[str, Any], path: Optional[Path] = None) -> str:
    """Append one immutable row. A repeated identity returns ``duplicate``."""
    journal = journal_file(path)
    lock = _lock_path(journal)
    fd = _acquire(lock)
    try:
        existing = _read_lines(journal)
        forecast_id = str(record["forecast_id"])
        if any(row.get("forecast_id") == forecast_id for row in existing):
            return "duplicate"
        journal.parent.mkdir(parents=True, exist_ok=True)
        with open(journal, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return "written"
    finally:
        _release(fd, lock)


def open_commitment(
    rows: list[dict[str, Any]],
    *,
    market: str,
    horizon: int,
    as_of: str,
) -> Optional[dict[str, Any]]:
    """Oldest published initial of this horizon whose target is still ahead."""
    if int(horizon) not in FROZEN_HORIZONS or not as_of:
        return None
    found = []
    for row in _published(rows):
        if row.get("market") != market or int(row.get("horizon") or -1) != int(horizon):
            continue
        if not row.get("forecast_id"):
            continue
        if row.get("opens_initial_id") != row.get("forecast_id") or row.get("price") is None:
            continue
        target = _day(row.get("target_date"))
        if not target or target <= _day(as_of):
            continue
        found.append(row)
    if not found:
        return None
    found.sort(key=lambda row: (str(row.get("origin_date") or ""), str(row.get("recorded_at") or "")))
    return found[0]


def sessions_between(start: Optional[str], end: Optional[str]) -> Optional[int]:
    """Trading sessions strictly after start, up to and including end."""
    start_day = _day(start)
    end_day = _day(end)
    if not start_day or not end_day:
        return None
    if end_day <= start_day:
        return 0
    count = 0
    cursor = start_day
    while cursor < end_day and count < 400:
        nxt = future_business_date(cursor, 1).date().isoformat()
        if nxt <= cursor:
            return None
        cursor = nxt
        count += 1
    return count


def _latest_same_target(
    rows: list[dict[str, Any]],
    initial: dict[str, Any],
) -> tuple[dict[str, Any], Optional[float], Optional[float], Optional[str]]:
    """Latest published estimate of this target, its gap, and a notice."""
    initial_id = str(initial.get("forecast_id") or "")
    revisions = [
        row
        for row in _published(rows)
        if row.get("market") == initial.get("market")
        and initial_id in (row.get("revises") or [])
        and row.get("forecast_id") != initial_id
    ]
    revisions.sort(key=lambda row: (str(row.get("origin_date") or ""), str(row.get("recorded_at") or "")))
    latest = revisions[-1] if revisions else initial
    initial_price = initial.get("price")
    latest_price = latest.get("price")
    gap = None
    gap_pct = None
    if latest_price is not None and initial_price is not None:
        gap = float(latest_price) - float(initial_price)
        if float(initial_price) != 0:
            gap_pct = gap / float(initial_price) * 100.0
    reference = initial.get("reference_price")
    if reference is None:
        reference = initial.get("origin_price")
    notice = None
    if latest.get("forecast_id") != initial_id and gap_pct is not None:
        opening = direction_from_reference(initial_price, reference)
        current = direction_from_reference(latest_price, reference)
        if opening in {"hausse", "baisse"} and current in {"hausse", "baisse"} and opening != current:
            notice = "inversion"
        elif abs(gap_pct) > NEUTRAL_BAND_PCT:
            notice = "revision"
    return latest, gap, gap_pct, notice


def open_commitments(
    rows: list[dict[str, Any]],
    *,
    market: str,
    as_of: str,
) -> dict[int, dict[str, Any]]:
    """Open J+7, J+14 and J+30 commitments still ahead of the latest session."""
    held: dict[int, dict[str, Any]] = {}
    for horizon in FROZEN_HORIZONS:
        row = open_commitment(rows, market=market, horizon=horizon, as_of=as_of)
        if row is not None:
            held[int(horizon)] = row
    return held


def _view(
    *,
    price: Optional[float],
    lower: Optional[float],
    upper: Optional[float],
    target_date: Optional[str],
    issued_date: Optional[str],
    model_version: Optional[str],
    status: Optional[str],
) -> dict[str, Any]:
    return {
        "price": None if price is None else float(price),
        "lower": None if lower is None else float(lower),
        "upper": None if upper is None else float(upper),
        "target_date": _day(target_date) or None,
        "issued_date": _day(issued_date) or None,
        "model_version": None if model_version is None else str(model_version),
        "status": status if status in {"experimental", "validated", "unavailable"} else None,
    }


def _pair_band(lower: Optional[float], upper: Optional[float]) -> Optional[list[float]]:
    if lower is None or upper is None:
        return None
    return [float(lower), float(upper)]


def _fresh_view(item: dict[str, Any]) -> dict[str, Any]:
    components = item.get("components") or {}
    interval = list(item.get("confidence_interval") or [])
    lower = interval[0] if len(interval) > 0 else None
    upper = interval[1] if len(interval) > 1 else None
    return _view(
        price=item.get("price"),
        lower=lower,
        upper=upper,
        target_date=item.get("target_date") or components.get("target_date"),
        issued_date=components.get("origin_date"),
        model_version=item.get("model_version"),
        status=item.get("status") or components.get("status"),
    )


def _failed(item: dict[str, Any]) -> bool:
    components = item.get("components") or {}
    return (
        item.get("price") is None
        or item.get("status") == "unavailable"
        or bool(components.get("calculation_failed"))
        or bool(components.get("engine_missing"))
    )


def overlay_commitments(
    items: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    market: str,
    as_of: str,
) -> list[dict[str, Any]]:
    """Keep the open initial intact. The fresh calculation stays in ``fresh``."""
    held = open_commitments(rows, market=market, as_of=as_of)
    overlaid = []
    for item in items:
        fresh = _fresh_view(item)
        row = held.get(int(item.get("horizon") or -1))
        if row is None:
            copied = dict(item)
            if not copied.get("target_date"):
                copied["target_date"] = fresh.get("target_date")
            copied["fresh"] = fresh
            overlaid.append(copied)
            continue
        copied = dict(item)
        initial_band = _pair_band(row.get("lower"), row.get("upper"))
        reference = row.get("reference_price")
        if reference is None:
            reference = row.get("origin_price")
        initial = _view(
            price=row.get("price"),
            lower=row.get("lower"),
            upper=row.get("upper"),
            target_date=row.get("target_date"),
            issued_date=row.get("origin_date"),
            model_version=row.get("model_version"),
            status=row.get("status"),
        )
        latest_row, gap, gap_pct, notice = _latest_same_target(rows, row)
        has_revision = latest_row.get("forecast_id") != row.get("forecast_id")
        latest = None
        latest_failed = False
        if has_revision:
            latest_failed = latest_row.get("price") is None or latest_row.get("status") == "unavailable"
            latest = _view(
                price=latest_row.get("price"),
                lower=latest_row.get("lower"),
                upper=latest_row.get("upper"),
                target_date=latest_row.get("target_date"),
                issued_date=latest_row.get("origin_date"),
                model_version=latest_row.get("model_version"),
                status=latest_row.get("status"),
            )
        fresh_same_target = _day(fresh.get("target_date")) == _day(row.get("target_date"))
        latest_unavailable = latest_failed or (not has_revision and fresh_same_target and _failed(item))
        today_remaining = sessions_between(as_of, row.get("target_date"))
        issued_remaining = None
        if latest is not None and latest.get("issued_date"):
            issued_remaining = sessions_between(latest.get("issued_date"), row.get("target_date"))
        revision_matches_today = bool(
            has_revision
            and not latest_failed
            and issued_remaining is not None
            and issued_remaining == today_remaining
            and _day(latest.get("issued_date")) == _day(as_of)
        )
        if latest_failed:
            shown_latest_price = None
            shown_latest_date = None
        elif has_revision and latest is not None:
            shown_latest_price = latest["price"]
            shown_latest_date = latest["issued_date"]
        else:
            shown_latest_price = initial["price"]
            shown_latest_date = initial["issued_date"]
        components = dict(item.get("components") or {})
        for key in ("calculation_failed", "engine_missing", "ensemble_weights"):
            components.pop(key, None)
        components["display_final"] = True
        shown_price = initial["price"]
        shown_band = initial_band
        # Même règle que publish_or_close : une initiale non validée publie le
        # cours de son jour d'émission ; le prix du modèle reste à côté.
        if (
            str(initial.get("status") or "experimental") != "validated"
            and initial["price"] is not None
            and reference is not None
        ):
            components["served_as_close"] = True
            components["candidate_price"] = initial["price"]
            components["label"] = "Cours inchangé — le modèle ne bat pas encore cette référence"
            components["candidate_label"] = "Prix du modèle — non validé"
            shown_price = float(reference)
            shown_band = None
        copied.update(
            {
                "price": shown_price,
                "confidence_interval": shown_band,
                "target_date": initial["target_date"],
                "reference_price": None if reference is None else float(reference),
                "held_commitment": True,
                "initial_horizon": int(row.get("horizon") or item.get("horizon") or 0),
                "sessions_remaining": today_remaining,
                "latest_sessions_remaining": issued_remaining,
                "revision_matches_today": revision_matches_today,
                "latest_price": shown_latest_price,
                "latest_date": shown_latest_date,
                "latest_gap": gap if has_revision and not latest_failed else None,
                "latest_gap_pct": gap_pct if has_revision and not latest_failed else None,
                "revision_notice": notice if has_revision and not latest_failed else None,
                "has_revision": has_revision and not latest_failed,
                "status": initial["status"] or "experimental",
                "components": components,
                "initial": initial,
                "latest": latest,
                "fresh": fresh,
                "latest_unavailable": latest_unavailable,
            }
        )
        overlaid.append(copied)
    return overlaid


def _status(status: str, release_mode: str) -> str:
    if status not in {"experimental", "validated", "unavailable"}:
        status = "unavailable"
    if release_mode == "experimental" and status == "validated":
        return "experimental"
    return status


def _served_kind(status: str, served_as_close: bool) -> str:
    """Ce que le site publie (ou aurait publié) pour cette ligne."""
    if status == "unavailable":
        return "unavailable"
    if served_as_close:
        return "unchanged_close"
    return "validated_model" if status == "validated" else "experimental_model"


def _candidate_kind(status: str, price: Optional[float]) -> Optional[str]:
    """Le prix du modèle, toujours étiqueté : expérimental tant qu'il n'est pas validé."""
    if price is None:
        return None
    return "validated_model" if status == "validated" else "experimental_model"


def append_forecast(
    *,
    market: str,
    model_version: str,
    release_mode: str,
    horizon: int,
    price: Optional[float],
    lower: Optional[float],
    upper: Optional[float],
    origin_date: Optional[str],
    origin_price: Optional[float],
    target_date: Optional[str],
    status: str,
    procedure_fingerprint: str = "",
    data_fingerprint: str = "",
    source_fingerprint: str = "",
    train_end_date: Optional[str] = None,
    diagnostic: bool = False,
    previous_model_version: Optional[str] = None,
    previous_price: Optional[float] = None,
    comparison: Optional[str] = None,
    rejected_lower: Optional[float] = None,
    rejected_upper: Optional[float] = None,
    served_as_close: bool = False,
    path: Optional[Path] = None,
) -> str:
    """Journal one served forecast. Official horizons also open their own initial.

    ``price`` is the model price. ``served_as_close`` says the dashboard
    showed the unchanged close instead, because the horizon is not validated.
    """
    kind = "diagnostic" if diagnostic else "published"
    identity = forecast_identity(
        market=market,
        origin_date=origin_date,
        target_date=target_date,
        model_version=model_version,
        procedure_fingerprint=procedure_fingerprint,
        data_fingerprint=data_fingerprint,
        kind=kind,
    )
    journal = journal_file(path)
    lock = _lock_path(journal)
    fd = _acquire(lock)
    try:
        existing = _read_lines(journal)
        as_of = _day(origin_date)
        suppress_open = False
        if (
            not diagnostic
            and int(horizon) in FROZEN_HORIZONS
            and price is not None
            and as_of
            and open_commitment(existing, market=market, horizon=int(horizon), as_of=as_of) is not None
        ):
            # An open card blocks a second initial. It does not block a
            # revision of an initial that already aims at this target date.
            if _initials(existing, market, target_date):
                suppress_open = True
            else:
                diagnostic = True
                identity = forecast_identity(
                    market=market,
                    origin_date=origin_date,
                    target_date=target_date,
                    model_version=model_version,
                    procedure_fingerprint=procedure_fingerprint,
                    data_fingerprint=data_fingerprint,
                    kind="diagnostic",
                )
        prior = next((row for row in existing if row.get("forecast_id") == identity), None)
        replacement = False
        if prior is not None:
            # A missing engine wrote an unavailable row. A later price for the
            # same target revises that row and does not open a second initial.
            if not (
                prior.get("status") == "unavailable"
                and prior.get("price") is None
                and price is not None
            ):
                return "duplicate"
            identity = hashlib.sha256(f"{prior['forecast_id']}|served".encode()).hexdigest()[:32]
            if any(row.get("forecast_id") == identity for row in existing):
                return "duplicate"
            replacement = True
        opens = None
        revises: list[str] = []
        reference_price = None
        if (
            not diagnostic
            and not suppress_open
            and int(horizon) in OFFICIAL_HORIZONS
            and not replacement
        ):
            opens = identity
            reference_price = None if origin_price is None else float(origin_price)
        journal_only = int(horizon) in JOURNAL_ONLY_HORIZONS
        if not diagnostic and not journal_only:
            revises = [
                item
                for item in _initials(existing, market, target_date)
                if item != identity
            ]
        record_status = _status(status, release_mode)
        record = {
            "forecast_id": identity,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "market": market,
            "model_version": model_version,
            "release_mode": release_mode,
            "horizon": int(horizon),
            "price": price,
            "lower": lower,
            "upper": upper,
            "rejected_lower": None if rejected_lower is None else float(rejected_lower),
            "rejected_upper": None if rejected_upper is None else float(rejected_upper),
            "origin_date": origin_date,
            "origin_price": origin_price,
            "target_date": target_date,
            "status": record_status,
            "procedure_fingerprint": procedure_fingerprint,
            "data_fingerprint": data_fingerprint,
            "source_fingerprint": source_fingerprint,
            "train_end_date": train_end_date or origin_date,
            "opens_initial_id": opens,
            "revises": revises,
            "reference_price": reference_price,
            "published": not diagnostic,
            "diagnostic": diagnostic,
            "previous_model_version": previous_model_version,
            "previous_price": previous_price,
            "comparison": comparison,
            "served_as_close": bool(served_as_close),
            # Deux valeurs à ne jamais confondre : la référence publiée (ou qui l'aurait été)
            # et le prix du modèle expérimental, gardé à part tant qu'il n'est pas validé.
            "naive_price": origin_price,
            "served_kind": _served_kind(record_status, served_as_close),
            "candidate_kind": _candidate_kind(record_status, price),
            "journal_only": journal_only,
            "trace": "complete",
        }
        journal.parent.mkdir(parents=True, exist_ok=True)
        with open(journal, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return identity
    finally:
        _release(fd, lock)


def direction_from_reference(
    price: Optional[float],
    reference: Optional[float],
    band: float = NEUTRAL_BAND_PCT,
) -> Optional[str]:
    """Hausse, neutre or baisse against one frozen starting price."""
    if price is None or reference is None or float(reference) <= 0:
        return None
    change = (float(price) / float(reference) - 1.0) * 100.0
    if change > band:
        return "hausse"
    if change < -band:
        return "baisse"
    return "neutre"


def _published(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    for row in rows:
        if row.get("diagnostic") or row.get("published") is False:
            continue
        kept.append(row)
    return kept


def _day(value: Optional[str]) -> str:
    return str(value or "")[:10]


def count_inversions(directions: list[Optional[str]]) -> int:
    """Count hausse ↔ baisse flips.

    Neutre, an unavailable price and a missing session are not flips by
    themselves. Hausse, then neutre, then baisse counts as one flip, because
    the next polar direction is compared with the previous one.
    """
    polar = [item for item in directions if item in {"hausse", "baisse"}]
    flips = 0
    for previous, current in zip(polar, polar[1:]):
        if previous != current:
            flips += 1
    return flips


def _emission_points(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not series:
        return []
    by_day = {_day(row.get("origin_date")): row for row in series}
    start = min(by_day)
    end = max(by_day)
    if not start or not end:
        return []
    points = []
    cursor = start
    guard = 0
    while cursor <= end and guard < 400:
        row = by_day.get(cursor)
        price = None if row is None else row.get("price")
        points.append(
            {
                "date": cursor,
                "price": price,
                "model_version": None if row is None else row.get("model_version"),
                "gap": row is None,
            }
        )
        nxt = future_business_date(cursor, 1).date().isoformat()
        if nxt <= cursor:
            break
        cursor = nxt
        guard += 1
    return points


def _select_initials(
    initials: list[dict[str, Any]],
    *,
    target_date: Optional[str],
    horizon: Optional[int],
    offset: int,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    selected = initials
    if target_date:
        selected = [row for row in selected if _day(row.get("target_date")) == _day(target_date)]
    if horizon is not None:
        selected = [row for row in selected if int(row.get("horizon") or -1) == int(horizon)]
    start = max(0, int(offset))
    return selected[start : start + int(limit)], len(selected)


def tracking_view(
    rows: list[dict[str, Any]],
    *,
    market: str,
    initial_id: Optional[str] = None,
    realized_price: Optional[float] = None,
    target_reached: bool = False,
    limit: int = 12,
    offset: int = 0,
    target_date: Optional[str] = None,
    horizon: Optional[int] = None,
) -> dict[str, Any]:
    """One initial and the later estimates that revise that same target."""
    market_rows = collapse_fingerprint_twins([row for row in rows if row.get("market") == market])
    published = _published(market_rows)
    initials = [
        row
        for row in published
        if row.get("forecast_id")
        and row.get("opens_initial_id") == row.get("forecast_id")
    ]
    initials.sort(key=lambda row: (str(row.get("origin_date") or ""), str(row.get("recorded_at") or "")), reverse=True)
    distinct = []
    seen = set()
    for row in initials:
        key = (
            _day(row.get("origin_date")),
            _day(row.get("target_date")),
            int(row.get("horizon") or -1),
        )
        if key in seen:
            continue
        seen.add(key)
        distinct.append(row)
    initials_for_page = distinct
    page, total = _select_initials(
        initials_for_page, target_date=target_date, horizon=horizon, offset=offset, limit=limit
    )
    if initial_id is None:
        return {
            "market": market,
            "initials": [_summary(row) for row in page],
            "total_initials": total,
            "offset": offset,
        }
    initial = next((row for row in initials if row.get("forecast_id") == initial_id), None)
    recent = [_summary(row) for row in page]
    if initial is None:
        return {
            "market": market,
            "initial_id": initial_id,
            "found": False,
            "initials": recent,
            "total_initials": total,
            "offset": offset,
        }
    revisions = [
        row
        for row in published
        if initial_id in (row.get("revises") or [])
        and row.get("forecast_id") != initial_id
    ]
    revisions.sort(key=lambda row: (str(row.get("origin_date") or ""), str(row.get("recorded_at") or "")))
    reference = initial.get("reference_price")
    if reference is None:
        reference = initial.get("origin_price")
    series = [initial, *revisions]
    latest = series[-1]
    directions = [
        direction_from_reference(row.get("price"), reference) for row in series
    ]
    latest_price = latest.get("price")
    initial_price = initial.get("price")
    gap = None
    gap_pct = None
    if latest_price is not None and initial_price is not None:
        gap = float(latest_price) - float(initial_price)
        if float(initial_price) != 0:
            gap_pct = gap / float(initial_price) * 100.0
    target = _day(initial.get("target_date"))
    del target
    mature = bool(target_reached)
    diagnostics = [
        row
        for row in market_rows
        if row.get("diagnostic")
        and row.get("target_date") == initial.get("target_date")
        and _day(row.get("origin_date")) == _day(latest.get("origin_date"))
    ]
    comparison = None
    if diagnostics:
        diag = diagnostics[-1]
        comparison = {
            "available": diag.get("comparison") != "indisponible" and diag.get("previous_price") is not None,
            "comparison": diag.get("comparison") or "disponible",
            "previous_model_version": diag.get("previous_model_version"),
            "previous_price": diag.get("previous_price"),
            "new_price": diag.get("price"),
            "note": "Diagnostic du changement de lot. Il n'est pas une révision publiée.",
        }
    return {
        "market": market,
        "found": True,
        "initials": recent,
        "total_initials": total,
        "offset": offset,
        "initial": _with_error(_summary(initial), realized_price, mature),
        "reference_price": reference,
        "revisions": [_with_error(_summary(row), realized_price, mature) for row in revisions],
        "latest": _with_error(_summary(latest), realized_price, mature),
        "gap": gap,
        "gap_pct": gap_pct,
        "direction": directions[-1] if directions else None,
        "direction_vs_emission_spot": direction_from_reference(
            latest_price, latest.get("origin_price")
        ),
        "inversions": count_inversions(directions),
        "points": _emission_points(series),
        "remaining_horizon": None if mature else latest.get("horizon"),
        "mature": mature,
        "realized_price": realized_price if mature else None,
        "initial_error": _price_error(initial_price, realized_price, mature),
        "latest_error": _price_error(latest_price, realized_price, mature),
        "model_comparison": comparison,
    }


def _price_error(
    price: Optional[float],
    realized: Optional[float],
    mature: bool,
) -> Optional[float]:
    if not mature or price is None or realized is None:
        return None
    return float(price) - float(realized)


def _with_error(summary: dict[str, Any], realized: Optional[float], mature: bool) -> dict[str, Any]:
    copied = dict(summary)
    copied["error"] = _price_error(copied.get("price"), realized, mature)
    return copied


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "forecast_id": row.get("forecast_id"),
        "origin_date": row.get("origin_date"),
        "origin_price": row.get("origin_price"),
        "target_date": row.get("target_date"),
        "horizon": row.get("horizon"),
        "price": row.get("price"),
        "lower": row.get("lower"),
        "upper": row.get("upper"),
        "status": row.get("status"),
        "model_version": row.get("model_version"),
        "reference_price": row.get("reference_price"),
        "trace": row.get("trace") or "complete",
        "revises": list(row.get("revises") or []),
    }


def price_on_date(rows: list[dict[str, Any]], target_date: Optional[str]) -> Optional[float]:
    """Test helper. Production uses the price table, not the journal."""
    del rows, target_date
    return None


def lookup_calls(loader: Callable[[str], Optional[float]], target_date: Optional[str]) -> Optional[float]:
    if not target_date:
        return None
    return loader(str(target_date)[:10])
