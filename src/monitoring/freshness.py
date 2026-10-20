"""Fraîcheur des prix : la dernière séance en base doit être la veille ouvrée.

Les séances suivent le calendrier ICE Futures Europe du projet
(``config/exchange_holidays.json`` via ``is_closed_collection_date``). La date
de référence est la date civile de Londres : un contrôle lancé le lundi attend
le vendredi précédent.

Un rapport par marché est écrit dans ``data/status/freshness_<marché>.json``.
L'API le relit pour ``/health`` et le tableau de bord ; le traitement du soir
s'en sert pour son code de sortie « source indisponible ».
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STATUS_DIR = ROOT / "data" / "status"
LONDON = "Europe/London"
# Couverture minimale des séances sur les 30 derniers jours civils.
MIN_COVERAGE = 0.9


def _default_is_closed(day) -> bool:
    from src.models.hybrid_features import is_closed_collection_date

    return is_closed_collection_date(day)


def london_today(now: Optional[datetime] = None) -> pd.Timestamp:
    stamp = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(timezone.utc)
    return stamp.tz_convert(LONDON).tz_localize(None).normalize()


def expected_last_session(today, is_closed: Optional[Callable] = None) -> pd.Timestamp:
    """Dernière séance strictement avant ``today``."""
    closed = is_closed or _default_is_closed
    cursor = pd.Timestamp(today).normalize() - pd.Timedelta(days=1)
    while closed(cursor):
        cursor -= pd.Timedelta(days=1)
    return cursor


def sessions_between(start, end, is_closed: Optional[Callable] = None) -> list[pd.Timestamp]:
    """Séances de ``start`` exclu à ``end`` inclus."""
    closed = is_closed or _default_is_closed
    days = pd.date_range(pd.Timestamp(start).normalize() + pd.Timedelta(days=1), pd.Timestamp(end).normalize())
    return [d for d in days if not closed(d)]


def session_coverage(dates: Iterable, today, days: int = 30, is_closed: Optional[Callable] = None) -> float:
    """Part des séances des ``days`` derniers jours civils présentes dans ``dates``."""
    end = expected_last_session(today, is_closed)
    expected = sessions_between(end - pd.Timedelta(days=days), end, is_closed)
    if not expected:
        return 1.0
    present = {pd.Timestamp(d).normalize() for d in dates}
    return sum(1 for d in expected if d in present) / len(expected)


def freshness_report(
    market_id: str,
    last_session,
    *,
    today=None,
    source: Optional[str] = None,
    coverage: Optional[float] = None,
    is_closed: Optional[Callable] = None,
) -> Dict[str, Any]:
    """État de fraîcheur d'un marché, lisible par un dirigeant."""
    day = london_today() if today is None else pd.Timestamp(today).normalize()
    expected = expected_last_session(day, is_closed)
    last = None if last_session is None else pd.Timestamp(last_session).normalize()
    stale = len(sessions_between(last, expected, is_closed)) if last is not None else None
    status = "ok"
    reasons = []
    if last is None:
        status = "stale"
        reasons.append("aucune séance en base")
    elif stale and stale > 0:
        status = "stale"
        reasons.append(f"retard de {stale} séance(s)")
    if coverage is not None and coverage < MIN_COVERAGE:
        status = "stale"
        reasons.append(f"séances présentes sur 30 jours : {coverage:.0%}")
    return {
        "market": market_id,
        "status": status,
        "last_session": None if last is None else last.strftime("%Y-%m-%d"),
        "expected_last_session": expected.strftime("%Y-%m-%d"),
        "stale_sessions": stale,
        "coverage_30d": None if coverage is None else round(float(coverage), 3),
        "source": source,
        "reason": "; ".join(reasons) or None,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def status_path(market_id: str, status_dir: Optional[Path] = None) -> Path:
    return Path(status_dir or STATUS_DIR) / f"freshness_{market_id}.json"


def write_report(report: Dict[str, Any], status_dir: Optional[Path] = None) -> Path:
    path = status_path(report["market"], status_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def read_report(market_id: str, status_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = status_path(market_id, status_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def check(markets: Iterable[str], status_dir: Optional[Path] = None) -> int:
    """0 si tous les marchés sont à jour, 2 sinon (rapport absent compris)."""
    code = 0
    for market_id in markets:
        report = read_report(market_id, status_dir)
        if report is None:
            print(f"[ALERTE] {market_id} : aucun rapport de fraîcheur")
            code = 2
        elif report.get("status") != "ok":
            print(f"[ALERTE] {market_id} : {banner(report)}")
            code = 2
        else:
            print(f"[OK] {market_id} : dernière séance {report.get('last_session')}")
    return code


def banner(report: Optional[Dict[str, Any]]) -> Optional[str]:
    """Phrase du bandeau ; None quand les données sont à jour."""
    if not report or report.get("status") == "ok":
        return None
    last = report.get("last_session")
    when = "aucune séance" if not last else f"dernière séance : {pd.Timestamp(last):%d/%m}"
    source = f" ({report['source']})" if report.get("source") else ""
    return f"Données en retard — {when}, {report.get('reason') or 'cause inconnue'}{source}"


if __name__ == "__main__":
    import sys

    raise SystemExit(check(sys.argv[1:] or ["cocoa", "coffee_robusta"]))
