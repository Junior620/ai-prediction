"""
Collecte du café robusta ICE Futures Europe depuis Databento (RC.c.0..2, ohlcv-1d).

Investing.com renvoie HTTP 403 (Cloudflare) depuis le 14/10/2026. Chaque passage :
1. lit la plage publiée par Databento (gratuit) : dernière journée complète ;
2. estime le coût (gratuit) puis télécharge la dernière semaine (0,004 $, plafond 0,05 $) ;
3. ajoute les versions nouvelles au brut versionné (data/databento_robusta_calendar_raw.csv) ;
4. reconstruit la série sans saut (data/robusta_roll_series.csv) ;
5. écrit dans coffee_robusta_prices chaque séance absente, au cours de clôture de l'échéance
   tenue : jamais au-delà de la dernière journée complète, jamais un jour fermé ;
6. écrit le rapport de fraîcheur et logs/databento_robusta_collection_AAAAMMJJ.json.

Une ligne déjà en base n'est jamais modifiée. L'historique (1,226 $) se télécharge une seule
fois, à part : python scripts/build_robusta_roll_series.py --history.

Usage :
    python collect_coffee_robusta_price.py [--dry-run] [--offline]
      --dry-run : métadonnées gratuites seulement, rien n'est écrit (base, brut, série, fraîcheur)
      --offline : sans Databento, depuis le brut local
Codes de sortie : 0 ok, 1 échec, 2 source indisponible ou base en retard,
4 quarantaine (dernière séance non écrite : jour fermé ou prix hors bornes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data_collection.robusta_roll import (  # noqa: E402
    DAILY_COST_CAP_USD,
    RAW_PATH,
    SERIES_PATH,
    CostRefused,
    build_series,
    estimate_cost,
    fetch_into_store,
    incremental_start,
    write_series,
)
from src.data_collection.versioned_store import VersionedBarStore  # noqa: E402
from src.models.market_registry import get_market_config  # noqa: E402

SOURCE = "databento"
# Codes lus par scripts/run_pipeline.py (1 reste l'échec réel).
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_SOURCE_UNAVAILABLE = 2
EXIT_QUARANTINE = 4
JOURNAL_PREFIX = "databento_robusta_collection"


@dataclass
class Paths:
    raw: Path = RAW_PATH
    series: Path = SERIES_PATH
    quarantine: Path = ROOT / "data" / "quarantine" / "coffee_robusta_sessions.jsonl"
    status_dir: Path = ROOT / "data" / "status"
    logs_dir: Path = ROOT / "logs"


def report_freshness(supabase, table: str, status_dir: Optional[Path] = None, *, write: bool = True) -> dict:
    """Écrit data/status/freshness_coffee_robusta.json : retard et couverture sur 30 jours."""
    from src.models.hybrid_features import is_closed_collection_date
    from src.monitoring.freshness import freshness_report, london_today, session_coverage, write_report

    try:
        rows = supabase.table(table).select("date").order("date", desc=True).limit(60).execute().data or []
    except Exception as exc:
        rows = []
        print(f"[WARN] Fraîcheur: lecture de la base impossible ({exc})")
    sessions = [str(r["date"])[:10] for r in rows if not is_closed_collection_date(str(r["date"])[:10])]
    coverage = session_coverage(sessions, london_today()) if sessions else 0.0
    report = freshness_report(
        "coffee_robusta", sessions[0] if sessions else None, source=SOURCE, coverage=coverage
    )
    if report["status"] != "ok":
        print(f"[ALERTE] Robusta : {report['reason']} (dernière séance {report['last_session']})")
    else:
        print(f"[OK] Fraîcheur : dernière séance {report['last_session']}")
    if not write:
        return report
    write_report(report, status_dir)
    if report["status"] != "ok":
        try:
            from src.monitoring.alert_system import get_alert_system

            get_alert_system().send_data_source_failure_alert("freshness_coffee_robusta", report["reason"], report)
        except Exception:
            pass
    return report


def decide_robusta_quote(data: dict | None) -> dict:
    """Choose insert or quarantine. A missing session date is never today."""
    if not data:
        return {"action": "failed", "reason": "collecte_vide"}
    session = str(data.get("session_date") or "")[:10]
    reason = data.get("quarantine_reason")
    if not reason and len(session) == 10:
        from src.models.hybrid_features import is_closed_collection_date

        if is_closed_collection_date(session):
            reason = "jour_ferme"
    if reason or len(session) != 10:
        return {
            "action": "quarantine",
            "reason": reason or "date_de_seance_absente",
            "record": {
                "market": "coffee_robusta",
                "session_date": session or None,
                "collected_at": data.get("collected_at") or datetime.now().isoformat(),
                "price": data.get("price"),
                "symbol": data.get("symbol"),
                "reason": reason or "date_de_seance_absente",
            },
        }
    return {"action": "insert", "session_date": session, "price": data.get("price")}


def plan_rows(
    series: pd.DataFrame,
    existing: set[str],
    *,
    until: str,
    price_bounds: tuple[float, float],
) -> tuple[list[dict], list[dict]]:
    """Séances de la série absentes de la base : à insérer, ou en quarantaine. Rien après ``until``."""
    inserts: list[dict] = []
    quarantined: list[dict] = []
    lo, hi = price_bounds
    collected_at = datetime.now(timezone.utc).isoformat()
    for row in series.sort_values("date").itertuples(index=False):
        day = pd.Timestamp(row.date).strftime("%Y-%m-%d")
        if day > until[:10] or day in existing:
            continue
        price = float(row.raw_close)
        quote = {"price": price, "symbol": row.symbol, "session_date": day, "collected_at": collected_at}
        if not (lo <= price <= hi):
            quote["quarantine_reason"] = "hors_bornes"
        decision = decide_robusta_quote(quote)
        if decision["action"] == "insert":
            inserts.append(
                {"date": day, "price": price, "symbol": row.symbol, "source": SOURCE, "collected_at": collected_at}
            )
        else:
            quarantined.append({**decision["record"], "source": SOURCE})
    return inserts, quarantined


def existing_dates(supabase, table: str, since: str) -> set[str]:
    """Toutes les dates en base depuis ``since`` (pages de 1 000)."""
    dates: set[str] = set()
    offset = 0
    while True:
        rows = (
            supabase.table(table).select("date").gte("date", since).order("date")
            .range(offset, offset + 999).execute().data or []
        )
        dates.update(str(r["date"])[:10] for r in rows)
        if len(rows) < 1000:
            return dates
        offset += 1000


def append_quarantine(records: list[dict], path: Path) -> int:
    """Une séance déjà en quarantaine pour la même raison n'est pas réécrite chaque jour."""
    seen = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen.add((item.get("session_date"), item.get("reason"), item.get("source")))
    fresh = [r for r in records if (r.get("session_date"), r.get("reason"), r.get("source")) not in seen]
    if fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in fresh:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(fresh)


def _make_client():
    from src.data_collection.databento_london_collector import _make_client as make

    return make()


def main(
    argv: Optional[list[str]] = None,
    *,
    client: Any = None,
    supabase: Any = None,
    paths: Optional[Paths] = None,
    now: Optional[datetime] = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Métadonnées gratuites seulement, aucune écriture")
    parser.add_argument("--offline", action="store_true", help="Sans Databento, depuis le brut local")
    args = parser.parse_args(argv)
    paths = paths or Paths()
    now = now or datetime.now(timezone.utc)
    write = not args.dry_run
    prefix = "[DRY-RUN] " if args.dry_run else ""

    print("=" * 80)
    print(f"{prefix}COLLECTE DU CAFE ROBUSTA (Databento IFEU.IMPACT, RC.c.0..2)")
    print("=" * 80)
    market = get_market_config("coffee_robusta")
    if supabase is None:
        from supabase import create_client

        supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    store = VersionedBarStore(paths.raw)
    journal: dict[str, Any] = {"timestamp": now.isoformat(), "dry_run": args.dry_run, "attempts": []}
    code = EXIT_OK

    print("\n[1/4] Plage publiée et téléchargement de la dernière semaine...")
    if args.offline:
        from src.models.retrain_policy import clock_last_complete_day

        until = clock_last_complete_day(now)
        print(f"[INFO] Hors ligne : dernière journée complète selon l'horloge {until}")
    else:
        from src.data_collection.databento_london_collector import daily_bounds

        client = client or _make_client()
        if client is None:
            print("[ERREUR] DATABENTO_API_KEY manquante ou paquet databento absent")
            report_freshness(supabase, market.price_table, paths.status_dir, write=write)
            return EXIT_SOURCE_UNAVAILABLE
        bounds = daily_bounds(client, now=now.astimezone(timezone.utc).replace(tzinfo=None))
        until = bounds["last_complete_day"]
        attempt: dict[str, Any] = {"strategy": "databento_robusta_ohlcv", **bounds}
        start = incremental_start(store)
        if start is None:
            print(
                f"[ERREUR] Aucun historique dans {store.path.name} : "
                "python scripts/build_robusta_roll_series.py --history (1,226 $, accord du propriétaire)"
            )
            attempt.update(ok=False, detail="historique_absent")
            journal["attempts"].append(attempt)
            if write:
                _write_journal(journal, paths)
            report_freshness(supabase, market.price_table, paths.status_dir, write=write)
            return EXIT_SOURCE_UNAVAILABLE
        end = bounds["end_exclusive"]
        attempt.update(start=start, end_exclusive=end)
        if start >= end:
            print(f"[INFO] Rien de nouveau publié après {start}")
            attempt.update(ok=True, skipped="rien_a_demander")
        elif args.dry_run:
            cost = estimate_cost(client, start, end)
            attempt.update(ok=True, skipped="dry_run", cost_usd=round(cost, 4))
            print(f"[DRY-RUN] Demanderait {start} -> {end} (coût estimé {cost:.4f} $, plafond {DAILY_COST_CAP_USD:.2f} $)")
        else:
            try:
                entry = fetch_into_store(store, client, start, end, cap=DAILY_COST_CAP_USD, bounds=bounds)
                attempt.update(ok=True, **{k: entry[k] for k in ("fetched_at", "cost_usd", "rows", "new_versions")})
                print(
                    f"[OK] {entry['rows']} barres {start} -> {end}, {entry['new_versions']} version(s) nouvelle(s), "
                    f"coût estimé {entry['cost_usd']:.4f} $"
                )
            except CostRefused as exc:
                print(f"[ERREUR] {exc} : rien n'est demandé")
                attempt.update(ok=False, detail=str(exc))
                code = EXIT_SOURCE_UNAVAILABLE
            except Exception as exc:
                print(f"[ERREUR] Databento indisponible ({exc}) : la série reste celle du brut local")
                attempt.update(ok=False, detail=str(exc))
                code = EXIT_SOURCE_UNAVAILABLE
        journal["attempts"].append(attempt)

    print(f"\n[2/4] Série sans saut jusqu'au {until}...")
    series = build_series(store, until=until)
    if series.empty:
        print(f"[ERREUR] Aucune barre robusta dans {store.path.name}")
        return EXIT_FAILED
    last = series.iloc[-1]
    print(
        f"[OK] {len(series)} séances ({series['date'].min():%Y-%m-%d} -> {series['date'].max():%Y-%m-%d}), "
        f"{int(series['roll'].sum())} changements d'échéance, dernier cours {last['raw_close']:,.2f} "
        f"{market.unit} ({last['symbol']})"
    )
    if write:
        write_series(series, store, paths.series, until=until)

    print("\n[3/4] Séances absentes de la base...")
    try:
        existing = existing_dates(supabase, market.price_table, f"{series['date'].min():%Y-%m-%d}")
    except Exception as exc:
        print(f"[ERREUR] Lecture de {market.price_table} impossible ({exc})")
        return EXIT_FAILED
    inserts, quarantined = plan_rows(series, existing, until=until, price_bounds=market.price_bounds)
    journal.update(planned_inserts=len(inserts), quarantined=[q["session_date"] for q in quarantined])
    latest = f"{series['date'].max():%Y-%m-%d}"
    if inserts:
        shown = ", ".join(r["date"] for r in inserts[-5:])
        print(f"{prefix}[INFO] {len(inserts)} séance(s) à écrire (dernières : {shown})")
    else:
        print(f"[OK] Aucune séance manquante jusqu'au {until}")
    for record in quarantined:
        print(f"{prefix}[QUARANTAINE] {record['session_date']} {record['reason']} prix {record['price']:,.2f} non écrit")
    if write:
        if quarantined:
            append_quarantine(quarantined, paths.quarantine)
        try:
            for start_at in range(0, len(inserts), 500):
                supabase.table(market.price_table).insert(inserts[start_at:start_at + 500]).execute()
        except Exception as exc:
            print(f"[ERREUR] Écriture dans {market.price_table} : {exc}")
            journal["status"] = "failed"
            _write_journal(journal, paths)
            return EXIT_FAILED
        if inserts:
            print(f"[OK] {len(inserts)} séance(s) écrite(s) dans {market.price_table}")
    if any(record["session_date"] == latest for record in quarantined):
        # La dernière séance n'est pas écrite : relancer n'y change rien.
        code = EXIT_QUARANTINE

    print("\n[4/4] Fraîcheur...")
    freshness = report_freshness(supabase, market.price_table, paths.status_dir, write=write)
    if args.dry_run and inserts:
        print("[DRY-RUN] Fraîcheur lue avant écriture : les séances ci-dessus la changeraient")
    elif freshness.get("status") != "ok" and code == EXIT_OK:
        code = EXIT_SOURCE_UNAVAILABLE
    journal.update(status="ok" if code == EXIT_OK else "degraded", exit_code=code, last_session=latest, until=until)
    if write:
        print(f"\nJournal : {_write_journal(journal, paths)}")
    print("\n" + "=" * 80)
    print(f"{prefix}COLLECTE ROBUSTA TERMINEE (code {code})")
    print("=" * 80)
    return code


def _write_journal(journal: dict, paths: Paths) -> Path:
    from src.data_collection.databento_london_collector import write_collection_journal

    return write_collection_journal(journal, paths.logs_dir, prefix=JOURNAL_PREFIX)


if __name__ == "__main__":
    raise SystemExit(main())
