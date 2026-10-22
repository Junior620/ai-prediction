"""Construit data/robusta_roll_series.csv, la série robusta sans saut de contrat.

Usage:
    python scripts/build_robusta_roll_series.py                 # reconstruit depuis le brut versionné local
    python scripts/build_robusta_roll_series.py --estimate      # coût de l'historique (gratuit), rien n'est demandé
    python scripts/build_robusta_roll_series.py --history       # historique Databento payant, une seule fois
    python scripts/build_robusta_roll_series.py --as-of 2026-10-20T23:00:00Z --series out.csv
                                                                # la série telle qu'on la connaissait à cet instant

La mise à jour quotidienne se fait dans collect_coffee_robusta_price.py (une semaine, 0,004 $).
``--history`` refuse un brut déjà rempli et tout coût estimé au-delà de 1,23 $.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.data_collection.robusta_roll import (  # noqa: E402
    HISTORY_COST_CAP_USD,
    HISTORY_START,
    RAW_PATH,
    SERIES_PATH,
    CostRefused,
    build_series,
    estimate_cost,
    fetch_into_store,
    write_series,
)
from src.data_collection.versioned_store import VersionedBarStore  # noqa: E402


def main(argv: list[str] | None = None, *, client=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", action="store_true", help="Télécharger l'historique (payant, plafonné)")
    parser.add_argument("--estimate", action="store_true", help="Afficher le coût de l'historique sans rien demander")
    parser.add_argument("--start", default=HISTORY_START)
    parser.add_argument("--as-of", default=None, help="Série telle que connue à cet instant (UTC)")
    parser.add_argument("--raw", default=str(RAW_PATH))
    parser.add_argument("--series", default=str(SERIES_PATH))
    args = parser.parse_args(argv)
    store = VersionedBarStore(Path(args.raw))
    until = None
    if args.history or args.estimate:
        from src.data_collection.databento_london_collector import _make_client, daily_bounds

        client = client or _make_client()
        if client is None:
            print("ERREUR: DATABENTO_API_KEY manquante ou paquet databento absent")
            return 1
        bounds = daily_bounds(client)
        end, until = bounds["end_exclusive"], bounds["last_complete_day"]
        if args.estimate:
            cost = estimate_cost(client, args.start, end)
            print(f"[COÛT] RC.c.0..2 ohlcv-1d {args.start} -> {end} : {cost:.4f} $ (plafond {HISTORY_COST_CAP_USD:.2f} $)")
            return 0
        if store.last_event() is not None:
            print(f"ERREUR: {store.path.name} contient déjà un historique ; la collecte quotidienne le complète.")
            return 1
        try:
            entry = fetch_into_store(store, client, args.start, end, cap=HISTORY_COST_CAP_USD, bounds=bounds, kind="history")
        except CostRefused as exc:
            print(f"ERREUR: {exc}. Accord du propriétaire nécessaire.")
            return 2
        print(
            f"[OK] Historique {entry['start']} -> {entry['end_exclusive']} : {entry['rows']} barres, "
            f"{entry['new_versions']} versions, coût estimé {entry['cost_usd']:.4f} $"
        )
    series = build_series(store, as_of=args.as_of, until=until)
    if series.empty:
        print(f"ERREUR: aucune barre robusta connue dans {store.path}")
        return 1
    path = write_series(series, store, Path(args.series), until=until)
    print(
        f"[OK] {path.name} : {len(series)} séances "
        f"({series['date'].min():%Y-%m-%d} -> {series['date'].max():%Y-%m-%d}), "
        f"{int(series['roll'].sum())} changements d'échéance, dernier cours {series['raw_close'].iloc[-1]:.0f} "
        f"({series['symbol'].iloc[-1]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
