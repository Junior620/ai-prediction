"""
Backfill historique London Cocoa via Databento (OHLCV + Open Interest).

Ecrase le bootstrap NY→GBP sur les dates couvertes (upsert on_conflict=date).
Optionnel : remplit aussi cocoa_london_contracts (C.v.0 … C.v.3).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.data_collection.databento_london_collector import (
    bars_to_contract_rows,
    bars_to_supabase_rows,
    continuous_symbols,
    fetch_daily_bars_with_oi,
    write_collection_journal,
)
from src.models.market_registry import get_market_config


def _year_chunks(start_year: int, end_year: int) -> list[tuple[str, str]]:
    chunks = []
    for year in range(start_year, end_year + 1):
        chunks.append((f"{year}-01-01", f"{year}-12-31"))
    return chunks


def _table_has_column(supabase, table: str, column: str) -> bool:
    try:
        supabase.table(table).select(column).limit(1).execute()
        return True
    except Exception:
        return False


def _upsert_batches(supabase, table: str, rows: list, conflict: str) -> int:
    inserted = 0
    batch_size = 500
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        supabase.table(table).upsert(chunk, on_conflict=conflict).execute()
        inserted += len(chunk)
        print(f"  ... {inserted}/{len(rows)} upsertes -> {table}")
    return inserted


def _count_by_source(supabase, table: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    offset = 0
    while True:
        resp = (
            supabase.table(table)
            .select("source")
            .range(offset, offset + 999)
            .execute()
        )
        if not resp.data:
            break
        for row in resp.data:
            src = row.get("source") or "unknown"
            counts[src] = counts.get(src, 0) + 1
        if len(resp.data) < 1000:
            break
        offset += 1000
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill Databento London Cocoa")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=datetime.utcnow().year)
    parser.add_argument(
        "--with-contracts",
        action="store_true",
        help="Remplit aussi cocoa_london_contracts (C.v.0..3)",
    )
    parser.add_argument("--skip-oi", action="store_true")
    args = parser.parse_args()

    market = get_market_config("cocoa")
    table = market.price_table
    bounds = market.price_bounds

    print("=" * 80)
    print("BACKFILL DATABENTO — LONDON COCOA")
    print("=" * 80)
    print(f"Table: {table} | bornes: {bounds} | annees: {args.start_year}-{args.end_year}")

    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    has_oi_col = _table_has_column(supabase, table, "open_interest")
    has_contracts = _table_has_column(supabase, "cocoa_london_contracts", "close")
    if not has_oi_col:
        print("[WARN] Colonne open_interest absente — OHLCV seulement")
        print("       Executer scripts/sql/alter_databento_london.sql pour OI + contrats")

    symbols = continuous_symbols((0, 1, 2, 3)) if args.with_contracts else continuous_symbols((0,))
    all_bars = []

    for start, end in _year_chunks(args.start_year, args.end_year):
        print(f"\n[FETCH] {start} -> {end} ({', '.join(symbols)})...")
        result = fetch_daily_bars_with_oi(
            start=start,
            end=end,
            symbols=symbols,
            price_bounds=bounds,
            include_oi=not args.skip_oi,
        )
        for att in result.attempts:
            print(f"  attempt: {att}")
        if not result.ok:
            print(f"  [WARN] Aucune barre ({result.error})")
            continue
        print(f"  [OK] {len(result.bars)} barres")
        all_bars.extend(result.bars)

    if not all_bars:
        print("\n[ERREUR] Aucune donnee Databento recuperee")
        write_collection_journal(
            {
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "action": "backfill",
                "error": "no_bars",
            }
        )
        return 1

    price_rows = bars_to_supabase_rows(
        all_bars, front_only=True, include_oi=has_oi_col and not args.skip_oi
    )
    print(f"\n[UPSERT] {len(price_rows)} lignes front-month -> {table}")
    n_prices = _upsert_batches(supabase, table, price_rows, conflict="date")

    n_contracts = 0
    if args.with_contracts:
        if not has_contracts:
            print("\n[WARN] Table cocoa_london_contracts absente — skip contrats")
            print("       Executer scripts/sql/alter_databento_london.sql")
        else:
            contract_rows = bars_to_contract_rows(all_bars)
            print(f"\n[UPSERT] {len(contract_rows)} lignes -> cocoa_london_contracts")
            try:
                n_contracts = _upsert_batches(
                    supabase,
                    "cocoa_london_contracts",
                    contract_rows,
                    conflict="date,contract_rank",
                )
            except Exception as exc:
                print(f"[WARN] cocoa_london_contracts upsert echoue: {exc}")
                print("       Executer scripts/sql/alter_databento_london.sql d'abord")

    print("\n[RESUME] Sources dans", table)
    try:
        counts = _count_by_source(supabase, table)
        for src, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {src}: {n}")
    except Exception as exc:
        print(f"  (compte sources indisponible: {exc})")

    journal = {
        "timestamp": datetime.now().isoformat(),
        "status": "ok",
        "action": "backfill",
        "price_rows": n_prices,
        "contract_rows": n_contracts,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "with_oi": not args.skip_oi,
    }
    path = write_collection_journal(journal)
    print(f"\nJournal: {path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
