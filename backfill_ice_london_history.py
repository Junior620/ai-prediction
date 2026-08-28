"""
Backfill historique cacao ICE London.

1. Tente le tableau ICE (souvent courbe contrats seulement, pas historique journalier)
2. Fallback : convertit cocoa_prices (ICE NY USD) -> GBP via taux USD/GBP (Frankfurter)
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

from src.data_collection.ice_london_collector import (
    bootstrap_london_from_ny_prices,
    fetch_ice_london_history,
    fetch_usd_gbp_rates,
)
from src.models.market_registry import get_market_config


def _insert_rows(supabase, table: str, rows: list, bounds: tuple) -> tuple[int, int]:
    lo, hi = bounds
    to_insert = []
    skipped = 0
    seen: set[str] = set()
    for row in rows:
        date_str = str(row["date"])[:10]
        if date_str in seen:
            continue
        seen.add(date_str)
        price = float(row["price"])
        if not (lo <= price <= hi):
            skipped += 1
            continue
        to_insert.append(
            {
                "date": date_str,
                "price": price,
                "symbol": row.get("symbol", "LCC"),
                "source": row.get("source", "ice_london"),
                "collected_at": datetime.now().isoformat(),
            }
        )

    inserted = 0
    batch_size = 500
    for i in range(0, len(to_insert), batch_size):
        chunk = to_insert[i : i + batch_size]
        supabase.table(table).upsert(chunk, on_conflict="date").execute()
        inserted += len(chunk)
        print(f"  ... {inserted}/{len(to_insert)} upsertes")

    return inserted, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill ICE London cocoa prices")
    parser.add_argument("--max-rows", type=int, default=0, help="0 = toute la serie NY")
    parser.add_argument("--skip-ice-scrape", action="store_true")
    args = parser.parse_args()

    market = get_market_config("cocoa")
    ice_url = getattr(market, "ice_london_url", None) or (
        "https://www.ice.com/products/37089076/London-Cocoa-Futures/data?marketId=7758984"
    )

    print("=" * 80)
    print("BACKFILL ICE LONDON")
    print("=" * 80)

    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    rows: list = []

    if not args.skip_ice_scrape:
        print("\n[1/2] Scraping tableau ICE...")
        rows = fetch_ice_london_history(
            url=ice_url,
            max_rows=args.max_rows or 120,
            price_bounds=market.price_bounds,
        )
        if rows:
            print(f"[OK] {len(rows)} lignes ICE")
        else:
            print("[INFO] Pas d'historique journalier sur la page ICE (courbe contrats)")

    if not rows:
        print("\n[2/2] Bootstrap depuis cocoa_prices (NY USD -> GBP via Frankfurter)...")
        q = (
            supabase.table("cocoa_prices")
            .select("date, price")
            .order("date", desc=False)
        )
        if args.max_rows and args.max_rows > 0:
            ny = q.limit(args.max_rows).execute()
        else:
            # Pagination par lots de 1000
            ny_rows: list = []
            offset = 0
            while True:
                batch = (
                    supabase.table("cocoa_prices")
                    .select("date, price")
                    .order("date", desc=False)
                    .range(offset, offset + 999)
                    .execute()
                )
                if not batch.data:
                    break
                ny_rows.extend(batch.data)
                if len(batch.data) < 1000:
                    break
                offset += 1000
            ny = type("R", (), {"data": ny_rows})()

        ny_data = ny.data or []
        if not ny_data:
            print("[ERREUR] cocoa_prices vide — impossible de bootstrapper")
            return 1

        start = str(ny_data[0]["date"])[:10]
        end = str(ny_data[-1]["date"])[:10]
        print(f"  NY: {len(ny_data)} points ({start} -> {end})")
        fx = fetch_usd_gbp_rates(start, end)
        print(f"  FX: {len(fx)} jours depuis Frankfurter")
        rows = bootstrap_london_from_ny_prices(ny_data, fx)

    print(f"\nInsertion dans {market.price_table}...")
    inserted, skipped = _insert_rows(supabase, market.price_table, rows, market.price_bounds)
    print(f"\nInsere: {inserted}, deja presents: {skipped}")
    print("=" * 80)
    return 0 if inserted > 0 or skipped > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
