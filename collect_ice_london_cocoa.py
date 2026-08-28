"""
Collecte quotidienne du prix cacao ICE London -> Supabase (cocoa_london_prices).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

from src.data_collection.ice_london_collector import (
    fetch_ice_london_spot,
    write_collection_journal,
)
from src.models.market_registry import get_market_config

try:
    from src.monitoring.alert_system import get_alert_system
except ImportError:
    get_alert_system = None  # type: ignore


def main() -> int:
    print("=" * 80)
    print("COLLECTE PRIX CACAO ICE LONDON")
    print("=" * 80)

    market = get_market_config("cocoa")
    ice_url = getattr(market, "ice_london_url", None) or (
        "https://www.ice.com/products/37089076/London-Cocoa-Futures/data?marketId=7758984"
    )

    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    print("\n[1/3] Dernier prix en base...")
    last = (
        supabase.table(market.price_table)
        .select("date, price")
        .order("date", desc=True)
        .limit(1)
        .execute()
    )
    if last.data:
        print(f"[OK] {last.data[0]['price']:,.2f} {market.unit} le {last.data[0]['date']}")
    else:
        print("[WARN] Aucune donnee en base")

    print(f"\n[2/3] Scraping ICE London ({ice_url})...")
    t0 = datetime.now()
    result = fetch_ice_london_spot(url=ice_url, price_bounds=market.price_bounds)
    duration_ms = int((datetime.now() - t0).total_seconds() * 1000)

    journal = {
        "timestamp": datetime.now().isoformat(),
        "status": "ok" if result else "failed",
        "strategy": result.strategy if result else None,
        "duration_ms": duration_ms,
        "attempts": result.attempts if result else [],
    }

    if result is None:
        write_collection_journal(journal)
        print("[ERREUR] Collecte ICE London echouee — aucun prix insere")
        if get_alert_system:
            try:
                get_alert_system().send_data_source_failure_alert(
                    "ice_london",
                    "Toutes les strategies de scraping ont echoue",
                    context={"url": ice_url},
                )
            except Exception:
                pass
        return 1

    lo, hi = market.price_bounds
    if not (lo <= result.price <= hi):
        journal["status"] = "failed"
        journal["error"] = f"prix_hors_bornes:{result.price}"
        write_collection_journal(journal)
        print(f"[ERREUR] Prix {result.price} hors bornes ({lo}-{hi})")
        return 1

    print(
        f"[OK] {result.price:,.2f} {market.unit} "
        f"({result.date}, source={result.source}, strategie={result.strategy})"
    )

    print("\n[3/3] Insertion Supabase...")
    check = (
        supabase.table(market.price_table)
        .select("id")
        .eq("date", result.date)
        .execute()
    )
    if check.data:
        print(f"[SKIP] Date {result.date} deja en base")
    else:
        supabase.table(market.price_table).insert(
            {
                "date": result.date,
                "price": float(result.price),
                "symbol": result.symbol,
                "source": result.source,
                "collected_at": datetime.now().isoformat(),
            }
        ).execute()
        print(f"[OK] Insere: {result.date} - {result.price:,.2f}")

    journal["price"] = result.price
    journal["date"] = result.date
    journal["source"] = result.source
    path = write_collection_journal(journal)
    print(f"\nJournal: {path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
