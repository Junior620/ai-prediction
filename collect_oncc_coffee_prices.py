"""
Collecte quotidienne des prix cafe ONCC (Arabica + Robusta) -> oncc_coffee_prices.
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

from src.data_collection.oncc_coffee_collector import (
    fetch_oncc_coffee_prices,
    write_collection_journal,
)

try:
    from src.monitoring.alert_system import get_alert_system
except ImportError:
    get_alert_system = None  # type: ignore


def main() -> int:
    print("=" * 80)
    print("COLLECTE PRIX CAFE ONCC (Arabica + Robusta)")
    print("=" * 80)

    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    collection = fetch_oncc_coffee_prices()

    journal = {
        "timestamp": datetime.now().isoformat(),
        "status": "ok" if collection.ok else "failed",
        "attempts": collection.attempts,
        "products": [p.product for p in collection.prices],
    }

    if not collection.ok:
        write_collection_journal(journal)
        print("[ERREUR] Aucun prix ONCC recupere")
        if get_alert_system:
            try:
                get_alert_system().send_data_source_failure_alert(
                    "oncc_coffee",
                    "Scraping ONCC vide",
                )
            except Exception:
                pass
        return 1

    inserted = 0
    for item in collection.prices:
        check = (
            supabase.table("oncc_coffee_prices")
            .select("id")
            .eq("date", item.date)
            .eq("product", item.product)
            .execute()
        )
        if check.data:
            print(f"[SKIP] {item.product} {item.date} deja en base")
            continue
        supabase.table("oncc_coffee_prices").insert(
            {
                "date": item.date,
                "product": item.product,
                "price": float(item.price),
                "unit": item.unit,
                "source": item.source,
                "trend": item.trend,
                "change_pct": item.change_pct,
                "collected_at": datetime.now().isoformat(),
            }
        ).execute()
        inserted += 1
        print(f"[OK] {item.product}: {item.price:,.0f} {item.unit}")

    journal["inserted"] = inserted
    path = write_collection_journal(journal)
    print(f"\nJournal: {path}")
    print(f"Insere: {inserted} ligne(s)")
    print("=" * 80)
    return 0 if inserted >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
