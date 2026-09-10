"""
Collecte des contrats a terme cacao ICE London mois nommes (DEC26, MAR27...).

Priorite :
1. Databento symboles bruts (C   FMZ0026! …) — fiable, ~J-1
2. Scrape ICE.com tableau CONTRACT/LAST — snapshot du jour

Stocke un snapshot dans cocoa_futures (currency=GBP).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.data_collection.databento_london_collector import fetch_named_london_curve
from src.data_collection.ice_london_collector import fetch_ice_london_named_contracts


LONDON_NAMED_SOURCES = ("databento_london_named", "ice_london_named")


def collect_london_named_futures(supabase=None) -> int:
    if supabase is None:
        supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    print("=" * 80)
    print("COLLECTE CONTRATS A TERME ICE LONDON (mois nommes DEC26…)")
    print("=" * 80)

    contracts: list = []
    source = None
    attempts: list = []

    print("\n[1/2] Databento raw symbols...")
    db_contracts, db_attempts = fetch_named_london_curve()
    attempts.extend(db_attempts)
    if db_contracts:
        contracts = db_contracts
        source = "databento_london_named"
        for c in contracts:
            print(f"   {c['contract']}: £{c['price_usd']:,.2f} ({c.get('date')})")
        print(f"[OK] Databento: {len(contracts)} contrats")
    else:
        print("[WARN] Databento named indisponible — fallback ICE scrape")

    if not contracts:
        print("\n[2/2] Scrape ICE.com...")
        ice_contracts, ice_attempts = fetch_ice_london_named_contracts()
        attempts.extend(ice_attempts)
        if ice_contracts:
            contracts = ice_contracts
            source = "ice_london_named"
            for c in contracts:
                print(f"   {c['contract']}: £{c['price_usd']:,.2f}")
            print(f"[OK] ICE scrape: {len(contracts)} contrats")
        else:
            print("[ERREUR] Aucune courbe Londres nommee recuperable")
            print(f"         attempts={attempts}")
            return 0

    payload = {
        "data": contracts,
        "source": source,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    supabase.table("cocoa_futures").insert(payload).execute()
    print(f"\n[OK] Snapshot {source} insere ({len(contracts)} contrats)")
    print("=" * 80)
    return len(contracts)


if __name__ == "__main__":
    n = collect_london_named_futures()
    raise SystemExit(0 if n > 0 else 1)
