"""
Collecte des contrats a terme cacao depuis Investing.com.

Remplace / complete collect_futures.py (Yahoo) pour la courbe ICE US Cocoa.
"""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client

from src.data_collection.investing_futures_collector import fetch_investing_futures

load_dotenv()

DEFAULT_URL = "https://fr.investing.com/commodities/us-cocoa-contracts"
FALLBACK_URL = "https://www.investing.com/commodities/us-cocoa-contracts"


def fetch_cocoa_futures_investing() -> list[dict]:
    """Essaie FR puis EN si besoin."""
    for url in (DEFAULT_URL, FALLBACK_URL):
        data = fetch_investing_futures(url)
        if data:
            return data
    return []


def store_cocoa_futures_investing(supabase=None) -> int:
    """Insere le snapshot courbe dans cocoa_futures. Retourne le nombre de contrats."""
    if supabase is None:
        supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    print("   Fetching cocoa futures from Investing.com...")
    contracts = fetch_cocoa_futures_investing()
    if not contracts:
        print("   No futures data fetched from Investing.com")
        return 0

    for c in contracts:
        print(f"   {c['contract']}: ${c['price_usd']:,.2f}")

    payload = {
        "data": contracts,
        "source": "investing_com",
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    supabase.table("cocoa_futures").insert(payload).execute()
    print(f"   Inserted {len(contracts)} futures contracts into Supabase")
    return len(contracts)


if __name__ == "__main__":
    print("=" * 80)
    print("COLLECTE DES CONTRATS A TERME (FUTURES) - CACAO INVESTING.COM")
    print("=" * 80)
    n = store_cocoa_futures_investing()
    if n == 0:
        raise SystemExit(1)
    print("=" * 80)
