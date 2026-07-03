"""
Collecte du prix du café robusta du jour depuis Investing.com (London Robusta).

Équivalent robusta de collect_latest_price.py (cacao/Yahoo).
Le symbole du contrat actif (ex. RCU6) est détecté sur la page — les rollovers
sont suivis automatiquement.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()

from src.data_collection.investing_price_collector import fetch_investing_price
from src.models.market_registry import get_market_config

print("=" * 80)
print("COLLECTE DU PRIX ACTUEL DU CAFE ROBUSTA (Investing.com)")
print("=" * 80)

market = get_market_config("coffee_robusta")

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# 1. Dernier prix en base
print("\n[1/3] Vérification du dernier prix en base...")
response = (
    supabase.table(market.price_table)
    .select("date, price")
    .order("date", desc=True)
    .limit(1)
    .execute()
)
if response.data:
    last_date = response.data[0]["date"]
    last_price = response.data[0]["price"]
    print(f"[OK] Dernier prix en base: {last_price:,.2f} {market.unit} le {last_date}")
else:
    last_date, last_price = None, None
    print("[WARN] Aucune donnée en base")

# 2. Scrape Investing.com
print(f"\n[2/3] Scraping Investing.com ({market.investing_url})...")
data = fetch_investing_price(
    url=market.investing_url,
    fallback_symbol=market.contract_symbol or "",
)

if data is None:
    print("[ERREUR] Impossible de récupérer le prix sur Investing.com")
    print("         (captcha, changement de page, ou réseau). Réessayer plus tard.")
    raise SystemExit(1)

price = data["price"]
symbol = data["symbol"]
today = data["date"]
print(f"[OK] Prix actuel: {price:,.2f} {market.unit} (contrat {symbol}, {today})")

lo, hi = market.price_bounds
if not (lo <= price <= hi):
    print(f"[ERREUR] Prix {price} hors bornes plausibles ({lo}-{hi}) — insertion annulée")
    raise SystemExit(1)

# 3. Insertion (si pas déjà présent)
print("\n[3/3] Insertion dans Supabase...")
check = supabase.table(market.price_table).select("id").eq("date", today).execute()
if check.data:
    print(f"[SKIP] Prix du {today} déjà en base")
else:
    supabase.table(market.price_table).insert(
        {
            "date": today,
            "price": float(price),
            "symbol": symbol,
            "source": data["source"],
            "collected_at": datetime.now().isoformat(),
        }
    ).execute()
    print(f"[OK] Inséré: {today} - {price:,.2f} {market.unit}")

if last_price:
    change = price - last_price
    change_pct = change / last_price * 100
    print(f"\nVariation depuis {last_date}: {change:+,.2f} ({change_pct:+.2f}%)")

print("\n" + "=" * 80)
print("COLLECTE ROBUSTA TERMINEE")
print("=" * 80)
