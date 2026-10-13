"""
Collecte du prix du café robusta du jour depuis Investing.com (London Robusta).

Équivalent robusta de collect_latest_price.py (cacao/Yahoo).
Le symbole du contrat actif (ex. RCU6) est détecté sur la page — les rollovers
sont suivis automatiquement.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()

from src.data_collection.investing_price_collector import fetch_investing_price
from src.models.market_registry import get_market_config

QUARANTINE = Path("data/quarantine/coffee_robusta_sessions.jsonl")


def decide_robusta_quote(data: dict | None) -> dict:
    """Choose insert or quarantine. A missing session date is never today."""
    if not data:
        return {"action": "failed", "reason": "collecte_vide"}
    session = str(data.get("session_date") or "")[:10]
    if data.get("quarantine_reason") or len(session) != 10:
        return {
            "action": "quarantine",
            "reason": data.get("quarantine_reason") or "date_de_seance_absente",
            "record": {
                "market": "coffee_robusta",
                "session_date": session or None,
                "collected_at": data.get("collected_at") or datetime.now().isoformat(),
                "price": data.get("price"),
                "symbol": data.get("symbol"),
                "reason": data.get("quarantine_reason") or "date_de_seance_absente",
            },
        }
    return {"action": "insert", "session_date": session, "price": data.get("price")}


def main() -> None:
    from supabase import create_client

    print("=" * 80)
    print("COLLECTE DU PRIX ACTUEL DU CAFE ROBUSTA (Investing.com)")
    print("=" * 80)
    market = get_market_config("coffee_robusta")
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
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

    print(f"\n[2/3] Scraping Investing.com ({market.investing_url})...")
    data = fetch_investing_price(
        url=market.investing_url,
        fallback_symbol=market.contract_symbol or "",
    )
    decision = decide_robusta_quote(data)
    if decision["action"] == "failed":
        print("[ERREUR] Impossible de récupérer le prix sur Investing.com")
        raise SystemExit(1)
    if decision["action"] == "quarantine":
        QUARANTINE.parent.mkdir(parents=True, exist_ok=True)
        with QUARANTINE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision["record"], ensure_ascii=False) + "\n")
        print(f"[QUARANTAINE] {decision['reason']} — aucune insertion, date du jour non utilisée")
        raise SystemExit(1)

    price = float(decision["price"])
    symbol = data["symbol"]
    session = decision["session_date"]
    print(f"[OK] Prix actuel: {price:,.2f} {market.unit} (contrat {symbol}, {session})")
    lo, hi = market.price_bounds
    if not (lo <= price <= hi):
        print(f"[ERREUR] Prix {price} hors bornes plausibles ({lo}-{hi}) — insertion annulée")
        raise SystemExit(1)
    print("\n[3/3] Insertion dans Supabase...")
    check = supabase.table(market.price_table).select("id").eq("date", session).execute()
    if check.data:
        print(f"[SKIP] Prix du {session} déjà en base")
    else:
        supabase.table(market.price_table).insert(
            {
                "date": session,
                "price": price,
                "symbol": symbol,
                "source": data["source"],
                "collected_at": datetime.now().isoformat(),
            }
        ).execute()
        print(f"[OK] Inséré: {session} - {price:,.2f} {market.unit}")
    if last_price:
        change = price - last_price
        print(f"\nVariation depuis {last_date}: {change:+,.2f} ({change / last_price * 100:+.2f}%)")
    print("\n" + "=" * 80)
    print("COLLECTE ROBUSTA TERMINEE")
    print("=" * 80)


if __name__ == "__main__":
    main()
