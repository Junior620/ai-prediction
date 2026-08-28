"""
Verifie la table ICE London cacao sur Supabase.

Etape 1 (manuelle) : executer la section cocoa_london_prices dans
scripts/sql/create_london_oncc_price_tables.sql (editeur SQL Supabase).

Etape 2 : python scripts/apply_london_oncc_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

SQL_FILE = ROOT / "scripts" / "sql" / "create_london_oncc_price_tables.sql"


def main() -> int:
    import os
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("[ERREUR] SUPABASE_URL / SUPABASE_KEY manquants dans .env")
        return 1

    print("Fichier SQL:", SQL_FILE)
    print("Si les tables n'existent pas, copiez le contenu dans Supabase > SQL Editor.\n")

    sb = create_client(url, key)
    ok = True
    for table in ("cocoa_london_prices",):
        try:
            sb.table(table).select("id").limit(1).execute()
            print(f"[OK] Table {table} accessible")
        except Exception as exc:
            ok = False
            print(f"[MANQUANT] {table}: {exc}")

    if not ok:
        print("\nExecutez le SQL puis relancez ce script.")
        return 1

    print("\nTables pretes. Prochaines etapes:")
    print("  python backfill_ice_london_history.py")
    print("  python collect_ice_london_cocoa.py")
    print("  python collect_coffee_robusta_price.py")
    print("  update_system.bat  (reentrainement + deploy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
