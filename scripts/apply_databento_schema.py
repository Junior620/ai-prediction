"""
Verifie / rappelle la migration SQL Databento (open_interest + cocoa_london_contracts).

Supabase Python ne peut pas ALTER TABLE via l'API REST — executer
scripts/sql/alter_databento_london.sql dans l'editeur SQL, puis relancer ce script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()


def main() -> int:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("[ERREUR] SUPABASE_URL / SUPABASE_KEY manquants")
        return 1

    sb = create_client(url, key)
    sql_path = Path(__file__).resolve().parent / "sql" / "alter_databento_london.sql"
    print("=" * 80)
    print("VERIFICATION SCHEMA DATABENTO")
    print("=" * 80)
    print(f"SQL a executer si besoin: {sql_path}")

    ok = True

    # open_interest sur cocoa_london_prices
    try:
        sb.table("cocoa_london_prices").select("date, open_interest").limit(1).execute()
        print("[OK] cocoa_london_prices.open_interest accessible")
    except Exception as exc:
        ok = False
        print(f"[MANQUANT] open_interest: {exc}")
        print("  -> Executer la section ALTER TABLE de alter_databento_london.sql")

    # table contrats
    try:
        sb.table("cocoa_london_contracts").select("date, contract_rank, close").limit(1).execute()
        print("[OK] cocoa_london_contracts accessible")
    except Exception as exc:
        ok = False
        print(f"[MANQUANT] cocoa_london_contracts: {exc}")
        print("  -> Executer le CREATE TABLE de alter_databento_london.sql")

    print("=" * 80)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
