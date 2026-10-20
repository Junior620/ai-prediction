"""Retire des tables de prix les lignes datées d'un jour fermé (samedi, dimanche, férié ICE).

Ces lignes ne sont pas des séances : les scrapes ICE du samedi reprenaient la
clôture du vendredi d'une autre échéance (Dec26 au lieu de Mar27). Chaque ligne
est d'abord copiée dans ``data/quarantine/closed_day_rows.jsonl``, puis retirée
de la table. Sans ``--apply``, le script ne fait que lister.

Usage:
    python scripts/quarantine_closed_day_rows.py            # liste seulement
    python scripts/quarantine_closed_day_rows.py --apply    # archive puis retire
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.models.hybrid_features import is_closed_collection_date  # noqa: E402

TABLES = ("cocoa_london_prices", "coffee_robusta_prices")
ARCHIVE = ROOT / "data" / "quarantine" / "closed_day_rows.jsonl"


def closed_day_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if is_closed_collection_date(str(row["date"])[:10])]


def _all_rows(client, table: str) -> list[dict]:
    rows, offset, page = [], 0, 1000
    while True:
        batch = client.table(table).select("*").order("date").range(offset, offset + page - 1).execute().data or []
        rows.extend(batch)
        if len(batch) < page:
            return rows
        offset += page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Archiver puis retirer les lignes")
    args = parser.parse_args()
    from supabase import create_client

    client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    found = 0
    for table in TABLES:
        bad = closed_day_rows(_all_rows(client, table))
        found += len(bad)
        for row in bad:
            print(f"{table} {str(row['date'])[:10]} {row.get('price')} source={row.get('source')}")
        if not args.apply or not bad:
            continue
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with ARCHIVE.open("a", encoding="utf-8") as handle:
            for row in bad:
                handle.write(json.dumps({"table": table, "removed_at": stamp, "row": row}, ensure_ascii=False, default=str) + "\n")
        for row in bad:
            client.table(table).delete().eq("date", row["date"]).execute()
        print(f"[OK] {table} : {len(bad)} ligne(s) archivée(s) dans {ARCHIVE.relative_to(ROOT)} puis retirée(s)")
    if not args.apply:
        print(f"{found} ligne(s) datée(s) d'un jour fermé. Relancer avec --apply pour les retirer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
