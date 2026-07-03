"""
Import du CSV historique café robusta (format Investing.com) dans Supabase.

Format attendu : Date ("MM/DD/YYYY"), "Price", "Open", "High", "Low", "Vol.", "Change %"
avec séparateurs de milliers (ex. "3,745.00") et volumes en K (ex. "9.03K").

Usage:
    python scripts/import_coffee_csv.py [--csv PATH] [--table coffee_robusta_prices] [--dry-run]
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

DEFAULT_CSV = "data/cafe/ROBUSTA/London Robusta Coffee Futures Historical Data.csv"
DEFAULT_TABLE = "coffee_robusta_prices"
BATCH_SIZE = 500


def parse_investing_csv(csv_path: str) -> pd.DataFrame:
    """Parse un export historique Investing.com en DataFrame (date, price)."""
    df = pd.read_csv(csv_path, thousands=",")
    df.columns = [c.strip() for c in df.columns]

    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df.dropna(subset=["date", "price"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df[["date", "price"]].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import robusta CSV into Supabase")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Chemin du CSV Investing.com")
    parser.add_argument("--table", default=DEFAULT_TABLE, help="Table Supabase cible")
    parser.add_argument("--symbol", default="RCU6", help="Symbole contrat pour la colonne symbol")
    parser.add_argument("--dry-run", action="store_true", help="Parse sans insérer")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERREUR] CSV introuvable: {csv_path}")
        return 1

    print("=" * 70)
    print(f"IMPORT ROBUSTA -> Supabase table '{args.table}'")
    print("=" * 70)

    df = parse_investing_csv(str(csv_path))
    print(f"[OK] {len(df)} lignes parsées "
          f"({df['date'].min().date()} -> {df['date'].max().date()}, "
          f"prix {df['price'].min():.0f}-{df['price'].max():.0f})")

    if args.dry_run:
        print("[DRY-RUN] Aucune insertion effectuée.")
        return 0

    from supabase import create_client

    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    # Dates déjà présentes (pagination pour dépasser la limite de 1000 lignes)
    existing = set()
    offset, page = 0, 1000
    while True:
        resp = (
            supabase.table(args.table)
            .select("date")
            .range(offset, offset + page - 1)
            .execute()
        )
        if not resp.data:
            break
        existing.update(row["date"] for row in resp.data)
        offset += page
    print(f"[INFO] {len(existing)} dates déjà en base")

    rows = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "price": float(row.price),
            "symbol": args.symbol,
            "source": "investing_com_csv",
            "collected_at": datetime.now().isoformat(),
        }
        for row in df.itertuples()
        if row.date.strftime("%Y-%m-%d") not in existing
    ]

    if not rows:
        print("[OK] Rien à insérer, la base est à jour.")
        return 0

    inserted = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        supabase.table(args.table).insert(batch).execute()
        inserted += len(batch)
        print(f"   ... {inserted}/{len(rows)} lignes insérées")

    print(f"[OK] Import terminé: {inserted} nouvelles lignes dans '{args.table}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
