"""Met en quarantaine les faux open interest écrits par un test dans le registre réel.

``tests/test_databento_london_collector.py::test_parse_oi_df`` écrivait jusqu'au
20/10/2026 ses deux valeurs factices (C.v.0, 2024-01-02, 10 000 et 10 500) dans
``data/cocoa_oi_receipts.csv``, que ``apply_oi_receipts`` fait lire à l'entraînement.
Seules les lignes identiques à ces valeurs de test sont déplacées vers
``data/quarantine/oi_receipts_test_rows.csv``, avec la cause et la date. Les
autres lignes restent. Sans ``--apply``, le script ne fait que lister.

Usage:
    python scripts/quarantine_test_oi_rows.py [--data-dir DIR] [--apply]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEDGER_NAME = "cocoa_oi_receipts.csv"
ARCHIVE_NAME = "oi_receipts_test_rows.csv"
REASON = "valeur factice de tests/test_databento_london_collector.py::test_parse_oi_df"

# Valeurs exactes du test (symbole, date de référence, réception, quantité).
TEST_ROWS = {
    ("C.v.0", "2024-01-02", "2024-01-02T18:00:00+00:00", 10000.0),
    ("C.v.0", "2024-01-02", "2024-01-02T20:00:00+00:00", 10500.0),
}


def _key(row) -> tuple:
    received = pd.to_datetime(row["ts_recv"], utc=True, errors="coerce")
    return (
        str(row["symbol"]),
        str(row["ts_ref"])[:10],
        received.isoformat() if not pd.isna(received) else "",
        float(row["quantity"]),
    )


def split_test_rows(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(lignes à garder, lignes factices du test)."""
    if ledger.empty:
        return ledger, ledger.iloc[0:0]
    mask = ledger.apply(lambda row: _key(row) in TEST_ROWS, axis=1)
    return ledger[~mask].copy(), ledger[mask].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    ledger_path = data_dir / LEDGER_NAME
    if not ledger_path.exists():
        print(f"[OK] {ledger_path} absent : rien à faire")
        return 0
    ledger = pd.read_csv(ledger_path)
    keep, fake = split_test_rows(ledger)
    print(f"{ledger_path} : {len(ledger)} ligne(s), dont {len(fake)} valeur(s) factice(s) du test")
    for row in fake.itertuples():
        print(f"  {row.symbol} {row.ts_ref} reçu {row.ts_recv} quantité {row.quantity}")
    if not args.apply or fake.empty:
        if not fake.empty:
            print("Relancer avec --apply pour les mettre en quarantaine.")
        return 0
    archive = data_dir / "quarantine" / ARCHIVE_NAME
    archive.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fake = fake.assign(quarantined_at=stamp, reason=REASON)
    if archive.exists():
        fake = pd.concat([pd.read_csv(archive), fake], ignore_index=True)
    fake.to_csv(archive, index=False)
    keep.to_csv(ledger_path, index=False)
    print(f"[OK] {len(fake)} ligne(s) archivée(s) dans {archive} ; registre réécrit avec {len(keep)} ligne(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
