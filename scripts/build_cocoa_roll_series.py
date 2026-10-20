"""Construit data/cocoa_roll_series.csv, la série cacao sans saut de contrat.

Usage:
    python scripts/build_cocoa_roll_series.py            # complète depuis Databento puis reconstruit
    python scripts/build_cocoa_roll_series.py --offline  # reconstruit depuis le brut local
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.data_collection.cocoa_roll import RAW_PATH, SERIES_PATH, refresh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Ne pas appeler Databento")
    args = parser.parse_args()
    try:
        series = refresh(RAW_PATH, SERIES_PATH, fetch=not args.offline)
    except Exception as exc:
        print(f"ERREUR: série sans saut non construite : {exc}")
        return 1
    rolls = int(series["roll"].sum())
    print(
        f"[OK] {SERIES_PATH.relative_to(ROOT)} : {len(series)} séances "
        f"({series['date'].min():%Y-%m-%d} -> {series['date'].max():%Y-%m-%d}), "
        f"{rolls} changements d'échéance, dernier cours {series['raw_close'].iloc[-1]:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
