"""Bloque l'entraînement si la série sans saut ne couvre pas l'instantané figé (cacao, robusta).

Après la fin de la série, ``apply_roll_series`` garde les prix bruts : une
ligne d'une autre échéance (scrape ICE = échéance de tête) se glisserait alors
derrière la série ajustée, sans signal. On refuse donc toute séance figée
postérieure à la fin de la série.

Usage:
    python scripts/check_roll_series.py --snapshot data/freeze/cocoa_prices.csv
    python scripts/check_roll_series.py --market coffee_robusta --snapshot data/freeze/coffee_robusta_prices.csv
Code de sortie : 0 si la série couvre l'instantané, 1 sinon.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_collection.cocoa_roll import SERIES_PATH, load_series  # noqa: E402
from src.data_collection.robusta_roll import SERIES_PATH as ROBUSTA_SERIES_PATH  # noqa: E402
from src.models.hybrid_features import is_closed_collection_date  # noqa: E402

SERIES = {"cocoa": SERIES_PATH, "coffee_robusta": ROBUSTA_SERIES_PATH}


def uncovered_sessions(snapshot: pd.DataFrame, series: pd.DataFrame | None) -> list[str]:
    """Séances de l'instantané postérieures à la fin de la série (toutes si la série manque)."""
    dates = pd.to_datetime(snapshot["date"]).dt.tz_localize(None).dt.normalize()
    sessions = sorted({d for d in dates if not is_closed_collection_date(d)})
    if series is None or series.empty:
        return [d.strftime("%Y-%m-%d") for d in sessions]
    end = pd.Timestamp(series["date"].max()).normalize()
    return [d.strftime("%Y-%m-%d") for d in sessions if d > end]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--market", choices=tuple(SERIES), default="cocoa")
    parser.add_argument("--series", default=None, help="Par défaut, la série du marché")
    args = parser.parse_args()
    snapshot = pd.read_csv(args.snapshot)
    series = load_series(Path(args.series) if args.series else SERIES[args.market])
    missing = uncovered_sessions(snapshot, series)
    if missing:
        end = "absente" if series is None or series.empty else f"{series['date'].max():%Y-%m-%d}"
        print(
            f"[ERREUR] Série sans saut (fin {end}) en retard sur l'instantané : "
            f"{len(missing)} séance(s) non couvertes ({', '.join(missing[:5])})"
        )
        return 1
    print(f"[OK] Série sans saut à jour jusqu'au {series['date'].max():%Y-%m-%d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
