#!/usr/bin/env python
"""Calibrate ensemble weights from walk-forward backtest CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validation.ensemble_calibrator import EnsembleCalibrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrer les poids d'ensemble sur backtest")
    parser.add_argument(
        "--walk-forward-csv",
        type=str,
        required=True,
        help="CSV walk-forward predictions",
    )
    parser.add_argument(
        "--nhits-csv",
        type=str,
        default=None,
        help="CSV N-HiTS cross_validation (optionnel)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "config" / "ensemble_weights.json"),
    )
    parser.add_argument(
        "--xgb-only",
        action="store_true",
        help="Calibration 2 modeles (XGBoost + Prophet) sans N-HiTS",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calibrator = EnsembleCalibrator()
    result = calibrator.run(
        walk_forward_csv=args.walk_forward_csv,
        nhits_csv=args.nhits_csv,
        xgb_only=args.xgb_only,
    )
    out_path = calibrator.save(result, args.output)

    print("=" * 60)
    print("CALIBRATION ENSEMBLE")
    print("=" * 60)
    for h in sorted(result.weights_payload.get("by_horizon", {}), key=int):
        w = result.weights_payload["by_horizon"][h]
        fixed = result.comparison["fixed"].get(h, float("nan"))
        cal = result.comparison["calibrated"].get(h, float("nan"))
        print(f"\nHorizon {h}d:")
        print(f"  Poids: xgb={w.get('xgb', 0):.2f} nhits={w.get('nhits', 0):.2f} prophet={w.get('prophet', 0):.2f}")
        print(f"  MAPE fixed 40/40/20: {fixed:.2f}%")
        print(f"  MAPE calibree:       {cal:.2f}%")
    print(f"\nSauvegarde: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
