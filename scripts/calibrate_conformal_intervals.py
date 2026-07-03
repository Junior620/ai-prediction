#!/usr/bin/env python
"""Calibrate conformal prediction intervals from walk-forward backtest CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validation.conformal_interval_calibrator import (
    ConformalIntervalCalibrator,
    ConformalIntervalCalibratorConfig,
)
from src.validation.report_loader import find_latest_walk_forward_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrer les intervalles conformes depuis un rapport walk-forward"
    )
    parser.add_argument(
        "--walk-forward-csv",
        type=str,
        default=None,
        help="CSV walk-forward (defaut: dernier rapport)",
    )
    parser.add_argument("--nhits-csv", type=str, default=None)
    parser.add_argument(
        "--ensemble-weights",
        type=str,
        default=str(ROOT / "config" / "ensemble_weights.json"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "config" / "conformal_intervals.json"),
    )
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--asymmetric", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    wf_csv = args.walk_forward_csv
    if wf_csv is None:
        wf_csv = find_latest_walk_forward_csv(str(ROOT / "reports" / "walk_forward"))
        if wf_csv is None:
            print("ERREUR: aucun CSV walk-forward trouve")
            return 1

    nhits_csv = args.nhits_csv
    if nhits_csv is None:
        ts = Path(wf_csv).stem.replace("_walk_forward_predictions", "")
        candidate = Path(wf_csv).parent / f"{ts}_nhits_predictions.csv"
        if candidate.exists():
            nhits_csv = str(candidate)

    config = ConformalIntervalCalibratorConfig(
        coverage_level=args.coverage,
        asymmetric=args.asymmetric,
    )
    calibrator = ConformalIntervalCalibrator(config)
    result = calibrator.run(
        walk_forward_csv=wf_csv,
        nhits_csv=nhits_csv,
        ensemble_weights_file=args.ensemble_weights,
    )
    out_path = calibrator.save(result, args.output)

    print(f"Intervalles conformes ({args.coverage:.0%}) sauvegardes: {out_path}")
    for h, meta in result.intervals_payload["by_horizon"].items():
        ml = meta["margin_lower"]
        mu = meta["margin_upper"]
        cov = meta["empirical_coverage"]
        n = meta["n"]
        print(
            f"  Horizon {h:>2}d: marge -${ml:.0f}/+${mu:.0f}  "
            f"coverage={cov:.1%}  (n={n})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
