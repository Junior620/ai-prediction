#!/usr/bin/env python
"""
Run honest walk-forward multi-horizon validation for cocoa price models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from supabase import create_client

load_dotenv(ROOT / "config" / ".env")
load_dotenv()

from src.models.hybrid_features import load_price_data_from_supabase
from src.validation.conformal_interval_calibrator import (
    ConformalIntervalCalibrator,
    ConformalIntervalCalibratorConfig,
)
from src.validation.ensemble_calibrator import EnsembleCalibrator
from src.validation.metrics import compute_holdout_baseline
from src.validation.nhits_validator import NHitsValidator, NHitsValidatorConfig
from src.validation.report import _serialize_summary, print_console_report, save_report
from src.validation.walk_forward_validator import WalkForwardConfig, WalkForwardValidator


def load_config_from_yaml() -> dict:
    config_path = ROOT / "config" / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        with open(config_path, encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
        return full.get("validation", {}).get("walk_forward", {})
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    yaml_cfg = load_config_from_yaml()
    nhits_cfg = yaml_cfg.get("nhits", {}) or {}

    parser = argparse.ArgumentParser(
        description="Validation walk-forward multi-horizon honnete"
    )
    parser.add_argument("--market", type=str, default="cocoa", help="Marche (cocoa, coffee_robusta)")
    parser.add_argument("--horizons", type=int, nargs="+", default=yaml_cfg.get("horizons", [1, 7, 30]))
    parser.add_argument("--min-train-days", type=int, default=yaml_cfg.get("min_train_days", 252))
    parser.add_argument("--step-size", type=int, default=yaml_cfg.get("step_size", 5))
    parser.add_argument("--max-origins", type=int, default=yaml_cfg.get("max_origins"))
    parser.add_argument("--output-dir", type=str, default=yaml_cfg.get("output_dir", "reports/walk_forward"))
    parser.add_argument("--skip-nhits", action="store_true", default=not nhits_cfg.get("enabled", True))
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--direct-hstep", action="store_true", help="Include direct h-step in walk-forward (slow)")
    parser.add_argument("--nhits-n-windows", type=int, default=nhits_cfg.get("n_windows", 12))
    parser.add_argument("--nhits-val-size", type=int, default=nhits_cfg.get("val_size", 30))
    parser.add_argument("--nhits-step-size", type=int, default=nhits_cfg.get("step_size", 5))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from src.models.market_registry import get_market_config

    market = get_market_config(args.market)
    if args.market != "cocoa" and args.output_dir == "reports/walk_forward":
        args.output_dir = f"reports/walk_forward/{args.market}"

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        print("ERREUR: SUPABASE_URL et SUPABASE_KEY requis dans .env")
        return 1

    print(f"Marche: {market.display_name} ({args.market})")
    print("Chargement des donnees depuis Supabase...")
    supabase = create_client(supabase_url, supabase_key)
    df = load_price_data_from_supabase(supabase, table_name=market.price_table)
    print(f"  {len(df)} points charges ({df['date'].min().date()} -> {df['date'].max().date()})")

    wf_config = WalkForwardConfig(
        horizons=args.horizons,
        min_train_days=args.min_train_days,
        step_size=args.step_size,
        max_origins=args.max_origins,
        include_recursive=True,
        include_direct_hstep=args.direct_hstep,
    )

    print("\nWalk-forward Prophet + XGBoost...")
    wf_result = WalkForwardValidator(wf_config).run(df)
    print(f"  Termine: {len(wf_result.predictions)} predictions sur {wf_result.n_origins} origines")

    output_dir = str(ROOT / args.output_dir)
    holdout = compute_holdout_baseline(df)
    wf_paths = save_report(output_dir, wf_result, None, holdout)
    print_console_report(wf_result, None, holdout)

    nhits_result = None
    nhits_csv = None

    if not args.skip_nhits:
        print("\nN-HiTS cross_validation (peut prendre 10-30 min)...")
        nhits_config = NHitsValidatorConfig(
            horizons=args.horizons,
            n_windows=args.nhits_n_windows,
            val_size=args.nhits_val_size,
            step_size=args.nhits_step_size,
            unique_id=market.nhits_unique_id,
        )
        try:
            nhits_result = NHitsValidator(nhits_config).run(df)
            ts = Path(wf_paths["summary_json"]).stem.replace("_summary", "")
            nhits_csv = str(Path(output_dir) / f"{ts}_nhits_predictions.csv")
            nhits_result.predictions.to_csv(nhits_csv, index=False)
            wf_paths["nhits_csv"] = nhits_csv
        except Exception as exc:
            print(f"\nAVERTISSEMENT: N-HiTS a echoue ({exc})")

    ensemble_payload = None
    conformal_payload = None
    if not args.skip_calibration and wf_paths.get("walk_forward_csv"):
        print("\nCalibration ensemble...")
        try:
            calibrator = EnsembleCalibrator()
            cal_result = calibrator.run(
                walk_forward_csv=wf_paths["walk_forward_csv"],
                nhits_csv=nhits_csv,
                xgb_only=nhits_csv is None or not Path(nhits_csv).exists(),
            )
            ensemble_path = str(ROOT / market.ensemble_weights_file)
            Path(ensemble_path).parent.mkdir(parents=True, exist_ok=True)
            calibrator.save(cal_result, ensemble_path)
            ensemble_payload = cal_result.weights_payload
            print(f"  Poids sauvegardes: {ensemble_path}")
        except Exception as exc:
            print(f"  AVERTISSEMENT calibration: {exc}")

        print("\nCalibration intervalles conformes...")
        try:
            import yaml

            pred_cfg = {}
            cfg_path = ROOT / "config" / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as f:
                    pred_cfg = (yaml.safe_load(f) or {}).get("prediction", {})

            coverage = pred_cfg.get("confidence_level", 0.90)
            conformal_path = str(ROOT / market.conformal_intervals_file)
            Path(conformal_path).parent.mkdir(parents=True, exist_ok=True)
            conformal_calibrator = ConformalIntervalCalibrator(
                ConformalIntervalCalibratorConfig(coverage_level=coverage)
            )
            conformal_result = conformal_calibrator.run(
                walk_forward_csv=wf_paths["walk_forward_csv"],
                nhits_csv=nhits_csv,
                ensemble_weights_file=str(ROOT / market.ensemble_weights_file),
            )
            conformal_calibrator.save(conformal_result, conformal_path)
            conformal_payload = conformal_result.intervals_payload
            print(f"  Intervalles sauvegardes: {conformal_path}")
            for h, meta in conformal_payload.get("by_horizon", {}).items():
                print(
                    f"    h{h}: marge +/-${meta['margin_lower']:.0f}  "
                    f"coverage={meta['empirical_coverage']:.1%}"
                )
        except Exception as exc:
            print(f"  AVERTISSEMENT calibration conforme: {exc}")

    summary_path = Path(wf_paths["summary_json"])
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            payload = json.load(f)
        if nhits_result is not None:
            payload["nhits_cross_validation"] = {
                "n_windows": nhits_result.n_windows,
                "summary_by_horizon": _serialize_summary(nhits_result.summary),
            }
        if ensemble_payload is not None:
            payload["ensemble_calibration"] = ensemble_payload
        if conformal_payload is not None:
            payload["conformal_intervals"] = conformal_payload
        if "xgb_pred_recursive" in wf_result.summary:
            payload["walk_forward"]["recursive_vs_frozen"] = _serialize_summary(
                {
                    str(h): {
                        "frozen_mape": wf_result.summary.get("xgb_pred", {}).get(h, {}).get("mape"),
                        "recursive_mape": wf_result.summary.get("xgb_pred_recursive", {}).get(h, {}).get("mape"),
                    }
                    for h in wf_result.config.horizons
                }
            )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    print(f"\nRapports: {output_dir}")
    for key, path in wf_paths.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
