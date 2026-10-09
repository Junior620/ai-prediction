#!/usr/bin/env python
"""
Run honest walk-forward multi-horizon validation for cocoa price models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# On Windows, NeuralForecast must be imported before pandas, scikit-learn and XGBoost.
if "--replay-nhits" in sys.argv:
    import neuralforecast  # noqa: F401
    from neuralforecast.models import NHITS  # noqa: F401

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from supabase import create_client

load_dotenv(ROOT / "config" / ".env")
load_dotenv()

from src.models.hybrid_features import load_price_data_from_supabase
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


def _feature_cols_for_market(market_id: str):
    from src.models.hybrid_features import resolve_feature_cols

    if market_id == "cocoa":
        return resolve_feature_cols(include_ohlcv=True, include_oi=True)
    return resolve_feature_cols()


def _guard_config(conformal_file: str):
    import yaml

    from src.models.conformal_intervals import load_conformal_margins

    caps = {"1": 6.0, "7": 14.0, "14": 18.0, "30": 22.0}
    cfg_path = ROOT / "config" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            pred = (yaml.safe_load(f) or {}).get("prediction", {})
        caps = pred.get("max_abs_change_pct", caps)
    margins = load_conformal_margins(str(ROOT / conformal_file))
    return caps, margins


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
    parser.add_argument(
        "--origin-window",
        choices=("recent", "historical"),
        default="recent",
        help="recent = dernieres origines realisees; historical = debut de l'historique",
    )
    parser.add_argument("--output-dir", type=str, default=yaml_cfg.get("output_dir", "reports/walk_forward"))
    parser.add_argument("--skip-nhits", action="store_true", default=not nhits_cfg.get("enabled", True))
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help="Calcule les bandes et les décisions sans activer un horizon",
    )
    parser.add_argument(
        "--test-origins",
        type=int,
        default=0,
        help="Dernières origines tenues comme test, après une calibration antérieure",
    )
    parser.add_argument("--direct-hstep", action="store_true", help="Include direct h-step in walk-forward (slow)")
    parser.add_argument(
        "--replay-nhits",
        action="store_true",
        help="Rejouer un N-HiTS neuf a chaque origine, aux reglages servis",
    )
    parser.add_argument("--nhits-n-windows", type=int, default=nhits_cfg.get("n_windows", 12))
    parser.add_argument("--nhits-val-size", type=int, default=nhits_cfg.get("val_size", 30))
    parser.add_argument("--nhits-step-size", type=int, default=nhits_cfg.get("step_size", 5))
    return parser.parse_args()


def _calibrate_if_better(
    walk_forward_csv: str,
    nhits_csv,
    market,
    *,
    step_size: int,
    price_bounds,
    measure_only: bool = False,
    test_origins: int = 0,
    replayed_nhits: bool = False,
    candidate: dict | None = None,
):
    """Calibrate on the early origins. Promote a horizon only when its later slice passes.

    Ensemble weights are not refit here. ``nhits_csv`` is ignored on purpose.
    """
    del nhits_csv
    import yaml

    from src.validation.served_backtest import (
        apply_frozen_bands,
        held_out_coverage,
        residual_margins,
        served_prediction_column,
        split_chronological,
        split_last_origins,
    )

    frame = pd.read_csv(walk_forward_csv)
    pred_col = served_prediction_column(frame)
    if int(test_origins) > 0:
        cal_df, test_df = split_last_origins(frame, int(test_origins))
    else:
        cal_df, test_df = split_chronological(frame)
    coverage = 0.90
    cfg_path = ROOT / "config" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            coverage = (yaml.safe_load(f) or {}).get("prediction", {}).get("confidence_level", 0.90)

    from src.validation.acceptance import evaluate_horizon, load_acceptance

    rules = load_acceptance(str(ROOT / "config" / "acceptance.json"))
    margins = residual_margins(cal_df, pred_col, coverage)
    naive_margins = residual_margins(cal_df, "naive_pred", coverage)
    for horizon, row in margins.items():
        row["provenance"] = "calibration_slice_of_this_candidate"
        row["coverage_is_not_a_held_out_proof"] = True
    bounds = tuple(price_bounds) if price_bounds else (1000.0, 15000.0)
    test_df = apply_frozen_bands(test_df, pred_col, margins, "published_lower", "published_upper", bounds)
    test_df = apply_frozen_bands(test_df, "naive_pred", naive_margins, "naive_lower", "naive_upper", bounds)
    test_df.to_csv(walk_forward_csv.replace(".csv", "") + "_test_bands.csv", index=False)
    test_cov = held_out_coverage(test_df, pred_col, margins)
    decisions = {}
    for horizon in rules.get("horizons", []):
        decision = evaluate_horizon(
            test_df,
            int(horizon),
            rules,
            step=int(step_size),
            frozen_margin=margins.get(str(horizon)),
            frozen_naive_margin=naive_margins.get(str(horizon)),
        )
        decisions[str(horizon)] = decision
    from src.models.ensemble_weights import load_ensemble_weights

    weights_payload = load_ensemble_weights(market.ensemble_weights_file)
    nhits_required = False
    for row in (weights_payload or {}).values():
        if isinstance(row, dict) and float(row.get("nhits", 0.0) or 0.0) > 0.0:
            nhits_required = True
            break
    incomplete = bool(nhits_required and not replayed_nhits)
    if incomplete:
        for row in decisions.values():
            row["validated"] = False
            row["reason"] = (
                "Mesure incomplète : un poids N-HiTS est positif et le rejeu n'a pas eu lieu. "
                + str(row.get("reason") or "")
            ).strip()
    if measure_only:
        for row in decisions.values():
            row["validated"] = False
            row["measurement"] = "verification"
            row["reason"] = (
                "Vérification seulement : cette exécution n'active aucun horizon. "
                + str(row.get("reason") or "")
            ).strip()
    accepted = [] if measure_only or incomplete else [
        horizon for horizon, row in decisions.items() if row.get("validated")
    ]
    promoted = bool(accepted)
    print(
        f"\nCalibration sur {cal_df['origin_date'].nunique() if not cal_df.empty else 0} origines, "
        f"test reserve sur {test_df['origin_date'].nunique() if not test_df.empty else 0}."
    )
    print(f"  Colonne servie: {pred_col}")
    print("  La couverture de calibration n'est pas une preuve hors echantillon.")
    for horizon, row in decisions.items():
        gain = row.get("relative_gain")
        gain_txt = "n/a" if gain is None else f"{gain:.1%}"
        print(
            f"    h{horizon}: valide={row['validated']} gain={gain_txt} "
            f"repli={row.get('fallback_rate')} n={row.get('n')} "
            f"n_eff={row.get('n_effective')}"
        )

    source_report = Path(walk_forward_csv).stem.replace("_walk_forward_predictions", "")
    conformal_payload = {
        "source_report": source_report,
        "calibrated_at": datetime.now().isoformat(),
        "coverage_level": coverage,
        "asymmetric": False,
        "coverage_is_held_out": False,
        "applies_to_candidate": source_report,
        "provenance": "calibration_margins_frozen_before_the_test",
        "by_horizon": margins,
        "naive_by_horizon": naive_margins,
        "test_coverage": test_cov,
        "acceptance": decisions,
    }
    from src.models.release_manifest import promote_accepted_horizons

    wrote = promote_accepted_horizons(
        market.market_id,
        accepted,
        source_report,
        conformal_payload if accepted else None,
        ROOT,
        candidate=candidate,
    )
    promotion = {
        "promoted": promoted,
        "accepted_horizons": accepted,
        "candidate_version": (candidate or {}).get("version"),
        "prediction_column": pred_col,
        "nhits_replayed": bool(replayed_nhits),
        "nhits_required": bool(nhits_required),
        "measurement": "verification" if measure_only else ("held_out" if promoted else "procedure"),
        "weights_refit": False,
        "decisions": decisions,
        "reason": (
            "Les horizons acceptés sont inscrits dans le manifeste. Les autres restent non validés."
            if wrote
            else "Aucun horizon n'est accepté. Le manifeste actif et ses marges restent."
        ),
    }
    print(f"  Promotion: {promotion['reason']}")
    return None, conformal_payload, promotion


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

    feature_cols = _feature_cols_for_market(args.market)
    caps, margins = _guard_config(market.conformal_intervals_file)
    import yaml

    pred_cfg = {}
    cfg_path = ROOT / "config" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            pred_cfg = (yaml.safe_load(f) or {}).get("prediction", {}) or {}
    wf_config = WalkForwardConfig(
        horizons=args.horizons,
        min_train_days=args.min_train_days,
        step_size=args.step_size,
        max_origins=args.max_origins,
        origin_window=args.origin_window,
        include_recursive=not args.direct_hstep,
        include_direct_hstep=args.direct_hstep,
        feature_cols=feature_cols,
        max_abs_change_pct=caps,
        conformal_margins=margins,
        ensemble_weights_file=market.ensemble_weights_file,
        replay_nhits=bool(args.replay_nhits),
        defer_bands=not args.skip_calibration,
        price_bounds=market.price_bounds,
        recent_range_days=int(pred_cfg.get("recent_range_days", 252)),
        recent_range_padding_pct=float(pred_cfg.get("recent_range_padding_pct", 15.0)),
        garch_enabled=bool(market.garch_enabled),
        confidence_level=float(pred_cfg.get("confidence_level", 0.90)),
        nhits_unique_id=market.nhits_unique_id,
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
    promotion_payload = None
    if not args.skip_calibration and wf_paths.get("walk_forward_csv"):
        try:
            from src.models.release_manifest import latest_candidate

            named_candidate = latest_candidate(market.market_id, ROOT)
            if named_candidate is not None:
                named_candidate["source_report"] = Path(wf_paths["walk_forward_csv"]).stem.replace(
                    "_walk_forward_predictions", ""
                )
            ensemble_payload, conformal_payload, promotion_payload = _calibrate_if_better(
                wf_paths["walk_forward_csv"],
                nhits_csv,
                market,
                step_size=args.step_size,
                price_bounds=market.price_bounds,
                measure_only=bool(args.measure_only),
                test_origins=int(args.test_origins),
                replayed_nhits=bool(args.replay_nhits),
                candidate=named_candidate,
            )
        except Exception as exc:
            print(f"  AVERTISSEMENT calibration: {exc}")
    elif args.skip_calibration:
        from src.models.ensemble_weights import load_ensemble_weights

        weights_payload = load_ensemble_weights(market.ensemble_weights_file)
        nhits_required = any(
            isinstance(row, dict) and float(row.get("nhits", 0.0) or 0.0) > 0.0
            for row in (weights_payload or {}).values()
        )
        if nhits_required and not args.replay_nhits:
            promotion_payload = {
                "promoted": False,
                "accepted_horizons": [],
                "nhits_replayed": False,
                "nhits_required": True,
                "measurement": "incomplete",
                "reason": (
                    "Mesure incomplète : le walk-forward quotidien n'a pas rejoué N-HiTS. "
                    "Aucun horizon n'est activé."
                ),
            }
            print("  " + promotion_payload["reason"])

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
        if promotion_payload is not None:
            payload["promotion"] = promotion_payload
        from src.models.release_manifest import data_fingerprint, procedure_fingerprint

        tail = df.tail(40)
        token_rows = []
        for _, row in tail.iterrows():
            interest = row["open_interest"] if "open_interest" in row.index else None
            token_rows.append(
                {
                    "date": str(pd.Timestamp(row["date"]).date()),
                    "price": None if pd.isna(row["price"]) else float(row["price"]),
                    "open_interest": None if interest is None or pd.isna(interest) else float(interest),
                }
            )
        payload["procedure_fingerprint"] = procedure_fingerprint(
            root=ROOT,
            parameters={
                "market": args.market,
                "horizons": list(args.horizons),
                "step_size": int(args.step_size),
                "max_origins": args.max_origins,
                "direct_hstep": bool(args.direct_hstep),
                "replay_nhits": bool(args.replay_nhits),
                "measure_only": bool(args.measure_only),
                "test_origins": int(args.test_origins),
                "skip_calibration": bool(args.skip_calibration),
            },
            frame_token=data_fingerprint(token_rows),
        )
        payload["data_window"] = {
            "min_date": str(pd.Timestamp(df["date"].min()).date()),
            "max_date": str(pd.Timestamp(df["date"].max()).date()),
            "rows": int(len(df)),
            "cut": "2020-01-01",
        }
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
