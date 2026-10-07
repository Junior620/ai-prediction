"""Report generation for walk-forward validation results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional  # noqa: F401

import pandas as pd

from src.validation.metrics import compute_holdout_baseline
from src.validation.nhits_validator import NHitsValidatorResult
from src.validation.walk_forward_validator import WalkForwardResult


def _evaluated_period(walk_forward: WalkForwardResult) -> Dict[str, Any]:
    preds = walk_forward.predictions
    if preds is None or preds.empty:
        return {
            "origin_start": None,
            "origin_end": None,
            "target_start": None,
            "target_end": None,
            "n_origins": walk_forward.n_origins,
        }
    origins = pd.to_datetime(preds["origin_date"])
    targets = pd.to_datetime(preds["target_date"])
    return {
        "origin_start": str(origins.min().date()),
        "origin_end": str(origins.max().date()),
        "target_start": str(targets.min().date()),
        "target_end": str(targets.max().date()),
        "n_origins": int(origins.nunique()),
    }


def _metric(summary: Dict[str, Any], column: str, horizon: int, key: str) -> Optional[float]:
    block = summary.get(column) or {}
    row = block.get(horizon, block.get(str(horizon), {}))
    value = row.get(key) if isinstance(row, dict) else None
    return None if value is None else float(value)


def _baseline_comparison(summary: Dict[str, Any]) -> Dict[str, Any]:
    """MAPE and MAE of the published forecast against the unchanged close and drift."""
    model_col = "published_pred" if "published_pred" in summary else "xgb_pred"
    model = summary.get(model_col) or {}
    out: Dict[str, Any] = {"model_column": model_col, "by_horizon": {}}
    for horizon in model:
        h = int(horizon)
        model_mape = _metric(summary, model_col, h, "mape")
        naive_mape = _metric(summary, "naive_pred", h, "mape")
        gap = None
        if model_mape is not None and naive_mape is not None:
            gap = model_mape - naive_mape
        out["by_horizon"][str(h)] = {
            "model_mape": model_mape,
            "model_mae": _metric(summary, model_col, h, "mae"),
            "naive_mape": naive_mape,
            "naive_mae": _metric(summary, "naive_pred", h, "mae"),
            "drift_mape": _metric(summary, "drift_pred", h, "mape"),
            "drift_mae": _metric(summary, "drift_pred", h, "mae"),
            "mape_minus_naive": gap,
        }
    return out


def _serialize_summary(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _serialize_summary(v) for k, v in obj.items()}
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def build_summary_payload(
    walk_forward: WalkForwardResult,
    nhits: Optional[NHitsValidatorResult] = None,
    holdout_baseline: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build JSON-serializable summary dict."""
    payload: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "validation_type": "walk_forward_multi_horizon",
        "notes": [
            "Sentiment FinBERT excluded from backtest (no historical news replay).",
            "published_pred is the served direct forecast after the guard.",
            "The hybrid column at one session uses the origin row to forecast the next close.",
            "Prophet fit only on training slice at each origin (no leakage).",
        ],
        "walk_forward": {
            "n_origins": walk_forward.n_origins,
            "horizons": walk_forward.config.horizons,
            "min_train_days": walk_forward.config.min_train_days,
            "step_size": walk_forward.config.step_size,
            "origin_window": walk_forward.config.origin_window,
            "evaluated_period": _evaluated_period(walk_forward),
            "baseline_comparison": _baseline_comparison(walk_forward.summary),
            "summary_by_component": _serialize_summary(walk_forward.summary),
        },
    }

    if nhits is not None:
        payload["nhits_cross_validation"] = {
            "n_windows": nhits.n_windows,
            "summary_by_horizon": _serialize_summary(nhits.summary),
        }

    if holdout_baseline is not None:
        payload["legacy_holdout_baseline"] = holdout_baseline

    return payload


def save_report(
    output_dir: str,
    walk_forward: WalkForwardResult,
    nhits: Optional[NHitsValidatorResult] = None,
    holdout_baseline: Optional[Dict[str, float]] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, str]:
    """Write JSON summary and CSV prediction files."""
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_path = out_path / f"{ts}_summary.json"
    wf_csv_path = out_path / f"{ts}_walk_forward_predictions.csv"

    payload = build_summary_payload(walk_forward, nhits, holdout_baseline)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    walk_forward.predictions.to_csv(wf_csv_path, index=False)

    paths = {
        "summary_json": str(summary_path),
        "walk_forward_csv": str(wf_csv_path),
    }

    if nhits is not None and not nhits.predictions.empty:
        nhits_csv_path = out_path / f"{ts}_nhits_predictions.csv"
        nhits.predictions.to_csv(nhits_csv_path, index=False)
        paths["nhits_csv"] = str(nhits_csv_path)

    return paths


def print_console_report(
    walk_forward: WalkForwardResult,
    nhits: Optional[NHitsValidatorResult] = None,
    holdout_baseline: Optional[Dict[str, float]] = None,
) -> None:
    """Print human-readable validation summary to stdout."""
    print("=" * 80)
    print("VALIDATION WALK-FORWARD MULTI-HORIZON (HONNETE)")
    print("=" * 80)

    period = _evaluated_period(walk_forward)
    print(f"\nOrigines evaluees: {walk_forward.n_origins}")
    print(
        f"Fenetre: origines {period['origin_start']} -> {period['origin_end']}, "
        f"cibles {period['target_start']} -> {period['target_end']} "
        f"({walk_forward.config.origin_window})"
    )
    print(f"Horizons: {walk_forward.config.horizons}")
    print(f"Fenetre min. entrainement: {walk_forward.config.min_train_days} jours")
    print(f"Pas entre origines: {walk_forward.config.step_size} jours")

    print("\n=== Walk-Forward (Prophet + XGBoost) ===")
    for component, label in [
        ("xgb_pred", "XGBoost (frozen)"),
        ("xgb_pred_recursive", "XGBoost (recursive)"),
        ("xgb_pred_direct", "XGBoost (direct h-step)"),
        ("published_pred", "Prevision publiee"),
        ("naive_pred", "Cours inchange"),
        ("drift_pred", "Derive"),
        ("prophet_pred", "Prophet"),
    ]:
        if component not in walk_forward.summary:
            continue
        print(f"\n  [{label}]")
        for horizon, metrics in sorted(walk_forward.summary[component].items()):
            print(
                f"    Horizon {horizon:2d}d: MAPE={metrics['mape']:.2f}%  "
                f"RMSE=${metrics['rmse']:.2f}  MAE=${metrics['mae']:.2f}  "
                f"Dir.Acc={metrics['directional_accuracy']:.2%}  "
                f"(n={metrics['n_predictions']})"
            )

    if nhits is not None:
        print(f"\n=== N-HiTS cross_validation ({nhits.n_windows} fenetres) ===")
        for horizon, metrics in sorted(nhits.summary.items()):
            print(
                f"    Horizon {horizon:2d}d: MAPE={metrics['mape']:.2f}%  "
                f"RMSE=${metrics['rmse']:.2f}  MAE=${metrics['mae']:.2f}  "
                f"Dir.Acc={metrics['directional_accuracy']:.2%}  "
                f"(n={metrics['n_predictions']})"
            )

    if holdout_baseline is not None:
        print("\n=== Comparaison baseline holdout (1-step, legacy) ===")
        print(
            f"    MAPE holdout 80/20 (1-step XGBoost): "
            f"{holdout_baseline['mape_1step_holdout']:.2f}% "
            f"(n={holdout_baseline['n_val']})"
        )
        xgb_h1 = walk_forward.summary.get("xgb_pred", {}).get(1, {})
        if xgb_h1.get("n_predictions", 0) > 0:
            print(
                f"    MAPE walk-forward honnete (h=1): {xgb_h1['mape']:.2f}% "
                f"(n={xgb_h1['n_predictions']})"
            )

    print("\n" + "=" * 80)


def run_full_report(
    df: pd.DataFrame,
    walk_forward: WalkForwardResult,
    nhits: Optional[NHitsValidatorResult],
    output_dir: str,
) -> Dict[str, str]:
    """Save files and print console report."""
    holdout = compute_holdout_baseline(df)
    paths = save_report(output_dir, walk_forward, nhits, holdout)
    print_console_report(walk_forward, nhits, holdout)
    print(f"\nRapports sauvegardes dans: {output_dir}")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    return paths
