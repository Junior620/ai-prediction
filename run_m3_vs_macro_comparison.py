"""
Comparaison hors-prod : M3 (OHLCV+OI) vs M3 + USD/GBP (Frankfurter).

N'active PAS le macro en production — archive un rapport sous reports/.
Usage:
  python run_m3_vs_macro_comparison.py
  python run_m3_vs_macro_comparison.py --split-date 2025-01-01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

from src.data_collection.ice_london_collector import fetch_usd_gbp_rates
from src.models.hybrid_features import (
    DEFAULT_XGB_PARAMS,
    build_prediction_row,
    build_technical_features,
    future_business_date,
    load_price_data_from_supabase,
    prepare_training_frame,
    resolve_feature_cols,
    add_prophet_features,
)
from src.models.market_registry import get_market_config

HORIZONS = (1, 7, 30)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": 0}
    return {
        "mae": float(np.mean(np.abs(yt - yp))),
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "mape": _mape(yt, yp),
        "n": int(len(yt)),
    }


def _attach_usd_gbp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["usd_gbp"] = np.nan
        return out
    start = out["date"].min().strftime("%Y-%m-%d")
    end = out["date"].max().strftime("%Y-%m-%d")
    rates = fetch_usd_gbp_rates(start, end)
    out["usd_gbp"] = out["date"].dt.strftime("%Y-%m-%d").map(rates)
    out["usd_gbp"] = out["usd_gbp"].ffill().bfill()
    return out


def _evaluate_xgb(
    train: pd.DataFrame,
    test: pd.DataFrame,
    full: pd.DataFrame,
    feature_cols: List[str],
    horizons: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    cols = list(feature_cols)
    feat_full, prophet_model = prepare_training_frame(train)
    feat_all = add_prophet_features(build_technical_features(full), prophet_model)
    for extra in ("usd_gbp",):
        if extra in full.columns and extra not in feat_all.columns:
            feat_all = feat_all.merge(full[["date", extra]], on="date", how="left")
    for c in cols:
        if c not in feat_all.columns:
            feat_all[c] = 0.0
    feat_all[cols] = feat_all[cols].fillna(0.0)

    train_mask = feat_all["date"].isin(train["date"])
    train_feat = feat_all.loc[train_mask].dropna(subset=cols + ["price"])
    if train_feat.empty:
        return {h: _metrics([], []) for h in horizons}

    model = XGBRegressor(**DEFAULT_XGB_PARAMS)
    model.fit(train_feat[cols], train_feat["price"])

    price_lookup = full.set_index(full["date"].dt.normalize())["price"]
    results: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        y_true, y_pred = [], []
        test_feat = feat_all[feat_all["date"].isin(test["date"])]
        for _, row in test_feat.iterrows():
            origin = row["date"]
            target = future_business_date(origin, h)
            actual = price_lookup.get(pd.Timestamp(target).normalize())
            if actual is None or (isinstance(actual, float) and np.isnan(actual)):
                continue
            pred_row = build_prediction_row(
                row, float(row["price"]), target, prophet_model, feature_cols=cols
            )
            for c in cols:
                if c not in pred_row.columns:
                    pred_row[c] = float(row[c]) if c in row.index else 0.0
            pred = float(model.predict(pred_row[cols])[0])
            y_true.append(float(actual))
            y_pred.append(pred)
        results[h] = _metrics(y_true, y_pred)
    return results


def _to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# M3 vs M3+USD/GBP (hors production)",
        "",
        f"- Généré : {report.get('generated_at')}",
        f"- Split : {report.get('split_date')}",
        f"- Marché : {report.get('market')} ({report.get('unit')})",
        "",
        "| Modèle | Horizon | MAPE % | MAE | RMSE | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, by_h in (report.get("models") or {}).items():
        for h, m in by_h.items():
            lines.append(
                f"| {name} | J+{h} | {m.get('mape'):.2f} | {m.get('mae'):.1f} | "
                f"{m.get('rmse'):.1f} | {m.get('n')} |"
            )
    lines.extend(
        [
            "",
            "## Décision",
            "",
            report.get("decision", ""),
            "",
            "> Ce rapport n'active **pas** USD/GBP en production.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 vs M3+USD/GBP comparison (report only)")
    parser.add_argument("--split-date", default="2025-01-01")
    parser.add_argument("--min-date", default="2020-01-01")
    args = parser.parse_args()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("[ERREUR] SUPABASE_URL / SUPABASE_KEY manquants")
        return 1

    market = get_market_config("cocoa")
    sb = create_client(url, key)
    raw = load_price_data_from_supabase(
        sb,
        min_date=args.min_date,
        table_name=market.price_table,
        extra_columns=["open", "high", "low", "volume", "open_interest"],
    )
    if raw.empty or len(raw) < 200:
        print(f"[ERREUR] Historique insuffisant ({len(raw)} lignes)")
        return 1

    raw = _attach_usd_gbp(raw)
    split = pd.Timestamp(args.split_date)
    train = raw[raw["date"] < split].copy()
    test = raw[raw["date"] >= split].copy()
    if train.empty or test.empty:
        print("[ERREUR] Split vide — ajustez --split-date")
        return 1

    m3_cols = resolve_feature_cols(include_ohlcv=True, include_oi=True)
    m3_macro_cols = m3_cols + ["usd_gbp"]

    print("=" * 80)
    print("COMPARAISON M3 vs M3+USD/GBP (rapport only)")
    print("=" * 80)
    print(f"Train < {args.split_date}: {len(train)} | Test: {len(test)}")
    print(f"M3 cols: {len(m3_cols)} | M3+macro: {len(m3_macro_cols)}")

    models = {
        "M3_OHLCV_OI": _evaluate_xgb(train, test, raw, m3_cols, HORIZONS),
        "M3_plus_USDGBP": _evaluate_xgb(train, test, raw, m3_macro_cols, HORIZONS),
    }

    mape_m3 = models["M3_OHLCV_OI"][1].get("mape", float("nan"))
    mape_macro = models["M3_plus_USDGBP"][1].get("mape", float("nan"))
    if np.isnan(mape_macro) or np.isnan(mape_m3):
        decision = "Indécis (métriques manquantes) — rester en M3 pur en prod."
    elif mape_macro + 0.15 < mape_m3:
        decision = (
            f"Lift J+1 détecté ({mape_macro:.2f}% < {mape_m3:.2f}%) — "
            "à revalider walk-forward avant prod."
        )
    else:
        decision = (
            f"Pas de lift clair J+1 (M3={mape_m3:.2f}%, M3+FX={mape_macro:.2f}%) — "
            "prod reste M3 pur."
        )

    report: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "market": "cocoa",
        "unit": market.unit,
        "split_date": args.split_date,
        "min_date": args.min_date,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "feature_cols_m3": m3_cols,
        "feature_cols_m3_macro": m3_macro_cols,
        "models": {
            name: {str(h): metrics for h, metrics in by_h.items()}
            for name, by_h in models.items()
        },
        "decision": decision,
        "production": False,
    }

    out_dir = Path("reports") / "macro_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"m3_vs_usdgbp_{stamp}.json"
    md_path = out_dir / f"m3_vs_usdgbp_{stamp}.md"
    latest_json = out_dir / "m3_vs_usdgbp_latest.json"
    latest_md = out_dir / "m3_vs_usdgbp_latest.md"

    payload = json.dumps(report, indent=2)
    json_path.write_text(payload, encoding="utf-8")
    latest_json.write_text(payload, encoding="utf-8")
    md = _to_markdown(report)
    md_path.write_text(md, encoding="utf-8")
    latest_md.write_text(md, encoding="utf-8")

    print()
    print(decision)
    print(f"[OK] Rapport JSON: {json_path}")
    print(f"[OK] Rapport MD:   {md_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
