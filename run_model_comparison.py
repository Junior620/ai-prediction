"""
Etude comparative M1–M4 pour le memoire (hors production).

M1 : Close seul → Prophet
M2 : OHLCV + features techniques → XGBoost
M3 : M2 + Open Interest → XGBoost
M4 : M3 + spreads d'echeances (+ sentiment si dispo) → XGBoost

Split temporel fixe : train jusqu'a --split-date, test apres (jamais vu).
Metriques : MAE / RMSE / MAPE par modele et horizon (1/7/30).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

from src.models.hybrid_features import (
    DEFAULT_XGB_PARAMS,
    FEATURE_COLS,
    build_prediction_row,
    build_technical_features,
    fit_prophet,
    future_business_date,
    load_price_data_from_supabase,
    load_term_structure_from_supabase,
    prepare_training_frame,
    resolve_feature_cols,
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


def _load_sentiment(supabase, min_date: str) -> pd.DataFrame:
    """Charge un score sentiment quotidien si la table existe."""
    for table, score_col in (
        ("news_sentiment_daily", "sentiment_score"),
        ("market_sentiment", "score"),
        ("news_articles", "sentiment"),
    ):
        try:
            resp = (
                supabase.table(table)
                .select(f"date, {score_col}")
                .gte("date", min_date)
                .order("date")
                .limit(5000)
                .execute()
            )
            if not resp.data:
                continue
            df = pd.DataFrame(resp.data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.rename(columns={score_col: "sentiment"})
            df = df.groupby("date", as_index=False)["sentiment"].mean()
            return df
        except Exception:
            continue
    return pd.DataFrame(columns=["date", "sentiment"])


def _merge_extras(df: pd.DataFrame, term_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not term_df.empty:
        out = out.merge(term_df, on="date", how="left")
        if "close_0" not in out.columns:
            out["close_0"] = out["price"]
    if not sentiment_df.empty:
        out = out.merge(sentiment_df, on="date", how="left")
        out["sentiment"] = out["sentiment"].fillna(0.0)
    else:
        out["sentiment"] = 0.0
    return out


def _evaluate_prophet(
    train: pd.DataFrame,
    test: pd.DataFrame,
    full: pd.DataFrame,
    horizons: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    model = fit_prophet(train)
    price_lookup = full.set_index(full["date"].dt.normalize())["price"]
    results: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        y_true, y_pred = [], []
        for _, row in test.iterrows():
            origin = row["date"]
            target = future_business_date(origin, h)
            actual = price_lookup.get(pd.Timestamp(target).normalize())
            if actual is None or (isinstance(actual, float) and np.isnan(actual)):
                continue
            fc = model.predict(pd.DataFrame({"ds": [target]}))
            y_true.append(float(actual))
            y_pred.append(float(fc["yhat"].iloc[0]))
        results[h] = _metrics(y_true, y_pred)
    return results


def _evaluate_xgb(
    train: pd.DataFrame,
    test: pd.DataFrame,
    full: pd.DataFrame,
    feature_cols: List[str],
    horizons: Sequence[int],
    include_sentiment: bool = False,
) -> Dict[int, Dict[str, float]]:
    cols = list(feature_cols)
    if include_sentiment and "sentiment" not in cols:
        cols = cols + ["sentiment"]

    feat_full, prophet_model = prepare_training_frame(train)
    # Rebuild features on full history for last_row context, but train only on train dates
    feat_all = build_technical_features(full)
    feat_all = feat_all.merge(
        feat_full[["date", "prophet_trend", "prophet_yearly", "prophet_yhat"]],
        on="date",
        how="left",
        suffixes=("", "_tr"),
    )
    # Recompute prophet on full using train-fitted model
    from src.models.hybrid_features import add_prophet_features

    feat_all = add_prophet_features(build_technical_features(full), prophet_model)

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
            pred = float(model.predict(pred_row[cols])[0])
            y_true.append(float(actual))
            y_pred.append(pred)
        results[h] = _metrics(y_true, y_pred)
    return results


def _to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Comparaison modeles cacao Londres (Databento)",
        "",
        f"- Split date : `{report['split_date']}`",
        f"- Train : {report['n_train']} jours | Test : {report['n_test']} jours",
        f"- Genere : {report['generated_at']}",
        "",
        "| Modele | Horizon | MAE | RMSE | MAPE (%) | N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name, by_h in report["models"].items():
        for h, m in by_h.items():
            lines.append(
                f"| {model_name} | J+{h} | {m['mae']:.2f} | {m['rmse']:.2f} | "
                f"{m['mape']:.2f} | {m['n']} |"
            )
    lines.append("")
    lines.append(
        "> M1=Prophet(Close) · M2=XGB(OHLCV) · M3=XGB(OHLCV+OI) · M4=XGB(complet+sentiment)"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Comparaison M1-M4 cacao Londres")
    parser.add_argument("--split-date", default="2025-01-01")
    parser.add_argument("--min-date", default="2019-01-01")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    market = get_market_config("cocoa")
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    print("=" * 80)
    print("COMPARAISON M1–M4 — LONDON COCOA")
    print("=" * 80)

    df = load_price_data_from_supabase(
        supabase, min_date=args.min_date, table_name=market.price_table
    )
    term_df = load_term_structure_from_supabase(supabase, min_date=args.min_date)
    sentiment_df = _load_sentiment(supabase, args.min_date)
    df = _merge_extras(df, term_df, sentiment_df)

    split = pd.Timestamp(args.split_date)
    train = df[df["date"] < split].copy()
    test = df[df["date"] >= split].copy()
    print(f"Train: {len(train)} | Test: {len(test)} | split={args.split_date}")

    if len(train) < 100 or len(test) < 10:
        print("[ERREUR] Pas assez de donnees pour le split")
        return 1

    models: Dict[str, Dict[str, Any]] = {}

    print("\n[M1] Prophet (Close)...")
    m1 = _evaluate_prophet(train, test, df, HORIZONS)
    models["M1_Prophet_Close"] = {str(k): v for k, v in m1.items()}
    for h, m in m1.items():
        print(f"  J+{h}: MAPE={m['mape']:.2f}% RMSE={m['rmse']:.2f} n={m['n']}")

    print("\n[M2] XGBoost (OHLCV)...")
    cols_m2 = resolve_feature_cols(include_ohlcv=True)
    m2 = _evaluate_xgb(train, test, df, cols_m2, HORIZONS)
    models["M2_XGB_OHLCV"] = {str(k): v for k, v in m2.items()}
    for h, m in m2.items():
        print(f"  J+{h}: MAPE={m['mape']:.2f}% RMSE={m['rmse']:.2f} n={m['n']}")

    print("\n[M3] XGBoost (OHLCV + OI)...")
    cols_m3 = resolve_feature_cols(include_ohlcv=True, include_oi=True)
    m3 = _evaluate_xgb(train, test, df, cols_m3, HORIZONS)
    models["M3_XGB_OHLCV_OI"] = {str(k): v for k, v in m3.items()}
    for h, m in m3.items():
        print(f"  J+{h}: MAPE={m['mape']:.2f}% RMSE={m['rmse']:.2f} n={m['n']}")

    print("\n[M4] XGBoost (complet + sentiment)...")
    cols_m4 = resolve_feature_cols(include_ohlcv=True, include_oi=True, include_term=True)
    m4 = _evaluate_xgb(train, test, df, cols_m4, HORIZONS, include_sentiment=True)
    models["M4_XGB_Full"] = {str(k): v for k, v in m4.items()}
    for h, m in m4.items():
        print(f"  J+{h}: MAPE={m['mape']:.2f}% RMSE={m['rmse']:.2f} n={m['n']}")

    report = {
        "generated_at": datetime.now().isoformat(),
        "split_date": args.split_date,
        "min_date": args.min_date,
        "n_train": len(train),
        "n_test": len(test),
        "price_table": market.price_table,
        "models": models,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"model_comparison_{stamp}.json"
    md_path = out_dir / f"model_comparison_{stamp}.md"
    csv_path = out_dir / f"model_comparison_{stamp}.csv"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    rows = []
    for model_name, by_h in models.items():
        for h, m in by_h.items():
            rows.append({"model": model_name, "horizon": int(h), **m})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nRapports:")
    print(f"  {json_path}")
    print(f"  {md_path}")
    print(f"  {csv_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
