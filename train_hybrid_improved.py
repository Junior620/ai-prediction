"""
MODÈLE HYBRIDE AMÉLIORÉ
XGBoost comme modèle principal, Prophet comme feature secondaire

Usage: python train_hybrid_improved.py [--market cocoa|coffee_robusta]
Cacao production = M3 (OHLCV + Open Interest).
"""

import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from supabase import create_client

from src.models.hybrid_features import (
    FEATURE_COLS,
    add_prophet_features,
    build_prediction_row,
    build_technical_features,
    future_business_date,
    load_price_data_from_supabase,
    resolve_feature_cols,
)
from src.models.hybrid_trainer import HybridModelTrainer, next_session_frame
from src.models.market_registry import get_market_config
from src.validation.report_loader import extract_walk_forward_reference, load_latest_summary

load_dotenv()

parser = argparse.ArgumentParser(description="Entraînement du modèle hybride")
parser.add_argument("--market", default="cocoa", help="Marché (cocoa, coffee_robusta)")
args = parser.parse_args()

market = get_market_config(args.market)
models_dir = Path(market.models_dir)
models_dir.mkdir(parents=True, exist_ok=True)

# M3 pour cacao entreprise ; baseline pour les autres marches
if args.market == "cocoa":
    feature_set = "m3"
    feature_cols = resolve_feature_cols(include_ohlcv=True, include_oi=True)
else:
    feature_set = "baseline"
    feature_cols = list(FEATURE_COLS)

print("=" * 80)
print("MODELE HYBRIDE AMELIORE")
print(f"   Marche: {market.display_name} ({args.market})")
print(f"   Feature set: {feature_set} ({len(feature_cols)} cols)")
print("   XGBoost = Patron | Prophet = Conseiller")
print("=" * 80)

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

print("\n[1/6] Recuperation des donnees...")
df = load_price_data_from_supabase(supabase, table_name=market.price_table)

print(f"[OK] {len(df)} points")
if len(df) < 100:
    print(f"[ERREUR] Pas assez de prix dans {market.price_table} ({len(df)} lignes).")
    print("         Verifier Supabase / reessayer la collecte, puis relancer.")
    raise SystemExit(1)
print(f"   Prix min: {df['price'].min():.2f}")
print(f"   Prix max: {df['price'].max():.2f}")
print(f"   Prix moyen: {df['price'].mean():.2f}")

# Imputation microstructure avant split
for col in ("open", "high", "low", "volume", "open_interest"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill()

print("\n[2/6] Preparation features (validation honnete sans fuite Prophet)...")

df_technical = build_technical_features(df)
for col in feature_cols:
    if col not in df_technical.columns and not col.startswith("prophet_"):
        df_technical[col] = 0.0
micro = [c for c in feature_cols if c not in FEATURE_COLS]
if micro:
    present = [c for c in micro if c in df_technical.columns]
    if present:
        df_technical[present] = df_technical[present].ffill().fillna(0.0)

base_feat = [c for c in FEATURE_COLS if c in df_technical.columns and not c.startswith("prophet_")]
valid_mask = df_technical[base_feat].notna().all(axis=1)
valid_indices = df_technical.index[valid_mask]
if len(valid_indices) < 50:
    raise SystemExit("[ERREUR] Pas assez de lignes valides apres features")
split_idx = int(len(valid_indices) * 0.8)
split_pos = valid_indices[split_idx] if split_idx < len(valid_indices) else valid_indices[-1]

train_raw = df.iloc[: split_pos + 1].copy()
val_raw = df.iloc[split_pos + 1 :].copy()

trainer = HybridModelTrainer(feature_cols=feature_cols)
prophet_val, xgb_val, train_features = trainer.fit(train_raw)

val_technical = build_technical_features(df)
for col in feature_cols:
    if col not in val_technical.columns and not col.startswith("prophet_"):
        val_technical[col] = 0.0
if micro:
    present = [c for c in micro if c in val_technical.columns]
    if present:
        val_technical[present] = val_technical[present].ffill().fillna(0.0)
val_with_prophet = add_prophet_features(val_technical, prophet_val)
for col in feature_cols:
    if col not in val_with_prophet.columns:
        val_with_prophet[col] = 0.0
val_df = val_with_prophet.iloc[split_pos + 1 :].dropna(subset=feature_cols)

train_df = next_session_frame(train_features.dropna(subset=feature_cols + ["price"]))
val_aligned = next_session_frame(val_df)
X_train = train_df[feature_cols]
y_train = train_df["target_price"]
X_val = val_aligned[feature_cols]
y_val = val_aligned["target_price"]

print(f"[OK] {len(train_df)} points train | {len(val_df)} points val")

print("\n[3/6] Entrainement final sur toutes les donnees...")
prophet_model, xgb_model, df_clean = trainer.fit(df)
df_clean = df_clean.dropna(subset=feature_cols + ["price"])
print(f"[OK] Prophet + XGBoost entraines sur {len(df_clean)} points")

print("\n[4/6] Metriques de validation (Prophet fit sur train uniquement)...")
val_pred = xgb_val.predict(X_val)
train_pred = xgb_val.predict(X_train)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_mae = mean_absolute_error(y_train, train_pred)
train_mape = np.mean(np.abs((y_train.values - train_pred) / y_train.values)) * 100

val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
val_mae = mean_absolute_error(y_val, val_pred)
val_mape = np.mean(np.abs((y_val.values - val_pred) / y_val.values)) * 100

print(f"\n   Performance:")
print(f"      Train RMSE: {train_rmse:.2f} | MAE: {train_mae:.2f} | MAPE: {train_mape:.2f}%")
print(f"      Val RMSE: {val_rmse:.2f} | MAE: {val_mae:.2f} | MAPE holdout seance suivante: {val_mape:.2f}%")

feature_importance = dict(zip(feature_cols, xgb_model.feature_importances_))
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

print(f"\n   Top 10 Features:")
for i, (feat, imp) in enumerate(sorted_features[:10], 1):
    print(f"      {i}. {feat}: {imp:.2%}")

print("\n[5/6] Test des predictions...")

current_price = df_clean["price"].iloc[-1]
current_date = df_clean["date"].iloc[-1]
last_row = df_clean.iloc[-1]

print(f"\nPrix actuel ({current_date.date()}): {current_price:,.2f}")
print("\n" + "=" * 80)
print("PREDICTIONS HYBRIDES (feature_set=%s)" % feature_set)
print("=" * 80)

for days in [1, 7, 14, 30]:
    future_date = future_business_date(current_date, days)
    features_future = build_prediction_row(
        last_row, current_price, future_date, prophet_model, feature_cols=feature_cols
    )
    pred_price = xgb_model.predict(features_future[feature_cols])[0]
    prophet_yhat_future = features_future["prophet_yhat"].iloc[0]
    change = pred_price - current_price
    change_pct = (change / current_price) * 100

    print(f"\n{days} jour(s) - {future_date.date()}:")
    print(f"   Prix predit: {pred_price:,.2f}")
    print(f"   Changement: {change:+,.2f} ({change_pct:+.2f}%)")
    print(f"   Prophet suggere: {prophet_yhat_future:,.2f}")

print("\n" + "=" * 80)
print("SAUVEGARDE DES MODELES")
print("=" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
models_dir = Path("models") / "candidates" / args.market / timestamp
models_dir.mkdir(parents=True, exist_ok=True)
print(f"[OK] Candidat, pas encore actif: {models_dir}")

prophet_path = str(models_dir / f"prophet_improved_{timestamp}.pkl")
with open(prophet_path, "wb") as f:
    pickle.dump(prophet_model, f)
print(f"[OK] Prophet: {prophet_path}")

xgb_path = str(models_dir / f"xgboost_improved_{timestamp}.pkl")
with open(xgb_path, "wb") as f:
    pickle.dump(xgb_model, f)
print(f"[OK] XGBoost: {xgb_path}")

model_info = {
    "timestamp": timestamp,
    "model_type": "hybrid_improved",
    "target": "next_session",
    "feature_set": feature_set,
    "feature_cols": feature_cols,
    "market": args.market,
    "description": f"XGBoost {feature_set} avec Prophet features",
    "data_period": "2020-2026",
    "training_points": len(X_train),
    "validation_points": len(X_val),
    "train_mape": float(train_mape),
    "val_mape_holdout_1step": float(val_mape),
    "val_mape": float(val_mape),
    "val_rmse": float(val_rmse),
    "val_mae": float(val_mae),
    "dampener_note": "Production dampener J+1/7/14/30 = 6/14/18/22 pct (config.yaml)",
    "feature_importance": {k: float(v) for k, v in sorted_features[:20]},
    "prophet_weight": float(feature_importance.get("prophet_yhat", 0)),
    "price_lag_1_weight": float(feature_importance.get("price_lag_1", 0)),
}
wf_reports_dir = (
    Path("reports/walk_forward")
    if args.market == "cocoa"
    else Path("reports/walk_forward") / args.market
)
wf_ref = extract_walk_forward_reference(load_latest_summary(str(wf_reports_dir)))
if wf_ref:
    model_info["walk_forward_reference"] = wf_ref

info_path = str(models_dir / f"model_info_improved_{timestamp}.json")
with open(info_path, "w") as f:
    json.dump(model_info, f, indent=2)
print(f"[OK] Infos: {info_path}")

# Direct horizons alignes sur le meme feature set
try:
    from src.models.direct_horizon_trainer import DirectHorizonTrainer

    print("\n[6/6] Direct horizons h=1,7,14,30...")
    dht = DirectHorizonTrainer(horizons=[1, 7, 14, 30], feature_cols=feature_cols)
    direct_models, direct_meta = dht.fit(df, prophet_model=prophet_model)
    direct_meta["feature_set"] = feature_set
    direct_meta["feature_cols"] = feature_cols
    paths = dht.save(direct_models, direct_meta, models_dir=str(models_dir), timestamp=timestamp)
    print(f"[OK] Direct horizons: {paths}")
except Exception as exc:
    print(f"[WARN] Direct horizons: {exc}")

print("\n" + "=" * 80)
print("[OK] ENTRAINEMENT TERMINE")
print("=" * 80)
print(f"""
RESUME:
   Feature set: {feature_set} ({len(feature_cols)} cols)
   MAPE holdout: {val_mape:.2f}%
   RMSE Validation: {val_rmse:.2f}
   MAE Validation: {val_mae:.2f}
""")
print("=" * 80)
