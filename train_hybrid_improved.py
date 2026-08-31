"""
MODÈLE HYBRIDE AMÉLIORÉ
XGBoost comme modèle principal, Prophet comme feature secondaire

Usage: python train_hybrid_improved.py [--market cocoa|coffee_robusta]
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
)
from src.models.hybrid_trainer import HybridModelTrainer
from src.models.market_registry import get_market_config
from src.validation.report_loader import extract_walk_forward_reference, load_latest_summary

load_dotenv()

parser = argparse.ArgumentParser(description="Entraînement du modèle hybride")
parser.add_argument("--market", default="cocoa", help="Marché (cocoa, coffee_robusta)")
args = parser.parse_args()

market = get_market_config(args.market)
models_dir = Path(market.models_dir)
models_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🚀 MODÈLE HYBRIDE AMÉLIORÉ")
print(f"   Marché: {market.display_name} ({args.market})")
print("   XGBoost = Patron | Prophet = Conseiller")
print("=" * 80)

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

print("\n[1/6] Récupération des données 2020-2026...")
df = load_price_data_from_supabase(supabase, table_name=market.price_table)

print(f"✅ {len(df)} points (2020-2026)")
print(f"   Prix min: ${df['price'].min():.2f}")
print(f"   Prix max: ${df['price'].max():.2f}")
print(f"   Prix moyen: ${df['price'].mean():.2f}")

print("\n[2/6] Préparation features (validation honnête sans fuite Prophet)...")

df_technical = build_technical_features(df)
# Ne dropper que sur les features de base (OHLCV/OI optionnels peuvent etre NaN)
base_feat = [c for c in FEATURE_COLS if c in df_technical.columns and not c.startswith("prophet_")]
valid_mask = df_technical[base_feat].notna().all(axis=1)
first_valid = df_technical.index[valid_mask][0]
valid_indices = df_technical.index[valid_mask]
split_idx = int(len(valid_indices) * 0.8)
split_pos = valid_indices[split_idx] if split_idx < len(valid_indices) else valid_indices[-1]

train_raw = df.iloc[: split_pos + 1].copy()
val_raw = df.iloc[split_pos + 1 :].copy()

trainer = HybridModelTrainer()
prophet_val, xgb_val, train_features = trainer.fit(train_raw)

# Validation set: technical features from full history, Prophet from train-only model
val_technical = build_technical_features(df)
val_with_prophet = add_prophet_features(val_technical, prophet_val)
val_df = val_with_prophet.iloc[split_pos + 1 :].dropna(subset=FEATURE_COLS)

train_df = train_features.dropna()
X_train = train_df[FEATURE_COLS]
y_train = train_df["price"]
X_val = val_df[FEATURE_COLS]
y_val = val_df["price"]

print(f"✅ {len(train_df)} points train | {len(val_df)} points val")

print("\n[3/6] Entraînement final sur toutes les données...")
prophet_model, xgb_model, df_clean = trainer.fit(df)
df_clean = df_clean.dropna()
print(f"✅ Prophet + XGBoost entraînés sur {len(df_clean)} points")

print("\n[4/6] Métriques de validation (Prophet fit sur train uniquement)...")
val_pred = xgb_val.predict(X_val)
train_pred = xgb_val.predict(X_train)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_mae = mean_absolute_error(y_train, train_pred)
train_mape = np.mean(np.abs((y_train.values - train_pred) / y_train.values)) * 100

val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
val_mae = mean_absolute_error(y_val, val_pred)
val_mape = np.mean(np.abs((y_val.values - val_pred) / y_val.values)) * 100

print(f"\n   📊 Performance:")
print(f"      Train RMSE: ${train_rmse:.2f} | MAE: ${train_mae:.2f} | MAPE: {train_mape:.2f}%")
print(f"      Val RMSE: ${val_rmse:.2f} | MAE: ${val_mae:.2f} | MAPE holdout 1-step: {val_mape:.2f}%")

feature_importance = dict(zip(FEATURE_COLS, xgb_model.feature_importances_))
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

print(f"\n   🔥 Top 10 Features:")
for i, (feat, imp) in enumerate(sorted_features[:10], 1):
    print(f"      {i}. {feat}: {imp:.2%}")

print("\n[5/6] Test des prédictions...")

current_price = df_clean["price"].iloc[-1]
current_date = df_clean["date"].iloc[-1]
last_row = df_clean.iloc[-1]

print(f"\n📊 Prix actuel ({current_date.date()}): ${current_price:,.2f}")
print("\n" + "=" * 80)
print("PRÉDICTIONS HYBRIDES AMÉLIORÉES")
print("=" * 80)

for days in [1, 7, 30]:
    future_date = future_business_date(current_date, days)
    features_future = build_prediction_row(last_row, current_price, future_date, prophet_model)
    pred_price = xgb_model.predict(features_future[FEATURE_COLS])[0]
    prophet_yhat_future = features_future["prophet_yhat"].iloc[0]
    change = pred_price - current_price
    change_pct = (change / current_price) * 100

    print(f"\n{days} jour(s) - {future_date.date()}:")
    print(f"   Prix prédit: ${pred_price:,.2f}")
    print(f"   Changement: ${change:+,.2f} ({change_pct:+.2f}%)")
    print(f"   Prophet suggère: ${prophet_yhat_future:,.2f} (XGBoost décide)")

print("\n" + "=" * 80)
print("SAUVEGARDE DES MODÈLES")
print("=" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

prophet_path = str(models_dir / f"prophet_improved_{timestamp}.pkl")
with open(prophet_path, "wb") as f:
    pickle.dump(prophet_model, f)
print(f"✅ Prophet sauvegardé: {prophet_path}")

xgb_path = str(models_dir / f"xgboost_improved_{timestamp}.pkl")
with open(xgb_path, "wb") as f:
    pickle.dump(xgb_model, f)
print(f"✅ XGBoost sauvegardé: {xgb_path}")

model_info = {
    "timestamp": timestamp,
    "model_type": "hybrid_improved",
    "market": args.market,
    "description": "XGBoost principal avec Prophet comme feature secondaire",
    "data_period": "2020-2026",
    "training_points": len(X_train),
    "validation_points": len(X_val),
    "train_mape": float(train_mape),
    "val_mape_holdout_1step": float(val_mape),
    "val_mape": float(val_mape),
    "val_rmse": float(val_rmse),
    "val_mae": float(val_mae),
    "feature_importance": {k: float(v) for k, v in sorted_features[:15]},
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
print(f"✅ Infos sauvegardées: {info_path}")

print("\n" + "=" * 80)
print("✅ ENTRAÎNEMENT TERMINÉ !")
print("=" * 80)
print(f"""
📊 RÉSUMÉ:
   Modèle: Hybride Amélioré (XGBoost principal)
   Période: 2020-2026
   Points d'entraînement: {len(X_train)}

   Performance (holdout 1-step, sans fuite Prophet):
   - MAPE holdout: {val_mape:.2f}%
   - RMSE Validation: ${val_rmse:.2f}
   - MAE Validation: ${val_mae:.2f}

   Reference honnete (walk-forward): voir walk_forward_reference dans model_info
   ou GET /api/v1/validation/metrics

🎯 PROCHAINE ÉTAPE:
   docker-compose restart api
""")
print("=" * 80)
