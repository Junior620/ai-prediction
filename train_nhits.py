# -*- coding: utf-8 -*-
"""
ENTRAINEMENT N-HiTS (NeuralForecast)
3eme moteur de prediction - complement a Prophet + XGBoost
"""

import argparse
import os
import sys
from pathlib import Path

# IMPORTANT: Import neuralforecast BEFORE sklearn to avoid DLL conflict on Windows
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="Entrainement N-HiTS")
parser.add_argument("--market", default="cocoa", help="Marche (cocoa, coffee_robusta)")
args = parser.parse_args()

from src.models.market_registry import get_market_config

market = get_market_config(args.market)
models_dir = Path(market.models_dir)
models_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("[N-HiTS] ENTRAINEMENT N-HiTS (NeuralForecast)")
print("   Marche: %s (%s)" % (market.display_name, args.market))
print("   3eme moteur de prediction")
print("=" * 80)

# ============================================================================
# 1. RECUPERER LES DONNEES
# ============================================================================
print("")
print("[1/5] Recuperation des donnees depuis Supabase...")

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

all_data = []
page_size = 1000
offset = 0
max_retries = 4

while True:
    response = None
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = (
                supabase.table(market.price_table)
                .select("date, price")
                .order("date")
                .range(offset, offset + page_size - 1)
                .execute()
            )
            break
        except Exception as e:
            last_err = e
            wait_s = attempt * 2
            print("   [WARN] Supabase page offset=%s attempt %s/%s: %s — retry in %ss" % (
                offset, attempt, max_retries, e, wait_s
            ))
            import time
            time.sleep(wait_s)
    if response is None:
        raise RuntimeError("Supabase fetch failed after retries: %s" % last_err)
    if not response.data:
        break
    all_data.extend(response.data)
    offset += page_size

df = pd.DataFrame(all_data)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')

df = df[df['date'] >= '2020-01-01'].copy()

mean_price = df['price'].mean()
std_price = df['price'].std()
df = df[
    (df['price'] >= mean_price - 3 * std_price) &
    (df['price'] <= mean_price + 3 * std_price)
].copy()

print("[OK] %d points de donnees (2020-2026)" % len(df))
print("   Prix min: $%.2f" % df['price'].min())
print("   Prix max: $%.2f" % df['price'].max())
print("   Prix actuel: $%.2f (%s)" % (df['price'].iloc[-1], df['date'].iloc[-1].date()))

# ============================================================================
# 2. FORMATER POUR NEURALFORECAST
# ============================================================================
print("")
print("[2/5] Formatage des donnees pour NeuralForecast...")

nf_df = df[['date', 'price']].copy()
nf_df.columns = ['ds', 'y']
nf_df['unique_id'] = market.nhits_unique_id
nf_df = nf_df.sort_values('ds').reset_index(drop=True)

print("[OK] DataFrame formate: %d lignes" % len(nf_df))

# ============================================================================
# 3. ENTRAINER N-HiTS
# ============================================================================
print("")
print("[3/5] Entrainement de N-HiTS...")

HORIZON = 30
INPUT_SIZE = 60

nhits_model = NHITS(
    h=HORIZON,
    input_size=INPUT_SIZE,
    stack_types=['identity', 'identity', 'identity'],
    n_blocks=[1, 1, 1],
    mlp_units=3 * [[256, 256]],
    n_pool_kernel_size=[4, 2, 1],
    n_freq_downsample=[4, 2, 1],
    learning_rate=1e-3,
    max_steps=500,
    early_stop_patience_steps=50,
    val_check_steps=25,
    dropout_prob_theta=0.1,
    scaler_type='robust',
    batch_size=32,
    windows_batch_size=256,
    random_seed=42,
    loss=MAE(),
    accelerator='cpu',
    enable_progress_bar=True,
)

nf = NeuralForecast(
    models=[nhits_model],
    freq='B'
)

val_size = HORIZON

print("   Horizon: %d jours" % HORIZON)
print("   Input size: %d jours" % INPUT_SIZE)
print("   Validation: derniers %d jours" % val_size)
print("   Entrainement en cours (CPU, ~2-5 min)...")
print("")
sys.stdout.flush()

import traceback
try:
    nf.fit(df=nf_df, val_size=val_size)
except BaseException as e:
    print("[ERREUR] nf.fit a echoue: %s" % str(e))
    traceback.print_exc()
    sys.exit(1)

print("")
print("[OK] N-HiTS entraine avec succes!")

# ============================================================================
# 4. EVALUER LES PERFORMANCES
# ============================================================================
print("")
print("[4/5] Evaluation des performances...")

cv_results = nf.cross_validation(df=nf_df, val_size=val_size, n_windows=1)

y_true = cv_results['y'].values
y_pred = cv_results['NHITS'].values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae_val = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("")
print("   Performance N-HiTS (validation %d jours):" % val_size)
print("      RMSE: $%.2f" % rmse)
print("      MAE:  $%.2f" % mae_val)
print("      MAPE: %.2f%%" % mape)

print("")
print("   Predictions detaillees:")
if len(y_true) >= 1:
    print("      J+1:  Predit $%.2f vs Reel $%.2f (erreur: %.2f)" % (y_pred[0], y_true[0], abs(y_pred[0]-y_true[0])))
if len(y_true) >= 7:
    print("      J+7:  Predit $%.2f vs Reel $%.2f (erreur: %.2f)" % (y_pred[6], y_true[6], abs(y_pred[6]-y_true[6])))
if len(y_true) >= 30:
    print("      J+30: Predit $%.2f vs Reel $%.2f (erreur: %.2f)" % (y_pred[29], y_true[29], abs(y_pred[29]-y_true[29])))

# ============================================================================
# 5. SAUVEGARDER LE MODELE
# ============================================================================
print("")
print("[5/5] Sauvegarde du modele...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

save_dir = str(models_dir / ("nhits_%s" % timestamp))
nf.save(path=save_dir, overwrite=True)
print("[OK] Modele N-HiTS sauvegarde: %s/" % save_dir)

import json
model_info = {
    "timestamp": timestamp,
    "model_type": "nhits",
    "market": args.market,
    "description": "N-HiTS (NeuralForecast) - 3eme moteur de prediction",
    "framework": "neuralforecast",
    "data_period": "2020-2026",
    "training_points": int(len(nf_df) - val_size),
    "validation_points": int(val_size),
    "horizon": HORIZON,
    "input_size": INPUT_SIZE,
    "val_rmse": float(rmse),
    "val_mae": float(mae_val),
    "val_mape": float(mape),
}

info_path = str(models_dir / ("model_info_nhits_%s.json" % timestamp))
with open(info_path, 'w') as f:
    json.dump(model_info, f, indent=2)
print("[OK] Infos sauvegardees: %s" % info_path)

# ============================================================================
# RESUME
# ============================================================================
print("")
print("=" * 80)
print("[OK] ENTRAINEMENT N-HiTS TERMINE !")
print("=" * 80)
print("   MAPE Validation: %.2f%%" % mape)
print("   RMSE Validation: $%.2f" % rmse)
print("   MAE Validation:  $%.2f" % mae_val)
print("=" * 80)
