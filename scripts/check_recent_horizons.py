"""Contrôle J+1/7/14/30 cacao avant déploiement.

1. Prévision live du dernier cours : écart au spot, bande, plafond.
2. Erreur réalisée sur les 10 dernières origines dont la cible est connue
   (modèle entraîné avant ces origines).
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.conformal_intervals import get_margins_for_horizon, load_conformal_margins
from src.models.direct_horizon_trainer import DirectHorizonTrainer
from src.models.hybrid_features import (
    build_price_lookup,
    direct_feature_row,
    fit_prophet,
    future_business_date,
    guard_forecast,
    load_price_data_from_supabase,
    prepare_training_frame,
    resolve_feature_cols,
)
from src.models.market_registry import get_market_config

HORIZONS = [1, 7, 14, 30]
# Walk-forward MAE de référence (reports/walk_forward/20261003_031839).
WF_MAE = {1: 31.0, 7: 52.0, 14: 66.0, 30: 83.0}
LIVE_MAE_LIMIT = {h: 2.0 * mae for h, mae in WF_MAE.items()}


def _load_prices():
    load_dotenv(ROOT / ".env")
    market = get_market_config("cocoa")
    sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    df = load_price_data_from_supabase(sb, table_name=market.price_table)
    for col in ("open", "high", "low", "volume", "open_interest"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").ffill()
    return market, df.sort_values("date").reset_index(drop=True)


def _feature_frame(df: pd.DataFrame, prophet_model, feature_cols):
    frame, _ = prepare_training_frame(df, prophet_model=prophet_model)
    for col in feature_cols:
        if col not in frame.columns:
            frame[col] = 0.0
    frame[feature_cols] = frame[feature_cols].ffill()
    clean = frame.dropna(subset=["price", "price_lag_30"]).copy()
    clean[feature_cols] = clean[feature_cols].fillna(0.0)
    return clean


def _origins(df: pd.DataFrame, horizon: int, n: int = 10) -> list[int]:
    lookup = build_price_lookup(df)
    found = []
    for i, raw_date in enumerate(df["date"]):
        target = pd.Timestamp(future_business_date(raw_date, horizon)).normalize()
        if target in lookup.index:
            found.append(i)
    return found[-n:]


def _latest_prophet(models_dir: Path):
    files = sorted(models_dir.glob("prophet_improved_*.pkl"))
    if not files:
        raise SystemExit(f"Prophet introuvable dans {models_dir}")
    with open(files[-1], "rb") as f:
        return pickle.load(f)


def live_check(market, df, feature_cols, caps, margins) -> bool:
    models = DirectHorizonTrainer.load_latest(str(Path(market.models_dir)))
    target_mode = str(
        DirectHorizonTrainer.load_latest_meta(str(Path(market.models_dir))).get("target")
        or "price"
    )
    missing = [h for h in HORIZONS if h not in models]
    if missing:
        print(f"[ECHEC] Modeles directs manquants: {missing}")
        return False

    prophet = _latest_prophet(Path(market.models_dir))
    clean = _feature_frame(df, prophet, feature_cols)
    last = clean.iloc[-1]
    spot = float(df["price"].iloc[-1])
    spot_date = pd.Timestamp(df["date"].iloc[-1]).date()
    model_spot = float(last["price"])
    model_date = pd.Timestamp(last["date"]).date()
    print(f"\nLive  cours {spot_date} {spot:.0f}  |  spot modele {model_date} {model_spot:.0f}")
    if model_date != spot_date or abs(model_spot - spot) > 0.5:
        print("[ECHEC] Le spot du modele n'est pas le dernier cours publie")
        return False

    ok = True
    row = direct_feature_row(last, feature_cols=feature_cols)
    for h in HORIZONS:
        raw_model = float(models[h].predict(row[feature_cols])[0])
        raw = DirectHorizonTrainer.level_from_prediction(raw_model, spot, target_mode)
        band = get_margins_for_horizon(h, margins)
        central, change, failed = guard_forecast(
            raw, spot, float(caps[str(h)]), band
        )
        lo, hi = (central - band[0], central + band[1]) if band else (np.nan, np.nan)
        covers = bool(band) and lo <= spot <= hi
        status = "OK" if covers else "ECHEC"
        if status != "OK":
            ok = False
        print(
            f"  h={h:2} raw={raw:8.1f} central={central:8.1f} "
            f"d={central - spot:+7.1f} cap_hit={failed} "
            f"bande=[{lo:.0f},{hi:.0f}] couvre_spot={covers} {status}"
        )
    return ok


def holdout_check(df, feature_cols, caps, margins) -> bool:
    print("\nHoldout 10 dernieres origines realisees")
    ok = True
    for h in HORIZONS:
        idxs = _origins(df, h, 10)
        if len(idxs) < 10:
            print(f"  h={h} seulement {len(idxs)} origines")
            ok = False
            continue
        train_df = df.iloc[: idxs[0]].copy()
        prophet = fit_prophet(train_df)
        trainer = DirectHorizonTrainer(horizons=[h], feature_cols=feature_cols)
        models, _ = trainer.fit(train_df, prophet_model=prophet)
        errors = []
        naive = []
        rejected = 0
        lookup = build_price_lookup(df)
        band = get_margins_for_horizon(h, margins)
        for i in idxs:
            origin_df = df.iloc[: i + 1].copy()
            clean = _feature_frame(origin_df, prophet, feature_cols)
            last = clean.iloc[-1]
            spot = float(origin_df["price"].iloc[-1])
            raw_model = float(
                models[h].predict(direct_feature_row(last, feature_cols)[feature_cols])[0]
            )
            raw = DirectHorizonTrainer.level_from_prediction(
                raw_model, spot, trainer.target
            )
            central, _, failed = guard_forecast(raw, spot, float(caps[str(h)]), band)
            if failed:
                rejected += 1
            target = pd.Timestamp(future_business_date(origin_df["date"].iloc[-1], h)).normalize()
            actual = float(lookup.loc[target])
            errors.append(abs(central - actual))
            naive.append(abs(actual - spot))
        med = float(np.median(errors))
        naive_med = float(np.median(naive))
        # Le walk-forward historique (~30/50/70/80 £) est un plancher.
        # Sur une fenêtre plus violente, l'erreur du cours inchangé est le
        # plancher réel : on refuse un modèle nettement pire que ça.
        limit = max(LIVE_MAE_LIMIT[h], naive_med * 1.25)
        status = "OK" if med <= limit else "ECHEC"
        if status != "OK":
            ok = False
        print(
            f"  h={h:2} median_|err|={med:6.1f} (limite {limit:.0f}, "
            f"naif {naive_med:.0f}) max={max(errors):.1f} "
            f"rejets={rejected}/10 {status}"
        )
    return ok


def main() -> int:
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        caps = yaml.safe_load(f)["prediction"]["max_abs_change_pct"]
    margins = load_conformal_margins(str(ROOT / "config" / "conformal_intervals.json"))
    market, df = _load_prices()
    feature_cols = resolve_feature_cols(include_ohlcv=True, include_oi=True)
    print(f"Serie cacao: {len(df)} seances, dernier={df['date'].iloc[-1].date()} {df['price'].iloc[-1]:.0f}")

    live_ok = live_check(market, df, feature_cols, caps, margins)
    holdout_ok = holdout_check(df, feature_cols, caps, margins)
    if live_ok and holdout_ok:
        print("\n[OK] Horizons dans le regime attendu")
        return 0
    print("\n[ECHEC] Ne pas deployer")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
