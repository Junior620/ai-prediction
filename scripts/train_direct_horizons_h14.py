"""Entrainement des modeles directs h=1,7,14,30 (cacao + robusta)."""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from supabase import create_client

from src.models.direct_horizon_trainer import DirectHorizonTrainer
from src.models.hybrid_features import (
    FEATURE_COLS,
    load_price_data_from_supabase,
    resolve_feature_cols,
)
from src.models.market_registry import get_market_config

load_dotenv(ROOT / ".env")


def latest_prophet(models_dir: Path):
    files = sorted(models_dir.glob("prophet_improved_*.pkl"), reverse=True)
    if not files:
        return None
    with open(files[0], "rb") as f:
        return pickle.load(f)


def train_market(market_id: str) -> None:
    market = get_market_config(market_id)
    models_dir = Path(market.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    df = load_price_data_from_supabase(sb, table_name=market.price_table)
    print(f"\n=== {market_id}: {len(df)} points ===")
    if len(df) < 100:
        raise SystemExit(f"Pas assez de donnees pour {market_id}")

    for col in ("open", "high", "low", "volume", "open_interest"):
        if col in df.columns:
            import pandas as pd

            df[col] = pd.to_numeric(df[col], errors="coerce").ffill()

    if market_id == "cocoa":
        feature_cols = resolve_feature_cols(include_ohlcv=True, include_oi=True)
    else:
        feature_cols = list(FEATURE_COLS)

    prophet = latest_prophet(models_dir)
    dht = DirectHorizonTrainer(horizons=[1, 7, 14, 30], feature_cols=feature_cols)
    models, meta = dht.fit(df, prophet_model=prophet)
    meta["market"] = market_id
    meta["feature_cols"] = feature_cols
    paths = dht.save(models, meta, models_dir=str(models_dir))
    print(f"[OK] {market_id} direct horizons: {paths}")


if __name__ == "__main__":
    train_market("cocoa")
    train_market("coffee_robusta")
    print("\n[OK] Direct h1/h7/h14/h30 entraines")
