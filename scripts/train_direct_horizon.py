#!/usr/bin/env python
"""Train direct h-step XGBoost models for horizons 7 and 30."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from supabase import create_client

load_dotenv(ROOT / "config" / ".env")
load_dotenv()

from src.models.direct_horizon_trainer import DirectHorizonTrainer
from src.models.hybrid_features import load_price_data_from_supabase


def main() -> int:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        print("ERREUR: SUPABASE_URL et SUPABASE_KEY requis")
        return 1

    print("Chargement des donnees...")
    supabase = create_client(supabase_url, supabase_key)
    df = load_price_data_from_supabase(supabase)
    print(f"  {len(df)} points")

    trainer = DirectHorizonTrainer(horizons=[7, 30])
    models, meta = trainer.fit(df)
    paths = trainer.save(models, meta, models_dir=str(ROOT / "models"))

    print("Modeles direct h-step sauvegardes:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
