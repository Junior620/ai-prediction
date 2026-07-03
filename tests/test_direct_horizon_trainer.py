"""Tests for direct h-step horizon training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.direct_horizon_trainer import DirectHorizonTrainer
from src.models.hybrid_features import future_business_date


def test_direct_dataset_target_alignment():
    dates = pd.bdate_range("2020-01-01", periods=80)
    prices = np.linspace(3000, 3200, 80)
    df = pd.DataFrame({"date": dates, "price": prices})

    trainer = DirectHorizonTrainer(horizons=[7])
    X_df, y = trainer._build_direct_dataset(df, horizon=7)

    assert len(X_df) == len(y)
    assert len(X_df) > 0

    idx = X_df.index[0]
    origin = df.loc[idx, "date"]
    target_date = pd.Timestamp(future_business_date(origin, 7)).normalize()
    expected = df[df["date"].dt.normalize() == target_date]["price"].iloc[0]
    assert y.loc[idx] == expected


def test_fit_and_save_roundtrip(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=100)
    prices = np.linspace(3000, 3300, 100)
    df = pd.DataFrame({"date": dates, "price": prices})

    trainer = DirectHorizonTrainer(horizons=[7])
    models, meta = trainer.fit(df)
    paths = trainer.save(models, meta, models_dir=str(tmp_path))

    loaded = DirectHorizonTrainer.load_latest(str(tmp_path))
    assert 7 in loaded
