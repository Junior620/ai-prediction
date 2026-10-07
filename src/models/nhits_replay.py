"""One fresh N-HiTS, trained only on the past, at the served settings."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def forecast_nhits(history: pd.DataFrame, unique_id: str) -> Dict[int, float]:
    """Price levels for the next 30 sessions. A new network every call."""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import NHITS

    frame = history[["date", "price"]].dropna().copy()
    frame.columns = ["ds", "y"]
    frame["unique_id"] = unique_id
    frame = frame.sort_values("ds")
    if len(frame) < 120:
        return {}

    model = NHITS(
        h=30,
        input_size=60,
        stack_types=["identity", "identity", "identity"],
        n_blocks=[1, 1, 1],
        mlp_units=3 * [[256, 256]],
        n_pool_kernel_size=[4, 2, 1],
        n_freq_downsample=[4, 2, 1],
        learning_rate=1e-3,
        max_steps=500,
        early_stop_patience_steps=50,
        val_check_steps=25,
        dropout_prob_theta=0.1,
        scaler_type="robust",
        batch_size=32,
        windows_batch_size=256,
        random_seed=42,
        loss=MAE(),
        accelerator="cpu",
        enable_progress_bar=False,
    )
    network = NeuralForecast(models=[model], freq="B")
    network.fit(df=frame, val_size=30)
    forecast = network.predict(df=frame)
    levels: Dict[int, float] = {}
    for step in range(1, min(30, len(forecast)) + 1):
        levels[step] = float(forecast["NHITS"].iloc[step - 1])
    return levels
