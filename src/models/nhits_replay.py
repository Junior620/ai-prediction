"""One fresh N-HiTS, trained only on the past, at the served settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd


def _nhits_frame(history: pd.DataFrame, unique_id: str) -> pd.DataFrame:
    frame = history[["date", "price"]].dropna().copy()
    frame.columns = ["ds", "y"]
    frame["unique_id"] = unique_id
    return frame.sort_values("ds")


def fit_nhits_network(history: pd.DataFrame, unique_id: str):
    """Train one network on this history. Later sessions call predict, not fit."""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import NHITS

    frame = _nhits_frame(history, unique_id)
    if len(frame) < 120:
        return None

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
    from pandas.tseries.offsets import CustomBusinessDay

    from src.models.hybrid_features import exchange_holidays

    network = NeuralForecast(models=[model], freq=CustomBusinessDay(holidays=exchange_holidays()))
    network.fit(df=frame, val_size=30)
    return network


def predict_nhits_levels(network, history: pd.DataFrame, unique_id: str) -> Dict[int, float]:
    """Reread a frozen network on the current history. The horizon dates move with it."""
    if network is None:
        return {}
    frame = _nhits_frame(history, unique_id)
    if frame.empty:
        return {}
    forecast = network.predict(df=frame)
    levels: Dict[int, float] = {}
    for step in range(1, min(30, len(forecast)) + 1):
        levels[step] = float(forecast["NHITS"].iloc[step - 1])
    return levels


def strip_custom_business_day(network) -> None:
    """NeuralForecast cannot store CustomBusinessDay without pickle."""
    seen: set[int] = set()

    def walk(obj) -> None:
        if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
            return
        marker = id(obj)
        if marker in seen:
            return
        seen.add(marker)
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if type(value).__name__ == "CustomBusinessDay":
                    obj[key] = "B"
                else:
                    walk(value)
            return
        if isinstance(obj, list):
            for index, value in enumerate(obj):
                if type(value).__name__ == "CustomBusinessDay":
                    obj[index] = "B"
                else:
                    walk(value)
            return
        mapping = getattr(obj, "__dict__", None)
        if isinstance(mapping, dict):
            walk(mapping)

    walk(network)


def restore_exchange_calendar(network, model_dir) -> None:
    """Put the exchange calendar back after a save that stored freq as B."""
    if network is None:
        return
    sidecar = Path(model_dir) / "freq_holidays.json"
    if not sidecar.exists():
        return
    from pandas.tseries.offsets import CustomBusinessDay

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    calendar = CustomBusinessDay(holidays=list(payload.get("holidays") or []))
    network.freq = calendar
    for model in getattr(network, "models", []) or []:
        if hasattr(model, "freq"):
            model.freq = calendar


def forecast_nhits(history: pd.DataFrame, unique_id: str) -> Dict[int, float]:
    """Price levels for the next 30 sessions. A new network every call."""
    network = fit_nhits_network(history, unique_id)
    return predict_nhits_levels(network, history, unique_id)
