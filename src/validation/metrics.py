"""Aggregate evaluation metrics for walk-forward validation."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def compute_horizon_metrics(
    predictions: pd.DataFrame,
    pred_col: str,
    origin_col: str = "origin_price",
    actual_col: str = "actual",
) -> Dict[str, float]:
    """Compute RMSE, MAE, MAPE, and directional accuracy for one predictor column."""
    subset = predictions.dropna(subset=[pred_col, actual_col, origin_col])
    if subset.empty:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "mape": float("nan"),
            "directional_accuracy": float("nan"),
            "n_predictions": 0,
        }

    y_true = subset[actual_col].values
    y_pred = subset[pred_col].values
    origin = subset[origin_col].values

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    actual_dir = np.sign(y_true - origin)
    pred_dir = np.sign(y_pred - origin)
    directional_accuracy = float(np.mean(actual_dir == pred_dir))

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
        "n_predictions": int(len(subset)),
    }


def aggregate_by_horizon(
    predictions: pd.DataFrame,
    pred_columns: List[str],
    horizon_col: str = "horizon",
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Aggregate metrics per horizon and per prediction column."""
    results: Dict[str, Dict[int, Dict[str, float]]] = {}
    for col in pred_columns:
        results[col] = {}
        for horizon in sorted(predictions[horizon_col].unique()):
            subset = predictions[predictions[horizon_col] == horizon]
            results[col][int(horizon)] = compute_horizon_metrics(subset, col)
    return results


def compute_holdout_baseline(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
) -> Optional[Dict[str, float]]:
    """
    Compute naive 1-step holdout MAPE on last val_fraction of rows (legacy baseline).
    """
    from src.models.hybrid_trainer import HybridModelTrainer

    if len(df) < 100:
        return None

    split_idx = int(len(df) * (1 - val_fraction))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    trainer = HybridModelTrainer()
    prophet_model, xgb_model, _ = trainer.fit(train_df)
    from src.models.hybrid_features import FEATURE_COLS, prepare_training_frame

    full_features, _ = prepare_training_frame(df, prophet_model=prophet_model)
    val_features = full_features.iloc[split_idx:].dropna()

    if val_features.empty:
        return None

    from src.models.hybrid_trainer import next_session_frame

    aligned = next_session_frame(val_features)
    if aligned.empty:
        return None
    preds = xgb_model.predict(aligned[FEATURE_COLS])
    actual = aligned["target_price"].values
    mape = float(np.mean(np.abs((actual - preds) / actual)) * 100)

    return {"mape_1step_holdout": mape, "n_val": int(len(aligned))}
