"""
N-HiTS multi-window cross-validation via NeuralForecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.validation.metrics import compute_horizon_metrics


@dataclass
class NHitsValidatorConfig:
    horizons: List[int] = field(default_factory=lambda: [1, 7, 30])
    n_windows: int = 12
    val_size: int = 30
    step_size: int = 5
    horizon: int = 30
    input_size: int = 60
    max_steps: int = 500
    early_stop_patience_steps: int = 50
    unique_id: str = "cocoa_ice_ny"


@dataclass
class NHitsValidatorResult:
    predictions: pd.DataFrame
    summary: Dict[int, Dict[str, float]]
    n_windows: int
    config: NHitsValidatorConfig


class NHitsValidator:
    """Run NeuralForecast cross_validation and extract per-horizon metrics."""

    def __init__(self, config: Optional[NHitsValidatorConfig] = None):
        self.config = config or NHitsValidatorConfig()

    def _build_nf_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        nf_df = df[["date", "price"]].copy()
        nf_df.columns = ["ds", "y"]
        nf_df["unique_id"] = self.config.unique_id
        return nf_df.sort_values("ds").reset_index(drop=True)

    def _create_model(self):
        from neuralforecast.losses.pytorch import MAE
        from neuralforecast.models import NHITS

        cfg = self.config
        return NHITS(
            h=cfg.horizon,
            input_size=cfg.input_size,
            stack_types=["identity", "identity", "identity"],
            n_blocks=[1, 1, 1],
            mlp_units=3 * [[256, 256]],
            n_pool_kernel_size=[4, 2, 1],
            n_freq_downsample=[4, 2, 1],
            learning_rate=1e-3,
            max_steps=cfg.max_steps,
            early_stop_patience_steps=cfg.early_stop_patience_steps,
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

    def run(self, df: pd.DataFrame) -> NHitsValidatorResult:
        from neuralforecast import NeuralForecast

        nf_df = self._build_nf_dataframe(df)
        cfg = self.config

        logger.info(
            f"N-HiTS CV: n_windows={cfg.n_windows}, val_size={cfg.val_size}, "
            f"step_size={cfg.step_size}"
        )

        nhits = self._create_model()
        nf = NeuralForecast(models=[nhits], freq="B")

        nf.fit(df=nf_df, val_size=cfg.val_size)
        cv_results = nf.cross_validation(
            df=nf_df,
            val_size=cfg.val_size,
            n_windows=cfg.n_windows,
            step_size=cfg.step_size,
        )

        cv_results = cv_results.copy()
        cv_results["ds"] = pd.to_datetime(cv_results["ds"])
        cv_results["cutoff"] = pd.to_datetime(cv_results["cutoff"])
        cv_results = cv_results.sort_values(["cutoff", "ds"]).reset_index(drop=True)
        cv_results["horizon_step"] = cv_results.groupby("cutoff").cumcount() + 1

        cutoff_prices = nf_df.set_index("ds")["y"]
        cv_results["origin_price"] = cv_results["cutoff"].map(cutoff_prices)

        records: List[Dict[str, Any]] = []
        for _, row in cv_results.iterrows():
            h_step = int(row["horizon_step"])
            if h_step not in cfg.horizons:
                continue
            records.append(
                {
                    "cutoff": row["cutoff"],
                    "target_date": row["ds"],
                    "horizon": h_step,
                    "actual": float(row["y"]),
                    "nhits_pred": float(row["NHITS"]),
                    "origin_price": float(row["origin_price"])
                    if pd.notna(row["origin_price"])
                    else np.nan,
                }
            )

        predictions_df = pd.DataFrame(records)
        summary: Dict[int, Dict[str, float]] = {}
        for horizon in cfg.horizons:
            subset = predictions_df[predictions_df["horizon"] == horizon]
            metrics = compute_horizon_metrics(subset, "nhits_pred")
            summary[int(horizon)] = metrics

        return NHitsValidatorResult(
            predictions=predictions_df,
            summary=summary,
            n_windows=cfg.n_windows,
            config=cfg,
        )
