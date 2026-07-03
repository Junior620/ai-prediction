"""Direct h-step XGBoost models for horizons 7 and 30."""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import xgboost as xgb

from src.models.hybrid_features import (
    DEFAULT_XGB_PARAMS,
    FEATURE_COLS,
    future_business_date,
    prepare_training_frame,
)


class DirectHorizonTrainer:
    """Train one XGBoost regressor per horizon with target price at t+h."""

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
    ):
        self.horizons = horizons or [7, 30]
        self.xgb_params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}

    def _build_direct_dataset(
        self,
        df: pd.DataFrame,
        horizon: int,
        prophet_model=None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Build features at t with target price at t+horizon (business days)."""
        df = df.sort_values("date").reset_index(drop=True)
        df_features, prophet = prepare_training_frame(
            df, prophet_model=prophet_model
        )

        targets = []
        valid_rows = []

        for idx, row in df_features.iterrows():
            if pd.isna(row.get("price_lag_30")):
                continue
            origin_date = row["date"]
            target_date = pd.Timestamp(future_business_date(origin_date, horizon)).normalize()
            match = df[df["date"].dt.normalize() == target_date]
            if match.empty:
                continue
            valid_rows.append(idx)
            targets.append(float(match["price"].iloc[-1]))

        if not valid_rows:
            raise ValueError(f"No valid direct h={horizon} training rows")

        subset = df_features.loc[valid_rows].copy()
        y = pd.Series(targets, index=subset.index)
        return subset, y

    def fit(
        self,
        df: pd.DataFrame,
        prophet_model=None,
    ) -> Tuple[Dict[int, xgb.XGBRegressor], Dict[str, Any]]:
        """
        Train direct models for each configured horizon.

        Returns:
            models dict, metadata dict
        """
        models: Dict[int, xgb.XGBRegressor] = {}
        meta: Dict[str, Any] = {"horizons": {}, "trained_at": datetime.now().isoformat()}

        shared_prophet = prophet_model
        if shared_prophet is None:
            _, shared_prophet = prepare_training_frame(df)

        for h in self.horizons:
            X_df, y = self._build_direct_dataset(df, h, prophet_model=shared_prophet)
            X = X_df[FEATURE_COLS]
            model = xgb.XGBRegressor(**self.xgb_params)
            model.fit(X, y, verbose=False)
            models[h] = model
            meta["horizons"][str(h)] = {"n_samples": int(len(X))}

        return models, meta

    def save(
        self,
        models: Dict[int, xgb.XGBRegressor],
        meta: Dict[str, Any],
        models_dir: str = "models",
        timestamp: Optional[str] = None,
    ) -> Dict[str, str]:
        """Persist direct horizon models and metadata."""
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        paths: Dict[str, str] = {}

        for h, model in models.items():
            path = Path(models_dir) / f"xgboost_h{h}_{ts}.pkl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(model, f)
            paths[f"h{h}"] = str(path)
            meta["horizons"][str(h)]["model_path"] = str(path)

        info_path = Path(models_dir) / f"model_info_direct_horizon_{ts}.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        paths["info"] = str(info_path)
        return paths

    @staticmethod
    def load_latest(models_dir: str = "models") -> Dict[int, xgb.XGBRegressor]:
        """Load most recent direct horizon models from models/."""
        root = Path(models_dir)
        info_files = sorted(root.glob("model_info_direct_horizon_*.json"), reverse=True)
        if not info_files:
            return {}

        with open(info_files[0], encoding="utf-8") as f:
            meta = json.load(f)

        loaded: Dict[int, xgb.XGBRegressor] = {}
        for h_str, h_meta in meta.get("horizons", {}).items():
            path = h_meta.get("model_path")
            if path and Path(path).exists():
                with open(path, "rb") as f:
                    loaded[int(h_str)] = pickle.load(f)
        return loaded
