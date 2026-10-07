"""Direct h-step XGBoost models for horizons 1, 7, 14 and 30."""

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
    PROPHET_LEVEL_COLS,
    future_business_date,
    prepare_training_frame,
)


class DirectHorizonTrainer:
    """Train one XGBoost regressor per horizon with target price at t+h."""

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        self.horizons = horizons or [1, 7, 14, 30]
        self.xgb_params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self.feature_cols = list(feature_cols) if feature_cols else list(FEATURE_COLS)
        self.target = "close_delta"

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
        for col in self.feature_cols:
            if col not in df_features.columns:
                df_features[col] = 0.0
        micro = [c for c in self.feature_cols if c not in FEATURE_COLS]
        if micro:
            df_features[micro] = df_features[micro].ffill().fillna(0.0)

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
            origin_price = float(row["price"])
            targets.append(float(match["price"].iloc[-1]) - origin_price)

        if not valid_rows:
            raise ValueError(f"No valid direct h={horizon} training rows")

        subset = df_features.loc[valid_rows].copy()
        for col in PROPHET_LEVEL_COLS:
            if col in subset.columns:
                subset[col] = 0.0
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
        meta: Dict[str, Any] = {
            "horizons": {},
            "trained_at": datetime.now().isoformat(),
            "feature_cols": self.feature_cols,
            # Variation depuis la clôture de t, pas le niveau absolu :
            # le cours servi reste ancré sur le dernier settlement.
            "target": "close_delta",
        }

        shared_prophet = prophet_model
        if shared_prophet is None:
            _, shared_prophet = prepare_training_frame(df)

        for h in self.horizons:
            X_df, y = self._build_direct_dataset(df, h, prophet_model=shared_prophet)
            X = X_df[self.feature_cols]
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
    def load_from_info(info_path: str) -> Dict[int, xgb.XGBRegressor]:
        """Load the direct models named by one metadata file, not the newest name."""
        path = Path(info_path)
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        root = path.parent
        loaded: Dict[int, xgb.XGBRegressor] = {}
        for h_str, h_meta in meta.get("horizons", {}).items():
            model_path = h_meta.get("model_path")
            if not model_path:
                continue
            candidate = Path(model_path)
            if not candidate.exists():
                candidate = root / Path(str(model_path).replace("\\", "/")).name
            if candidate.exists():
                with open(candidate, "rb") as f:
                    loaded[int(h_str)] = pickle.load(f)
        return loaded

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
            if not path:
                continue
            candidate = Path(path)
            if not candidate.exists():
                # Le JSON est écrit sous Windows (antislash) puis chargé sur Linux.
                candidate = root / Path(str(path).replace("\\", "/")).name
            if candidate.exists():
                with open(candidate, "rb") as f:
                    loaded[int(h_str)] = pickle.load(f)
        return loaded

    @staticmethod
    def load_latest_meta(models_dir: str = "models") -> Dict[str, Any]:
        """Metadata of the newest direct-horizon file, or empty dict."""
        root = Path(models_dir)
        info_files = sorted(root.glob("model_info_direct_horizon_*.json"), reverse=True)
        if not info_files:
            return {}
        with open(info_files[0], encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def level_from_prediction(raw: float, origin_price: float, target: str) -> float:
        """Rebuild a price level from a model output."""
        if target == "close_delta":
            return float(origin_price) + float(raw)
        return float(raw)
