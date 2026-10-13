"""Journal the forecasts of one named lot, including the old-lot diagnostic."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models.data_models import Prediction
from src.models.direct_horizon_trainer import DirectHorizonTrainer
from src.models.horizon_schedule import TRACKED_HORIZONS, schedule_table
from src.models.improved_price_predictor import ImprovedPricePredictor
from src.models.prediction_journal import JournalLockError, append_forecast
from src.models.release_manifest import artifact_path, data_fingerprint, procedure_fingerprint
from src.models.served_forecast import publish_or_close

ROOT = Path(__file__).resolve().parents[2]


class _QuietNews:
    def aggregate_sentiment(self, articles) -> float:
        del articles
        return 0.0


def frame_fingerprint(frame: pd.DataFrame) -> str:
    from src.models.hybrid_features import consumed_snapshot_rows

    return data_fingerprint(consumed_snapshot_rows(frame))


def train_bounds(release: Optional[dict]) -> dict:
    """Training dates live in the lot metadata. An unknown bound stays unknown."""
    if not release:
        return {"train_start_date": None, "train_end_date": None, "training_data_fingerprint": None}
    try:
        meta = json.loads(artifact_path(release, "improved_info").read_text(encoding="utf-8"))
    except (KeyError, OSError, json.JSONDecodeError):
        return {"train_start_date": None, "train_end_date": None, "training_data_fingerprint": None}
    return {
        "train_start_date": meta.get("train_start_date"),
        "train_end_date": meta.get("train_end_date"),
        "training_data_fingerprint": meta.get("training_data_fingerprint"),
    }


def procedure_for(release: dict, caps: dict) -> str:
    weights = {}
    try:
        weights_path = artifact_path(release, "ensemble_weights")
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        weights = payload.get("by_horizon") or {}
    except (KeyError, OSError, json.JSONDecodeError):
        weights = {}
    from src.models.hybrid_features import OHLCV_FILL, SERVICE_CALENDAR

    parameters = schedule_table(weights, caps)
    parameters["calendar"] = SERVICE_CALENDAR
    parameters["ohlcv_fill"] = OHLCV_FILL
    return procedure_fingerprint(
        artifact_paths=dict(release.get("artifacts") or {}),
        parameters=parameters,
        frame_token="",
    )


def load_named_predictor(release: dict, market_cfg, pred_cfg: dict) -> ImprovedPricePredictor:
    """Load the artifacts named by this release. Never the newest folder."""
    info_path = artifact_path(release, "improved_info")
    meta = json.loads(info_path.read_text(encoding="utf-8"))
    if meta.get("target") != "next_session":
        raise ValueError(f"Métadonnées sans cible next_session: {info_path.name}")
    with open(artifact_path(release, "prophet"), "rb") as handle:
        prophet_model = pickle.load(handle)
    with open(artifact_path(release, "xgboost"), "rb") as handle:
        xgboost_model = pickle.load(handle)
    direct_models = DirectHorizonTrainer.load_from_info(str(artifact_path(release, "direct_info")))
    nhits_model = None
    artifacts = release.get("artifacts") or {}
    needs_nhits = False
    try:
        weights_payload = json.loads(artifact_path(release, "ensemble_weights").read_text(encoding="utf-8"))
        for item in (weights_payload.get("by_horizon") or {}).values():
            if float((item or {}).get("nhits") or 0.0) > 0.0:
                needs_nhits = True
                break
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError):
        needs_nhits = bool(artifacts.get("nhits"))
    if artifacts.get("nhits") and needs_nhits:
        try:
            from neuralforecast import NeuralForecast

            nhits_dir = artifact_path(release, "nhits")
            nhits_model = NeuralForecast.load(path=str(nhits_dir))
            from src.models.nhits_replay import restore_exchange_calendar

            restore_exchange_calendar(nhits_model, nhits_dir)
        except Exception as exc:
            nhits_model = None
            print(f"[AVERTISSEMENT] N-HiTS illisible: {exc}")
    return ImprovedPricePredictor(
        prophet_model=prophet_model,
        xgboost_model=xgboost_model,
        nlp_analyzer=_QuietNews(),
        sentiment_weight=0.0,
        model_version=str(release.get("version")),
        nhits_model=nhits_model,
        ensemble_weights_file=str(artifact_path(release, "ensemble_weights")),
        multi_step_mode=pred_cfg.get("multi_step_mode", "recursive"),
        direct_horizon_models=direct_models,
        direct_info_path=str(artifact_path(release, "direct_info")),
        conformal_intervals_file=str(artifact_path(release, "conformal_intervals")),
        confidence_level=float(pred_cfg.get("confidence_level", 0.90)),
        price_bounds=tuple(market_cfg.price_bounds) if market_cfg.price_bounds else (1000.0, 15000.0),
        price_table=market_cfg.price_table,
        nhits_unique_id=market_cfg.nhits_unique_id,
        garch_enabled=bool(market_cfg.garch_enabled),
        models_dir=str(artifact_path(release, "prophet").parent),
        max_abs_change_pct=pred_cfg.get("max_abs_change_pct"),
        recent_range_days=int(pred_cfg.get("recent_range_days", 252)),
        recent_range_padding_pct=float(pred_cfg.get("recent_range_padding_pct", 15.0)),
        feature_cols=meta.get("feature_cols"),
        feature_set=meta.get("feature_set", "baseline"),
    )


def journal_served(
    predictions: list[Prediction],
    *,
    market: str,
    release: dict,
    procedure: str,
    data_token: str,
    diagnostic: bool = False,
    previous_version: Optional[str] = None,
    previous_prices: Optional[dict] = None,
    path: Optional[Path] = None,
) -> list[str]:
    """Write served rows. A lock failure is raised, not ignored."""
    publish_or_close(predictions, release, None, release_mode="experimental")
    written = []
    for item in predictions:
        parts = dict(item.components or {})
        status = str(getattr(item, "status", None) or parts.get("status") or "unavailable")
        interval = getattr(item, "confidence_interval", None) or []
        lower = upper = None
        if status != "unavailable" and len(interval) == 2:
            lower = float(interval[0]) if interval[0] is not None else None
            upper = float(interval[1]) if interval[1] is not None else None
        price = None if status == "unavailable" or item.price is None else float(item.price)
        old_price = None
        comparison = None
        if diagnostic:
            old_price = (previous_prices or {}).get(int(item.horizon))
            if old_price is None or price is None:
                comparison = "indisponible"
            else:
                comparison = "disponible"
        try:
            result = append_forecast(
                market=market,
                model_version=str(release.get("version")),
                release_mode="experimental",
                horizon=int(item.horizon),
                price=price,
                lower=lower,
                upper=upper,
                origin_date=parts.get("origin_date"),
                origin_price=parts.get("origin_price"),
                target_date=parts.get("target_date"),
                status=status,
                procedure_fingerprint=procedure,
                data_fingerprint=data_token,
                train_end_date=train_bounds(release).get("train_end_date"),
                diagnostic=diagnostic,
                previous_model_version=previous_version,
                previous_price=old_price,
                comparison=comparison,
                path=path,
            )
        except JournalLockError:
            raise
        written.append(result)
    return written
