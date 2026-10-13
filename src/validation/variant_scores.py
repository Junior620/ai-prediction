"""Score engine outputs already stored by one walk-forward. No second training."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.models.ensemble_weights import combine_ensemble, get_weights_for_horizon
from src.models.hybrid_features import resolve_capped_forecast
from src.validation.acceptance import evaluate_horizon, load_acceptance
from src.validation.served_backtest import apply_frozen_bands, residual_margins, split_chronological, split_last_origins

SANS_PROPHET_NAME = "sans_prophet_dans_le_melange"
SANS_PROPHET_NOTE = (
    "« Sans Prophet dans le mélange » retire Prophet du mélange final. "
    "XGBoost continue d'utiliser les variables produites par Prophet. "
    "Cette variante ne mesure pas une suppression complète de Prophet."
)

VARIANT_ORDER = (
    "cours_inchange",
    "ensemble_actuel",
    "xgboost_direct",
    "nhits_seul",
    SANS_PROPHET_NAME,
)


def weights_without_prophet(weights: Dict[str, float]) -> Dict[str, float]:
    """Renormalize XGBoost and N-HiTS before the blend, not after an absence."""
    xgb = float(weights.get("xgb") or 0.0)
    nhits = float(weights.get("nhits") or 0.0)
    total = xgb + nhits
    if total <= 0.0:
        return {"xgb": 0.0, "prophet": 0.0, "nhits": 0.0}
    return {"xgb": xgb / total, "prophet": 0.0, "nhits": nhits / total}


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _cap(price: float, spot: float, horizon: int, caps: Optional[Dict[str, float]]) -> tuple[float, bool]:
    table = caps or {}
    max_pct = float(table.get(str(horizon), table.get(horizon, 20.0)))  # type: ignore[arg-type]
    central, _change, warning = resolve_capped_forecast(float(price), float(spot), max_pct)
    return float(central), bool(warning)


def _row_weights(row: pd.Series, weights_by_horizon: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    raw = row.get("weights_used")
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                return {str(key): float(value) for key, value in loaded.items()}
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return get_weights_for_horizon(int(row["horizon"]), weights_by_horizon, {"xgb": 0.0, "prophet": 0.0, "nhits": 0.0})


def variant_prices(
    frame: pd.DataFrame,
    weights_by_horizon: Dict[str, Dict[str, float]],
    caps: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """One price column per variant. A required engine that is missing stays empty."""
    out = frame.copy()
    columns = {name: [] for name in VARIANT_ORDER}
    warnings = {name: [] for name in VARIANT_ORDER}
    for _, row in out.iterrows():
        horizon = int(row["horizon"])
        spot = float(row["origin_price"])
        weights = _row_weights(row, weights_by_horizon)
        xgb = row.get("xgb_pred_direct", row.get("xgb_pred"))
        prophet = row.get("prophet_pred")
        nhits = row.get("nhits_pred")
        bare = {
            "cours_inchange": spot,
            "xgboost_direct": float(xgb) if _finite(xgb) else float("nan"),
            "nhits_seul": float(nhits) if _finite(nhits) else float("nan"),
            "ensemble_actuel": combine_ensemble(xgb, prophet, nhits, weights),
            SANS_PROPHET_NAME: combine_ensemble(
                xgb, prophet, nhits, weights_without_prophet(weights)
            ),
        }
        for name, price in bare.items():
            if not _finite(price) or float(price) <= 0:
                columns[name].append(float("nan"))
                warnings[name].append(False)
                continue
            central, warning = _cap(float(price), spot, horizon, caps)
            columns[name].append(central)
            warnings[name].append(warning)
    for name in VARIANT_ORDER:
        out[name] = columns[name]
        out[f"{name}_warning"] = warnings[name]
    return out


def _split(frame: pd.DataFrame, test_origins: int):
    if int(test_origins) > 0:
        return split_last_origins(frame, int(test_origins))
    return split_chronological(frame)


def score_variants(
    frame: pd.DataFrame,
    weights_by_horizon: Dict[str, Dict[str, float]],
    *,
    caps: Optional[Dict[str, float]] = None,
    acceptance: Optional[Dict[str, Any]] = None,
    test_origins: int = 0,
    step_size: int = 5,
    price_bounds: tuple[float, float] = (1000.0, 15000.0),
    coverage_level: float = 0.90,
) -> Dict[str, Any]:
    """Calibrate each variant on the past, then read only the same test slice."""
    rules = acceptance or load_acceptance("config/acceptance.json")
    priced = variant_prices(frame, weights_by_horizon, caps)
    calibration, test = _split(priced, test_origins)
    by_variant: Dict[str, Any] = {}
    for name in VARIANT_ORDER:
        margins = residual_margins(calibration, name, coverage_level) if not calibration.empty else {}
        banded = apply_frozen_bands(
            test, name, margins, "published_lower", "published_upper", price_bounds
        )
        naive_margins = residual_margins(calibration, "cours_inchange", coverage_level) if not calibration.empty else {}
        banded = apply_frozen_bands(
            banded, "cours_inchange", naive_margins, "naive_lower", "naive_upper", price_bounds
        )
        work = banded.copy()
        work["published_pred"] = pd.to_numeric(work[name], errors="coerce")
        work["naive_pred"] = pd.to_numeric(work["cours_inchange"], errors="coerce")
        work["engine_missing"] = ~np.isfinite(work["published_pred"].to_numpy(dtype=float))
        work["feature_failure"] = False
        work["large_move_warning"] = work.get(f"{name}_warning", False)
        horizons = {}
        for horizon in sorted(int(value) for value in work["horizon"].unique()) if not work.empty else []:
            decision = evaluate_horizon(
                work, horizon, rules, step=step_size,
                frozen_margin=margins.get(str(horizon)),
                frozen_naive_margin=naive_margins.get(str(horizon)),
            )
            decision["validated"] = False
            decision["measurement"] = "verification"
            horizons[str(horizon)] = decision
        by_variant[name] = {"horizons": horizons, "margins": margins}
    return {
        "note": SANS_PROPHET_NOTE,
        "measurement": "verification",
        "promoted": False,
        "variants": by_variant,
        "ranking": _ranking(by_variant),
        "partition": _partition(test),
    }


def _partition(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {"partition": "test", "n_origins": 0}
    return {
        "partition": "test",
        "origin_start": str(pd.to_datetime(frame["origin_date"]).min().date()),
        "origin_end": str(pd.to_datetime(frame["origin_date"]).max().date()),
        "target_start": str(pd.to_datetime(frame["target_date"]).min().date()),
        "target_end": str(pd.to_datetime(frame["target_date"]).max().date()),
        "n_origins": int(pd.to_datetime(frame["origin_date"]).nunique()),
    }


def _ranking(by_variant: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Name one winner only when a single gap interval excludes zero."""
    names = [name for name in VARIANT_ORDER if name != "cours_inchange"]
    horizons = set()
    for name in names:
        horizons.update((by_variant.get(name) or {}).get("horizons") or {})
    report: Dict[str, Dict[str, str]] = {}
    for horizon in sorted(horizons, key=lambda value: int(value)):
        better = []
        worse = []
        for name in names:
            decision = ((by_variant.get(name) or {}).get("horizons") or {}).get(horizon) or {}
            reading = decision.get("gap_reading")
            if reading == "amélioration étayée":
                better.append(name)
            elif reading == "dégradation étayée":
                worse.append(name)
        report[str(horizon)] = {
            "meilleure": better[0] if len(better) == 1 else "indécidable",
            "moins_bonne": worse[0] if len(worse) == 1 else "indécidable",
            "lecture": "décidée" if len(better) == 1 or len(worse) == 1 else "indécidable",
        }
    return report


def compare_previous_snapshot(previous_csv: str, current: pd.DataFrame) -> Dict[str, Any]:
    """Compare values still readable. Do not fill a missing training history."""
    previous = pd.read_csv(previous_csv)
    note = (
        "L'instantané complet du premier calcul n'a pas été conservé. "
        "Les séances d'entraînement ne sont pas garanties identiques. "
        "Seules les colonnes encore lisibles sont comparées. Aucun trou n'est comblé."
    )
    left = previous.copy()
    right = current.copy()
    left["_origin"] = pd.to_datetime(left["origin_date"]).dt.strftime("%Y-%m-%d")
    right["_origin"] = pd.to_datetime(right["origin_date"]).dt.strftime("%Y-%m-%d")
    left["_horizon"] = left["horizon"].astype(int)
    right["_horizon"] = right["horizon"].astype(int)
    shared = ["origin_price", "actual"]
    merged = left.merge(right, on=["_origin", "_horizon"], suffixes=("_ancien", "_nouveau"))
    differences = {}
    for column in shared:
        old_name = f"{column}_ancien"
        new_name = f"{column}_nouveau"
        if old_name not in merged.columns or new_name not in merged.columns:
            continue
        old_values = pd.to_numeric(merged[old_name], errors="coerce")
        new_values = pd.to_numeric(merged[new_name], errors="coerce")
        gap = (old_values - new_values).abs()
        differences[column] = {
            "rows": int(len(merged)),
            "different": int((gap > 1e-6).fillna(False).sum()),
        }
    return {
        "note": note,
        "previous_rows": int(len(previous)),
        "current_rows": int(len(current)),
        "common_rows": int(len(merged)),
        "differences": differences,
    }


def render_variant_summary(payload: Dict[str, Any]) -> str:
    """Local comparative summary. A winning variant does not replace the served model."""
    lines = [
        "# Comparaison des variantes — diagnostic",
        "",
        payload.get("note") or SANS_PROPHET_NOTE,
        "",
        "Aucune variante ne remplace le modèle affiché.",
        "",
    ]
    previous = payload.get("previous_snapshot") or {}
    if previous:
        lines.append(previous.get("note", ""))
        lines.append("")
    ranking = payload.get("ranking") or {}
    for horizon, row in ranking.items():
        lines.append(
            f"- J+{horizon} : meilleure {row.get('meilleure')} ; "
            f"moins bonne {row.get('moins_bonne')} ; {row.get('lecture')}"
        )
    lines.append("")
    return "\n".join(lines)
