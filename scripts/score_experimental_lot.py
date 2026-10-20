"""Score the named cocoa lot against the unchanged close and revision rules.

The served price is not replaced. k and the revision rule are fit only on the
calibration targets. This script is not part of the evening update.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.hybrid_features import fill_missing_ohlcv, prepare_market_frame, trading_sessions
from src.models.market_registry import get_market_config
from src.models.revision_command import load_named_predictor
from src.validation.lot_variants import (
    _choose_k,
    _choose_policy,
    _level_errors,
    _policy_score,
    _split,
)

HORIZONS = (1, 7, 14, 30)
IN_SAMPLE_NOTE = (
    "Diagnostic d'ajustement : les modèles ont été entraînés sur les cibles notées. "
    "Ce résultat ne mesure pas une performance future et ne valide pas le lot. "
    "Aucune variante ne remplace le prix servi."
)


def _usable(item) -> bool:
    parts = dict(item.components or {})
    if parts.get("calculation_failed") or item.price is None:
        return False
    try:
        return float(item.price) > 0 and bool(parts.get("target_date"))
    except (TypeError, ValueError):
        return False


def _forecasts(predictor, frame: pd.DataFrame, dates: list[pd.Timestamp]) -> list[dict]:
    found = []
    for day in dates:
        history = frame.loc[pd.to_datetime(frame["date"]) <= day].copy()
        predictor.bind_frame(history)
        try:
            predictions = predictor.predict(list(HORIZONS))
        except Exception as exc:
            print(f"[AVERTISSEMENT] {day.date()} {exc}")
            continue
        spot = float(history["price"].iloc[-1])
        session = day.date().isoformat()
        for item in predictions:
            if not _usable(item):
                continue
            parts = dict(item.components or {})
            target = str(parts.get("target_date"))[:10]
            found.append(
                {
                    "session": session,
                    "target": target,
                    "horizon": int(item.horizon),
                    "price": float(item.price),
                    "spot": spot,
                    "reference": spot,
                }
            )
        print(f"[OK] {session} prévisions={len(found)}", flush=True)
    return found


def _chains(rows: list[dict], lookup: dict[str, float]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        realized = lookup.get(row["target"])
        if realized is None:
            continue
        copied = dict(row)
        copied["realized"] = realized
        grouped.setdefault(row["target"], []).append(copied)
    chained = []
    for chain in grouped.values():
        opening = max(chain, key=lambda item: int(item["horizon"]))
        for item in chain:
            item["initial_horizon"] = int(opening["horizon"])
            item["reference"] = float(opening["spot"])
            item["role"] = "initial" if item is opening else "revision"
        chained.extend(chain)
    return chained


def _score_horizon(rows: list[dict], horizon: int) -> dict:
    opening = [row for row in rows if int(row["horizon"]) == horizon and row["role"] == "initial"]
    chains = [row for row in rows if int(row["initial_horizon"]) == horizon]
    calibration, holdout = _split(chains, 0.25, key="target")
    cal_open = [row for row in calibration if row["role"] == "initial" and int(row["horizon"]) == horizon]
    hold_open = [row for row in holdout if row["role"] == "initial" and int(row["horizon"]) == horizon]
    if not cal_open:
        calibration_open, holdout_open = _split(opening, 0.25, key="target")
        cal_open, hold_open = calibration_open, holdout_open
    k = _choose_k(cal_open)
    policy = _choose_policy(calibration)
    return {
        "k": k,
        "revision_policy": policy,
        "chosen_on": "calibration",
        "calibration": {
            "levels": _level_errors(cal_open, k),
            "policies": [_policy_score(calibration, name) for name in ("initial", "previous", "latest")],
        },
        "holdout": {
            "levels": _level_errors(hold_open, k),
            "policies": [_policy_score(holdout, name) for name in ("initial", "previous", "latest")],
        },
    }


def main() -> int:
    release_path = ROOT / "config" / "experimental_release_cocoa.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("validated") or release.get("release_mode") != "experimental":
        print("[ERREUR] Le lot à noter doit rester expérimental et non validé.")
        return 1
    if release.get("calendar") != "trading_sessions":
        print("[ERREUR] Ce rejeu attend le lot préparé sur trading_sessions.")
        return 1
    raw = prepare_market_frame(pd.read_csv(ROOT / "data" / "freeze" / "cocoa_prices.csv"), market_id="cocoa")
    frame = fill_missing_ohlcv(trading_sessions(raw))
    dates = list(pd.to_datetime(frame["date"]).dt.normalize().drop_duplicates().sort_values())
    lookup = {pd.Timestamp(row.date).date().isoformat(): float(row.price) for row in frame.itertuples(index=False)}
    # Score only origins whose J+30 target is already known, and keep the last ones.
    matured = []
    from src.models.hybrid_features import future_business_date

    for day in dates:
        target = future_business_date(day, 30).date().isoformat()
        if target in lookup:
            matured.append(day)
    chosen = matured[-48:]
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as handle:
        pred_cfg = (yaml.safe_load(handle) or {}).get("prediction") or {}
    predictor = load_named_predictor(release, get_market_config("cocoa"), pred_cfg)
    predictor.active_release = release
    rows = _chains(_forecasts(predictor, frame, chosen), lookup)
    report = {
        "market": "cocoa",
        "version": release.get("version"),
        "calendar": "trading_sessions",
        "release_mode": "experimental",
        "validated": False,
        "evaluation_kind": "in_sample_diagnostic",
        "served_price": "direct_model",
        "variant_replaces_served_price": False,
        "note": IN_SAMPLE_NOTE,
        "origins": [day.date().isoformat() for day in chosen],
        "horizons": {str(horizon): _score_horizon(rows, horizon) for horizon in HORIZONS},
    }
    output = ROOT / "reports" / "revision_stability" / "cocoa" / "experimental_lot"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "variant_comparison.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
