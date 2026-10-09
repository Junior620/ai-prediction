"""Train a cocoa weather candidate and write a report.

The active release is not rewritten. Promotion stays false unless the MAE
gain, the MAPE, the fallback rate and the frozen J+30 close test all pass
on targets realized after 2026-10-02.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.hybrid_features import (
    DEFAULT_XGB_PARAMS,
    FEATURE_COLS,
    PROPHET_LEVEL_COLS,
    fit_prophet,
    future_business_date,
    load_price_data_from_supabase,
    prepare_training_frame,
    price_at_date,
    resolve_feature_cols,
)
from src.models.weather_features import FEATURE_COLS_WEATHER, attach_weather, build_weather_features
from src.data_collection.cocoa_weather import blend_country_daily
from src.validation.acceptance import evaluate_horizon, load_acceptance
from src.validation.weather_candidate import (
    SPLIT_DATE,
    UNUSED_TARGET_AFTER,
    WEATHER_HORIZONS,
    decide_weather_promotion,
    score_predictions,
)

LOCAL_POINTS = ROOT / "data" / "cocoa_weather_daily.csv"
LOCAL_BLEND = ROOT / "data" / "cocoa_weather_blend.csv"


def _load_blend() -> pd.DataFrame:
    if LOCAL_BLEND.exists():
        frame = pd.read_csv(LOCAL_BLEND, parse_dates=["date"])
        return frame
    if LOCAL_POINTS.exists():
        return blend_country_daily(pd.read_csv(LOCAL_POINTS))
    raise FileNotFoundError(
        "Historique météo absent. Lancer scripts/backfill_cocoa_weather.py"
    )


def _load_prices():
    from supabase import create_client

    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    return load_price_data_from_supabase(supabase, table_name="cocoa_london_prices")


def _fit_horizon(featured: pd.DataFrame, feature_cols: list, horizon: int, split: pd.Timestamp):
    rows = []
    targets = []
    train = featured[featured["date"] < split]
    for _, row in train.iterrows():
        target_day = pd.Timestamp(future_business_date(row["date"], horizon)).normalize()
        if target_day >= split:
            continue
        actual = price_at_date(featured, target_day)
        if actual is None:
            continue
        values = pd.to_numeric(row[feature_cols], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            continue
        rows.append(values.to_numpy(dtype=float))
        targets.append(float(actual) - float(row["price"]))
    if len(rows) < 30:
        raise RuntimeError(f"Pas assez de lignes d'entraînement pour J+{horizon} ({len(rows)})")
    model = xgb.XGBRegressor(**DEFAULT_XGB_PARAMS)
    model.fit(np.vstack(rows), np.asarray(targets), verbose=False)
    return model


def _predict_horizon(model, featured: pd.DataFrame, feature_cols: list, horizon: int, split: pd.Timestamp):
    records = []
    test = featured[featured["date"] >= split]
    for _, row in test.iterrows():
        target_day = pd.Timestamp(future_business_date(row["date"], horizon)).normalize()
        actual = price_at_date(featured, target_day)
        if actual is None:
            continue
        values = pd.to_numeric(row[feature_cols], errors="coerce").to_numpy(dtype=float)
        origin = float(row["price"])
        if not np.isfinite(values).all():
            predicted = float("nan")
        else:
            predicted = origin + float(model.predict(values.reshape(1, -1))[0])
        records.append(
            {
                "horizon": horizon,
                "origin_date": pd.Timestamp(row["date"]),
                "target_date": target_day,
                "actual": float(actual),
                "origin_price": origin,
                "naive_pred": origin,
                "published_pred": predicted,
                "predicted": predicted,
                "feature_failure": not np.isfinite(predicted),
            }
        )
    return records


def _acceptance_j30(records: list, acceptance: dict) -> bool:
    frame = pd.DataFrame(records)
    if frame.empty:
        return False
    cutoff = pd.Timestamp(UNUSED_TARGET_AFTER)
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    calibration = frame[(frame["horizon"] == 30) & (frame["target_date"] <= cutoff)]
    held = frame[(frame["horizon"] == 30) & (frame["target_date"] > cutoff)].copy()
    if held.empty or calibration.empty:
        return False
    from src.models.market_registry import get_market_config
    from src.validation.served_backtest import apply_frozen_bands, residual_margins

    bounds = get_market_config("cocoa").price_bounds
    margins = residual_margins(calibration, "published_pred", 0.90)
    naive = residual_margins(calibration, "naive_pred", 0.90)
    held = apply_frozen_bands(held, "published_pred", margins, "published_lower", "published_upper", bounds)
    held = apply_frozen_bands(held, "naive_pred", naive, "naive_lower", "naive_upper", bounds)
    decision = evaluate_horizon(
        held,
        30,
        acceptance,
        frozen_margin=margins.get("30"),
        frozen_naive_margin=naive.get("30"),
    )
    return bool(decision.get("validated"))


def run(split: str = SPLIT_DATE) -> dict:
    prices = _load_prices()
    weather = build_weather_features(_load_blend(), climatology_end=split)
    attached = attach_weather(prices, weather)
    for column in ("open", "high", "low", "volume", "open_interest"):
        if column in attached.columns:
            attached[column] = pd.to_numeric(attached[column], errors="coerce").ffill()

    split_ts = pd.Timestamp(split)
    prophet = fit_prophet(attached.loc[attached["date"] < split_ts, ["date", "price"]])
    featured, _ = prepare_training_frame(attached, prophet_model=prophet)
    for column in PROPHET_LEVEL_COLS:
        if column in featured.columns:
            featured[column] = 0.0
    featured["date"] = pd.to_datetime(featured["date"]).dt.normalize()

    current_cols = resolve_feature_cols(include_ohlcv=True, include_oi=True)
    weather_cols = resolve_feature_cols(include_ohlcv=True, include_oi=True, include_weather=True)
    for column in weather_cols:
        if column not in featured.columns:
            featured[column] = np.nan
    # Same as production: a missing microstructure column becomes 0.
    # Weather stays missing until a real observation exists.
    microstructure = [
        column
        for column in current_cols
        if column not in FEATURE_COLS and column not in FEATURE_COLS_WEATHER
    ]
    featured[microstructure] = featured[microstructure].ffill().fillna(0.0)

    per_horizon = []
    weather_models = {}
    current_records = []
    weather_records = []
    for horizon in WEATHER_HORIZONS:
        current_model = _fit_horizon(featured, current_cols, horizon, split_ts)
        weather_model = _fit_horizon(featured, weather_cols, horizon, split_ts)
        weather_models[horizon] = weather_model
        current_rows = _predict_horizon(current_model, featured, current_cols, horizon, split_ts)
        weather_rows = _predict_horizon(weather_model, featured, weather_cols, horizon, split_ts)
        current_records.extend(current_rows)
        weather_records.extend(weather_rows)
        current_score = score_predictions(
            [row["actual"] for row in current_rows],
            [row["predicted"] for row in current_rows],
            [row["naive_pred"] for row in current_rows],
            [row["target_date"] for row in current_rows],
        )
        weather_score = score_predictions(
            [row["actual"] for row in weather_rows],
            [row["predicted"] for row in weather_rows],
            [row["naive_pred"] for row in weather_rows],
            [row["target_date"] for row in weather_rows],
        )
        per_horizon.append(
            {
                "horizon": horizon,
                "current_mae": current_score["mae"],
                "candidate_mae": weather_score["mae"],
                "current_mape": current_score["mape"],
                "candidate_mape": weather_score["mape"],
                "current_fallback": current_score["fallback_rate"],
                "candidate_fallback": weather_score["fallback_rate"],
                "n": weather_score["n"],
                "n_after_cutoff": weather_score["n_after_cutoff"],
            }
        )
        print(
            f"J+{horizon} MAE actuel {current_score['mae']} | "
            f"météo {weather_score['mae']} | n={weather_score['n']} | "
            f"après {UNUSED_TARGET_AFTER}: {weather_score['n_after_cutoff']}"
        )

    acceptance = load_acceptance(str(ROOT / "config" / "acceptance.json"))
    j30_ok = _acceptance_j30(weather_records, acceptance)
    decision = decide_weather_promotion(per_horizon, acceptance_j30=j30_ok)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = ROOT / "models" / "candidates" / "cocoa" / f"weather_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    for horizon, model in weather_models.items():
        with open(folder / f"xgboost_weather_h{horizon}_{stamp}.pkl", "wb") as handle:
            pickle.dump(model, handle)
    with open(folder / f"prophet_weather_{stamp}.pkl", "wb") as handle:
        pickle.dump(prophet, handle)
    info = {
        "timestamp": stamp,
        "market": "cocoa",
        "feature_set": "m3_weather",
        "feature_cols": weather_cols,
        "weather_cols": FEATURE_COLS_WEATHER,
        "split_date": split,
        "unused_target_after": UNUSED_TARGET_AFTER,
        "horizons": list(WEATHER_HORIZONS),
        "per_horizon": per_horizon,
        "acceptance_j30": j30_ok,
        "promote": bool(decision["promote"]),
        "manifest_updated": False,
        "decision": decision,
        "note": "Candidat météo. Le manifeste actif n'est pas modifié.",
    }
    (folder / f"model_info_weather_{stamp}.json").write_text(
        json.dumps(info, indent=2, default=str),
        encoding="utf-8",
    )

    report_dir = ROOT / "reports" / "weather_candidate"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{stamp}.json"
    report_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    lines = [
        f"# Candidat météo cacao {stamp}",
        "",
        f"Split {split}. Climatologie figée avant cette date.",
        "Le manifeste actif n'est pas modifié.",
        "",
        f"Promotion : {decision['promote']}",
    ]
    for reason in decision["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    for row in per_horizon:
        lines.append(
            f"- J+{row['horizon']}: MAE actuel {row['current_mae']}, "
            f"météo {row['candidate_mae']}, n={row['n']}, "
            f"cibles après {UNUSED_TARGET_AFTER}: {row['n_after_cutoff']}"
        )
    (report_dir / f"{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Candidat {folder}")
    print(f"[OK] Rapport {report_path}")
    print(f"[OK] Manifeste inchangé. Promotion={decision['promote']}")
    return info


if __name__ == "__main__":
    run()
