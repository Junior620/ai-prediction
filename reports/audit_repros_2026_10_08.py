"""Local, synthetic audit reproductions. No database, network, or production writes.

Run from the repository with venv_py311/Scripts/python.exe.
Temporary release files are isolated in a TemporaryDirectory.
"""
from __future__ import annotations

import ast
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.models.hybrid_features import future_business_date
from src.models.improved_price_predictor import ImprovedPricePredictor
from src.models.release_manifest import load_active_release, promote_accepted_horizons
from src.models.served_forecast import compose_served_price, publish_or_close
from src.models.weather_features import attach_weather, build_weather_features
from src.validation.acceptance import evaluate_horizon, load_acceptance


def frame(n=1, horizon=1, actual=110.0, published=109.0):
    origins = pd.bdate_range("2027-01-04", periods=n)
    return pd.DataFrame({
        "origin_date": origins,
        "target_date": origins + pd.offsets.BDay(horizon),
        "horizon": horizon, "origin_price": 100.0, "actual": actual,
        "naive_pred": 100.0, "published_pred": published, "feature_failure": False,
    })


def decision(rows, horizon=1, model_margin=10.0, naive_margin=20.0):
    return evaluate_horizon(
        rows, horizon, load_acceptance(str(ROOT / "config/acceptance.json")),
        frozen_margin={"margin_lower": model_margin, "margin_upper": model_margin},
        frozen_naive_margin={"margin_lower": naive_margin, "margin_upper": naive_margin},
    )


def emit(name, values):
    print(json.dumps({"case": name, **values}, ensure_ascii=False, default=str))


def isolated_function(path, name, globals_dict):
    """Execute the actual function body without running application startup."""
    parsed = ast.parse(Path(path).read_text(encoding="utf-8"))
    node = next(n for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), globals_dict)
    return globals_dict[name]


one = decision(frame())
assert one["validated"] is True and one["n"] == 1
emit("one_observation_accepted", {k: one[k] for k in ("validated", "n", "n_effective", "gap_ci")})

missing = frame(100, published=110.0)
missing.loc[:79, "published_pred"] = np.nan
missing_result = decision(missing)
assert missing_result["validated"] and missing_result["fallback_rate"] == 0
emit("missing_forecasts_not_counted_as_fallback", {
    "missing_share": float(missing.published_pred.isna().mean()),
    **{k: missing_result[k] for k in ("validated", "fallback_rate", "relative_gain")},
})

served = compose_served_price(
    xgb_price=110.0, prophet_price=110.0, nhits_price=None,
    weights={"xgb": 1.0}, spot=100.0, horizon=30, price_bounds=(1.0, 120.0),
    conformal_margins={"30": {"margin_lower": 20.0, "margin_upper": 20.0}},
)
band_result = decision(frame(40, 30, actual=125.0, published=served["price"]), 30, 20.0, 30.0)
assert band_result["validated"] and not (served["lower"] <= 125 <= served["upper"])
emit("acceptance_band_differs_from_served_band", {
    "accepted": band_result["validated"], "gate_coverage": band_result["coverage"],
    "served_interval": [served["lower"], served["upper"]], "actual": 125.0,
    "served_coverage": 0.0,
})

with tempfile.TemporaryDirectory(prefix="prediction_audit_") as temp:
    root = Path(temp)
    (root / "config").mkdir()
    manifest = {"market": "cocoa", "version": "old", "validated": False,
                "artifacts": {"nhits": "models/nhits_old", "direct_info": "models/direct_old.json"},
                "horizons": {"1": {"validated": False}}}
    (root / "config/active_release_cocoa.json").write_text(json.dumps(manifest), encoding="utf-8")
    for stamp in ("20270101_100000", "20270102_100000"):
        folder = root / "models/candidates/cocoa" / stamp
        folder.mkdir(parents=True)
        for engine in ("prophet", "xgboost"):
            (folder / f"{engine}_improved_{stamp}.pkl").write_bytes(b"synthetic placeholder; never loaded")
    promote_accepted_horizons("cocoa", ["1"], "report_for_20270101_100000", root=root)
    release = load_active_release("cocoa", root=root)
    assert release["version"] == "improved_20270102_100000"
    emit("report_not_bound_to_promoted_candidate", {
        "report": release["source_report"], "promoted_version": release["version"],
        "retained_nhits": release["artifacts"]["nhits"],
        "retained_direct_info": release["artifacts"]["direct_info"],
    })

    metadata = root / "metadata"
    metadata.mkdir()
    (metadata / "model_info_direct_horizon_20270101.json").write_text(json.dumps({"target": "close_delta"}), encoding="utf-8")
    (metadata / "model_info_direct_horizon_20270102.json").write_text(json.dumps({"target": "price"}), encoding="utf-8")
    direct = Mock()
    direct.predict.return_value = np.array([5.0])
    with patch("src.models.improved_price_predictor.os.getenv", return_value=None):
        predictor = ImprovedPricePredictor(
            None, None, None, direct_horizon_models={1: direct}, models_dir=str(metadata),
            ensemble_weights_file=str(root / "absent_weights.json"),
            conformal_intervals_file=str(root / "absent_margins.json"),
            feature_cols=["price_lag_1"],
        )
    price = predictor._predict_xgb_price(1, pd.DataFrame(), pd.Series({"price_lag_1": 99}), 100.0, pd.Timestamp("2027-01-04"))
    assert price == 5.0
    emit("active_direct_uses_latest_metadata", {"expected_delta_level": 105.0, "actual_level": price, "target_loaded": predictor.direct_target})

old_item = SimpleNamespace(horizon=1, price=105.0, confidence_interval=[95, 115], components={})
old_response = SimpleNamespace(model_version="old_A", current_date="2027-01-04", current_price=100.0, predictions=[old_item])
new_release = {"validated": True, "version": "new_B", "horizons": {"1": {"validated": True}}}
cache = Mock()
cache.get_prediction.return_value = old_response
namespace = {
    "Depends": lambda x: None, "verify_token": None, "logger": Mock(),
    "resolve_api_market": lambda market: SimpleNamespace(market_id="cocoa", price_table="prices"),
    "predictors": {"cocoa": SimpleNamespace(model_version="new_B", active_release=new_release)},
    "redis_cache": cache, "_latest_price_date": lambda table: "2027-01-04",
}
predict = isolated_function(ROOT / "src/api/app.py", "predict_price", namespace)
response = asyncio.run(predict(SimpleNamespace(market="ICE_NY", horizons=[1], include_sentiment=False)))
assert response.model_version == "old_A" and response.predictions[0].components["served_as_close"] is False
emit("old_cache_served_under_new_release", {"loaded_version": "new_B", "returned_version": response.model_version,
                                        "served_as_close": response.predictions[0].components["served_as_close"]})

candidate = SimpleNamespace(horizon=1, price=110.0, confidence_interval=[90.0, 120.0], components={})
publish_or_close([candidate], {"validated": False}, 100.0)
assert candidate.components["candidate_price"] == 110.0 and candidate.confidence_interval == [100.0, 100.0]
emit("candidate_band_lost_before_journal", {"scored_candidate": candidate.components["candidate_price"],
                                         "journal_band": candidate.confidence_interval,
                                         "candidate_band_expected": [90.0, 120.0]})

landing = future_business_date("2026-04-02", 1).date().isoformat()
assert landing == "2026-04-03"
emit("holiday_target", {"origin": "2026-04-02", "horizon": 1, "computed_target": landing,
                        "ICE_next_agricultural_session": "2026-04-07"})

dates = pd.date_range("2023-01-01", "2025-01-08")
weather = pd.DataFrame({"date": dates, "precip_mm": 1.0, "temp_c": 25.0})
features = build_weather_features(weather, "2025-01-01")
before = attach_weather(pd.DataFrame({"date": ["2025-01-08"], "price": [100.0]}), features)
weather.loc[weather.date == pd.Timestamp("2025-01-08"), "precip_mm"] = 101.0
after = attach_weather(pd.DataFrame({"date": ["2025-01-08"], "price": [100.0]}), build_weather_features(weather, "2025-01-01"))
assert before.rain_sum_7.iloc[0] != after.rain_sum_7.iloc[0]
emit("weather_uses_same_day_unavailable_daily_total", {"origin": "2025-01-08", "rain_sum_before": float(before.rain_sum_7.iloc[0]),
                                                       "rain_sum_after": float(after.rain_sum_7.iloc[0])})
