"""Second audit on 2026-10-13: local reads and isolated mock/temp counterexamples.

No training, production journal writes, network requests or source changes.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import runpy
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import pandas as pd

from src.models.hybrid_features import future_business_date
from src.models.prediction_journal import append_forecast, load_forecasts, overlay_commitments
from src.validation.revision_value import (
    evaluate_records, load_daily_records, load_prices, paired_expiries, summarise_horizon,
)


def emit(check, **data):
    print(json.dumps({"check": check, **data}, ensure_ascii=False, default=str), flush=True)


def day(origin, sessions):
    return future_business_date(origin, sessions).strftime("%Y-%m-%d")


def append(path, *, origin, horizon, price, version, lower=None, upper=None):
    return append_forecast(
        market="ICE_NY", model_version=version, release_mode="experimental",
        horizon=horizon, price=price, lower=lower, upper=upper,
        origin_date=origin, origin_price=4200.0, target_date=day(origin, horizon),
        status="experimental", procedure_fingerprint="audit", data_fingerprint=version,
        path=path,
    )


def study_comparison():
    for market in ("cocoa", "coffee_robusta"):
        directory = ROOT / "reports/revision_stability" / market / "study_corrected"
        summary = json.loads(sorted(directory.glob("*_summary.json"))[-1].read_text(encoding="utf-8"))
        cutoff = summary["as_of"]
        records = load_daily_records(directory / "sessions.jsonl")
        prices = load_prices(ROOT / "data/freeze" / f"{market}_prices.csv")
        pairs = paired_expiries(records, prices, cutoff)
        groups = defaultdict(list)
        for row in records:
            if row["role"] == "revision" and row.get("price") is not None:
                groups[(int(row["initial_horizon"]), row["target"])].append(row)
        detail = []
        for horizon, rows in pairs.items():
            first_rows = []
            first_horizons, last_horizons = [], []
            naive_latest = []
            for row in rows:
                revisions = [r for r in groups[(horizon, row["target"])] if r["session"] > row["session"]]
                first = min(revisions, key=lambda r: r["session"])
                last = max(revisions, key=lambda r: r["session"])
                first_error = abs(float(first["price"]) - row["realized"])
                first_rows.append({**row, "latest_error": first_error, "gain": row["initial_error"] - first_error})
                first_horizons.append(int(first["horizon"]))
                last_horizons.append(int(last["horizon"]))
                if last.get("spot") is not None:
                    naive_latest.append(abs(float(last["spot"]) - row["realized"]))
            detail.append({
                "horizon": horizon,
                "initial_vs_first_revision": summarise_horizon(first_rows, horizon),
                "initial_vs_final_revision": summarise_horizon(rows, horizon),
                "first_remaining_horizons": dict(Counter(first_horizons)),
                "last_remaining_horizons": dict(Counter(last_horizons)),
                "mean_naive_at_last_revision": sum(naive_latest) / len(naive_latest) if naive_latest else None,
            })
        emit("historical_revision_comparison", market=market, as_of=cutoff, comparisons=detail)


def bootstrap_counterexample():
    rows = [
        {"session": str(i), "target": str(i), "gain": gain,
         "initial_error": 100.0, "latest_error": 100.0-gain}
        for i, gain in enumerate([10.0] * 7 + [-60.0])
    ]
    emit("bootstrap_short_sample", result=summarise_horizon(rows, 14))


def journal_counterexamples():
    from src.models.served_forecast import publish_or_close
    with tempfile.TemporaryDirectory(prefix="prediction-audit-v2-") as folder:
        path = Path(folder) / "journal.jsonl"
        origin = "2026-10-01"
        original = append(path, origin=origin, horizon=30, price=4500.0, version="day0")
        append(path, origin=origin, horizon=14, price=4400.0, version="day0")
        append(path, origin=day(origin, 14), horizon=14, price=4350.0, version="day14")
        attempted = append(path, origin=day(origin, 16), horizon=14, price=4100.0, version="day16")
        record = next(r for r in load_forecasts(path) if r["forecast_id"] == attempted)
        emit("lost_same_target_revision", initial_id=original, same_target=record["target_date"] == day(origin, 30),
             attempted_revision={k: record.get(k) for k in ("horizon", "origin_date", "target_date", "price", "published", "diagnostic", "revises")})

        path2 = Path(folder) / "bands.jsonl"
        append(path2, origin=origin, horizon=7, price=4500.0, version="opening")
        rows = load_forecasts(path2)
        fresh = {"horizon": 7, "price": 4000.0, "status": "experimental", "confidence_interval": [3900.0, 4100.0],
                 "components": {"origin_date": day(origin, 1), "target_date": day(origin, 8)}}
        held = overlay_commitments([fresh], rows, market="ICE_NY", as_of=day(origin, 1))[0]
        emit("mixed_forecast_band", overlaid=held)
        failure = {"horizon": 7, "price": None, "status": "unavailable", "confidence_interval": None,
                   "components": {"calculation_failed": True, "engine_missing": True}}
        held_failure = overlay_commitments([failure], rows, market="ICE_NY", as_of=day(origin, 1))[0]
        cached = SimpleNamespace(**held_failure)
        publish_or_close([cached], {"release_mode": "experimental"}, None, release_mode="experimental")
        emit("held_price_cache_inconsistency", before_cache={"price": held_failure["price"], "status": held_failure["status"]},
             after_cache={"price": cached.price, "status": cached.status})


def command_counterexamples():
    from src.api.models import PredictionItem
    from scripts.verify_served_forecast import assess_served
    item = PredictionItem(horizon=1, price=4300.0, confidence_interval=[4200., 4400.], confidence_level=.9,
                          timestamp=datetime(2026, 10, 13), components={"target_date": "2026-10-14"})
    predictions = [item.model_dump()]
    predictions.extend({"horizon": h, "price": 4300.0, "status": "experimental", "target_date": day("2026-10-13", h)} for h in (7, 14, 30))
    code, lines = assess_served({"model_version": "new", "current_date": "2026-10-13", "predictions": predictions}, "new", "2026-10-13")
    emit("j1_target_contract", top_level_target=item.target_date, component_target=item.components["target_date"], code=code, lines=lines)

    from src.data_collection import investing_price_collector
    mocked = {"price": 3475., "symbol": "RCU6", "session_date": None,
              "collected_at": "2026-10-13T12:00:00", "source": "investing_com", "price_kind": "intraday", "quarantine_reason": "date_de_seance_absente"}
    table = MagicMock()
    table.select.return_value = table
    table.order.return_value = table
    table.limit.return_value = table
    table.execute.return_value = SimpleNamespace(data=[])
    client = MagicMock()
    client.table.return_value = table
    output = io.StringIO()
    error = None
    with patch("supabase.create_client", return_value=client), patch.object(investing_price_collector, "fetch_investing_price", return_value=mocked), contextlib.redirect_stdout(output):
        try:
            runpy.run_path(str(ROOT / "collect_coffee_robusta_price.py"), run_name="__main__")
        except (KeyError, SystemExit) as exc:
            error = f"{type(exc).__name__}: {exc}"
    emit("robusta_collection_contract", error=error, database_write_attempted=table.insert.called)


def history_and_preparation():
    from src.validation.report_index import list_market_reports
    cocoa = list_market_reports("cocoa", ROOT)
    robusta = list_market_reports("coffee_robusta", ROOT)
    wrong = [r["report_id"] for r in cocoa if "coffee_robusta" in r["report_id"]]
    emit("history_market_assignment", robusta_in_cocoa=wrong, absent_from_robusta=[r for r in wrong if r not in {x["report_id"] for x in robusta}])
    from src.models.hybrid_features import prepare_market_frame, fill_missing_ohlcv, session_anomalies, trading_sessions
    from src.models.improved_price_predictor import ImprovedPricePredictor
    raw = prepare_market_frame(pd.read_csv(ROOT / "data/freeze/cocoa_prices.csv"), market_id="cocoa")
    predictor = ImprovedPricePredictor.__new__(ImprovedPricePredictor)
    predictor._frozen_frame = raw
    actual = predictor._fetch_historical_data()
    expected = fill_missing_ohlcv(raw)
    pd.testing.assert_frame_equal(actual, expected)
    emit("preparation_fix_and_calendar", frozen_matches_api_fill=True, service_rows=len(raw), study_rows=len(trading_sessions(raw)),
         closed_dates=session_anomalies(raw), raw_vs_served_fingerprints_distinct=predictor._data_fingerprint != predictor._served_fingerprint)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    study_comparison()
    bootstrap_counterexample()
    journal_counterexamples()
    command_counterexamples()
    history_and_preparation()
