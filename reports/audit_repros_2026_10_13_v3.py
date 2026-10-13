"""Read-only audit: archived data and isolated counterexamples; no training or serving writes."""
from __future__ import annotations

import ast
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from src.validation.revision_value import (
    _opening_groups, delay_steps, load_daily_records, load_prices,
    revision_value_for_market,
)
from src.models.prediction_journal import (
    append_forecast, load_forecasts, overlay_commitments, sessions_between,
)
from src.models.hybrid_features import future_business_date


def emit(check, **data):
    print(json.dumps({"check": check, **data}, ensure_ascii=False, default=str), flush=True)


def study_metrics():
    for market in ("cocoa", "coffee_robusta"):
        study = ROOT / "reports/revision_stability" / market / "study_corrected"
        records = load_daily_records(study / "sessions.jsonl")
        prices = load_prices(ROOT / "data/freeze" / f"{market}_prices.csv")
        payload = revision_value_for_market(market)
        unique = list({json.dumps(row, sort_keys=True): row for row in records}.values())
        dedup = delay_steps(unique, prices, payload["as_of"])
        initials, revisions = _opening_groups(records)
        summaries = []
        for item in payload["horizons"]:
            h = item["horizon"]
            mature = [r for (hh, target), r in initials.items()
                      if hh == h and target <= payload["as_of"] and r.get("price") is not None]
            observable = [r for r in mature if r["target"] in prices]
            selected = [s for s in item["steps"]
                        if s["remaining"] in {h - 1, h - 2, 9}]
            selected_dedup = [s for s in dedup[h]
                              if s["remaining"] in {h - 1, h - 2, 9}]
            summaries.append({"horizon": h, "mature_openings": len(mature),
                              "with_realized_price": len(observable),
                              "raw_steps": selected, "dedup_steps": selected_dedup,
                              "final_comparison": {k: v for k, v in item.items() if k != "steps"}})
        batches = [json.loads(line) for line in (study / "sessions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        emit("archived_metrics", market=market, as_of=payload["as_of"],
             batches=len(batches), unique_sessions=len({r["session"] for r in batches}),
             records=len(records), unique_records=len(unique), summaries=summaries)


def absent_revision():
    records = [
        {"policy": "daily", "role": "initial", "initial_horizon": 7, "horizon": 7,
         "session": "2026-06-01", "target": "2026-06-10", "price": 110.0},
        {"policy": "daily", "role": "initial", "initial_horizon": 7, "horizon": 7,
         "session": "2026-06-02", "target": "2026-06-11", "price": 110.0},
        {"policy": "daily", "role": "revision", "initial_horizon": 7, "horizon": 6,
         "session": "2026-06-02", "target": "2026-06-10", "price": 105.0, "spot": 100.0},
    ]
    scores = delay_steps(records, {"2026-06-10": 100.0, "2026-06-11": 100.0}, "2026-06-30", draws=20)
    emit("absent_revision", matured_initials=2, present_revisions=1, result=scores[7])


def wrong_targets():
    from scripts.verify_served_forecast import assess_served
    for target in ("2026-10-01", "2026-99-99"):
        payload = {"model_version": "audit-lot", "current_date": "2026-10-13",
                   "predictions": [{"horizon": h, "fresh": {"price": 4200.0,
                       "status": "experimental", "target_date": target}}
                       for h in (1, 7, 14, 30)]}
        code, lines = assess_served(payload, "audit-lot", "2026-10-13")
        emit("verifier_accepts_wrong_target", target=target, missing_per_item_origin_and_version=True,
             code=code, lines=lines)


def rejected_band():
    # Compile the current API helper alone to avoid importing/startup of the app.
    from src.models.served_forecast import publish_or_close
    import src.models.prediction_journal as journal
    source = ROOT / "src/api/app.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_journal_forecasts")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"Optional": __import__("typing").Optional, "logger": logging.getLogger("audit"),
                 "_mark_journal_unavailable": lambda exc: None}
    exec(compile(module, str(source), "exec"), namespace)
    item = SimpleNamespace(horizon=14, price=4500.0, confidence_interval=[4100.0, 4300.0],
                           status="experimental", components={"origin_date": "2026-10-13",
                           "origin_price": 4200.0, "target_date": "2026-11-02"})
    publish_or_close([item], {"release_mode": "experimental"}, None, release_mode="experimental")
    captured = []
    with patch.object(journal, "append_forecast", side_effect=lambda **kw: captured.append(kw) or "audit"):
        namespace["_journal_forecasts"]([item], market="ICE_NY", model_version="audit",
                                        release_mode="experimental")
    emit("api_restores_rejected_band", served_band=item.confidence_interval,
         captured_journal={k: captured[0].get(k) for k in ("price", "lower", "upper", "status")})


def dated_collection():
    from src.data_collection.investing_price_collector import fetch_investing_price
    from collect_coffee_robusta_price import decide_robusta_quote
    with patch("curl_cffi.requests.get", return_value=SimpleNamespace(
        status_code=200, text='<title>London Coffee (RCU6)</title><span data-test="instrument-price-last">3,453.26</span>'
    )):
        result = fetch_investing_price("https://example.invalid/audit")
    emit("robusta_successful_html_still_quarantined", quote=result, decision=decide_robusta_quote(result))


def stale_same_target():
    def add(path, origin, h, price, version):
        return append_forecast(market="ICE_NY", model_version=version, release_mode="experimental",
            horizon=h, price=price, lower=None, upper=None, origin_date=origin,
            origin_price=4200.0, target_date=future_business_date(origin, h).date().isoformat(),
            status="experimental", procedure_fingerprint="audit", data_fingerprint=version, path=path)
    with tempfile.TemporaryDirectory(prefix="prediction-audit-v3-") as temp:
        path = Path(temp) / "journal.jsonl"
        add(path, "2026-10-01", 14, 4500.0, "initial")
        add(path, "2026-10-02", 13, 4400.0, "last")
        item = {"horizon": 14, "price": 4300.0, "confidence_interval": None,
                "status": "experimental", "model_version": "today",
                "components": {"origin_date": "2026-10-05",
                               "target_date": future_business_date("2026-10-05", 14).date().isoformat()}}
        shown = overlay_commitments([item], load_forecasts(path), market="ICE_NY", as_of="2026-10-05")[0]
        emit("stale_revision_uses_current_remaining", latest=shown["latest"],
             sessions_remaining=shown["sessions_remaining"],
             revision_horizon_at_issue=sessions_between(shown["latest"]["issued_date"], shown["latest"]["target_date"]),
             latest_unavailable=shown["latest_unavailable"], has_revision=shown["has_revision"])


if __name__ == "__main__":
    absent_revision()
    wrong_targets()
    rejected_band()
    dated_collection()
    stale_same_target()
    study_metrics()
