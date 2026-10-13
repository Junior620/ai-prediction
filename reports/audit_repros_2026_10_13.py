"""Read-only audit counterexamples: no training, HTTP request or journal writes.

Run from the repository root with venv_py311/Scripts/python.exe.
The only writes are inside an automatically removed temporary directory.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import pickle
import sys
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.models.hybrid_features import (
    consumed_snapshot_rows, prepare_market_frame, session_anomalies, trading_sessions,
)
from src.models.release_manifest import data_fingerprint, procedure_fingerprint
from src.validation.revision_stability import annotate_realized, score_policies


def emit(name, result):
    print(json.dumps({"check": name, **result}, ensure_ascii=False, default=str), flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    cocoa = prepare_market_frame(pd.read_csv(ROOT / "data/freeze/cocoa_prices.csv"), market_id="cocoa")
    old = cocoa[cocoa.date <= pd.Timestamp("2026-10-06")].copy()
    lookup = {pd.Timestamp(r.date).normalize(): float(r.price) for r in old.itertuples()}
    checkpoint = ROOT / "reports/revision_stability/cocoa/study_corrected"
    batches = [json.loads(s) for s in (checkpoint / "sessions.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    unique = {}
    conflicts = []
    for batch in batches:
        key = batch["session"]
        if key in unique and unique[key] != batch:
            conflicts.append(key)
        unique.setdefault(key, batch)
    all_records = [r for batch in batches for r in batch["rows"]]
    unique_records = [r for batch in unique.values() for r in batch["rows"]]
    scores = score_policies(annotate_realized(all_records, lookup, "2026-10-06"))
    scores_unique = score_policies(annotate_realized(unique_records, lookup, "2026-10-06"))
    identity = json.loads((checkpoint / "checkpoint_identity.json").read_text(encoding="utf-8"))
    emit("checkpoint", {
        "batches": len(batches), "unique_sessions": len(unique),
        "counts_distribution": dict(Counter(Counter(b["session"] for b in batches).values())),
        "conflicting_duplicates": conflicts,
        "current_snapshot_through_cutoff_matches_study": data_fingerprint(consumed_snapshot_rows(old)) == identity["data_fingerprint"],
        "daily_j30_after_reload": next(r for r in scores if r["policy"] == "daily" and r["initial_horizon"] == 30),
        "daily_j30_unique": next(r for r in scores_unique if r["policy"] == "daily" and r["initial_horizon"] == 30),
    })

    # Same saved model, same last spot, different preprocessing paths.
    from scripts.run_revision_stability import _last_row
    from src.models.direct_horizon_trainer import DirectHorizonTrainer
    from src.models.served_forecast import direct_level
    release = json.loads((ROOT / "config/experimental_release_cocoa.json").read_text(encoding="utf-8"))
    artifacts = release["artifacts"]
    info = json.loads((ROOT / artifacts["improved_info"]).read_text(encoding="utf-8"))
    cols = info["feature_cols"]
    direct_info = json.loads((ROOT / artifacts["direct_info"]).read_text(encoding="utf-8"))
    with open(ROOT / artifacts["prophet"], "rb") as handle:
        prophet = pickle.load(handle)
    models = DirectHorizonTrainer.load_from_info(str(ROOT / artifacts["direct_info"]))
    service = cocoa.copy()
    for col in ("open", "high", "low", "volume"):
        if col in service:
            service[col] = pd.to_numeric(service[col], errors="coerce").ffill()
    frames = {"frozen_command": cocoa, "api_fill": service, "study_calendar": trading_sessions(cocoa)}
    last = {key: _last_row(frame, prophet, cols) for key, frame in frames.items()}
    prices = {
        key: {str(h): direct_level(models[h], row, cols, float(row["price"]), direct_info["target"]) for h in (1, 7, 14, 30)}
        for key, row in last.items()
    }
    changes = {
        col: {key: float(row[col]) for key, row in last.items()}
        for col in cols if len({round(float(row[col]), 10) for row in last.values()}) > 1
    }
    emit("preprocessing", {
        "version": release["version"], "last_date": str(cocoa.date.max()),
        "rows": {key: len(frame) for key, frame in frames.items()},
        "anomalies": session_anomalies(cocoa),
        "missing_ohlcv": cocoa[["open", "high", "low", "volume"]].isna().sum().to_dict(),
        "direct_component_prices": prices, "different_features": changes,
        "raw_and_api_data_fingerprints_equal": data_fingerprint(consumed_snapshot_rows(cocoa)) == data_fingerprint(consumed_snapshot_rows(service)),
    })
    journal = ROOT / "data/prediction_journal.jsonl"
    if journal.exists():
        journal_rows = [json.loads(s) for s in journal.read_text(encoding="utf-8").splitlines() if s.strip()]
        actual = [
            row for row in journal_rows
            if row.get("model_version") == release["version"]
            and row.get("origin_date") == str(cocoa.date.max().date())
            and row.get("horizon") in (1, 7, 14, 30) and row.get("published")
        ]
        emit("actual_journal", {"rows": [{k: row.get(k) for k in (
            "forecast_id", "recorded_at", "horizon", "target_date", "origin_price", "price", "status", "data_fingerprint"
        )} for row in actual]})

    # Fully intercepted HTTP: the real API is never called.
    from scripts import verify_served_forecast as verifier
    payload = {"model_version": release["version"], "predictions": [{"horizon": h, "status": "unavailable", "price": None} for h in verifier.HORIZONS]}
    output = io.StringIO()
    with patch.object(sys, "argv", ["verify_served_forecast.py", "--market", "cocoa"]), patch.dict(os.environ, {"API_TOKEN": "audit-dummy"}), patch.object(verifier.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(payload).encode("utf-8"))), contextlib.redirect_stdout(output):
        code = verifier.main()
    emit("unavailable_verifier", {"exit_code": code, "stdout": output.getvalue()})

    from src.data_collection import ice_london_collector as collector
    html = "<table><tr><td>2026-10-09</td><td>4277</td></tr></table>"
    with patch.object(collector, "_scrape_playwright", return_value=(html, "playwright_css", 4277.0)):
        result = collector.fetch_ice_london_spot()
    named = collector.parse_table_rows_from_html("<tr><td>DEC26</td><td>4277</td></tr>")
    emit("collection_date", {"html_quote_date": "2026-10-09", "returned_date": result.date, "contract_label_parsed_as_daily_row": named})

    with tempfile.TemporaryDirectory(prefix="prediction-audit-") as temp:
        base = Path(temp)
        path = base / "src/models/hybrid_trainer.py"
        path.parent.mkdir(parents=True)
        path.write_text("PARAMETER = 1\n", encoding="utf-8")
        before = procedure_fingerprint(root=base)
        path.write_text("PARAMETER = 999\n", encoding="utf-8")
        after = procedure_fingerprint(root=base)
        emit("procedure_hash", {"hybrid_trainer_changed_but_fingerprint_identical": before == after})

    ordinary = {"kind": "published", "policy": "daily", "role": "initial", "initial_horizon": 7, "realized": 100.0, "price": 101.0, "reference": 100.0}
    missing = {**ordinary, "price": None}
    scored = score_policies([ordinary, missing])[0]
    emit("missing_prediction_denominator", {"total_mature_targets": 2, "missing_predictions": 1, "reported": scored})


if __name__ == "__main__":
    main()
