"""Read-only audit reproductions; generated fixtures stay in a temporary directory.

Run from the repository: venv_py311\\Scripts\\python.exe reports\\audit_repros_2026_10_10.py
No training, network call, production journal write or manifest change.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import scripts.run_revision_stability as study
from src.models.hybrid_features import load_price_data_from_supabase
from src.models.improved_price_predictor import ImprovedPricePredictor
from src.models.prediction_journal import count_inversions
from src.models.release_manifest import procedure_fingerprint
from src.models.revision_command import frame_fingerprint


class Query:
    def __init__(self, rows):
        self.rows = rows

    def table(self, *args, **kwargs): return self
    def select(self, *args, **kwargs): return self
    def order(self, *args, **kwargs): return self
    def range(self, *args, **kwargs): return self
    def limit(self, *args, **kwargs): return self
    def execute(self): return SimpleNamespace(data=self.rows)


class X:
    def predict(self, frame): return [frame["price_lag_1"].iloc[0]]


class ProphetStub:
    def predict(self, frame):
        return pd.DataFrame({"yhat": [100.] * len(frame), "trend": [100.] * len(frame), "yearly": [0.] * len(frame)})


def resume_repro(folder):
    dates = pd.bdate_range("2026-03-02", periods=60)
    frame = pd.DataFrame({"date": dates, "price": range(100, 160)})
    snapshot = folder / "snapshot.csv"
    frame.to_csv(snapshot, index=False)
    calls = {"n": 0, "fail_at": None}

    def fake_train(history, *args):
        calls["n"] += 1
        if calls["n"] == calls["fail_at"]:
            raise RuntimeError("simulated interruption")
        return {"anchor": float(history.price.iloc[-1])}

    def fake_price(bundle, cols, spot, day, horizon, *args):
        return float(spot + bundle["anchor"] / 10 + horizon)

    def run(output):
        argv = ["run_revision_stability.py", "--market", "cocoa", "--snapshot", str(snapshot),
                "--as-of", str(dates[-1].date()), "--sessions", "8", "--output-dir", str(output)]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            return study.main()

    full, resumed = folder / "full", folder / "resumed"
    with patch.object(study, "_train", fake_train), patch.object(study, "_price", fake_price):
        run(full)
        calls.update(n=0, fail_at=8)
        try:
            run(resumed)
        except RuntimeError as exc:
            assert str(exc) == "simulated interruption"
        calls.update(n=0, fail_at=None)
        run(resumed)
    read_last = lambda p: json.loads((p / "sessions.jsonl").read_text(encoding="utf-8").splitlines()[-1])["rows"]
    a, b = read_last(full), read_last(resumed)
    initial = lambda rows: next(r for r in rows if r.get("policy") == "five" and r.get("role") == "initial" and r["horizon"] == 7)
    revision = lambda rows: next(r for r in rows if r.get("policy") == "daily" and r.get("role") == "revision")
    return {
        "five_price_uninterrupted": initial(a)["price"], "five_price_resumed": initial(b)["price"],
        "daily_revision_delta_uninterrupted": revision(a)["delta"], "daily_revision_delta_resumed": revision(b)["delta"],
        "diagnostics_uninterrupted": sum(r.get("kind") == "diagnostic" for r in a),
        "diagnostics_resumed": sum(r.get("kind") == "diagnostic" for r in b),
    }


def main():
    result = {}
    f = pd.DataFrame({"date": pd.to_datetime(["2026-10-01", "2026-10-02"]), "price": [100., 105.],
                      "open_interest": [10., 11.], "high": [110., 111.], "low": [90., 91.], "volume": [1000., 1100.]})
    changed = f.copy(); changed.loc[1, "volume"] = 90000
    changed_oi = f.copy(); changed_oi.loc[1, "open_interest"] = 90000
    result["journal_hash_ignores_volume_change"] = frame_fingerprint(f) == frame_fingerprint(changed)
    result["study_hash_ignores_oi_change"] = study._identity(f, "cocoa", "2026-10-06", 60, "study", {}, {}) == study._identity(changed_oi, "cocoa", "2026-10-06", 60, "study", {}, {})
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "src/models/direct_horizon_trainer.py"; source.parent.mkdir(parents=True)
        source.write_text("before")
        old = procedure_fingerprint(root=root); source.write_text("after")
        result["procedure_hash_ignores_direct_training_code"] = old == procedure_fingerprint(root=root)
        meta = root / "direct.json"; meta.write_text('{"horizons":{"1":{"model_path":"h1.pkl"}}}')
        artifact = root / "h1.pkl"; artifact.write_bytes(b"old model")
        old = procedure_fingerprint(root=root, artifact_paths={"direct_info": "direct.json"})
        artifact.write_bytes(b"different model")
        result["procedure_hash_ignores_referenced_direct_model"] = old == procedure_fingerprint(root=root, artifact_paths={"direct_info": "direct.json"})
        result["resume"] = resume_repro(root)
    p = ImprovedPricePredictor.__new__(ImprovedPricePredictor)
    p._frozen_frame = None; p._historical_data = f.copy(); p._last_data_fetch = datetime.now()
    p.price_table = "cocoa_london_prices"; p.supabase_client = Query([{"date": "2026-10-02", "price": 999.}])
    result["same_day_correction"] = {"database_price": 999., "returned_price": float(p._fetch_historical_data().price.iloc[-1])}
    rows = f.to_dict("records"); p.supabase_client = Query(rows)
    train_frame = load_price_data_from_supabase(Query(rows), table_name="cocoa_london_prices")
    live_frame = p._fetch_historical_data(force_refresh=True)
    result["oi_preparation"] = {"training_non_null": int(train_frame.open_interest.notna().sum()), "serving_non_null": int(live_frame.open_interest.notna().sum())}
    bundle = {"models": {1: X()}, "last": pd.Series({"price_lag_1": 10.}), "prophet": ProphetStub(), "target": "close_delta", "nhits": {1: 111.}}
    columns = ["price_lag_1", "prophet_yhat"]
    old = study._price(bundle, columns, 200., pd.Timestamp("2026-10-05"), 1, {"xgb": 1.}, {}, {})
    bundle["last"] = pd.Series({"price_lag_1": 190.})
    new = study._price(bundle, columns, 200., pd.Timestamp("2026-10-05"), 1, {"xgb": 1.}, {}, {})
    result["stale_features_same_model"] = {"old_features": old, "current_features": new}
    directions = ["hausse", "neutre", "baisse"]
    result["neutral_flip"] = {"tracking_count": count_inversions(directions), "study_count": sum(a in {"hausse", "baisse"} and b in {"hausse", "baisse"} and a != b for a, b in zip(directions, directions[1:]))}
    result["reports"] = {}
    for market in ("cocoa", "coffee_robusta"):
        path = ROOT / "reports/revision_stability" / market / "study/sessions.jsonl"
        sessions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [r for s in sessions for r in s["rows"]]
        revisions = [r for r in rows if r.get("role") == "revision"]
        scores = []
        for h in (1, 7, 14, 30):
            valid = [r for r in rows if r.get("role") == "initial" and r["policy"] == "daily" and r["horizon"] == h and r.get("price") is not None and r.get("realized")]
            mape = lambda key: sum(abs(r[key] - r["realized"]) / r["realized"] * 100 for r in valid) / len(valid)
            scores.append({"horizon": h, "n": len(valid), "model_mape": mape("price"), "unchanged_mape": mape("reference")})
        result["reports"][market] = {"revisions": len(revisions), "revisions_with_realized": sum(r.get("realized") is not None for r in revisions), "scores": scores}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
