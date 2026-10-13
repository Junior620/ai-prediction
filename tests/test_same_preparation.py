"""One OHLCV preparation, one study resume, and a priced end-of-update check."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-10-08", "2026-10-09", "2026-10-13"]),
            "price": [4200.0, 4241.0, 4277.0],
            "open": [4200.0, 4241.0, None],
            "high": [4210.0, 4250.0, None],
            "low": [4190.0, 4230.0, None],
            "volume": [10.0, 12.0, None],
        }
    )


def test_missing_ohlcv_is_filled_before_features_on_both_paths():
    from src.models.hybrid_features import build_technical_features, fill_missing_ohlcv
    from src.models.improved_price_predictor import ImprovedPricePredictor
    from src.models.release_manifest import data_fingerprint
    from src.models.hybrid_features import consumed_snapshot_rows

    frame = _frame()
    predictor = ImprovedPricePredictor.__new__(ImprovedPricePredictor)
    predictor._frozen_frame = frame
    predictor.price_table = "cocoa_london_prices"
    fetched = predictor._fetch_historical_data()
    direct = fill_missing_ohlcv(frame)
    assert float(fetched["open"].iloc[-1]) == float(direct["open"].iloc[-1]) == 4241.0
    left = build_technical_features(fetched)
    right = build_technical_features(direct)
    assert float(left["hl_range_pct"].iloc[-1]) == float(right["hl_range_pct"].iloc[-1])
    raw = data_fingerprint(consumed_snapshot_rows(frame))
    served = data_fingerprint(consumed_snapshot_rows(fetched))
    assert predictor._data_fingerprint == raw
    assert predictor._served_fingerprint == served
    assert raw != served


def test_service_keeps_closed_rows_and_names_that_calendar():
    from src.models.hybrid_features import SERVICE_CALENDAR, prepare_market_frame, trading_sessions

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-10-09", "2026-10-10", "2026-10-12"]),
            "price": [1.0, 2.0, 3.0],
        }
    )
    kept = prepare_market_frame(frame, market_id="cocoa", ledger_path=Path("missing-ledger.csv"))
    sessions = trading_sessions(frame)
    assert SERVICE_CALENDAR == "keep_closed_rows"
    assert "2026-10-10" in set(pd.to_datetime(kept["date"]).dt.strftime("%Y-%m-%d"))
    assert "2026-10-10" not in set(pd.to_datetime(sessions["date"]).dt.strftime("%Y-%m-%d"))


def test_procedure_fingerprint_includes_the_trainer_and_the_calendar(tmp_path: Path):
    from src.models.release_manifest import procedure_fingerprint

    first = procedure_fingerprint(root=tmp_path, parameters={"calendar": "keep_closed_rows"})
    changed = procedure_fingerprint(root=tmp_path, parameters={"calendar": "trading_sessions"})
    assert first != changed
    trainer = tmp_path / "src" / "models" / "hybrid_trainer.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("changed = True\n", encoding="utf-8")
    assert procedure_fingerprint(root=tmp_path, parameters={"calendar": "keep_closed_rows"}) != first


def test_score_counts_an_abstention_and_pairs_the_same_rows():
    from src.validation.revision_stability import score_policies

    records = [
        {
            "kind": "published",
            "policy": "daily",
            "initial_horizon": 7,
            "role": "initial",
            "price": 100.0,
            "realized": 110.0,
            "reference": 105.0,
        },
        {
            "kind": "published",
            "policy": "daily",
            "initial_horizon": 7,
            "role": "initial",
            "price": None,
            "realized": 110.0,
            "reference": 105.0,
        },
        {
            "kind": "published",
            "policy": "daily",
            "initial_horizon": 7,
            "role": "revision",
            "horizon": 1,
            "price": 108.0,
            "realized": 110.0,
            "spot": 107.0,
            "delta": 1.0,
        },
        {
            "kind": "published",
            "policy": "daily",
            "initial_horizon": 7,
            "role": "revision",
            "horizon": 1,
            "price": 109.0,
            "realized": 110.0,
            "spot": None,
            "delta": 1.0,
        },
    ]
    row = score_policies(records)[0]
    assert row["expected_initials"] == 2
    assert row["available_prices"] == 1
    assert row["abstentions"] == 1
    assert row["availability_rate"] == 0.5
    assert row["mature_initials"] == 1
    assert row["paired_revision_mape_by_remaining"]["1"] is not None
    assert row["paired_unchanged_mape_by_remaining"]["1"] is not None


def test_identical_sessions_collapse_and_a_contradiction_is_refused(tmp_path: Path):
    from src.validation.revision_stability import (
        ArchivePreserved,
        ContradictorySession,
        append_unique_session,
        load_unique_sessions,
    )

    path = tmp_path / "sessions.jsonl"
    first = {"session": "2026-06-02", "rows": [{"price": 1}]}
    path.write_text(json.dumps(first) + "\n" + json.dumps(first) + "\n", encoding="utf-8")
    unique = load_unique_sessions(path, study_id="study")
    assert len(unique) == 1
    original = path.read_text(encoding="utf-8")
    with pytest.raises(ArchivePreserved):
        append_unique_session(path, {"session": "2026-06-03", "rows": [{"price": 2}]}, study_id="study")
    assert path.read_text(encoding="utf-8") == original
    with pytest.raises(ContradictorySession):
        load_unique_sessions(
            _write(tmp_path / "bad.jsonl", [first, {"session": "2026-06-02", "rows": [{"price": 9}]}]),
            study_id="study",
        )


def test_resume_matches_a_continuous_write(tmp_path: Path):
    from src.validation.revision_stability import append_unique_session, load_unique_sessions

    sessions = [
        {"session": "2026-06-02", "rows": [{"price": 1}]},
        {"session": "2026-06-03", "rows": [{"price": 2}]},
        {"session": "2026-06-04", "rows": [{"price": 3}]},
    ]
    continuous = tmp_path / "continuous.jsonl"
    resumed = tmp_path / "resumed.jsonl"
    for batch in sessions:
        assert append_unique_session(continuous, batch, study_id="study") == "written"
    for batch in sessions[:2]:
        append_unique_session(resumed, batch, study_id="study")
    assert append_unique_session(resumed, sessions[1], study_id="study") == "duplicate"
    assert append_unique_session(resumed, sessions[2], study_id="study") == "written"
    left = load_unique_sessions(continuous, study_id="study")
    right = load_unique_sessions(resumed, study_id="study")
    assert [item["session"] for item in left] == [item["session"] for item in right]
    assert [item["rows"] for item in left] == [item["rows"] for item in right]


def test_two_writers_keep_both_sessions(tmp_path: Path):
    import threading

    from src.validation.revision_stability import append_unique_session, load_unique_sessions

    path = tmp_path / "sessions.jsonl"
    errors = []

    def write(session: str) -> None:
        try:
            append_unique_session(
                path, {"session": session, "rows": [{"price": session}]}, study_id="study"
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(day,)) for day in ("2026-06-02", "2026-06-03")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert {item["session"] for item in load_unique_sessions(path, study_id="study")} == {
        "2026-06-02",
        "2026-06-03",
    }


def test_verify_requires_finite_prices_for_every_horizon():
    from scripts.verify_served_forecast import assess_served

    def payload(**changes):
        base = {
            "model_version": "improved_x",
            "current_date": "2026-10-13",
            "predictions": [
                {
                    "horizon": horizon,
                    "status": "experimental",
                    "price": 900,
                    "target_date": "2026-12-01",
                    "fresh": {
                        "price": 1000 + horizon,
                        "status": "experimental",
                        "target_date": "2026-11-01",
                        "issued_date": "2026-10-13",
                        "model_version": "improved_x",
                    },
                }
                for horizon in (1, 7, 14, 30)
            ],
        }
        base.update(changes)
        return base

    assert assess_served(payload(), "improved_x", "2026-10-13")[0] == 0
    missing = payload()
    for item in missing["predictions"]:
        item["fresh"]["price"] = None
        item["fresh"]["status"] = "unavailable"
    code, lines = assess_served(missing, "improved_x", "2026-10-13")
    assert code == 3
    assert sum("prix absent" in line for line in lines) == 4
    assert not any(line.startswith("[OK]") for line in lines)
    one = payload()
    one["predictions"][1]["fresh"]["price"] = None
    kept = payload()
    for item in kept["predictions"]:
        item.pop("fresh")
    assert assess_served(kept, "improved_x", "2026-10-13")[0] == 3
    assert assess_served(one, "improved_x", "2026-10-13")[0] == 3
    assert assess_served(payload(), "improved_old", "2026-10-13")[0] == 1
    assert assess_served(payload(), "improved_x", "2026-10-06")[0] == 1


def test_history_index_includes_revision_studies_without_validating_the_lot(tmp_path: Path):
    from src.validation.report_index import list_market_reports

    walk = tmp_path / "reports" / "walk_forward"
    walk.mkdir(parents=True)
    (walk / "20261006_214926_summary.json").write_text(
        json.dumps({"timestamp": "2026-10-06", "n_origins": 60, "origin_start": "2026-01-01", "origin_end": "2026-08-01"}),
        encoding="utf-8",
    )
    study = tmp_path / "reports" / "revision_stability" / "cocoa" / "study_corrected"
    study.mkdir(parents=True)
    (study / "20261012_183344_summary.json").write_text(
        json.dumps(
            {
                "as_of": "2026-10-06",
                "origins": ["2026-06-02", "2026-08-24"],
                "policies": [{"policy": "daily", "initial_horizon": 30, "mature_initials": 60}],
            }
        ),
        encoding="utf-8",
    )
    rows = list_market_reports("cocoa", tmp_path)
    assert rows[0]["kind"] == "revision_stability"
    assert rows[0]["timestamp"] == "2026-10-12T18:33:44"
    assert all(row["compatible_with_current_lot"] is False for row in rows)
    assert any(row["period"] == "2026-06-02 – 2026-08-24" for row in rows)


def test_robusta_measure_trees_are_not_indexed_as_cocoa(tmp_path: Path):
    from src.validation.report_index import assigned_market, list_market_reports

    reports = tmp_path / "reports"
    measure = reports / "walk_forward" / "measure" / "coffee_robusta"
    trial = reports / "walk_forward" / "trial_measure" / "coffee_robusta"
    clash = reports / "walk_forward" / "measure" / "cocoa"
    measure.mkdir(parents=True)
    trial.mkdir(parents=True)
    clash.mkdir(parents=True)
    (reports / "walk_forward" / "20261006_214926_summary.json").write_text("{}", encoding="utf-8")
    (measure / "20261009_150625_summary.json").write_text("{}", encoding="utf-8")
    (trial / "20261009_124610_summary.json").write_text(
        json.dumps({"market": "coffee_robusta"}),
        encoding="utf-8",
    )
    clash_path = clash / "20261009_134348_summary.json"
    clash_path.write_text(json.dumps({"market": "coffee_robusta"}), encoding="utf-8")
    cocoa_ids = {row["report_id"] for row in list_market_reports("cocoa", tmp_path)}
    robusta_ids = {row["report_id"] for row in list_market_reports("coffee_robusta", tmp_path)}
    assert "walk_forward/20261006_214926_summary.json" in cocoa_ids
    assert "walk_forward/measure/coffee_robusta/20261009_150625_summary.json" in robusta_ids
    assert "walk_forward/trial_measure/coffee_robusta/20261009_124610_summary.json" in robusta_ids
    assert not any("coffee_robusta" in item for item in cocoa_ids)
    clash_id = "walk_forward/measure/cocoa/20261009_134348_summary.json"
    assert clash_id not in cocoa_ids
    assert clash_id not in robusta_ids
    assert assigned_market(clash_path, {"market": "coffee_robusta"}, reports) is None


def test_robusta_quote_uses_the_session_date_and_quarantines_a_missing_one():
    from collect_coffee_robusta_price import decide_robusta_quote

    ready = decide_robusta_quote({"price": 3400, "symbol": "RCZ6", "session_date": "2026-10-13"})
    assert ready == {"action": "insert", "session_date": "2026-10-13", "price": 3400}
    missing = decide_robusta_quote(
        {
            "price": 3400,
            "symbol": "RCZ6",
            "session_date": None,
            "quarantine_reason": "date_de_seance_absente",
        }
    )
    assert missing["action"] == "quarantine"
    assert missing["record"]["session_date"] is None
    assert "date" not in missing["record"]
    bare = decide_robusta_quote({"price": 3400, "symbol": "RCZ6"})
    assert bare["action"] == "quarantine"
    assert bare["reason"] == "date_de_seance_absente"


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_october_13_cocoa_paths_share_the_served_prices():
    release_path = ROOT / "config" / "experimental_release_cocoa.json"
    snapshot = ROOT / "data" / "freeze" / "cocoa_prices.csv"
    if not release_path.exists() or not snapshot.exists():
        pytest.skip("instantané ou lot cacao absent")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("version") != "improved_20261013_045358":
        pytest.skip("le lot du 13 octobre n'est pas le lot expérimental")
    import yaml

    from src.models.hybrid_features import clean_price_dataframe, fill_missing_ohlcv
    from src.models.market_registry import get_market_config
    from src.models.revision_command import load_named_predictor

    frame = clean_price_dataframe(pd.read_csv(snapshot))
    last = frame.iloc[-1]
    if str(pd.to_datetime(last["date"]).date()) != "2026-10-13" or float(last["price"]) != 4277.0:
        pytest.skip("l'instantané cacao n'est plus celui du 13 octobre")
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as handle:
        pred_cfg = (yaml.safe_load(handle) or {}).get("prediction") or {}
    market = get_market_config("cocoa")
    predictor = load_named_predictor(release, market, pred_cfg)
    predictor.bind_frame(frame)
    first = {int(item.horizon): float(item.price) for item in predictor.predict([1, 7, 14, 30])}
    assert predictor._data_fingerprint != predictor._served_fingerprint
    predictor.bind_frame(fill_missing_ohlcv(frame))
    second = {int(item.horizon): float(item.price) for item in predictor.predict([1, 7, 14, 30])}
    assert first == second
    expected = {1: 4278.70, 7: 4328.63, 14: 4084.28, 30: 3579.32}
    for horizon, price in expected.items():
        assert abs(first[horizon] - price) < 0.05
