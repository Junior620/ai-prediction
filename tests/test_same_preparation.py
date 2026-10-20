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


def test_the_inventory_lists_closed_and_missing_sessions_without_editing_the_file(tmp_path: Path):
    from src.models.hybrid_features import session_inventory

    path = tmp_path / "prices.csv"
    path.write_text("date,price\n2026-10-09,1\n2026-10-10,2\n", encoding="utf-8")
    before = path.read_bytes()
    report = session_inventory(pd.read_csv(path), "2026-10-01", "2026-10-13")
    assert path.read_bytes() == before
    assert "2026-10-10" in report["closed_dates"]
    assert "2026-10-12" in report["missing_sessions"]
    assert "2026-10-09" not in report["missing_sessions"]


def test_omitting_a_declared_calendar_is_refused():
    from src.models.improved_price_predictor import ImprovedPricePredictor
    from src.models.release_calendar import CalendarOmitted

    predictor = ImprovedPricePredictor.__new__(ImprovedPricePredictor)
    predictor.active_release = {
        "artifacts": {
            "improved_info": (
                "models/candidates/cocoa/20261014_112728/"
                "model_info_improved_20261014_112728.json"
            )
        }
    }
    frame = pd.DataFrame(
        {"date": pd.to_datetime(["2026-10-09", "2026-10-12"]), "price": [1.0, 3.0]}
    )
    with pytest.raises(CalendarOmitted):
        predictor._lot_calendar(frame)


@pytest.mark.production
def test_the_journal_descriptor_matches_the_api_manifest():
    from scripts.write_experimental_release import release_descriptor

    release_path = ROOT / "config" / "experimental_release_cocoa.json"
    # Chemin servi : un manifeste absent est une erreur, pas un test sauté.
    assert release_path.exists(), "manifeste expérimental cacao absent"
    served = json.loads(release_path.read_text(encoding="utf-8"))
    journal = release_descriptor(
        "cocoa",
        served["version"].removeprefix("improved_"),
        "config/ensemble_weights.json",
        "config/conformal_intervals.json",
        ROOT / "models",
    )
    assert journal["calendar"] == "trading_sessions"
    assert served["calendar"] == journal["calendar"]
    assert journal["version"] == served["version"]
    assert journal["artifacts"]["prophet"] == served["artifacts"]["prophet"]


@pytest.mark.production
def test_both_cocoa_paths_keep_the_same_prices():
    import yaml

    from scripts.write_experimental_release import release_descriptor
    from src.models.hybrid_features import clean_price_dataframe
    from src.models.market_registry import get_market_config
    from src.models.revision_command import load_named_predictor

    release_path = ROOT / "config" / "experimental_release_cocoa.json"
    snapshot = ROOT / "data" / "freeze" / "cocoa_prices.csv"
    assert release_path.exists(), "manifeste expérimental cacao absent"
    assert snapshot.exists(), "instantané cacao absent"
    served = json.loads(release_path.read_text(encoding="utf-8"))
    assert served.get("calendar") == "trading_sessions", "le lot cacao servi doit déclarer trading_sessions"
    journal = release_descriptor(
        "cocoa",
        served["version"].removeprefix("improved_"),
        "config/ensemble_weights.json",
        "config/conformal_intervals.json",
        ROOT / "models",
    )
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as handle:
        pred_cfg = (yaml.safe_load(handle) or {}).get("prediction") or {}
    frame = clean_price_dataframe(pd.read_csv(snapshot))
    market = get_market_config("cocoa")
    predictor = load_named_predictor(served, market, pred_cfg)

    def priced(release: dict) -> dict:
        predictor.active_release = release
        predictor.bind_frame(frame)
        rows = predictor.predict([1, 7, 14, 30])
        return {
            "rows": len(predictor._lot_calendar(frame)),
            "raw": predictor._data_fingerprint,
            "served": predictor._served_fingerprint,
            "prices": {
                int(item.horizon): (
                    round(float(item.price), 2),
                    str((item.components or {}).get("target_date") or "")[:10],
                )
                for item in rows
            },
        }

    api_prices = priced(served)
    journal_prices = priced(journal)
    assert api_prices["rows"] == journal_prices["rows"]
    assert api_prices["raw"] == journal_prices["raw"]
    assert api_prices["served"] == journal_prices["served"]
    assert api_prices["prices"] == journal_prices["prices"]


def test_a_calendar_disagreement_is_not_a_partial_success():
    from scripts.verify_served_forecast import assess_calendars

    release = {
        "version": "improved_x",
        "calendar": "keep_closed_rows",
        "artifacts": {},
    }
    code, lines = assess_calendars("cocoa", release, "keep_closed_rows")
    assert code == 1
    assert any("trading_sessions" in line for line in lines)
    agreed = {
        "version": "improved_x",
        "calendar": "trading_sessions",
        "artifacts": {},
    }
    assert assess_calendars("cocoa", agreed, "trading_sessions")[0] == 0
    assert assess_calendars("cocoa", agreed, None)[0] == 1


def test_a_named_lot_can_read_trading_sessions_without_changing_the_default():
    from src.models.hybrid_features import SERVICE_CALENDAR
    from src.models.improved_price_predictor import ImprovedPricePredictor

    frame = pd.DataFrame(
        {"date": pd.to_datetime(["2026-10-09", "2026-10-10", "2026-10-12"]), "price": [1.0, 2.0, 3.0]}
    )
    predictor = ImprovedPricePredictor.__new__(ImprovedPricePredictor)
    predictor.active_release = {"calendar": "trading_sessions"}
    kept = ImprovedPricePredictor.__new__(ImprovedPricePredictor)
    kept.active_release = {}
    filtered = predictor._lot_calendar(frame)
    untouched = kept._lot_calendar(frame)
    assert SERVICE_CALENDAR == "keep_closed_rows"
    assert len(filtered) == 2
    assert len(untouched) == 3


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
    from scripts.verify_served_forecast import assess_served, expected_target

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
                        "target_date": expected_target("2026-10-13", horizon),
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
    past = payload()
    for item in past["predictions"]:
        item["fresh"]["target_date"] = "2026-10-01"
    assert assess_served(past, "improved_x", "2026-10-13")[0] == 3
    invalid = payload()
    for item in invalid["predictions"]:
        item["fresh"]["target_date"] = "2026-99-99"
    assert assess_served(invalid, "improved_x", "2026-10-13")[0] == 3
    closed = payload()
    for item in closed["predictions"]:
        item["fresh"]["target_date"] = "2026-10-17"
    assert assess_served(closed, "improved_x", "2026-10-13")[0] == 3
    swapped = payload()
    swapped["predictions"][1]["fresh"]["target_date"] = expected_target("2026-10-13", 1)
    assert assess_served(swapped, "improved_x", "2026-10-13")[0] == 3
    unsigned = payload()
    for item in unsigned["predictions"]:
        item["fresh"].pop("issued_date")
        item["fresh"].pop("model_version")
    assert assess_served(unsigned, "improved_x", "2026-10-13")[0] == 3


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


def test_the_batch_can_start_the_served_forecast_check():
    import subprocess
    import sys

    from scripts.verify_served_forecast import assess_served, expected_target

    launched = subprocess.run(
        [sys.executable, "scripts/verify_served_forecast.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert launched.returncode == 0, launched.stderr
    assert "ModuleNotFoundError" not in launched.stderr

    def item(horizon: int) -> dict:
        return {
            "horizon": horizon,
            "price": 4000.0,
            "fresh": {
                "price": 4000.0,
                "issued_date": "2026-10-13",
                "model_version": "improved_x",
                "target_date": expected_target("2026-10-13", horizon),
                "status": "experimental",
            },
        }

    payload = {
        "model_version": "improved_x",
        "current_date": "2026-10-13",
        "predictions": [item(horizon) for horizon in (1, 7, 14, 30)],
    }
    assert assess_served(payload, "improved_x", "2026-10-13")[0] == 0


def test_a_close_date_stays_with_the_close_price_of_that_contract():
    from src.data_collection.investing_price_collector import linked_quote

    mixed = (
        "<title>London Coffee (RCX6)</title>"
        '<span data-test="instrument-price-last">3,500.00</span>'
        '{"last":3500,"last_close":3400,"last_close_date":"2026-10-13","symbol":"RCX6"}'
        '{"related":{"session_date":"2026-10-12","last":9999,"last_close_date":"2026-10-12"}}'
    )
    quote = linked_quote(mixed)
    assert quote["price"] == 3400.0
    assert quote["session_date"] == "2026-10-13"
    assert quote["symbol"] == "RCX6"
    assert quote["price_kind"] == "session_close"
    assert quote["quarantine_reason"] is None

    borrowed = (
        '<span data-test="instrument-price-last">3,500</span>'
        '{"related":{"last_close":3400,"last_close_date":"2026-10-13","symbol":"RCX6"}}'
    )
    refused = linked_quote(borrowed, fallback_symbol="RCX6")
    assert refused["session_date"] is None
    assert refused["price_kind"] != "session_close"
    assert refused["quarantine_reason"]

    labeled = '{"session_date":"2026-10-13","last":3500,"symbol":"RCX6"}'
    marked = linked_quote(labeled)
    assert marked["price_kind"] != "session_close"
    assert marked["session_date"] is None

    several = (
        "<title>London Coffee (RCX6)</title>"
        '{"symbol":"CCZ6","last_close":4500,"last_close_date":"2026-10-10"}'
        '{"symbol":"RCX6","last_close":3400,"last_close_date":"2026-10-13"}'
    )
    chosen = linked_quote(several)
    assert chosen["price"] == 3400.0
    assert chosen["session_date"] == "2026-10-13"
    assert chosen["symbol"] == "RCX6"
    doubled = several + '{"symbol":"RCX6","last_close":3300,"last_close_date":"2026-10-12"}'
    crowded = linked_quote(doubled)
    assert crowded["session_date"] is None
    assert crowded["quarantine_reason"]

    from collect_coffee_robusta_price import decide_robusta_quote

    nested = (
        "<title>London Coffee (RCZ6)</title>"
        '<script>{"related":{"instrument":{"last_close_date":"2026-10-13","close":9000}}}</script>'
    )
    foreign = (
        "<title>London Coffee (RCZ6)</title>"
        '<span data-test="instrument-price-last">3,500.00</span>'
        '<aside data-instrument="cocoa">'
        '<span data-test="instrument-session-date">2026-10-12</span>'
        "</aside>"
    )
    nested_child = (
        "<title>London Coffee (RCX6)</title>"
        '<span data-test="instrument-price-last">3,500.00</span>'
        '<div data-instrument="cocoa"><div>widget</div>'
        '<span data-test="instrument-session-date">2026-10-12</span>'
    )
    for page in (nested, foreign, nested_child):
        quote = linked_quote(page)
        decision = decide_robusta_quote(quote)
        assert quote["session_date"] is None
        assert quote["quarantine_reason"]
        assert decision["action"] == "quarantine"

    closed = (
        "<title>London Coffee (RCX6)</title>"
        '<span data-test="instrument-price-last">3,481.00</span>'
        '<span data-test="trading-state-label">Closed</span>'
        '<time dateTime="2026-10-14T16:29:57.958Z" data-test="trading-time-label">12:29:57</time>'
    )
    settled = linked_quote(closed)
    assert settled["price"] == 3481.0
    assert settled["symbol"] == "RCX6"
    assert settled["session_date"] == "2026-10-14"
    assert settled["price_kind"] == "closed_header"
    assert settled["quarantine_reason"] is None
    assert decide_robusta_quote(settled)["action"] == "insert"

    still_open = closed.replace("Closed", "Open")
    assert linked_quote(still_open)["session_date"] is None
    news_clock = (
        "<title>London Coffee (RCX6)</title>"
        '<span data-test="instrument-price-last">3,481.00</span>'
        '<time dateTime="2026-09-16T15:48:17Z" data-test="article-publish-date">Sep 16, 2026</time>'
    )
    assert linked_quote(news_clock)["session_date"] is None
    foreign_clock = (
        "<title>London Coffee (RCX6)</title>"
        '<span data-test="instrument-price-last">3,481.00</span>'
        '<aside data-instrument="cocoa">'
        '<span data-test="trading-state-label">Closed</span>'
        '<time dateTime="2026-10-12T16:00:00Z" data-test="trading-time-label"></time>'
        "</aside>"
    )
    assert linked_quote(foreign_clock)["session_date"] is None


def test_investing_keeps_a_page_date_and_quarantines_an_undated_price():
    from unittest.mock import patch
    from types import SimpleNamespace

    from src.data_collection.investing_price_collector import fetch_investing_price, session_quote_from_html

    dated = (
        '<title>London Coffee (RCU6)</title>'
        '<span data-test="instrument-price-last">3,453.26</span>'
        '<span data-test="instrument-session-date">2026-10-13</span>'
    )
    undated = (
        '<title>London Coffee (RCU6)</title>'
        '<span data-test="instrument-price-last">3,453.26</span>'
    )
    assert session_quote_from_html(dated) == ("2026-10-13", "marked_session")
    assert session_quote_from_html(undated) == (None, "intraday")
    with patch("curl_cffi.requests.get", return_value=SimpleNamespace(status_code=200, text=dated)):
        quoted = fetch_investing_price("https://example.invalid/audit")
    from collect_coffee_robusta_price import decide_robusta_quote

    assert decide_robusta_quote(quoted)["action"] == "insert"
    assert decide_robusta_quote(quoted)["session_date"] == "2026-10-13"
    with patch("curl_cffi.requests.get", return_value=SimpleNamespace(status_code=200, text=undated)):
        blank = fetch_investing_price("https://example.invalid/audit")
    decision = decide_robusta_quote(blank)
    assert decision["action"] == "quarantine"
    assert decision["record"]["session_date"] is None
    with patch("curl_cffi.requests.get", side_effect=OSError("down")):
        assert fetch_investing_price("https://example.invalid/down") is None
        assert decide_robusta_quote(None)["action"] == "failed"


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_october_13_cocoa_paths_share_the_served_prices():
    # Test historique : l'instantané du 13/10 (4 277 £, échéance de tête) a été remplacé
    # par la série Databento (4 450 £, Mar27). Il ne tourne plus ; le chemin servi est
    # couvert par les tests marqués « production » ci-dessus.
    snapshot = ROOT / "data" / "freeze" / "cocoa_prices.csv"
    release = None
    for name in ("experimental_release_cocoa.json", "previous_experimental_release_cocoa.json"):
        path = ROOT / "config" / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") == "improved_20261013_045358":
            release = payload
            break
    if release is None or not snapshot.exists():
        pytest.skip("instantané ou lot cacao du 13 octobre absent")
    assert str(release.get("calendar") or "keep_closed_rows") == "keep_closed_rows"
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
