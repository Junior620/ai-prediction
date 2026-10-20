"""Same-target gain of the latest estimate over the opening forecast."""

from __future__ import annotations

import json
from pathlib import Path

from src.validation.revision_value import delay_steps, evaluate_records, paired_expiries, summarise_horizon


def _initial(horizon, session, target, price):
    return {
        "role": "initial",
        "policy": "daily",
        "initial_horizon": horizon,
        "horizon": horizon,
        "session": session,
        "target": target,
        "price": price,
    }


def _revision(horizon, session, target, remaining, price):
    return {
        "role": "revision",
        "policy": "daily",
        "initial_horizon": horizon,
        "horizon": remaining,
        "session": session,
        "target": target,
        "price": price,
    }


def test_latest_same_target_is_compared_with_the_opening_forecast():
    records = [
        _initial(14, "2026-06-01", "2026-06-19", 4500.0),
        _revision(14, "2026-06-02", "2026-06-19", 13, 4300.0),
        _revision(14, "2026-06-03", "2026-06-19", 12, 4700.0),
        _initial(14, "2026-06-02", "2026-06-22", 1000.0),
        _revision(14, "2026-06-03", "2026-06-22", 13, 1000.0),
        {
            "role": "revision",
            "policy": "five",
            "initial_horizon": 14,
            "horizon": 13,
            "session": "2026-06-04",
            "target": "2026-06-19",
            "price": 1.0,
        },
    ]
    paired = paired_expiries(records, {"2026-06-19": 4350.0, "2026-06-22": 1000.0}, "2026-06-30")
    assert len(paired[14]) == 2
    first = paired[14][0]
    assert first["target"] == "2026-06-19"
    assert first["initial_price"] == 4500.0
    assert first["latest_price"] == 4700.0
    assert first["initial_error"] == 150.0
    assert first["latest_error"] == 350.0
    assert first["gain"] == -200.0


def test_clear_gain_is_established_and_clear_loss_is_unfavourable():
    better = summarise_horizon(
        [{"session": f"2026-01-{day:02d}", "target": f"2026-02-{day:02d}", "gain": 10.0,
          "initial_error": 20.0, "latest_error": 10.0} for day in range(1, 13)],
        1,
        draws=300,
    )
    assert better["verdict"] == "etabli"
    assert better["improved_share"] == 1.0
    worse = summarise_horizon(
        [{"session": f"2026-01-{day:02d}", "target": f"2026-02-{day:02d}", "gain": -10.0,
          "initial_error": 10.0, "latest_error": 20.0} for day in range(1, 13)],
        1,
        draws=300,
    )
    assert worse["verdict"] == "defavorable"


def test_many_small_gains_do_not_hide_a_few_large_losses():
    rows = [
        {"session": f"2026-03-{day:02d}", "target": f"2026-04-{day:02d}", "gain": 1.0,
         "initial_error": 11.0, "latest_error": 10.0}
        for day in range(1, 11)
    ]
    rows.extend(
        {"session": f"2026-05-{day:02d}", "target": f"2026-06-{day:02d}", "gain": -20.0,
         "initial_error": 10.0, "latest_error": 30.0}
        for day in range(1, 3)
    )
    summary = summarise_horizon(rows, 1, draws=400)
    assert summary["improved_share"] > 0.5
    assert summary["mean_gain"] < 0
    assert summary["degradation_sum"] > summary["improvement_sum"]
    assert summary["verdict"] != "etabli"


def test_a_short_sample_stays_uncertain():
    summary = summarise_horizon(
        [{"session": "2026-01-02", "target": "2026-01-12", "gain": 100.0,
          "initial_error": 100.0, "latest_error": 0.0}],
        7,
        draws=50,
    )
    assert summary["verdict"] == "incertain"


def test_overlapping_block_can_keep_a_small_gain_uncertain():
    rows = []
    for day in range(1, 21):
        gain = 1.0 if day % 2 else -1.0
        rows.append(
            {
                "session": f"2026-01-{day:02d}",
                "target": f"2026-03-{day:02d}",
                "gain": gain,
                "initial_error": 10.0,
                "latest_error": 10.0 - gain,
            }
        )
    summary = summarise_horizon(rows, 14, draws=300)
    assert summary["mean_gain"] == 0
    assert summary["usable_blocks"] < 2
    assert summary["verdict"] == "incertain"
    assert summary["gain_low"] is None and summary["gain_high"] is None
    assert summary["interval_level"] == 0.90


def test_unreached_and_other_horizon_rows_are_left_out():
    records = [
        _initial(7, "2026-06-01", "2026-06-10", 110.0),
        _revision(7, "2026-06-02", "2026-06-10", 6, 100.0),
        _initial(30, "2026-06-01", "2026-07-13", 200.0),
        _revision(30, "2026-06-02", "2026-07-13", 29, 150.0),
    ]
    scored = evaluate_records(records, {"2026-06-10": 100.0}, "2026-06-10", draws=20)
    by_horizon = {row["horizon"]: row for row in scored}
    assert by_horizon[7]["expiries"] == 1
    assert by_horizon[7]["mean_gain"] == 10.0
    assert by_horizon[14]["expiries"] == 0
    assert by_horizon[30]["expiries"] == 0


def test_remaining_horizon_is_not_replaced_by_the_last_estimate():
    records = [
        _initial(14, "2026-06-01", "2026-06-19", 4500.0),
        _revision(14, "2026-06-02", "2026-06-19", 13, 4300.0) | {"spot": 4400.0},
        _revision(14, "2026-06-03", "2026-06-19", 12, None),
        _revision(14, "2026-06-18", "2026-06-19", 1, 4700.0) | {"spot": 4360.0},
        _initial(7, "2026-06-04", "2026-06-15", 100.0),
        _initial(14, "2026-06-01", "2026-06-22", 1000.0),
        _revision(14, "2026-06-02", "2026-06-22", 13, 1100.0),
        _revision(14, "2026-06-03", "2026-06-22", 12, 1050.0) | {"spot": 1000.0},
    ]
    scored = evaluate_records(
        records,
        {"2026-06-19": 4350.0, "2026-06-22": 1000.0},
        "2026-06-22",
        draws=30,
    )
    horizon = {row["horizon"]: row for row in scored}[14]
    assert horizon["comparison"] == "initiale contre dernière estimation avant échéance"
    assert horizon["mean_gain"] == -125.0
    early = next(
        step for step in horizon["steps"]
        if step["comparison"] == "initial_vs_remaining" and step["remaining"] == 13
    )
    assert early["available"] == 2
    assert early["mean_initial_error"] == 75.0
    assert early["mean_revised_error"] == 75.0
    assert early["mean_gain"] == 0.0
    assert early["mean_spot_error"] == 50.0
    assert early["block"] == 14
    missing = next(
        step for step in horizon["steps"]
        if step["comparison"] == "initial_vs_remaining" and step["remaining"] == 12
    )
    assert missing["expected"] == 2
    assert missing["available"] == 1
    assert missing["abstentions"] == 1
    assert missing["missing_emissions"] == 0
    assert missing["availability_rate"] == 0.5
    successive = next(
        step for step in horizon["steps"]
        if step["comparison"] == "successive" and step["remaining"] == 12
    )
    assert successive["mean_gain"] == 50.0
    jumped = next(
        step for step in horizon["steps"]
        if step["comparison"] == "successive" and step["remaining"] == 1
    )
    assert jumped["session_jumps"] >= 1
    assert jumped["mean_gain"] is None


def test_doubling_the_archive_does_not_change_the_score(tmp_path: Path):
    from src.validation.revision_value import load_daily_records

    opening = _initial(7, "2026-06-01", "2026-06-10", 110.0) | {"kind": "published"}
    revision = _revision(7, "2026-06-02", "2026-06-10", 6, 100.0) | {"kind": "published"}
    first = {"session": "2026-06-01", "rows": [opening]}
    second = {"session": "2026-06-02", "rows": [revision]}
    path = tmp_path / "sessions.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in (first, second, first, second)),
        encoding="utf-8",
    )
    once = load_daily_records(path)
    assert len(once) == 2
    prices = {"2026-06-10": 100.0}
    single = delay_steps(once, prices, "2026-06-10", draws=20)
    doubled = delay_steps(once + once, prices, "2026-06-10", draws=20)
    assert doubled == single


def test_a_missing_revision_stays_in_the_denominator():
    records = [
        _initial(7, "2026-06-01", "2026-06-10", 110.0),
        _initial(7, "2026-06-02", "2026-06-11", 110.0),
        _revision(7, "2026-06-02", "2026-06-10", 6, 105.0) | {"spot": 100.0},
    ]
    scores = delay_steps(records, {"2026-06-10": 100.0, "2026-06-11": 100.0}, "2026-06-30", draws=20)
    step = next(item for item in scores[7] if item["comparison"] == "initial_vs_remaining" and item["remaining"] == 6)
    assert step["expected"] == 2
    assert step["available"] == 1
    assert step["missing_emissions"] == 1
    assert step["abstentions"] == 0
    assert step["availability_rate"] == 0.5


def test_eight_gains_and_one_loss_stay_uncertain_when_the_block_covers_them():
    gains = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, -60.0]
    rows = [
        {
            "session": f"2026-01-{day:02d}",
            "target": f"2026-02-{day:02d}",
            "gain": gain,
            "initial_error": 20.0,
            "latest_error": 20.0 - gain,
        }
        for day, gain in enumerate(gains, start=1)
    ]
    summary = summarise_horizon(rows, 14, draws=40)
    assert summary["mean_gain"] == 1.25
    assert summary["verdict"] == "incertain"
    assert summary["gain_low"] is None
    assert summary["gain_high"] is None


def test_a_dependent_series_is_not_established_without_two_blocks():
    rows = [
        {
            "session": f"2026-04-{day:02d}",
            "target": f"2026-05-{day:02d}",
            "gain": 10.0,
            "initial_error": 20.0,
            "latest_error": 10.0,
        }
        for day in range(1, 21)
    ]
    summary = summarise_horizon(rows, 14, draws=40)
    assert summary["mean_gain"] == 10.0
    assert summary["usable_blocks"] < 2
    assert summary["verdict"] == "incertain"


def test_a_calendar_hole_is_not_counted_as_the_next_session():
    records = [
        _initial(14, "2026-08-20", "2026-09-09", 100.0),
        _revision(14, "2026-08-21", "2026-09-09", 5, 80.0),
        _revision(14, "2026-08-26", "2026-09-09", 2, 90.0),
    ]
    scored = evaluate_records(records, {"2026-09-09": 100.0}, "2026-09-09", draws=20)
    steps = {row["horizon"]: row for row in scored}[14]["steps"]
    jumped = next(
        step for step in steps if step["comparison"] == "successive" and step["remaining"] == 2
    )
    assert jumped["session_jumps"] >= 1
    assert jumped["missing_emissions"] >= 1
    assert jumped["mean_gain"] is None
    assert all(
        step["mean_gain"] is None
        for step in steps
        if step["comparison"] == "successive"
    )


def test_a_price_correction_or_a_new_cutoff_is_another_measurement(tmp_path: Path):
    from src.validation.revision_value import (
        bind_realized_prices,
        evaluate_records,
        load_prices,
        measurement_identity,
    )

    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text("", encoding="utf-8")
    live = tmp_path / "live.csv"
    live.write_text("date,price\n2026-06-10,100\n", encoding="utf-8")
    summary = tmp_path / "20260609_summary.json"
    summary.write_text(json.dumps({"as_of": "2026-06-09"}), encoding="utf-8")
    bound, digest, differs = bind_realized_prices(tmp_path, live)
    assert differs is False
    assert digest
    first = measurement_identity(sessions, bound, summary, 20)
    gain_before = evaluate_records(
        [_initial(7, "2026-06-01", "2026-06-10", 110.0), _revision(7, "2026-06-02", "2026-06-10", 6, 100.0)],
        load_prices(bound),
        "2026-06-09",
        draws=20,
    )
    live.write_text("date,price\n2026-06-10,80\n", encoding="utf-8")
    again, same_digest, live_differs = bind_realized_prices(tmp_path, live)
    assert again == bound
    assert same_digest == digest
    assert live_differs is True
    assert measurement_identity(sessions, again, summary, 20) == first
    gain_after = evaluate_records(
        [_initial(7, "2026-06-01", "2026-06-10", 110.0), _revision(7, "2026-06-02", "2026-06-10", 6, 100.0)],
        load_prices(again),
        "2026-06-09",
        draws=20,
    )
    assert gain_after[0]["mean_gain"] == gain_before[0]["mean_gain"]
    bound.write_text("date,price\n2026-06-10,120\n", encoding="utf-8")
    versioned, new_digest, _changed = bind_realized_prices(tmp_path, live)
    assert new_digest != digest
    assert versioned != bound
    corrected = evaluate_records(
        [_initial(7, "2026-06-01", "2026-06-10", 110.0), _revision(7, "2026-06-02", "2026-06-10", 6, 100.0)],
        load_prices(versioned),
        "2026-06-10",
        draws=20,
    )
    assert corrected[0]["mean_gain"] != gain_before[0]["mean_gain"]
    summary.write_text(json.dumps({"as_of": "2026-06-10"}), encoding="utf-8")
    assert measurement_identity(sessions, versioned, summary, 20) != first
