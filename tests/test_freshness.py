"""Fraîcheur : la veille ouvrée ICE est attendue, le retard et la couverture sont comptés."""

import json

import pandas as pd

from scripts.check_roll_series import uncovered_sessions
from src.monitoring.freshness import (
    banner,
    check,
    expected_last_session,
    freshness_report,
    london_today,
    read_report,
    session_coverage,
    write_report,
)

HOLIDAYS = {pd.Timestamp("2026-12-25"), pd.Timestamp("2026-12-28")}


def _closed(day):
    day = pd.Timestamp(day)
    return day.weekday() >= 5 or day.normalize() in HOLIDAYS


def test_monday_expects_friday_and_holidays_are_skipped():
    assert expected_last_session("2026-10-19", _closed) == pd.Timestamp("2026-10-16")
    assert expected_last_session("2026-10-20", _closed) == pd.Timestamp("2026-10-19")
    # Mardi 29/12 après Noël (vendredi 25) et le lundi 28 fermés : jeudi 24.
    assert expected_last_session("2026-12-29", _closed) == pd.Timestamp("2026-12-24")


def test_report_counts_missing_sessions():
    ok = freshness_report("cocoa", "2026-10-16", today="2026-10-19", is_closed=_closed)
    assert (ok["status"], ok["stale_sessions"], banner(ok)) == ("ok", 0, None)
    late = freshness_report("cocoa", "2026-10-13", today="2026-10-16", is_closed=_closed)
    assert late["status"] == "stale"
    assert late["stale_sessions"] == 2  # 14 et 15 octobre
    assert "13/10" in banner(late) and "2 séance" in banner(late)
    empty = freshness_report("cocoa", None, today="2026-10-16", is_closed=_closed)
    assert empty["status"] == "stale" and empty["last_session"] is None


def test_low_coverage_is_stale_even_when_yesterday_is_present():
    dates = ["2026-10-16", "2026-10-13", "2026-10-06"]
    coverage = session_coverage(dates, "2026-10-19", is_closed=_closed)
    assert coverage < 0.2
    report = freshness_report("coffee_robusta", "2026-10-16", today="2026-10-19", coverage=coverage, is_closed=_closed)
    assert report["stale_sessions"] == 0
    assert report["status"] == "stale"
    assert "30 jours" in report["reason"]


def test_reports_round_trip_and_check_exit_code(tmp_path, capsys):
    write_report(freshness_report("cocoa", "2026-10-16", today="2026-10-19", is_closed=_closed), tmp_path)
    assert read_report("cocoa", tmp_path)["status"] == "ok"
    assert check(["cocoa"], tmp_path) == 0
    assert check(["cocoa", "coffee_robusta"], tmp_path) == 2  # rapport robusta absent
    write_report(freshness_report("coffee_robusta", "2026-10-13", today="2026-10-19", is_closed=_closed), tmp_path)
    assert check(["coffee_robusta"], tmp_path) == 2
    assert "retard" in capsys.readouterr().out
    assert json.loads((tmp_path / "freshness_cocoa.json").read_text(encoding="utf-8"))["market"] == "cocoa"


def test_london_today_uses_london_date():
    assert london_today(pd.Timestamp("2026-10-19 23:30", tz="UTC")) == pd.Timestamp("2026-10-20")
    assert london_today(pd.Timestamp("2026-12-01 23:30", tz="UTC")) == pd.Timestamp("2026-12-01")


def test_snapshot_sessions_beyond_the_roll_series_are_reported():
    series = pd.DataFrame({"date": pd.to_datetime(["2026-10-15", "2026-10-16"])})
    snapshot = pd.DataFrame({"date": ["2026-10-15", "2026-10-16", "2026-10-17", "2026-10-19"]})
    # Le samedi 17 n'est pas une séance ; le lundi 19 n'est pas couvert.
    assert uncovered_sessions(snapshot, series) == ["2026-10-19"]
    assert uncovered_sessions(snapshot.iloc[:2], series) == []
    assert uncovered_sessions(snapshot.iloc[:1], None) == ["2026-10-15"]
