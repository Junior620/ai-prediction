"""Disponibilité Databento journalisée, et open interest coupé tant qu'il ne rapporte rien."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import src.data_collection.databento_london_collector as mod
from src.data_collection.databento_london_collector import daily_bounds, fetch_daily_bars, open_interest_enabled

ROOT = Path(__file__).resolve().parents[1]
TUESDAY_MORNING = datetime(2026, 10, 20, 4, 14)


class _Metadata:
    def __init__(self, end=None, fail=False):
        self._end = end
        self._fail = fail

    def get_dataset_range(self, dataset):
        if self._fail:
            raise RuntimeError("réseau")
        return {"end": self._end, "schema": {"ohlcv-1d": {"end": self._end}}}


class _Timeseries:
    def __init__(self, frame=None, fail=False):
        self.frame = frame if frame is not None else pd.DataFrame()
        self.fail = fail

    def get_range(self, **kwargs):
        if self.fail:
            raise RuntimeError("422 dataset_unavailable_range")
        return SimpleNamespace(to_df=lambda: self.frame)


def _client(end=None, fail=False, frame=None, series_fail=False):
    return SimpleNamespace(metadata=_Metadata(end, fail), timeseries=_Timeseries(frame, series_fail))


def test_bounds_record_what_databento_published():
    bounds = daily_bounds(_client("2026-10-19T22:05:00Z"), now=TUESDAY_MORNING)
    assert bounds["available_end"] == "2026-10-19T22:05:00Z"
    assert bounds["end_exclusive"] == "2026-10-19T22:04:00"
    assert bounds["last_complete_day"] == "2026-10-19"
    assert bounds["fetched_at"] == "2026-10-20T04:14:00Z"
    assert bounds["source"] == "metadata"


def test_bounds_fall_back_without_inventing_a_session():
    bounds = daily_bounds(_client(fail=True), now=TUESDAY_MORNING)
    assert bounds["available_end"] is None
    assert bounds["last_complete_day"] == "2026-10-18"
    assert bounds["source"] == "fallback"


def test_a_day_in_progress_is_not_complete():
    # Plage ouverte jusqu'au mardi 13:00 : la dernière journée complète reste le lundi.
    bounds = daily_bounds(_client("2026-10-20T13:00:00Z"), now=datetime(2026, 10, 20, 13, 30))
    assert bounds["last_complete_day"] == "2026-10-19"


def _patch_now(monkeypatch):
    class _Now(datetime):
        @classmethod
        def utcnow(cls):
            return TUESDAY_MORNING

    monkeypatch.setattr(mod, "datetime", _Now)


def test_each_attempt_logs_availability(monkeypatch):
    _patch_now(monkeypatch)
    frame = pd.DataFrame(
        {"symbol": ["C.v.0"], "open": [4380.0], "high": [4400.0], "low": [4350.0], "close": [4377.0], "volume": [6708.0]},
        index=pd.to_datetime(["2026-10-19"], utc=True),
    )
    result = fetch_daily_bars(start="2026-10-10", client=_client("2026-10-19T22:05:00Z", frame=frame))
    attempt = result.attempts[-1]
    assert attempt["available_end"] == "2026-10-19T22:05:00Z"
    assert attempt["last_complete_day"] == "2026-10-19"
    assert attempt["last_bar_day"] == "2026-10-19"
    assert attempt["fetched_at"] == "2026-10-20T04:14:00Z"
    failed = fetch_daily_bars(start="2026-10-10", client=_client("2026-10-19T22:05:00Z", series_fail=True))
    assert failed.attempts[-1]["ok"] is False
    assert failed.attempts[-1]["last_complete_day"] == "2026-10-19"


def test_open_interest_is_off_by_default_and_can_be_forced(monkeypatch):
    monkeypatch.delenv("DATABENTO_OPEN_INTEREST", raising=False)
    assert open_interest_enabled() is False
    monkeypatch.setenv("DATABENTO_OPEN_INTEREST", "1")
    assert open_interest_enabled() is True
    monkeypatch.setenv("DATABENTO_OPEN_INTEREST", "0")
    assert open_interest_enabled() is False


def test_collection_skips_the_statistics_requests_when_disabled():
    source = (ROOT / "collect_ice_london_cocoa.py").read_text(encoding="utf-8")
    assert "include_oi=True)" not in source
    assert "fetch_latest_curve(price_bounds=market.price_bounds, include_oi=fetch_oi)" in source
    backfill = source.index("backfill_recent_oi(start=start_bf")
    guard = source.index("if not fetch_oi:\n        # La courbe ci-dessus")
    assert guard < backfill
