"""Borne de fin Databento : la séance de la veille est demandée dès qu'elle est publiée."""

from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from src.data_collection.databento_london_collector import _inclusive, daily_end_exclusive, fetch_daily_bars


class _Metadata:
    def __init__(self, end=None, fail=False):
        self._end = end
        self._fail = fail

    def get_dataset_range(self, dataset):
        if self._fail:
            raise RuntimeError("réseau")
        return {"start": "2018-12-23T00:00:00Z", "end": self._end, "schema": {"ohlcv-1d": {"end": self._end}}}


class _Timeseries:
    def __init__(self):
        self.calls = []

    def get_range(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(to_df=lambda: pd.DataFrame())


def _client(end=None, fail=False):
    return SimpleNamespace(metadata=_Metadata(end, fail), timeseries=_Timeseries())


TUESDAY_MORNING = datetime(2026, 10, 20, 4, 14)


def test_monday_is_requested_without_going_past_the_licensed_range():
    # Constaté le 20/10 : plage jusqu'au lundi 19/10 22:05 UTC. Minuit du 20 renvoie
    # « 422 dataset_unavailable_range » (au-delà, licence requise). La barre du lundi,
    # datée du 19 à minuit, est dans la borne.
    client = _client("2026-10-19T22:05:00.000000000Z")
    end = daily_end_exclusive(client, now=TUESDAY_MORNING)
    assert end == "2026-10-19T22:04:00"
    assert _inclusive(end) == "2026-10-19"
    assert _inclusive("2026-10-20") == "2026-10-19"


def test_a_day_still_in_progress_is_not_requested():
    # Plage ouverte jusqu'au mardi 13:00 : la barre du mardi serait partielle.
    client = _client("2026-10-20T13:00:00Z")
    assert daily_end_exclusive(client, now=datetime(2026, 10, 20, 13, 30)) == "2026-10-20"


def test_publication_delay_keeps_the_previous_day():
    # Lundi pas encore traité (plage jusqu'au lundi 10:00) : on s'arrête au vendredi inclus.
    client = _client("2026-10-19T10:00:00Z")
    assert daily_end_exclusive(client, now=datetime(2026, 10, 19, 12, 0)) == "2026-10-19"


def test_never_beyond_today_and_fallback_when_metadata_fails():
    assert daily_end_exclusive(_client("2026-10-25T23:00:00Z"), now=TUESDAY_MORNING) == "2026-10-20"
    assert daily_end_exclusive(_client(fail=True), now=TUESDAY_MORNING) == "2026-10-19"


def test_fetch_daily_bars_asks_for_yesterday(monkeypatch):
    import src.data_collection.databento_london_collector as mod

    class _Now(datetime):
        @classmethod
        def utcnow(cls):
            return TUESDAY_MORNING

    monkeypatch.setattr(mod, "datetime", _Now)
    client = _client("2026-10-19T22:05:00Z")
    fetch_daily_bars(start="2026-10-10", client=client)
    # Exclusif, sous la fin de plage publiée : la barre du lundi 19 est incluse.
    assert client.timeseries.calls[0]["end"] == "2026-10-19T22:04:00"


def test_midnight_is_used_only_when_the_range_goes_past_it():
    # Plage arrêtée pile à minuit : on reste juste en dessous. Plage au-delà : minuit.
    client = _client("2026-10-20T00:00:00Z")
    assert daily_end_exclusive(client, now=datetime(2026, 10, 20, 6, 0)) == "2026-10-19T23:59:00"
    client = _client("2026-10-20T03:00:00Z")
    assert daily_end_exclusive(client, now=datetime(2026, 10, 20, 6, 0)) == "2026-10-20"
