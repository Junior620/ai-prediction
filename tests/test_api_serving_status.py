"""API : séances seulement, sentiment absent plutôt que 0.00, robusta suspendu si incomplet."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

import src.api.app as app_module


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *_):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, n):
        self._rows = self._rows[:n]
        return self

    def execute(self):
        return SimpleNamespace(data=self._rows)


class _Client:
    def __init__(self, rows):
        self._rows = rows

    def table(self, _name):
        return _Query(list(self._rows))


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    app_module._FRESHNESS_CACHE.clear()
    yield
    app_module._FRESHNESS_CACHE.clear()


def test_latest_session_skips_a_saturday_scrape(monkeypatch):
    rows = [{"date": "2026-10-10"}, {"date": "2026-10-09"}, {"date": "2026-10-08"}]
    monkeypatch.setattr(app_module, "supabase_client", _Client(rows))
    assert app_module._latest_price_date("cocoa_london_prices") == "2026-10-09"


def test_sentiment_needs_a_scored_article_in_the_last_24_hours():
    now = datetime.utcnow()
    old = SimpleNamespace(published_at=now - timedelta(days=3), sentiment_score=0.4)
    unscored = SimpleNamespace(published_at=now - timedelta(hours=1), sentiment_score=None)
    fresh = SimpleNamespace(published_at=now - timedelta(hours=2), sentiment_score=-0.1)
    assert app_module._has_scored_news([]) is False
    assert app_module._has_scored_news([old, unscored]) is False
    assert app_module._has_scored_news([old, fresh]) is True


def _items():
    return [
        SimpleNamespace(horizon=h, price=3500.0, confidence_interval=None, status="experimental",
                        components={"status": "experimental", "served_as_close": True})
        for h in (1, 7)
    ]


def test_sparse_series_withholds_every_horizon(monkeypatch):
    # Trois séances sur les 30 derniers jours, comme le robusta en octobre.
    rows = [{"date": d} for d in ("2026-10-16", "2026-10-13", "2026-10-06")]
    monkeypatch.setattr(app_module, "supabase_client", _Client(rows))
    monkeypatch.setattr(
        "src.monitoring.freshness.london_today", lambda now=None: pd.Timestamp("2026-10-19")
    )
    items = _items()
    reason = app_module._withhold_if_incomplete(items, "coffee_robusta", "coffee_robusta_prices")
    assert reason and "30 jours" in reason
    for item in items:
        assert item.price is None and item.status == "unavailable"
        assert item.components["unavailable_reason"] == reason
        assert item.components["served_as_close"] is False


def test_complete_series_is_left_alone(monkeypatch):
    days = pd.bdate_range(end="2026-10-16", periods=40)
    rows = [{"date": d.strftime("%Y-%m-%d")} for d in reversed(days)]
    monkeypatch.setattr(app_module, "supabase_client", _Client(rows))
    monkeypatch.setattr(
        "src.monitoring.freshness.london_today", lambda now=None: pd.Timestamp("2026-10-19")
    )
    monkeypatch.setattr(
        "src.models.hybrid_features.is_closed_collection_date",
        lambda day, holidays=None: pd.Timestamp(day).weekday() >= 5,
    )
    items = _items()
    assert app_module._withhold_if_incomplete(items, "coffee_robusta", "coffee_robusta_prices") is None
    assert [item.price for item in items] == [3500.0, 3500.0]


def test_unreadable_base_does_not_withhold(monkeypatch):
    monkeypatch.setattr(app_module, "supabase_client", None)
    items = _items()
    assert app_module._withhold_if_incomplete(items, "cocoa", "cocoa_london_prices") is None
    assert items[0].price == 3500.0
