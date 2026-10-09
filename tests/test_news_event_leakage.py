"""Future articles, late corrections and unverified dates stay out of training rows."""

import importlib.util
from pathlib import Path

from src.models.hybrid_features import resolve_feature_cols
from src.models.served_forecast import compose_served_price
from src.nlp.event_features import feature_row
from src.nlp.nlp_analyzer import NLPAnalyzer

_spec = importlib.util.spec_from_file_location(
    "run_gdelt_pilot",
    Path(__file__).resolve().parents[1] / "scripts" / "run_gdelt_pilot.py",
)
_pilot = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_pilot)
_parse_seendate = _pilot._parse_seendate


def _event(**overrides):
    base = {
        "url": "https://example.test/a",
        "event_key": "evt-a",
        "title": "Swollen shoot cuts Ivory Coast cocoa",
        "category": "disease",
        "country": "CI",
        "severity": 1.0,
        "economic_direction": "hausse",
        "important": True,
        "available_at": "2024-03-01T08:00:00+00:00",
        "classified_at": "2024-03-01T08:00:00+00:00",
        "published_at": "2024-03-01T07:00:00+00:00",
        "published_at_verified": True,
    }
    base.update(overrides)
    return base


def test_event_available_after_origin_is_excluded():
    rows = [
        _event(),
        _event(
            url="https://example.test/later",
            event_key="evt-later",
            available_at="2024-03-20T08:00:00+00:00",
            classified_at="2024-03-20T08:00:00+00:00",
        ),
    ]
    features = feature_row(rows, "2024-03-10T12:00:00+00:00")
    assert features["disease_count_30"] == 1.0


def test_late_correction_does_not_change_an_earlier_origin():
    rows = [
        _event(severity=1.0, classified_at="2024-03-01T08:00:00+00:00"),
        _event(severity=5.0, classified_at="2024-03-25T08:00:00+00:00", content_hash="new"),
    ]
    early = feature_row(rows, "2024-03-10T12:00:00+00:00")
    late = feature_row(rows, "2024-03-26T12:00:00+00:00")
    assert early["disease_severity_30"] == 1.0
    assert late["disease_severity_30"] == 5.0


def test_published_date_without_availability_is_excluded():
    rows = [
        _event(
            available_at=None,
            published_at="2024-02-01T00:00:00+00:00",
            published_at_verified=True,
            classified_at="2024-03-01T08:00:00+00:00",
        )
    ]
    features = feature_row(rows, "2024-03-10T12:00:00+00:00")
    assert features["disease_count_30"] == 0.0


def test_duplicate_after_origin_does_not_merge_into_the_past_count():
    rows = [
        _event(event_key="same", url="https://example.test/1"),
        _event(
            event_key="same",
            url="https://example.test/2",
            available_at="2024-03-18T08:00:00+00:00",
            classified_at="2024-03-18T08:00:00+00:00",
        ),
    ]
    features = feature_row(rows, "2024-03-10T12:00:00+00:00")
    assert features["disease_count_14"] == 1.0


def test_direction_stays_uncertain_without_an_economic_cue():
    label = NLPAnalyzer.classify_event(
        "Cocoa market note",
        "Traders watched the session.",
        source="Ecofin",
        source_weight=1.0,
        published_at_verified=False,
        first_seen_at="2024-03-01T08:00:00+00:00",
    )
    assert label["economic_direction"] == "incertain"
    assert label["rules_version"]


def test_direction_is_not_implied_by_category_alone():
    label = NLPAnalyzer.classify_event(
        "Ivory Coast harvest update",
        "Officials published the harvest figures.",
        source="ICCO",
        source_weight=1.5,
        published_at_verified=True,
        published_at="2024-03-01T08:00:00+00:00",
        first_seen_at="2024-03-01T09:00:00+00:00",
    )
    assert label["category"] == "harvest"
    assert label["economic_direction"] == "incertain"
    assert label["country"] == "CI"


def test_gdelt_seendate_with_separator_is_utc():
    assert _parse_seendate("20240315T153000Z") == "2024-03-15T15:30:00+00:00"
    assert _parse_seendate("") is None


def test_published_price_still_ignores_sentiment_and_event_columns():
    served = compose_served_price(
        xgb_price=2000.0,
        prophet_price=2000.0,
        nhits_price=None,
        weights={"xgb": 0.7, "prophet": 0.3, "nhits": 0.0},
        spot=2000.0,
        horizon=30,
    )
    assert served["sentiment"] == 0.0
    columns = resolve_feature_cols(include_ohlcv=True, include_oi=True)
    assert "disease_count_30" not in columns
    source = Path("train_hybrid_improved.py").read_text(encoding="utf-8")
    assert "event_features" not in source
