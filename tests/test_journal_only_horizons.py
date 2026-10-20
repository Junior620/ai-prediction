"""J+60 et J+90 : mesurés dans le journal, jamais affichés, référence naïve à part."""

import json
from pathlib import Path

from src.models.horizon_schedule import (
    JOURNAL_ONLY_HORIZONS,
    OFFICIAL_HORIZONS,
    TRACKED_HORIZONS,
    scheduled_cap,
    scheduled_weights,
)
from src.models.prediction_journal import append_forecast, tracking_view

ROOT = Path(__file__).resolve().parents[1]


def _append(path, horizon, price, target, *, served_as_close=True, status="experimental", origin="2026-10-19"):
    return append_forecast(
        market="ICE_NY",
        model_version="improved_test",
        release_mode="experimental",
        horizon=horizon,
        price=price,
        lower=None,
        upper=None,
        origin_date=origin,
        origin_price=4377.0,
        target_date=target,
        status=status,
        served_as_close=served_as_close,
        path=path,
    )


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_tracked_horizons_add_60_and_90_without_touching_the_official_ones():
    assert JOURNAL_ONLY_HORIZONS == (60, 90)
    assert set(JOURNAL_ONLY_HORIZONS) <= set(TRACKED_HORIZONS)
    assert OFFICIAL_HORIZONS == (1, 7, 14, 30)
    assert not set(JOURNAL_ONLY_HORIZONS) & set(OFFICIAL_HORIZONS)
    # Poids et plafond : ceux du dernier ancrage, J+30.
    weights = {"30": {"xgboost": 0.8, "prophet": 0.2}, "1": {"xgboost": 1.0}}
    assert scheduled_weights(90, weights) == scheduled_weights(30, weights)
    assert scheduled_cap(60, {1: 6.0, 30: 22.0}) == 22.0


def test_journal_only_rows_keep_naive_and_model_apart(tmp_path):
    journal = tmp_path / "journal.jsonl"
    _append(journal, 60, 4410.0, "2027-01-14")
    _append(journal, 90, None, "2027-02-25", served_as_close=False, status="unavailable")
    row60, row90 = _rows(journal)
    assert row60["journal_only"] is True
    assert row60["opens_initial_id"] is None and row60["revises"] == []
    assert row60["naive_price"] == 4377.0
    assert row60["served_kind"] == "unchanged_close"
    assert row60["candidate_kind"] == "experimental_model" and row60["price"] == 4410.0
    # Sans modèle à 90 jours (lots actuels), la référence naïve reste journalisée.
    assert row90["naive_price"] == 4377.0
    assert row90["candidate_kind"] is None and row90["served_kind"] == "unavailable"


def test_journal_only_rows_never_reach_the_tracking_view(tmp_path):
    journal = tmp_path / "journal.jsonl"
    # Une initiale J+30 existe déjà pour le 30/11 ; une prévision J+60 plus ancienne vise la
    # même échéance. Elle ne doit ni la réviser ni apparaître dans le suivi.
    _append(journal, 30, 4400.0, "2026-11-30", origin="2026-10-19")
    _append(journal, 60, 4420.0, "2026-11-30", origin="2026-09-04")
    rows = _rows(journal)
    assert len(rows) == 2 and rows[1]["horizon"] == 60
    assert rows[1]["revises"] == [] and rows[1]["opens_initial_id"] is None
    view = tracking_view(rows, market="ICE_NY", limit=50)
    assert {item.get("horizon") for item in view["initials"]} == {30}


def test_the_site_only_asks_for_the_official_horizons():
    dashboard = (ROOT / "frontend" / "src" / "components" / "SimplifiedMarketDashboard.tsx").read_text(encoding="utf-8")
    verify = (ROOT / "scripts" / "verify_served_forecast.py").read_text(encoding="utf-8")
    assert "horizons: [1, 7, 14, 30]," in dashboard
    assert "HORIZONS = (1, 7, 14, 30)" in verify
