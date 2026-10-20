"""Réentraînement : samedi, lot vieux, demande, choc ou dégradation ; jamais sur séance incomplète."""

import json
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.models import retrain_policy as policy

OK_FRESHNESS = {"status": "ok", "expected_last_session": "2026-10-19"}
GUARD_OK = policy.training_guard(OK_FRESHNESS, "2026-10-19", "2026-10-19")
TUESDAY = date(2026, 10, 20)
SATURDAY = date(2026, 10, 24)


def _closes(n=400, seed=3, shock_at=None, shock=0.0, vol_from=None, vol=1.0):
    # Bruit borné : aucune séance ordinaire ne dépasse 4 fois la médiane des rendements absolus.
    rng = np.random.default_rng(seed)
    returns = rng.uniform(-0.01, 0.01, n)
    if vol_from is not None:
        returns[vol_from:] *= vol
    if shock_at is not None:
        returns[shock_at] = shock
    days = pd.bdate_range("2025-03-03", periods=n)
    return pd.Series(4000.0 * np.exp(np.cumsum(returns)), index=days)


def test_guard_blocks_missing_and_incomplete_sessions():
    assert GUARD_OK["allowed"] is True
    stale = policy.training_guard({"status": "stale", "reason": "retard de 1 séance(s)"}, "2026-10-16", "2026-10-19")
    assert stale["allowed"] is False and "séance attendue manquante" in stale["reasons"][0]
    partial = policy.training_guard(OK_FRESHNESS, "2026-10-20", "2026-10-19")
    assert partial["allowed"] is False and "incomplète" in partial["reasons"][0]
    assert policy.training_guard(None, "2026-10-19", "2026-10-19")["allowed"] is False
    assert policy.training_guard(OK_FRESHNESS, "2026-10-19", None)["allowed"] is False


def test_scheduled_triggers():
    fresh_lot = "improved_20261019_052744"
    assert not policy.decide("cocoa", today=TUESDAY, lot_version=fresh_lot, guard=GUARD_OK).retrain
    saturday = policy.decide("cocoa", today=SATURDAY, lot_version=fresh_lot, guard=GUARD_OK)
    assert saturday.retrain and "samedi" in saturday.reasons
    # Deux passages le samedi : le lot du matin n'est pas réentraîné le soir.
    assert not policy.decide("cocoa", today=SATURDAY, lot_version="improved_20261024_061500", guard=GUARD_OK).retrain
    old = policy.decide("cocoa", today=TUESDAY, lot_version="improved_20261012_045358", guard=GUARD_OK)
    assert old.retrain and old.lot_age_days == 8
    forced = policy.decide("cocoa", today=TUESDAY, lot_version=fresh_lot, guard=GUARD_OK, force=True)
    assert forced.retrain and "--retrain demandé" in forced.reasons
    weekly = policy.decide("cocoa", today=TUESDAY, lot_version=fresh_lot, guard=GUARD_OK, weekly=True)
    assert weekly.retrain
    assert policy.decide("cocoa", today=TUESDAY, lot_version=None, guard=GUARD_OK).retrain


def test_a_blocked_retrain_keeps_the_lot_and_says_why():
    guard = policy.training_guard(OK_FRESHNESS, "2026-10-20", "2026-10-19")
    decision = policy.decide("cocoa", today=SATURDAY, lot_version="improved_20261019_052744", guard=guard)
    assert decision.wanted and not decision.retrain
    assert "incomplète" in decision.blocked_by[0]
    assert decision.summary().startswith("réentraînement reporté")


def test_a_return_shock_after_the_lot_triggers():
    closes = _closes(shock_at=390, shock=0.12)
    trained = closes.index[385].date()
    shock = policy.market_shock(closes, trained)
    assert shock["evaluated"] and shock["triggered"]
    assert shock["return_shocks"] == [closes.index[390].strftime("%Y-%m-%d")]
    # Le lot entraîné après le choc l'a vu : pas de nouveau déclenchement.
    assert not policy.market_shock(closes, closes.index[391].date())["triggered"]


def test_a_volatility_regime_triggers_once_at_its_crossing():
    closes = _closes(vol_from=370, vol=4.0)
    shock = policy.market_shock(closes, closes.index[300].date())
    assert shock["triggered"] and len(shock["volatility_crossings"]) >= 1
    # Le régime persiste : le lot entraîné après le franchissement n'est pas relancé par lui.
    later = policy.market_shock(closes, pd.Timestamp(shock["volatility_crossings"][-1]).date())
    assert later["volatility_crossings"] == []


def test_a_quiet_series_and_a_contract_roll_do_not_trigger():
    closes = _closes()
    assert not policy.market_shock(closes, closes.index[300].date())["triggered"]
    # Sans série ajustée, un roulement n'est pas pris pour un choc : rien n'est évalué.
    unadjusted = policy.market_shock(None, date(2026, 10, 1))
    assert unadjusted == {"evaluated": False, "triggered": False, "reason": "pas de série sans saut pour ce marché"}
    short = policy.market_shock(_closes(n=100), None)
    assert short["evaluated"] is False


def _journal(n, model_error, naive_error=10.0, version="improved_20260901_000000", start="2026-01-05"):
    rows, actuals = [], {}
    for i, day in enumerate(pd.bdate_range(start, periods=n)):
        target = day.strftime("%Y-%m-%d")
        actuals[target] = 4000.0
        rows.append(
            {
                "market": "ICE_London",
                "horizon": 1,
                "model_version": version,
                "price": 4000.0 + model_error(i),
                "naive_price": 4000.0 + naive_error,
                "target_date": target,
                "diagnostic": False,
            }
        )
    return rows, actuals


def test_degradation_compares_recent_and_reference_windows():
    rows, actuals = _journal(80, lambda i: 8.0 if i < 60 else 12.0)
    result = policy.degradation(rows, actuals, {"ICE_London"})
    assert result["evaluated"] and result["triggered"]
    assert result["recent_ratio"] == 1.2 and result["reference_ratio"] == 0.8
    stable, actuals = _journal(80, lambda i: 8.0)
    assert not policy.degradation(stable, actuals, {"ICE_London"})["triggered"]


def test_degradation_needs_fifteen_matured_and_skips_contaminated_lots():
    rows, actuals = _journal(30, lambda i: 12.0)
    assert policy.degradation(rows, actuals, {"ICE_London"})["evaluated"] is False
    rows, actuals = _journal(80, lambda i: 8.0 if i < 60 else 12.0, version="improved_20261010_182417")
    excluded = policy.degradation(rows, actuals, {"ICE_London"}, excluded={"improved_20261010_182417"})
    assert excluded["matured"] == 0 and not excluded["triggered"]
    # Diagnostics, horizons du journal seul et autres marchés ne comptent pas.
    for row in rows:
        row["diagnostic"] = True
    assert policy.degradation(rows, actuals, {"ICE_London"})["matured"] == 0


def test_early_triggers_wait_three_days_after_a_retrain():
    shock = {"triggered": True, "return_shocks": ["2026-10-19"], "volatility_crossings": []}
    recent = policy.decide("cocoa", today=TUESDAY, lot_version="improved_20261019_052744", guard=GUARD_OK, shock=shock)
    assert not recent.retrain and recent.ignored
    older = policy.decide("cocoa", today=TUESDAY, lot_version="improved_20261016_052744", guard=GUARD_OK, shock=shock)
    assert older.retrain and older.reasons[0].startswith("choc de marché")
    degraded = {"triggered": True, "recent_ratio": 1.3, "reference_ratio": 0.9}
    worse = policy.decide("cocoa", today=TUESDAY, lot_version="improved_20261016_052744", guard=GUARD_OK, degraded=degraded)
    assert worse.retrain and "dégradation J+1" in worse.reasons[0]


def test_the_decision_is_written_with_its_reason(tmp_path):
    decision = policy.decide("cocoa", today=SATURDAY, lot_version="improved_20261019_052744", guard=GUARD_OK)
    path = policy.write_decision(decision, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "retrain_cocoa.json"
    assert payload["retrain"] is True and payload["reasons"] == ["samedi"]
    assert payload["guard"]["last_complete_day"] == "2026-10-19"


def test_contaminated_versions_read_the_inventory(tmp_path):
    path = tmp_path / "oi_contamination.json"
    assert policy.contaminated_versions(path) == set()
    path.write_text(json.dumps({"contaminated_versions": ["improved_20261010_182417"]}), encoding="utf-8")
    assert policy.contaminated_versions(path) == {"improved_20261010_182417"}


def test_clock_rule_matches_databento():
    closed = lambda day: pd.Timestamp(day).weekday() >= 5  # noqa: E731
    assert policy.clock_last_complete_day(datetime(2026, 10, 20, 4, 0), closed) == "2026-10-19"
    assert policy.clock_last_complete_day(datetime(2026, 10, 20, 15, 0), closed) == "2026-10-19"
    assert policy.clock_last_complete_day(datetime(2026, 10, 20, 21, 30), closed) == "2026-10-20"
    assert policy.clock_last_complete_day(datetime(2026, 10, 24, 22, 0), closed) == "2026-10-23"
    assert policy.lot_trained_on("improved_20261020_055547") == date(2026, 10, 20)
