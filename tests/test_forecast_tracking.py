"""Journal d'une échéance : double rôle, verrou, cadence, cours de départ."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.models.horizon_schedule import scheduled_cap, scheduled_weights
from src.models.hybrid_features import future_business_date
from src.models.improved_price_predictor import ImprovedPricePredictor
from src.models.prediction_journal import (
    JournalLockError,
    append_forecast,
    load_forecasts,
    overlay_commitments,
    sessions_between,
    tracking_view,
)
from src.validation.revision_stability import lot_index, params_frozen, study_window


def _model(card: dict) -> float | None:
    """Model price of a card; a non-validated card publishes the close beside it."""
    parts = card.get("components") or {}
    if parts.get("served_as_close"):
        return parts.get("candidate_price")
    return card["price"]

COMMON = dict(
    market="ICE_NY",
    model_version="improved_a",
    release_mode="experimental",
    procedure_fingerprint="proc",
    data_fingerprint="data",
    status="experimental",
)


def _write(path: Path, **overrides):
    payload = dict(COMMON)
    payload.update(overrides)
    return append_forecast(path=path, **payload)


def test_intermediate_weights_are_interpolated_then_normalised():
    weights = {
        "1": {"xgb": 0.45, "nhits": 0.45, "prophet": 0.10},
        "7": {"xgb": 0.40, "nhits": 0.45, "prophet": 0.15},
    }
    mixed = scheduled_weights(4, weights)
    assert mixed["xgb"] == pytest.approx(0.425)
    assert mixed["nhits"] == pytest.approx(0.45)
    assert mixed["prophet"] == pytest.approx(0.125)
    assert sum(mixed.values()) == pytest.approx(1.0)
    cocoa = {"1": {"xgb": 1.0, "nhits": 0.0, "prophet": 0.0}, "7": {"xgb": 1.0, "nhits": 0.0, "prophet": 0.0}}
    assert scheduled_weights(4, cocoa)["xgb"] == pytest.approx(1.0)
    assert scheduled_weights(4, cocoa)["nhits"] == pytest.approx(0.0)
    assert scheduled_cap(4, {"1": 6.0, "7": 14.0}) == pytest.approx(10.0)


def test_missing_direct_model_is_not_replaced():
    predictor = ImprovedPricePredictor.__new__(ImprovedPricePredictor)
    predictor.direct_horizon_models = {}
    predictor.feature_cols = []
    predictor.direct_target = "close_delta"
    price = predictor._predict_xgb_price(7, None, None, 100.0, None)
    assert math.isnan(price)


def test_unavailable_row_is_revised_when_the_price_arrives(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    first = _write(
        journal,
        horizon=1,
        price=None,
        lower=None,
        upper=None,
        status="unavailable",
        origin_date="2026-10-13",
        origin_price=3475.0,
        target_date="2026-10-14",
    )
    second = _write(
        journal,
        horizon=1,
        price=3400.0,
        lower=3300.0,
        upper=3500.0,
        origin_date="2026-10-13",
        origin_price=3475.0,
        target_date="2026-10-14",
    )
    third = _write(
        journal,
        horizon=1,
        price=3400.0,
        lower=3300.0,
        upper=3500.0,
        origin_date="2026-10-13",
        origin_price=3475.0,
        target_date="2026-10-14",
    )
    assert first != "duplicate"
    assert second not in {"duplicate", first}
    assert third == "duplicate"
    rows = load_forecasts(journal)
    assert rows[1]["opens_initial_id"] is None
    assert first in rows[1]["revises"]
    view = tracking_view(rows, market="ICE_NY", initial_id=first)
    assert view["latest"]["price"] == 3400.0


def test_j7_opens_its_target_and_revises_an_older_j14(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    older = _write(
        journal,
        horizon=14,
        price=110.0,
        lower=100.0,
        upper=120.0,
        origin_date="2026-09-01",
        origin_price=100.0,
        target_date="2026-09-21",
    )
    newer = _write(
        journal,
        horizon=7,
        price=111.0,
        lower=101.0,
        upper=121.0,
        origin_date="2026-09-10",
        origin_price=104.0,
        target_date="2026-09-21",
    )
    rows = load_forecasts(journal)
    revised = next(row for row in rows if row["forecast_id"] == newer)
    assert revised["opens_initial_id"] == newer
    assert older in revised["revises"]
    view = tracking_view(rows, market="ICE_NY", initial_id=older)
    assert view["revisions"][0]["forecast_id"] == newer
    own = tracking_view(rows, market="ICE_NY", initial_id=newer)
    assert own["revisions"] == []


def test_j7_then_j6_keeps_the_same_target(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    initial = _write(
        journal,
        horizon=7,
        price=110.0,
        lower=None,
        upper=None,
        origin_date="2026-09-01",
        origin_price=100.0,
        target_date="2026-09-10",
    )
    _write(
        journal,
        horizon=6,
        price=109.0,
        lower=None,
        upper=None,
        origin_date="2026-09-02",
        origin_price=101.0,
        target_date="2026-09-10",
    )
    view = tracking_view(load_forecasts(journal), market="ICE_NY", initial_id=initial)
    assert view["revisions"][0]["horizon"] == 6
    assert view["revisions"][0]["target_date"] == "2026-09-10"
    assert view["latest"]["forecast_id"] != initial


def test_closure_moves_the_target_without_filling_the_gap():
    origin = datetime(2026, 10, 6)
    opened = future_business_date(origin, 1, holidays=[])
    closed = future_business_date(origin, 1, holidays=["2026-10-07"])
    assert closed.date() != opened.date()
    assert closed.date().isoformat() != "2026-10-07"


def test_duplicate_call_does_not_add_a_row(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    first = _write(
        journal,
        horizon=1,
        price=10.0,
        lower=9.0,
        upper=11.0,
        origin_date="2026-10-01",
        origin_price=10.0,
        target_date="2026-10-02",
    )
    second = _write(
        journal,
        horizon=1,
        price=10.0,
        lower=9.0,
        upper=11.0,
        origin_date="2026-10-01",
        origin_price=10.0,
        target_date="2026-10-02",
    )
    assert second == "duplicate"
    assert len(load_forecasts(journal)) == 1
    assert load_forecasts(journal)[0]["forecast_id"] == first


def test_resume_after_lock_failure_reuses_the_same_id(tmp_path: Path, monkeypatch):
    journal = tmp_path / "journal.jsonl"
    state = {"fail": True}
    real = __import__("src.models.prediction_journal", fromlist=["_acquire"])._acquire

    def wrapped(lock, retries=8, delay=0.25):
        if state["fail"]:
            state["fail"] = False
            raise JournalLockError("verrou")
        return real(lock, retries=1, delay=0.0)

    monkeypatch.setattr("src.models.prediction_journal._acquire", wrapped)
    payload = dict(
        horizon=1,
        price=10.0,
        lower=9.0,
        upper=11.0,
        origin_date="2026-10-01",
        origin_price=10.0,
        target_date="2026-10-02",
    )
    with pytest.raises(JournalLockError):
        _write(journal, **payload)
    assert not journal.exists()
    written = _write(journal, **payload)
    again = _write(journal, **payload)
    assert again == "duplicate"
    assert load_forecasts(journal)[0]["forecast_id"] == written


def test_new_lot_and_data_correction_add_rows_without_rewriting(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    first = _write(
        journal,
        horizon=7,
        price=100.0,
        lower=90.0,
        upper=110.0,
        origin_date="2026-10-01",
        origin_price=100.0,
        target_date="2026-10-12",
    )
    corrected = _write(
        journal,
        horizon=7,
        price=101.0,
        lower=91.0,
        upper=111.0,
        origin_date="2026-10-01",
        origin_price=100.0,
        target_date="2026-10-12",
        data_fingerprint="corrected",
    )
    new_lot = _write(
        journal,
        horizon=7,
        price=102.0,
        lower=92.0,
        upper=112.0,
        origin_date="2026-10-01",
        origin_price=100.0,
        target_date="2026-10-12",
        model_version="improved_b",
    )
    rows = load_forecasts(journal)
    assert [row["forecast_id"] for row in rows] == [first, corrected, new_lot]
    assert rows[0]["price"] == 100.0


def test_old_row_stays_readable_and_incomplete(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    journal.write_text(json.dumps({"market": "ICE_NY", "price": 1, "horizon": 1}) + "\n", encoding="utf-8")
    loaded = load_forecasts(journal)
    assert loaded[0]["trace"] == "incomplete"
    assert "trace" not in journal.read_text(encoding="utf-8")


def test_unavailable_initial_is_not_rewritten(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    initial = _write(
        journal,
        horizon=7,
        price=None,
        lower=None,
        upper=None,
        origin_date="2026-09-01",
        origin_price=100.0,
        target_date="2026-09-10",
        status="unavailable",
    )
    _write(
        journal,
        horizon=6,
        price=108.0,
        lower=None,
        upper=None,
        origin_date="2026-09-02",
        origin_price=101.0,
        target_date="2026-09-10",
    )
    view = tracking_view(load_forecasts(journal), market="ICE_NY", initial_id=initial)
    assert view["initial"]["status"] == "unavailable"
    assert view["initial"]["price"] is None
    assert view["latest"]["price"] == 108.0


def test_direction_uses_the_frozen_reference_not_the_moving_spot(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    initial = _write(
        journal,
        horizon=7,
        price=110.0,
        lower=None,
        upper=None,
        origin_date="2026-09-01",
        origin_price=100.0,
        target_date="2026-09-10",
    )
    _write(
        journal,
        horizon=6,
        price=110.0,
        lower=None,
        upper=None,
        origin_date="2026-09-02",
        origin_price=112.0,
        target_date="2026-09-10",
    )
    view = tracking_view(load_forecasts(journal), market="ICE_NY", initial_id=initial)
    assert view["direction"] == "hausse"
    assert view["direction_vs_emission_spot"] == "baisse"
    assert view["inversions"] == 0
    assert view["gap"] == 0.0


def test_real_inversion_is_counted_once(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    initial = _write(
        journal,
        horizon=14,
        price=110.0,
        lower=None,
        upper=None,
        origin_date="2026-09-01",
        origin_price=100.0,
        target_date="2026-09-21",
    )
    _write(
        journal,
        horizon=7,
        price=97.0,
        lower=None,
        upper=None,
        origin_date="2026-09-10",
        origin_price=100.0,
        target_date="2026-09-21",
    )
    view = tracking_view(load_forecasts(journal), market="ICE_NY", initial_id=initial)
    assert view["direction"] == "baisse"
    assert view["inversions"] == 1


def test_separate_targets_are_not_mixed(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    first = _write(
        journal,
        horizon=14,
        price=110.0,
        lower=None,
        upper=None,
        origin_date="2026-09-01",
        origin_price=100.0,
        target_date="2026-09-21",
    )
    other = _write(
        journal,
        horizon=7,
        price=120.0,
        lower=None,
        upper=None,
        origin_date="2026-09-10",
        origin_price=104.0,
        target_date="2026-09-21-other",
    )
    view = tracking_view(load_forecasts(journal), market="ICE_NY", initial_id=first)
    assert view["revisions"] == []
    assert other != first


def test_maturity_and_missing_realized_price(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    initial = _write(
        journal,
        horizon=1,
        price=110.0,
        lower=None,
        upper=None,
        origin_date="2026-10-01",
        origin_price=100.0,
        target_date="2026-10-02",
    )
    _write(
        journal,
        horizon=1,
        price=104.0,
        lower=None,
        upper=None,
        origin_date="2026-10-01",
        origin_price=100.0,
        target_date="2026-10-02",
        model_version="improved_b",
        data_fingerprint="revision",
    )
    rows = load_forecasts(journal)
    mature = tracking_view(rows, market="ICE_NY", initial_id=initial, realized_price=108.0, target_reached=True)
    assert mature["mature"] is True
    assert mature["realized_price"] == 108.0
    assert mature["initial_error"] == pytest.approx(2.0)
    assert mature["latest_error"] == pytest.approx(-4.0)
    assert mature["initial"]["error"] == pytest.approx(2.0)
    assert mature["revisions"][0]["error"] == pytest.approx(-4.0)
    missing = tracking_view(rows, market="ICE_NY", initial_id=initial, realized_price=None, target_reached=True)
    assert missing["mature"] is True
    assert missing["realized_price"] is None
    assert missing["initial_error"] is None
    assert missing["latest_error"] is None
    assert missing["revisions"][0]["error"] is None


def test_diagnostic_is_not_a_published_revision(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    initial = _write(
        journal,
        horizon=7,
        price=110.0,
        lower=None,
        upper=None,
        origin_date="2026-09-01",
        origin_price=100.0,
        target_date="2026-09-10",
    )
    _write(
        journal,
        horizon=7,
        price=112.0,
        lower=None,
        upper=None,
        origin_date="2026-09-01",
        origin_price=100.0,
        target_date="2026-09-10",
        diagnostic=True,
        previous_model_version="improved_old",
        previous_price=109.0,
        comparison="disponible",
        model_version="improved_b",
    )
    view = tracking_view(load_forecasts(journal), market="ICE_NY", initial_id=initial)
    assert view["revisions"] == []
    assert view["inversions"] == 0
    assert view["model_comparison"]["previous_price"] == 109.0


def test_five_session_lot_stays_frozen_then_renews():
    assert [lot_index(index) for index in range(6)] == [0, 0, 0, 0, 0, 1]
    assert params_frozen(0) is False
    assert params_frozen(4) is True
    assert params_frozen(5) is False


def test_missing_sessions_are_not_moved():
    dates = pd.bdate_range("2026-06-01", periods=80)
    frame = pd.DataFrame({"date": dates, "price": range(len(dates))})
    dropped = frame.drop(frame.index[10]).reset_index(drop=True)
    origins, sessions, _missing = study_window(dropped, as_of="2026-09-30", n_origins=60, horizon=30)
    assert dates[10] not in set(sessions)
    assert origins
    assert origins[0] in set(sessions)


def test_concurrent_writers_keep_one_row(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, r'E:\\\\prediction')\n"
        "from src.models.prediction_journal import append_forecast\n"
        "append_forecast(market='ICE_NY', model_version='v', release_mode='experimental',\n"
        " horizon=1, price=10, lower=9, upper=11, origin_date='2026-10-01',\n"
        " origin_price=10, target_date='2026-10-02', status='experimental',\n"
        " procedure_fingerprint='p', data_fingerprint='d',\n"
        f" path=Path(r'{journal}'))\n",
        encoding="utf-8",
    )
    import subprocess
    import sys

    processes = [
        subprocess.Popen([sys.executable, str(worker)], cwd=r"E:\prediction"),
        subprocess.Popen([sys.executable, str(worker)], cwd=r"E:\prediction"),
    ]
    codes = [item.wait(timeout=30) for item in processes]
    assert codes == [0, 0]
    assert len(load_forecasts(journal)) == 1


@pytest.mark.skipif(os.environ.get("RUN_DOCKER_JOURNAL") != "1", reason="journal Docker sur demande")
def test_docker_and_windows_share_the_journal(tmp_path: Path):
    del tmp_path
    shared = Path(r"E:\prediction\data\_journal_lock_test.jsonl")
    shared.unlink(missing_ok=True)
    import subprocess
    import sys

    local = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from src.models.prediction_journal import append_forecast; "
            "append_forecast(market='ICE_NY', model_version='win', release_mode='experimental', "
            "horizon=1, price=1, lower=None, upper=None, origin_date='2026-01-01', origin_price=1, "
            "target_date='2026-01-02', status='experimental', procedure_fingerprint='p', "
            "data_fingerprint='d', path=__import__('pathlib').Path(r'E:\\prediction\\data\\_journal_lock_test.jsonl'))",
        ],
        cwd=r"E:\prediction",
    )
    docker = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "api",
            "python",
            "-c",
            "from src.models.prediction_journal import append_forecast; "
            "append_forecast(market='COFFEE_ROBUSTA', model_version='docker', release_mode='experimental', "
            "horizon=1, price=2, lower=None, upper=None, origin_date='2026-01-01', origin_price=2, "
            "target_date='2026-01-02', status='experimental', procedure_fingerprint='p', "
            "data_fingerprint='d', path=__import__('pathlib').Path('/app/data/_journal_lock_test.jsonl'))",
        ],
        cwd=r"E:\prediction",
        check=False,
    )
    local.wait(timeout=30)
    assert docker.returncode == 0
    rows = load_forecasts(shared)
    assert {row["model_version"] for row in rows} == {"win", "docker"}
    shared.unlink(missing_ok=True)


def test_j30_initial_stays_findable_after_ten_days(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    first_target = None
    for day in range(11):
        origin = (pd.Timestamp("2026-01-05") + pd.offsets.BDay(day)).date().isoformat()
        for horizon in (1, 7, 14, 30):
            target = future_business_date(origin, horizon).date().isoformat()
            if day == 0 and horizon == 30:
                first_target = target
            _write(
                journal,
                horizon=horizon,
                price=100.0 + day,
                lower=90.0,
                upper=110.0,
                origin_date=origin,
                origin_price=100.0,
                target_date=target,
                model_version=f"lot-{day}",
                data_fingerprint=f"data-{day}",
            )
    rows = load_forecasts(journal)
    published_j30 = [
        row for row in rows
        if row.get("horizon") == 30
        and row.get("opens_initial_id") == row.get("forecast_id")
        and not row.get("diagnostic")
    ]
    assert len(published_j30) == 1
    assert published_j30[0]["target_date"] == first_target
    found = tracking_view(rows, market="ICE_NY", target_date=first_target, horizon=30, limit=12)
    assert found["total_initials"] == 1
    assert found["initials"][0]["target_date"] == first_target
    assert found["initials"][0]["horizon"] == 30


def test_next_day_j7_does_not_replace_the_open_commitment(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    _write(
        journal,
        horizon=7,
        price=110.0,
        lower=100.0,
        upper=120.0,
        origin_date="2026-10-01",
        origin_price=100.0,
        target_date="2026-10-12",
    )
    _write(
        journal,
        horizon=7,
        price=90.0,
        lower=80.0,
        upper=100.0,
        origin_date="2026-10-02",
        origin_price=120.0,
        target_date="2026-10-13",
        model_version="improved_b",
        data_fingerprint="next-day",
    )
    rows = load_forecasts(journal)
    assert rows[1]["diagnostic"] is True
    assert rows[1]["opens_initial_id"] is None
    assert rows[1]["revises"] == []
    overlaid = overlay_commitments(
        [
            {"horizon": 7, "price": 90.0, "confidence_interval": [80.0, 100.0]},
            {"horizon": 1, "price": 121.0, "confidence_interval": [110.0, 130.0]},
        ],
        rows,
        market="ICE_NY",
        as_of="2026-10-02",
    )
    assert _model(overlaid[0]) == 110.0
    assert overlaid[0]["reference_price"] == 100.0
    # Initiale non validée : la carte publie le cours de son jour d'émission.
    assert overlaid[0]["price"] == 100.0
    assert overlaid[0]["components"]["served_as_close"] is True
    assert overlaid[0]["target_date"] == "2026-10-12"
    assert overlaid[0]["has_revision"] is False
    assert overlaid[0]["latest_price"] == 110.0
    assert overlaid[0]["revision_notice"] is None
    assert overlaid[1]["price"] == 121.0
    assert "held_commitment" not in overlaid[1]
    frozen_pct = (_model(overlaid[0]) - overlaid[0]["reference_price"]) / overlaid[0]["reference_price"] * 100.0
    spot_pct = (_model(overlaid[0]) - 120.0) / 120.0 * 100.0
    assert frozen_pct > 0
    assert spot_pct < 0


def test_reached_target_opens_the_next_long_horizon(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    _write(
        journal,
        horizon=7,
        price=110.0,
        lower=100.0,
        upper=120.0,
        origin_date="2026-10-01",
        origin_price=100.0,
        target_date="2026-10-08",
    )
    opened = _write(
        journal,
        horizon=7,
        price=130.0,
        lower=120.0,
        upper=140.0,
        origin_date="2026-10-08",
        origin_price=125.0,
        target_date="2026-10-19",
        model_version="improved_b",
        data_fingerprint="mature",
    )
    rows = load_forecasts(journal)
    assert rows[1]["diagnostic"] is False
    assert rows[1]["opens_initial_id"] == opened
    overlaid = overlay_commitments(
        [{"horizon": 7, "price": 130.0, "confidence_interval": [120.0, 140.0]}],
        rows,
        market="ICE_NY",
        as_of="2026-10-08",
    )
    assert _model(overlaid[0]) == 130.0
    assert overlaid[0]["target_date"] == "2026-10-19"


def test_remaining_sessions_shrink_as_the_target_approaches():
    origin = "2026-10-06"
    target = future_business_date(origin, 30).date().isoformat()
    assert sessions_between(origin, target) == 30
    later = future_business_date(origin, 20).date().isoformat()
    assert sessions_between(later, target) == 10


def test_same_target_revision_is_shown_beside_the_initial(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    _write(
        journal,
        horizon=7,
        price=110.0,
        lower=100.0,
        upper=120.0,
        origin_date="2026-10-01",
        origin_price=100.0,
        target_date="2026-10-12",
    )
    _write(
        journal,
        horizon=5,
        price=111.0,
        lower=100.0,
        upper=120.0,
        origin_date="2026-10-05",
        origin_price=104.0,
        target_date="2026-10-12",
        model_version="improved_b",
        data_fingerprint="small",
    )
    rows = load_forecasts(journal)
    quiet = overlay_commitments(
        [{"horizon": 7, "price": 80.0, "confidence_interval": [70.0, 90.0]}],
        rows,
        market="ICE_NY",
        as_of="2026-10-05",
    )[0]
    assert _model(quiet) == 110.0
    assert quiet["has_revision"] is True
    assert quiet["latest_price"] == 111.0
    assert quiet["latest_date"] == "2026-10-05"
    assert quiet["latest_gap"] == pytest.approx(1.0)
    assert quiet["revision_notice"] is None
    assert quiet["sessions_remaining"] == sessions_between("2026-10-05", "2026-10-12")
    assert quiet["initial_horizon"] == 7

    _write(
        journal,
        horizon=4,
        price=130.0,
        lower=120.0,
        upper=140.0,
        origin_date="2026-10-06",
        origin_price=105.0,
        target_date="2026-10-12",
        model_version="improved_c",
        data_fingerprint="large",
    )
    moved = overlay_commitments(
        [{"horizon": 7, "price": 80.0, "confidence_interval": [70.0, 90.0]}],
        load_forecasts(journal),
        market="ICE_NY",
        as_of="2026-10-06",
    )[0]
    assert _model(moved) == 110.0
    assert moved["latest_price"] == 130.0
    assert moved["revision_notice"] == "revision"

    _write(
        journal,
        horizon=3,
        price=90.0,
        lower=80.0,
        upper=100.0,
        origin_date="2026-10-07",
        origin_price=106.0,
        target_date="2026-10-12",
        model_version="improved_d",
        data_fingerprint="flip",
    )
    flipped = overlay_commitments(
        [{"horizon": 7, "price": 80.0, "confidence_interval": [70.0, 90.0]}],
        load_forecasts(journal),
        market="ICE_NY",
        as_of="2026-10-07",
    )[0]
    assert _model(flipped) == 110.0
    assert flipped["reference_price"] == 100.0
    assert flipped["latest_price"] == 90.0
    assert flipped["latest_gap"] == pytest.approx(-20.0)
    assert flipped["revision_notice"] == "inversion"
    assert (_model(flipped) - flipped["reference_price"]) > 0
    assert (flipped["latest_price"] - flipped["reference_price"]) < 0


def test_legacy_row_without_an_id_does_not_hide_the_revision(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    _write(
        journal,
        horizon=7,
        price=110.0,
        lower=100.0,
        upper=120.0,
        origin_date="2026-10-01",
        origin_price=100.0,
        target_date="2026-10-12",
    )
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "market": "ICE_NY",
            "horizon": 7,
            "price": 200.0,
            "origin_date": "2026-09-01",
            "origin_price": 100.0,
            "target_date": "2026-10-12",
            "status": "experimental",
            "recorded_at": "2026-09-01T00:00:00+00:00",
        }) + "\n")
    _write(
        journal,
        horizon=5,
        price=90.0,
        lower=80.0,
        upper=100.0,
        origin_date="2026-10-05",
        origin_price=104.0,
        target_date="2026-10-12",
        model_version="improved_b",
        data_fingerprint="legacy-gap",
    )
    shown = overlay_commitments(
        [{"horizon": 7, "price": 80.0, "confidence_interval": [70.0, 90.0]}],
        load_forecasts(journal),
        market="ICE_NY",
        as_of="2026-10-05",
    )[0]
    assert _model(shown) == 110.0
    assert shown["has_revision"] is True
    assert shown["latest_price"] == 90.0
    assert shown["revision_notice"] == "inversion"


def test_same_target_revision_does_not_open_a_second_card(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    j30 = _write(
        journal,
        horizon=30,
        price=4500.0,
        lower=None,
        upper=None,
        origin_date="2026-10-01",
        origin_price=4200.0,
        target_date="2026-11-12",
    )
    j14 = _write(
        journal,
        horizon=14,
        price=4400.0,
        lower=4300.0,
        upper=4500.0,
        origin_date="2026-10-21",
        origin_price=4300.0,
        target_date="2026-11-10",
        data_fingerprint="j14-open",
    )
    _write(
        journal,
        horizon=7,
        price=4300.0,
        lower=4200.0,
        upper=4400.0,
        origin_date="2026-10-22",
        origin_price=4250.0,
        target_date="2026-10-30",
        data_fingerprint="j7-open",
    )
    later14 = _write(
        journal,
        horizon=14,
        price=4000.0,
        lower=3900.0,
        upper=4100.0,
        origin_date="2026-10-23",
        origin_price=4100.0,
        target_date="2026-11-12",
        model_version="improved_b",
        data_fingerprint="same-target-14",
    )
    later7 = _write(
        journal,
        horizon=7,
        price=4050.0,
        lower=3950.0,
        upper=4150.0,
        origin_date="2026-10-23",
        origin_price=4100.0,
        target_date="2026-11-12",
        model_version="improved_c",
        data_fingerprint="same-target-7",
    )
    second = _write(
        journal,
        horizon=14,
        price=4010.0,
        lower=3910.0,
        upper=4110.0,
        origin_date="2026-10-23",
        origin_price=4100.0,
        target_date="2026-11-12",
        model_version="improved_d",
        data_fingerprint="second-calc",
    )
    rows = load_forecasts(journal)
    by_id = {row["forecast_id"]: row for row in rows}
    assert by_id[later14]["published"] is True
    assert by_id[later14]["diagnostic"] is False
    assert by_id[later14]["opens_initial_id"] is None
    assert by_id[later14]["revises"] == [j30]
    assert by_id[later7]["opens_initial_id"] is None
    assert by_id[later7]["revises"] == [j30]
    assert by_id[second]["opens_initial_id"] is None
    assert by_id[second]["revises"] == [j30]
    opened_14 = [
        row for row in rows
        if row["horizon"] == 14 and row.get("opens_initial_id") == row.get("forecast_id")
    ]
    assert [row["forecast_id"] for row in opened_14] == [j14]


def test_fresh_band_stays_off_the_initial_and_a_failure_survives_republish(tmp_path: Path):
    from types import SimpleNamespace

    from src.models.served_forecast import publish_or_close

    journal = tmp_path / "journal.jsonl"
    _write(
        journal,
        horizon=14,
        price=4500.0,
        lower=None,
        upper=None,
        origin_date="2026-10-01",
        origin_price=4200.0,
        target_date="2026-11-12",
    )
    shown = overlay_commitments(
        [{
            "horizon": 14,
            "price": 4000.0,
            "confidence_interval": [3900.0, 4100.0],
            "status": "experimental",
            "model_version": "improved_fresh",
            "target_date": "2026-11-12",
            "components": {
                "origin_date": "2026-10-23",
                "target_date": "2026-11-12",
                "status": "experimental",
            },
        }],
        load_forecasts(journal),
        market="ICE_NY",
        as_of="2026-10-23",
    )[0]
    assert _model(shown) == 4500.0
    assert shown["confidence_interval"] is None
    assert shown["fresh"]["price"] == 4000.0
    assert shown["fresh"]["lower"] == 3900.0
    assert shown["fresh"]["upper"] == 4100.0
    assert shown["initial"]["model_version"] == "improved_a"
    assert shown["fresh"]["model_version"] == "improved_fresh"
    failed = overlay_commitments(
        [{
            "horizon": 14,
            "price": None,
            "confidence_interval": None,
            "status": "unavailable",
            "model_version": "improved_fresh",
            "target_date": "2026-11-12",
            "components": {
                "origin_date": "2026-10-23",
                "target_date": "2026-11-12",
                "calculation_failed": True,
                "status": "unavailable",
            },
        }],
        load_forecasts(journal),
        market="ICE_NY",
        as_of="2026-10-23",
    )[0]
    assert _model(failed) == 4500.0
    assert failed["latest_unavailable"] is True
    assert "calculation_failed" not in failed["components"]
    assert failed["components"]["display_final"] is True
    card = SimpleNamespace(
        price=_model(failed),
        status=failed["status"],
        confidence_interval=failed["confidence_interval"],
        components=failed["components"],
    )
    publish_or_close([card], {"release_mode": "experimental"}, 4200.0, release_mode="experimental")
    assert card.price == 4500.0
    assert card.status == "experimental"


def test_verifier_reads_the_fresh_calculation_built_like_the_api(tmp_path: Path):
    from scripts.verify_served_forecast import assess_served, expected_target

    journal = tmp_path / "journal.jsonl"
    _write(
        journal,
        horizon=14,
        price=4500.0,
        lower=None,
        upper=None,
        origin_date="2026-10-01",
        origin_price=4200.0,
        target_date="2026-11-12",
    )
    raw = {
        "horizon": 14,
        "price": 4000.0,
        "status": "experimental",
        "target_date": None,
        "model_version": "improved_x",
        "confidence_interval": [3900.0, 4100.0],
        "components": {
            "target_date": expected_target("2026-10-13", 14),
            "origin_date": "2026-10-13",
            "status": "experimental",
        },
    }
    raw["target_date"] = str(raw["components"]["target_date"])[:10]
    shown = overlay_commitments(
        [raw],
        load_forecasts(journal),
        market="ICE_NY",
        as_of="2026-10-13",
    )[0]

    def other(horizon: int) -> dict:
        return {
            "horizon": horizon,
            "price": 4500.0,
            "target_date": "2026-12-01",
            "fresh": {
                "price": 1000.0 + horizon,
                "status": "experimental",
                "target_date": expected_target("2026-10-13", horizon),
                "issued_date": "2026-10-13",
                "model_version": "improved_x",
            },
        }

    payload = {
        "model_version": "improved_x",
        "current_date": "2026-10-13",
        "predictions": [other(1), other(7), shown, other(30)],
    }
    assert _model(shown) == 4500.0
    assert shown["fresh"]["price"] == 4000.0
    assert assess_served(payload, "improved_x", "2026-10-13")[0] == 0
    shown["fresh"]["price"] = None
    shown["fresh"]["status"] = "unavailable"
    assert assess_served(payload, "improved_x", "2026-10-13")[0] == 3


def test_a_rejected_band_is_not_written_back_into_the_card(tmp_path: Path):
    from types import SimpleNamespace

    from src.models.served_forecast import publish_or_close, served_journal_bounds

    journal = tmp_path / "journal.jsonl"
    item = SimpleNamespace(
        horizon=14,
        price=4500.0,
        confidence_interval=[4100.0, 4300.0],
        status="experimental",
        components={
            "origin_date": "2026-10-13",
            "origin_price": 4200.0,
            "target_date": "2026-11-02",
        },
    )
    publish_or_close([item], {"release_mode": "experimental"}, None, release_mode="experimental")
    bounds = served_journal_bounds(item)
    assert item.confidence_interval is None
    assert bounds["lower"] is None
    assert bounds["upper"] is None
    assert bounds["rejected_lower"] == 4100.0
    assert bounds["rejected_upper"] == 4300.0
    _write(
        journal,
        horizon=14,
        price=4500.0,
        lower=bounds["lower"],
        upper=bounds["upper"],
        origin_date="2026-10-01",
        origin_price=4200.0,
        target_date="2026-11-02",
    )
    shown = overlay_commitments(
        [{
            "horizon": 14,
            "price": 4000.0,
            "confidence_interval": [3900.0, 4100.0],
            "status": "experimental",
            "model_version": "fresh",
            "components": {"origin_date": "2026-10-13", "target_date": "2026-11-02"},
        }],
        load_forecasts(journal),
        market="ICE_NY",
        as_of="2026-10-13",
    )[0]
    assert _model(shown) == 4500.0
    assert shown["confidence_interval"] is None
    assert shown["initial"]["lower"] is None


def test_an_older_revision_does_not_take_todays_delay(tmp_path: Path):
    journal = tmp_path / "journal.jsonl"
    _write(
        journal,
        horizon=14,
        price=4500.0,
        lower=None,
        upper=None,
        origin_date="2026-10-01",
        origin_price=4200.0,
        target_date=future_business_date("2026-10-01", 14).date().isoformat(),
    )
    target = future_business_date("2026-10-01", 14).date().isoformat()
    _write(
        journal,
        horizon=13,
        price=4400.0,
        lower=None,
        upper=None,
        origin_date="2026-10-02",
        origin_price=4200.0,
        target_date=target,
        model_version="last",
        data_fingerprint="last",
    )
    shown = overlay_commitments(
        [{
            "horizon": 14,
            "price": 4300.0,
            "confidence_interval": None,
            "status": "experimental",
            "model_version": "today",
            "components": {
                "origin_date": "2026-10-05",
                "target_date": future_business_date("2026-10-05", 14).date().isoformat(),
            },
        }],
        load_forecasts(journal),
        market="ICE_NY",
        as_of="2026-10-05",
    )[0]
    assert shown["has_revision"] is True
    assert shown["revision_matches_today"] is False
    assert shown["latest_sessions_remaining"] == sessions_between("2026-10-02", target)
    assert shown["sessions_remaining"] == sessions_between("2026-10-05", target)
    assert shown["latest_sessions_remaining"] != shown["sessions_remaining"]
