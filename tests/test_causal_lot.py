"""The causal comparison keeps future targets out of training."""

from scripts.score_experimental_lot import IN_SAMPLE_NOTE
from src.validation.causal_lot import known_targets, next_session_present, split_windows, walk_days


def test_a_future_target_is_not_a_training_example():
    dates = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"]
    learned = known_targets(dates, "2026-06-03", 2)
    assert learned
    assert all(day <= "2026-06-03" for day in learned)
    assert "2026-06-05" not in learned


def test_the_reserved_period_starts_after_the_calibration_targets():
    dates = []
    cursor = __import__("datetime").date(2024, 1, 2)
    while len(dates) < 120:
        if cursor.weekday() < 5:
            dates.append(cursor.isoformat())
        cursor += __import__("datetime").timedelta(days=1)
    lookup = {day: 1.0 for day in dates}
    windows = split_windows(dates, lookup, horizon=30, calibration=8, reserved=8, gap=30)
    assert len(windows["reserved"]) == 8
    assert len(windows["calibration"]) == 8
    last_origin = windows["calibration"][-1]
    from src.models.hybrid_features import future_business_date

    last_target = future_business_date(last_origin, 30).date().isoformat()
    gap_end = future_business_date(last_target, 30).date().isoformat()
    assert gap_end <= windows["reserved"][0]
    assert windows["calibration"][-1] < windows["reserved"][0]


def test_a_chain_without_the_next_session_is_not_a_revision():
    walked = walk_days("2026-06-01", 3)
    assert walked[1][0] != walked[0][0]
    assert walked[0][2] == walked[1][2]
    opening = {"session": "2026-06-01", "horizon": 14, "target": "2026-06-19"}
    assert next_session_present([opening]) is False
    nxt = {"session": walked[1][0], "horizon": 13, "target": walked[0][2]}
    assert next_session_present([opening, nxt]) is True


def test_the_in_sample_script_keeps_its_warning():
    assert "ne valide pas le lot" in IN_SAMPLE_NOTE
    assert "entraînés sur les cibles" in IN_SAMPLE_NOTE
