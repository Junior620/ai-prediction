"""Variants stay beside the served price."""

from src.validation.lot_variants import evaluate_variants


def _row(session, price, spot, realized, role="revision", horizon=14, target="2026-08-01"):
    return {
        "session": session,
        "target": target,
        "horizon": 1 if role == "revision" else horizon,
        "initial_horizon": horizon,
        "role": role,
        "price": price,
        "spot": spot,
        "realized": realized,
        "reference": 100.0,
    }


def test_k_and_the_revision_rule_are_chosen_before_the_holdout():
    rows = []
    for day, model in enumerate((80, 90, 70, 85, 75, 88, 82, 79), start=1):
        session = f"2026-06-{day:02d}"
        rows.append(_row(session, model, 100, 100, role="initial" if day == 1 else "revision"))
    # The last two sessions would prefer the raw model. They must not choose k.
    rows.append(_row("2026-06-20", 100, 100, 140, role="revision"))
    rows.append(_row("2026-06-21", 140, 100, 140, role="revision"))
    scored = evaluate_variants(rows, holdout_fraction=0.25)
    assert scored["variant_replaces_served_price"] is False
    assert scored["status"] == "experimental"
    assert scored["k"] == 0.0
    assert scored["chosen_on"] == "calibration"
    assert scored["holdout"]["levels"]["count"] > 0


def test_fewer_direction_changes_do_not_beat_a_closer_price():
    rows = []
    for day in range(1, 9):
        session = f"2026-07-{day:02d}"
        price = 160.0 if day == 1 else (115.0 if day % 2 else 85.0)
        rows.append(
            _row(session, price, 100, 100, role="initial" if day == 1 else "revision", target="2026-08-03")
        )
    scored = evaluate_variants(rows, holdout_fraction=0.25)
    latest = next(item for item in scored["calibration"]["policies"] if item["policy"] == "latest")
    kept = next(item for item in scored["calibration"]["policies"] if item["policy"] == "initial")
    assert latest["inversions"] > kept["inversions"]
    assert latest["mean_absolute_error"] < kept["mean_absolute_error"]
    assert scored["revision_policy"] == "latest"


def test_beating_the_initial_but_losing_to_the_day_is_not_an_advantage():
    from src.validation.lot_variants import delay_balances

    rows = []
    for index in range(4):
        target = f"2026-06-{index + 10:02d}"
        rows.append(
            {
                "session": f"2026-05-{index + 1:02d}",
                "target": target,
                "horizon": 14,
                "initial_horizon": 14,
                "role": "initial",
                "price": 50.0,
                "spot": 100.0,
                "realized": 100.0,
                "reference": 100.0,
            }
        )
        rows.append(
            {
                "session": f"2026-06-{index + 1:02d}",
                "target": target,
                "horizon": 1,
                "initial_horizon": 14,
                "role": "revision",
                "price": 80.0,
                "spot": 100.0,
                "realized": 100.0,
                "reference": 100.0,
            }
        )
    scored = delay_balances(rows)
    day = scored["1"]
    assert day["sample_status"] == "intervalle disponible"
    assert day["latest_mean_absolute_error"] < day["initial_mean_absolute_error"]
    assert day["latest_mean_absolute_error"] > day["unchanged_mean_absolute_error"]
    assert day["advantage_established"] is False


def test_a_short_sample_is_named_insufficient_and_does_not_pick_a_winner():
    from scripts.score_causal_cocoa import _summarize

    rows = [
        {
            "session": "2026-06-01",
            "target": "2026-06-08",
            "horizon": 7,
            "initial_horizon": 7,
            "role": "initial",
            "price": 110.0,
            "spot": 100.0,
            "realized": 120.0,
            "reference": 100.0,
        }
    ]
    summary = _summarize(rows, k=0.0, policy="latest", horizon=7)
    assert summary["sample_status"] == "échantillon insuffisant"
    assert summary["chosen_policy_on_this_slice"] is None
    assert summary["by_remaining_delay"]["7"]["sample_status"] == "échantillon insuffisant"
    assert summary["by_remaining_delay"]["7"]["advantage_established"] is False
