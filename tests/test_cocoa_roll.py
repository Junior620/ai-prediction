"""Série cacao sans saut de contrat et cible en rendement des modèles directs."""

import math

import numpy as np
import pandas as pd
import pytest

from src.data_collection.cocoa_roll import apply_roll_series, liquid_roll_series, on_exchange_bars

DEC, MAR, MAY = 101, 202, 303


def _bars(rows):
    """rows: (date, rank, instrument_id, close, volume)."""
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(d),
                "rank": r,
                "instrument_id": i,
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "volume": v,
            }
            for d, r, i, c, v in rows
        ]
    )


def test_alternating_volume_never_moves_back_and_creates_no_jump():
    # C.v.0 alternerait DEC (5 300) / MAR (4 600) au gré du volume.
    bars = _bars(
        [
            ("2024-09-10", 0, DEC, 5300, 9000), ("2024-09-10", 1, MAR, 4600, 5000),
            ("2024-09-11", 0, DEC, 5310, 4000), ("2024-09-11", 1, MAR, 4610, 6000),
            ("2024-09-12", 0, DEC, 5320, 7000), ("2024-09-12", 1, MAR, 4620, 3000),
            ("2024-09-13", 0, DEC, 5330, 2000), ("2024-09-13", 1, MAR, 4630, 8000),
        ]
    )
    series = liquid_roll_series(bars)
    assert series["instrument_id"].tolist() == [DEC, MAR, MAR, MAR]
    assert series["roll"].tolist() == [False, True, False, False]
    returns = series["price"].pct_change().dropna()
    assert returns.abs().max() < 0.01
    # Le jour du changement, le rendement est celui de MAR seul.
    assert series["ret"].iloc[1] == pytest.approx(4610 / 4600 - 1)


def test_expired_contract_rolls_forward_on_the_same_contract_return():
    bars = _bars(
        [
            ("2024-12-16", 0, DEC, 9000, 5000), ("2024-12-16", 1, MAR, 8000, 1000), ("2024-12-16", 2, MAY, 7900, 500),
            ("2024-12-17", 0, MAR, 8080, 1000), ("2024-12-17", 1, MAY, 7950, 900),
        ]
    )
    series = liquid_roll_series(bars)
    assert series["instrument_id"].tolist() == [DEC, MAR]
    assert bool(series["roll"].iloc[1])
    assert series["ret"].iloc[1] == pytest.approx(0.01)


def test_level_is_anchored_on_the_last_close_and_ohlc_scaled():
    bars = _bars(
        [
            ("2024-09-10", 0, DEC, 5000, 9000), ("2024-09-10", 1, MAR, 4000, 1000),
            ("2024-09-11", 0, DEC, 5100, 1000), ("2024-09-11", 1, MAR, 4040, 9000),
        ]
    )
    series = liquid_roll_series(bars)
    assert series["price"].iloc[-1] == pytest.approx(4040)
    assert series["price"].iloc[0] == pytest.approx(4000)
    assert series["open"].iloc[0] == pytest.approx(4000)


def test_on_exchange_bar_wins_over_off_market():
    raw = pd.DataFrame(
        {
            "ts_event": ["2024-09-10T00:00:00Z", "2024-09-10T00:00:00Z"],
            "publisher_id": [84, 57],
            "instrument_id": [DEC, DEC],
            "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0],
            "volume": [9999, 10],
            "symbol": ["C.c.0", "C.c.0"],
        }
    )
    bars = on_exchange_bars(raw)
    assert bars["close"].tolist() == [2.0]


def _series():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-10-08", "2026-10-09", "2026-10-12"]),
            "price": [100.0, 101.0, 102.0],
            "open": [100.0, 101.0, 102.0],
            "high": [100.0, 101.0, 102.0],
            "low": [100.0, 101.0, 102.0],
            "volume": [1.0, 2.0, 3.0],
        }
    )


def test_apply_drops_unquoted_sessions_keeps_closed_rows_and_later_rows():
    frame = pd.DataFrame(
        {
            # 10-07 n'existe pas chez Databento (ligne bootstrap), 10-10 est un samedi.
            "date": pd.to_datetime(["2026-10-07", "2026-10-08", "2026-10-09", "2026-10-10", "2026-10-12", "2026-10-13"]),
            "price": [90.0, 50.0, 51.0, 51.0, 52.0, 103.0],
        }
    )
    out = apply_roll_series(frame, _series(), is_closed=lambda d: d.weekday() >= 5)
    dates = out["date"].dt.strftime("%Y-%m-%d").tolist()
    assert dates == ["2026-10-08", "2026-10-09", "2026-10-10", "2026-10-12", "2026-10-13"]
    assert out["price"].tolist() == [100.0, 101.0, 101.0, 102.0, 103.0]
    assert out["roll_adjusted"].tolist() == [True, True, True, True, False]
    again = apply_roll_series(out, _series(), is_closed=lambda d: d.weekday() >= 5)
    assert again["price"].tolist() == out["price"].tolist()


def test_prepare_market_frame_uses_the_series_for_cocoa_only(tmp_path):
    from src.models.hybrid_features import prepare_market_frame

    path = tmp_path / "roll.csv"
    _series().to_csv(path, index=False)
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-10-08", "2026-10-09"]), "price": [50.0, 51.0]})
    cocoa = prepare_market_frame(frame, market_id="cocoa", ledger_path=tmp_path / "none.csv", roll_series_path=path)
    coffee = prepare_market_frame(frame, market_id="coffee_robusta", roll_series_path=path)
    assert cocoa["price"].tolist() == [100.0, 101.0]
    assert coffee["price"].tolist() == [50.0, 51.0]


def test_log_return_target_and_level():
    from src.models.direct_horizon_trainer import DirectHorizonTrainer

    assert DirectHorizonTrainer.level_from_prediction(math.log(1.1), 4000.0, "log_return") == pytest.approx(4400.0)
    n = 80
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=n),
            "price": 1000.0 * np.exp(np.linspace(0, 0.4, n)),
        }
    )
    trainer = DirectHorizonTrainer(horizons=[1], target="log_return")
    from src.models.hybrid_features import build_technical_features

    _, y = trainer._build_direct_dataset(df, horizon=1, df_features=build_technical_features(df))
    assert y.iloc[0] == pytest.approx(0.4 / (n - 1), rel=0.05)
    with pytest.raises(ValueError):
        DirectHorizonTrainer(target="level")


def test_cocoa_direct_recipe_has_no_price_level_or_year():
    from src.models.hybrid_features import FEATURE_COLS_RELATIVE, direct_recipe

    recipe = direct_recipe("cocoa", ["price_lag_1"])
    assert recipe["target"] == "log_return"
    assert recipe["xgb_params"]["max_depth"] <= 3
    banned = {"year", "day_of_year", "price_lag_1", "price_ma_30", "volume", "open_interest", "prophet_yhat"}
    assert not banned & set(FEATURE_COLS_RELATIVE)
    # Sans source réelle, l'open interest n'était qu'une indicatrice de date (registre de test).
    assert not {"oi_change_1d", "oi_change_7d"} & set(FEATURE_COLS_RELATIVE)
    assert direct_recipe("coffee_robusta", ["price_lag_1"])["target"] == "close_delta"
