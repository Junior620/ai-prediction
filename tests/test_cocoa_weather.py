"""Weather stays out of the published price until a candidate clears both gates."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_collection.cocoa_weather import (
    blend_country_daily,
    latest_origin_weather,
    parse_nasa_power_payload,
)
from src.models.hybrid_features import resolve_feature_cols
from src.models.release_manifest import latest_candidate
from src.models.weather_features import attach_weather, build_weather_features
from src.validation.weather_candidate import decide_weather_promotion, score_predictions


def _passing_row(horizon: int) -> dict:
    return {
        "horizon": horizon,
        "current_mae": 100.0,
        "candidate_mae": 80.0,
        "current_mape": 8.0,
        "candidate_mape": 7.0,
        "current_fallback": 0.05,
        "candidate_fallback": 0.05,
        "n_after_cutoff": 12,
    }


def test_snapshot_keeps_latest_origin_and_drops_other_cities():
    rows = [
        {"location": "Abidjan,Ivory Coast", "country": "Ivory Coast", "temperature_c": 24, "collected_at": "2026-10-08T08:00:00"},
        {"location": "Abidjan,Ivory Coast", "country": "Ivory Coast", "temperature_c": 31, "collected_at": "2026-10-08T12:00:00"},
        {"location": "London,UK", "country": "United Kingdom", "temperature_c": 12, "collected_at": "2026-10-08T12:00:00"},
        {"location": "Kumasi,Ghana", "country": "Ghana", "temperature_c": 28, "precipitation_mm": 1.5, "collected_at": "2026-10-08T11:00:00"},
    ]
    picked = latest_origin_weather(rows)
    by_location = {row["location"]: row for row in picked}
    assert set(by_location) == {"Abidjan,Ivory Coast", "Kumasi,Ghana"}
    assert by_location["Abidjan,Ivory Coast"]["temperature_c"] == 31
    assert by_location["Kumasi,Ghana"]["precipitation_mm"] == 1.5


def test_nasa_parser_drops_missing_sentinel():
    point = {"region": "soubre", "country": "CI", "latitude": 5.79, "longitude": -6.59}
    payload = {
        "properties": {
            "parameter": {
                "T2M": {"20200101": 26.5, "20200102": -999.0},
                "T2M_MIN": {"20200101": 22.0, "20200102": 21.0},
                "T2M_MAX": {"20200101": 31.0, "20200102": 30.0},
                "PRECTOTCORR": {"20200101": 4.2, "20200102": 0.0},
            }
        }
    }
    frame = parse_nasa_power_payload(payload, point)
    assert frame.loc[0, "precip_mm"] == 4.2
    assert frame.loc[0, "temp_c"] == 26.5
    assert pd.isna(frame.loc[1, "temp_c"])


def test_blend_weights_ivory_coast_twice_ghana():
    frame = pd.DataFrame(
        [
            {"date": "2024-01-02", "country": "CI", "precip_mm": 3.0, "temp_c": 24.0, "temp_min_c": 20.0, "temp_max_c": 30.0},
            {"date": "2024-01-02", "country": "GH", "precip_mm": 9.0, "temp_c": 30.0, "temp_min_c": 22.0, "temp_max_c": 33.0},
        ]
    )
    blended = blend_country_daily(frame)
    assert blended.loc[0, "precip_mm"] == 5.0
    assert blended.loc[0, "temp_c"] == 26.0


def test_future_rain_does_not_change_past_features_or_climatology():
    dates = pd.date_range("2022-01-01", periods=200, freq="D")
    base = pd.DataFrame({"date": dates, "precip_mm": 1.0, "temp_c": 25.0})
    spiked = base.copy()
    spiked.loc[spiked.index[-1], "precip_mm"] = 400.0
    cutoff = "2022-06-01"
    left = build_weather_features(base, cutoff)
    right = build_weather_features(spiked, cutoff)
    past = left["date"] < pd.Timestamp("2022-06-15")
    compared = [
        "rain_sum_7",
        "rain_sum_30",
        "rain_sum_90",
        "rain_lag_1",
        "rain_anomaly",
        "drought_index",
    ]
    for column in compared:
        assert np.allclose(
            left.loc[past, column].to_numpy(dtype=float),
            right.loc[past, column].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_attach_uses_weather_on_or_before_the_price_date():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-07"]),
            "price": [100.0, 110.0],
        }
    )
    weather = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-04", "2024-01-08"]),
            "precip_mm": [2.0, 50.0],
            "temp_c": [25.0, 40.0],
            "rain_sum_7": [2.0, 50.0],
            "rain_sum_30": [2.0, 50.0],
            "rain_sum_90": [2.0, 50.0],
            "temp_ma_7": [25.0, 40.0],
            "temp_ma_30": [25.0, 40.0],
            "rain_lag_1": [1.0, 9.0],
            "rain_lag_7": [1.0, 9.0],
            "rain_lag_14": [1.0, 9.0],
            "rain_lag_30": [1.0, 9.0],
            "temp_lag_1": [24.0, 30.0],
            "temp_lag_7": [24.0, 30.0],
            "temp_lag_14": [24.0, 30.0],
            "temp_lag_30": [24.0, 30.0],
            "rain_anomaly": [0.1, 9.0],
            "temp_anomaly": [0.1, 9.0],
            "drought_index": [-0.2, 4.0],
        }
    )
    merged = attach_weather(prices, weather)
    assert merged.loc[0, "rain_sum_7"] == 2.0
    assert pd.isna(merged.loc[1, "rain_sum_7"]) is False
    assert merged.loc[1, "rain_sum_7"] == 2.0


def test_production_feature_set_excludes_weather():
    columns = resolve_feature_cols(include_ohlcv=True, include_oi=True)
    assert "drought_index" not in columns
    assert "rain_sum_30" not in columns
    source = Path("train_hybrid_improved.py").read_text(encoding="utf-8")
    assert "include_weather=True" not in source


def test_missing_prediction_counts_as_the_close():
    scored = score_predictions(
        actual=[100.0, 100.0],
        predicted=[110.0, float("nan")],
        naive=[100.0, 100.0],
        target_dates=["2026-10-03", "2026-10-04"],
    )
    assert scored["fallback_rate"] == 0.5
    assert scored["mae"] == 5.0
    assert scored["n_after_cutoff"] == 2


def test_promotion_requires_mae_gain_and_unused_targets_and_close_gate():
    rows = [_passing_row(30), _passing_row(60), _passing_row(90)]
    assert decide_weather_promotion(rows, acceptance_j30=True)["promote"] is True

    weak = [_passing_row(30), _passing_row(60), _passing_row(90)]
    weak[0]["candidate_mae"] = 95.0
    refused = decide_weather_promotion(weak, acceptance_j30=True)
    assert refused["promote"] is False

    early = [_passing_row(horizon) for horizon in (30, 60, 90)]
    early[1]["n_after_cutoff"] = 0
    assert decide_weather_promotion(early, acceptance_j30=True)["promote"] is False

    assert decide_weather_promotion(rows, acceptance_j30=False)["promote"] is False


def test_weather_folder_does_not_hide_the_served_candidate(tmp_path):
    served = tmp_path / "models" / "candidates" / "cocoa" / "20261006_213603"
    served.mkdir(parents=True)
    (served / "prophet_improved_20261006_213603.pkl").write_bytes(b"prophet")
    (served / "xgboost_improved_20261006_213603.pkl").write_bytes(b"xgb")
    weather = tmp_path / "models" / "candidates" / "cocoa" / "weather_20261008_160000"
    weather.mkdir()
    (weather / "xgboost_weather_h30_20261008_160000.pkl").write_bytes(b"weather")

    found = latest_candidate("cocoa", root=tmp_path)
    assert found is not None
    assert found["version"] == "improved_20261006_213603"
