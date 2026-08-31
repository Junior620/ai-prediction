"""Unit tests for Databento London collector (mapping + bornes, no network)."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data_collection.databento_london_collector import (
    DatabentoBar,
    _ohlcv_df_to_bars,
    _parse_oi_df,
    attach_open_interest,
    bars_to_contract_rows,
    bars_to_supabase_rows,
    get_api_key,
    fetch_daily_bars,
)


def _sample_ohlcv_df() -> pd.DataFrame:
    idx = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04"], utc=True
    )
    return pd.DataFrame(
        {
            "symbol": ["C.v.0", "C.v.0", "C.v.0"],
            "open": [4100.0, 4150.0, 4200.0],
            "high": [4200.0, 4250.0, 4300.0],
            "low": [4050.0, 4100.0, 4150.0],
            "close": [4150.0, 4220.0, 4280.0],
            "volume": [1000.0, 1100.0, 1200.0],
        },
        index=idx,
    )


class TestDatabentoMapping:
    def test_ohlcv_df_to_bars(self):
        bars = _ohlcv_df_to_bars(_sample_ohlcv_df(), price_bounds=(1500, 6000))
        assert len(bars) == 3
        assert bars[-1].price == 4280.0
        assert bars[-1].volume == 1200.0
        assert bars[-1].symbol == "C.v.0"
        assert bars[-1].date == "2024-01-04"

    def test_price_bounds_filter(self):
        df = _sample_ohlcv_df()
        df.iloc[1, df.columns.get_loc("close")] = 50.0  # hors bornes
        bars = _ohlcv_df_to_bars(df, price_bounds=(1500, 6000))
        assert len(bars) == 2
        assert all(1500 <= b.price <= 6000 for b in bars)

    def test_parse_oi_df(self):
        idx = pd.to_datetime(["2024-01-02T18:00:00Z", "2024-01-02T20:00:00Z"])
        df = pd.DataFrame(
            {
                "stat_type": [9, 9],
                "symbol": ["C.v.0", "C.v.0"],
                "quantity": [10000.0, 10500.0],
                "ts_ref": pd.to_datetime(["2024-01-02", "2024-01-02"], utc=True),
                "ts_recv": idx,
            },
            index=idx,
        )
        oi = _parse_oi_df(df)
        assert oi[("2024-01-02", "C.v.0")] == 10500.0

    def test_attach_open_interest(self):
        bars = [
            DatabentoBar(date="2024-01-02", price=4150.0, symbol="C.v.0", contract_rank=0),
            DatabentoBar(date="2024-01-03", price=4220.0, symbol="C.v.0", contract_rank=0),
        ]
        oi_map = {("2024-01-02", "C.v.0"): 9000.0}
        out = attach_open_interest(bars, oi_map)
        assert out[0].open_interest == 9000.0
        assert out[1].open_interest is None

    def test_bars_to_supabase_rows(self):
        bars = [
            DatabentoBar(
                date="2024-01-02",
                price=4150.0,
                open=4100.0,
                high=4200.0,
                low=4050.0,
                volume=1000.0,
                open_interest=9000.0,
                contract_rank=0,
            ),
            DatabentoBar(date="2024-01-02", price=4180.0, contract_rank=1),
        ]
        rows = bars_to_supabase_rows(bars, front_only=True)
        assert len(rows) == 1
        assert rows[0]["price"] == 4150.0
        assert rows[0]["open_interest"] == 9000.0
        assert rows[0]["source"] == "databento"

    def test_bars_to_contract_rows(self):
        bars = [
            DatabentoBar(date="2024-01-02", price=4150.0, contract_rank=0, symbol="C.v.0"),
            DatabentoBar(date="2024-01-02", price=4180.0, contract_rank=1, symbol="C.v.1"),
        ]
        rows = bars_to_contract_rows(bars)
        assert len(rows) == 2
        assert rows[1]["contract_rank"] == 1
        assert rows[1]["close"] == 4180.0


def test_prefer_on_exchange_publisher():
    from src.data_collection.databento_london_collector import _ohlcv_df_to_bars

    idx = pd.to_datetime(
        ["2024-06-07", "2024-06-07"], utc=True
    )
    df = pd.DataFrame(
        {
            "publisher_id": [57, 84],
            "symbol": ["C.v.0", "C.v.0"],
            "open": [7208.0, 7213.0],
            "high": [7347.0, 7232.0],
            "low": [7128.0, 3920.0],
            "close": [7347.0, 3920.0],
            "volume": [4184.0, 63.0],
        },
        index=idx,
    )
    bars = _ohlcv_df_to_bars(df, price_bounds=(1000, 15000))
    assert len(bars) == 1
    assert bars[0].price == 7347.0
    assert bars[0].volume == 4184.0


class TestDatabentoClientGuards:
    def test_missing_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        assert get_api_key() is None
        result = fetch_daily_bars("2024-01-01", "2024-01-10", api_key="")
        assert result.ok is False
        assert result.error == "missing_api_key_or_package"
        assert result.bars == []
