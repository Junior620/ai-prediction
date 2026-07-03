"""Tests du registre de marchés (config.yaml section markets)."""

import pytest

from src.models.market_registry import (
    get_market_config,
    list_api_markets,
    load_all_markets,
    resolve_api_market,
)


class TestMarketRegistry:
    def test_load_all_markets_contains_cocoa_and_robusta(self):
        markets = load_all_markets()
        assert "cocoa" in markets
        assert "coffee_robusta" in markets

    def test_cocoa_config(self):
        cfg = get_market_config("cocoa")
        assert cfg.price_table == "cocoa_prices"
        assert cfg.source == "yahoo_finance"
        assert cfg.yahoo_symbol == "CC=F"
        assert cfg.tradingview_symbol == "PEPPERSTONE:COCOA"
        assert cfg.tradingview_embed_symbol == "PEPPERSTONE:COCOA"
        assert cfg.tradingview_alert_symbol == "ICEEUR:C1!"
        assert cfg.nhits_unique_id == "cocoa_ice_ny"
        assert cfg.garch_enabled is False

    def test_coffee_robusta_config(self):
        cfg = get_market_config("coffee_robusta")
        assert cfg.price_table == "coffee_robusta_prices"
        assert cfg.source == "investing_com"
        assert cfg.investing_url == "https://www.investing.com/commodities/london-coffee"
        assert cfg.contract_symbol == "RCU6"
        assert cfg.tradingview_symbol == "ICEEUR:RC1!"
        assert cfg.tradingview_embed_symbol == "ROBCOFFEE"
        assert cfg.unit == "USD/T"
        assert cfg.garch_enabled is True
        assert cfg.models_dir == "models/coffee_robusta"
        assert cfg.price_bounds == (1000.0, 8000.0)

    def test_unknown_market_raises(self):
        with pytest.raises(KeyError):
            get_market_config("arabica")

    def test_resolve_api_market(self):
        assert resolve_api_market("ICE_NY").market_id == "cocoa"
        assert resolve_api_market("ICE_London").market_id == "cocoa"
        assert resolve_api_market("COFFEE_ROBUSTA").market_id == "coffee_robusta"
        assert resolve_api_market("coffee_robusta").market_id == "coffee_robusta"
        assert resolve_api_market("UNKNOWN") is None

    def test_list_api_markets(self):
        markets = list_api_markets()
        assert "ICE_NY" in markets
        assert "COFFEE_ROBUSTA" in markets
