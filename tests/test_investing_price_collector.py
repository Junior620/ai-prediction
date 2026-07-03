"""Tests du collecteur Investing.com (détection de symbole, sans Selenium)."""

from src.data_collection.investing_price_collector import SYMBOL_PATTERN


class TestSymbolDetection:
    def test_detects_rcu6_from_page_title(self):
        match = SYMBOL_PATTERN.search("London Coffee Futures Overview (RCU6) - Investing.com")
        assert match is not None
        assert match.group(1) == "RCU6"

    def test_detects_rollover_symbol(self):
        # Après expiration de RCU6 (sept. 2026), la page affichera p.ex. RCZ6
        match = SYMBOL_PATTERN.search("London Coffee (RCZ6)")
        assert match is not None
        assert match.group(1) == "RCZ6"

    def test_no_symbol_in_title(self):
        assert SYMBOL_PATTERN.search("London Coffee - Investing.com") is None
