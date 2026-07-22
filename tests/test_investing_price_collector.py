"""Tests du collecteur Investing.com (parsing HTML, sans reseau)."""

from src.data_collection.investing_price_collector import (
    SYMBOL_PATTERN,
    _extract_price,
    _extract_symbol,
)


class TestSymbolDetection:
    def test_detects_rcu6_from_page_title(self):
        match = SYMBOL_PATTERN.search("London Coffee Futures Overview (RCU6) - Investing.com")
        assert match is not None
        assert match.group(1) == "RCU6"

    def test_detects_rollover_symbol(self):
        match = SYMBOL_PATTERN.search("London Coffee (RCZ6)")
        assert match is not None
        assert match.group(1) == "RCZ6"

    def test_no_symbol_in_title(self):
        assert SYMBOL_PATTERN.search("London Coffee - Investing.com") is None


class TestPriceExtraction:
    def test_extract_from_data_test_attribute(self):
        html = '<span data-test="instrument-price-last">3,751.00</span>'
        assert _extract_price(html) == 3751.0

    def test_extract_from_embedded_json_last(self):
        html = '{"currency":"USD","last":3758,"lastDecimalPrecision":2}'
        assert _extract_price(html) == 3758.0

    def test_extract_symbol_from_heading(self):
        html = "<h1>London Coffee (RCU6)</h1>" + '<span data-test="instrument-price-last">3,751.00</span>'
        assert _extract_symbol(html, fallback_symbol="RCZ6") == "RCU6"

    def test_extract_symbol_fallback(self):
        html = "<title>London Coffee - Investing.com</title>"
        assert _extract_symbol(html, fallback_symbol="RCU6") == "RCU6"
