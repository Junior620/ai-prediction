"""Unit tests for ONCC coffee price collector."""

from src.data_collection.oncc_coffee_collector import (
    _extract_by_css,
    _extract_by_label,
    _parse_fcfa,
)

SAMPLE_ONCC_HTML = """
<html><body><main>
<div>
  <div>
    <section><section><section><span>3 779</span></section></section>
  </div>
</div>
<div>
  <div>
    <section><section><section><span>1 952</span></section></section>
  </div>
</div>
<p>Cafe Arabica prix 3779 FCFA</p>
<p>Cafe Robusta prix 1952 FCFA</p>
</main></body></html>
"""


class TestOnccCoffeeCollector:
    def test_parse_fcfa(self):
        assert _parse_fcfa("3 779") == 3779.0
        assert _parse_fcfa("1,952") == 1952.0

    def test_label_fallback(self):
        prices = _extract_by_label(SAMPLE_ONCC_HTML)
        assert "arabica" in prices
        assert "robusta" in prices
        assert prices["arabica"] == 3779.0
        assert prices["robusta"] == 1952.0

    def test_css_may_fail_on_simplified_html(self):
        # Selecteurs nth-child fragiles — label fallback doit rester utilisable
        css = _extract_by_css(SAMPLE_ONCC_HTML)
        assert isinstance(css, dict)
