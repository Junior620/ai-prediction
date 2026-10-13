"""Unit tests for ICE London collector (HTML parsing, no network)."""

from datetime import date

from src.data_collection.ice_london_collector import (
    extract_price_regex,
    parse_table_rows_from_html,
    _parse_price,
)

SAMPLE_ICE_TABLE = """
<table>
<tbody>
<tr><td>2026-08-27</td><td>2,356.00</td></tr>
<tr><td>2026-08-26</td><td>2,340.50</td></tr>
<tr><td>2026-08-25</td><td>2,320.00</td></tr>
</tbody>
</table>
"""

EMPTY_TABLE = "<table><tbody></tbody></table>"

REDESIGN_HTML = """
<div class="market-data">
  <span>Settlement</span><strong>2,410.75</strong>
</div>
"""


class TestIceLondonCollector:
    def test_parse_price(self):
        assert _parse_price("2,356.00") == 2356.0
        assert _parse_price("") is None

    def test_parse_table_rows(self):
        rows = parse_table_rows_from_html(SAMPLE_ICE_TABLE, max_rows=10)
        assert len(rows) == 3
        assert rows[0]["price"] == 2356.0
        assert rows[0]["date"] == "2026-08-27"
        assert rows[0]["quarantine_reason"] is None

    def test_missing_session_date_is_not_replaced_by_today(self):
        html = "<table><tr><td>DEC26</td><td>4277</td></tr><tr><td>n/a</td><td>4100</td></tr></table>"
        rows = parse_table_rows_from_html(html)
        assert rows[0]["session_date"] is None
        assert rows[0]["contract"] == "DEC26"
        assert rows[0]["quarantine_reason"] == "contrat_sans_date_de_seance"
        assert rows[1]["quarantine_reason"] == "date_de_seance_absente"
        assert all(row["date"] != date.today().isoformat() for row in rows)

    def test_empty_table_returns_empty(self):
        assert parse_table_rows_from_html(EMPTY_TABLE) == []

    def test_regex_settlement_fallback(self):
        price = extract_price_regex(REDESIGN_HTML)
        assert price == 2410.75
