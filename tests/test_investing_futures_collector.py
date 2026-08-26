"""Tests parsing table contrats Investing.com (sans reseau)."""

from src.data_collection.investing_futures_collector import parse_investing_futures_table

SAMPLE_TABLE = """
<table id="BarchartDataTable" class="genTbl closedTbl crossRatesTbl">
<thead><tr><th>Mois</th><th>Dernier</th><th>Var.</th><th>Ouverture</th><th>+ Haut</th><th>+ Bas</th><th>Volume</th><th>Heure</th></tr></thead>
<tbody>
<tr id="pair_1">
  <td class="center"></td>
  <td class="bold left"><a href="?page=chart&symbol=CCY00">Cash&nbsp;</a></td>
  <td>5819s</td>
  <td class="greenFont" dir="ltr">+18</td>
  <td>5819</td><td>5819</td><td>5819</td><td>0</td><td>08/25/26</td><td></td>
</tr>
<tr id="pair_2">
  <td class="center"></td>
  <td class="bold left"><a href="?page=chart&symbol=CCU26">Sep&nbsp;26&nbsp;</a></td>
  <td>5759s</td>
  <td class="redFont" dir="ltr">-61</td>
  <td>5818</td><td>5819</td><td>5759</td><td>36</td><td>08/25/26</td><td></td>
</tr>
<tr id="pair_3">
  <td class="center"></td>
  <td class="bold left"><a href="?page=chart&symbol=CCZ26">Dec&nbsp;26&nbsp;</a></td>
  <td>5829s</td>
  <td class="redFont" dir="ltr">-116</td>
  <td>5944</td><td>5980</td><td>5722</td><td>18612</td><td>08/25/26</td><td></td>
</tr>
</tbody>
</table>
"""


class TestParseInvestingFutures:
    def test_parses_contract_rows(self):
        rows = parse_investing_futures_table(SAMPLE_TABLE)
        assert len(rows) == 3
        assert rows[0]["contract"] == "Cash"
        assert rows[0]["symbol"] == "CCY00"
        assert rows[0]["price_usd"] == 5819.0
        assert rows[0]["settlement"] is True

    def test_parses_futures_symbol_and_volume(self):
        rows = parse_investing_futures_table(SAMPLE_TABLE)
        sep = rows[1]
        assert sep["symbol"] == "CCU26"
        assert sep["contract"] == "Sep 26"
        assert sep["change"] == -61.0
        assert sep["volume"] == 36

    def test_empty_html(self):
        assert parse_investing_futures_table("") == []
