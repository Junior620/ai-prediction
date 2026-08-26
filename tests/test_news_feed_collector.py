"""Tests du collecteur news cacao (sans reseau)."""

from src.data_collection.news_feed_collector import (
    _parse_html_list,
    _parse_rss_items,
    passes_keyword_filter,
)
from src.data_collection.news_sources import MVP_SOURCES


SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Investir</title>
    <item>
      <title>Cacao: la production commercialisee chute au Cameroun</title>
      <link>https://www.investiraucameroun.com/agriculture/cacao-1</link>
      <description>Les feves de cacao en baisse.</description>
      <pubDate>Mon, 06 Aug 2026 10:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Recette de cookies au chocolat pour aout</title>
      <link>https://www.investiraucameroun.com/lifestyle/cookies</link>
      <description>Dessert facile.</description>
      <pubDate>Mon, 06 Aug 2026 11:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""

SAMPLE_ECOFIN_HTML = """
<html><body>
<a href="/cacao/0910-122284-cacao-les-prix-chutent-a-new-york">Cacao: les prix chutent a New York</a>
<a href="/cacao/1709-121598-ghana-contrebande-cacao">Ghana: contrebande de cacao</a>
<a href="/telecom/123-orange-mtn">Orange et MTN au Cameroun</a>
</body></html>
"""


class TestKeywordFilter:
    def test_keeps_cocoa_price(self):
        assert passes_keyword_filter("Cocoa prices fall in New York", "futures ICE")

    def test_rejects_recipe_noise(self):
        assert not passes_keyword_filter(
            "Chocolate chip cookie recipe",
            "baking challenge snack dessert",
        )

    def test_force_keep(self):
        assert passes_keyword_filter("Daily bulletin", "", force_keep=True)


class TestParsers:
    def test_parse_rss_items(self):
        items = _parse_rss_items(SAMPLE_RSS, "Investir au Cameroun", max_items=10)
        assert len(items) == 2
        assert "Cacao" in items[0]["title"]
        assert items[0]["url"].endswith("cacao-1")

    def test_parse_html_ecofin(self):
        ecofin = next(s for s in MVP_SOURCES if s["id"] == "ecofin")
        items = _parse_html_list(SAMPLE_ECOFIN_HTML, ecofin)
        assert len(items) >= 2
        assert any("prix" in i["title"].lower() or "Cacao" in i["title"] for i in items)
        urls = [i["url"] for i in items]
        assert all("agenceecofin.com" in u for u in urls)
