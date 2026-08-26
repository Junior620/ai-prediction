"""Unit tests for FR→EN sentiment scoring."""

from unittest.mock import patch

from src.data_collection.sentiment_scoring import score_sentiment


def test_english_positive_without_translate():
    with patch(
        "src.data_collection.sentiment_scoring.detect_language", return_value="en"
    ), patch(
        "src.data_collection.sentiment_scoring.translate_to_english"
    ) as tr:
        score, label, translated = score_sentiment(
            "Cocoa prices surge on strong demand",
            "Markets rally amid tight supply",
        )
        tr.assert_not_called()
        assert translated is False
        assert label in ("positive", "neutral", "negative")
        assert -1.0 <= score <= 1.0


def test_french_triggers_translation():
    with patch(
        "src.data_collection.sentiment_scoring.detect_language", return_value="fr"
    ), patch(
        "src.data_collection.sentiment_scoring.translate_to_english",
        return_value="Cocoa prices rise sharply on export growth",
    ) as tr:
        score, label, translated = score_sentiment(
            "Les prix du cacao montent fortement",
            "Hausse des exportations",
        )
        tr.assert_called_once()
        assert translated is True
        assert -1.0 <= score <= 1.0
