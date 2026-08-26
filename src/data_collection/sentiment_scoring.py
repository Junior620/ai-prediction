"""Sentiment scoring with optional FR→EN translation for TextBlob."""

from __future__ import annotations

from typing import Optional, Tuple

from textblob import TextBlob


def detect_language(text: str) -> Optional[str]:
    sample = (text or "").strip()
    if len(sample) < 20:
        return None
    try:
        from langdetect import detect

        return detect(sample)
    except Exception:
        return None


def translate_to_english(text: str) -> str:
    try:
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source="auto", target="en").translate(text) or text
    except Exception:
        return text


def score_sentiment(
    title: str,
    description: str = "",
    *,
    force_translate: bool = False,
) -> Tuple[float, str, bool]:
    """
    Returns (score, label, translated).

    French (or force_translate) text is translated to English before TextBlob.
    """
    raw = f"{title}. {description}".strip()
    lang = detect_language(raw)
    translated = False
    text = raw
    if force_translate or lang == "fr":
        text = translate_to_english(raw)
        translated = text != raw or lang == "fr"

    blob = TextBlob(text)
    score = float(blob.sentiment.polarity)
    if score > 0.1:
        label = "positive"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"
    return score, label, translated
