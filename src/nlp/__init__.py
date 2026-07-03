"""
NLP module for sentiment analysis of financial news.

This module provides sentiment analysis capabilities using:
- FinBERT (ProsusAI/finbert) when torch + transformers are installed
- TextBlob as a lightweight fallback

Classes:
    NLPAnalyzer: Full-featured analyzer with FinBERT/TextBlob auto-detection
    SentimentAnalyzer: Legacy lightweight TextBlob-only analyzer
"""

from src.nlp.nlp_analyzer import NLPAnalyzer
from src.nlp.sentiment_analyzer import SentimentAnalyzer

__all__ = ["NLPAnalyzer", "SentimentAnalyzer"]
