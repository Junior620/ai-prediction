"""
NLP Analyzer for sentiment analysis of financial news.

Supports two backends:
- **FinBERT** (ProsusAI/finbert) — high-quality financial sentiment analysis
  Requires `torch` and `transformers` packages.
- **TextBlob** — lightweight fallback when FinBERT is unavailable.

The analyzer automatically selects the best available backend on import.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

from src.models.data_models import NewsArticle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend availability detection
# ---------------------------------------------------------------------------
FINBERT_AVAILABLE = False
TEXTBLOB_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    FINBERT_AVAILABLE = True
    logger.info("FinBERT backend available (torch + transformers installed)")
except ImportError:
    logger.info("FinBERT not available (torch/transformers not installed)")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    pass

if not FINBERT_AVAILABLE and not TEXTBLOB_AVAILABLE:
    logger.warning(
        "No NLP backend available. Install 'torch transformers' for FinBERT "
        "or 'textblob' for the lightweight fallback."
    )


class NLPAnalyzer:
    """
    FinBERT-based sentiment analysis for financial news, with TextBlob fallback.

    This class provides comprehensive sentiment analysis capabilities for
    cocoa market news, including:
    - Single and batch sentiment analysis
    - Market shock keyword extraction
    - High-risk article flagging
    - Temporal sentiment aggregation

    Attributes:
        backend: Name of the active backend ("finbert" or "textblob")
        model_name: HuggingFace model identifier (FinBERT only)
        device: Computation device (FinBERT only)
        market_shock_keywords: Keywords indicating potential market shocks
    """

    # Market shock keywords for cocoa markets
    MARKET_SHOCK_KEYWORDS = [
        "disease", "swollen shoot", "black pod", "frosty pod",
        "el nino", "la nina", "drought", "flood", "hurricane",
        "policy", "export ban", "export tax", "quota", "regulation",
        "speculation", "hedge fund", "short squeeze",
        "strike", "protest", "conflict", "war",
        "crop failure", "harvest", "yield", "production cut",
        "supply shortage", "supply disruption",
        "price surge", "price spike", "volatility"
    ]

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None
    ):
        """
        Initialize NLP Analyzer with the best available backend.

        If torch and transformers are installed, FinBERT is used.
        Otherwise, TextBlob is used as a lightweight fallback.

        Args:
            model_name: HuggingFace model identifier (default: ProsusAI/finbert)
            device: Computation device. If None, auto-selects cuda if available
        """
        self.model_name = model_name
        self.market_shock_keywords = self.MARKET_SHOCK_KEYWORDS

        if FINBERT_AVAILABLE:
            self.backend = "finbert"
            # Automatically select device
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"✅ NLP Analyzer initialized with FinBERT on {self.device}")

        elif TEXTBLOB_AVAILABLE:
            self.backend = "textblob"
            self.device = "cpu"
            self.tokenizer = None
            self.model = None
            logger.info("✅ NLP Analyzer initialized with TextBlob fallback")

        else:
            self.backend = "none"
            self.device = "cpu"
            self.tokenizer = None
            self.model = None
            logger.warning("⚠️ NLP Analyzer initialized WITHOUT any backend — "
                           "sentiment analysis will return neutral scores")

    # ------------------------------------------------------------------
    # Sentiment analysis
    # ------------------------------------------------------------------

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.

        Uses FinBERT when available, otherwise falls back to TextBlob.

        Args:
            text: Text to analyze (article title + content recommended)

        Returns:
            Dictionary with keys:
                - positive: Probability of positive sentiment (0-1)
                - negative: Probability of negative sentiment (0-1)
                - neutral: Probability of neutral sentiment (0-1)
                - score: Normalized sentiment score (-1 to +1)
                         Computed as: positive - negative

        Example:
            >>> analyzer = NLPAnalyzer()
            >>> result = analyzer.analyze_sentiment("Cocoa prices surge on supply concerns")
            >>> print(result["score"])
            -0.45
        """
        if self.backend == "finbert":
            return self._analyze_finbert(text)
        elif self.backend == "textblob":
            return self._analyze_textblob(text)
        else:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "score": 0.0}

    def _analyze_finbert(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using FinBERT."""
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]

        # FinBERT outputs: [positive, negative, neutral]
        positive = probabilities[0].item()
        negative = probabilities[1].item()
        neutral = probabilities[2].item()

        # Compute normalized sentiment score: positive - negative
        # Range: -1 (very negative) to +1 (very positive)
        score = positive - negative

        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "score": score
        }

    def _analyze_textblob(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob (fallback)."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1

        # Map TextBlob polarity to FinBERT-compatible output
        if polarity > 0:
            positive = min(polarity, 1.0)
            negative = 0.0
            neutral = 1.0 - positive
        elif polarity < 0:
            positive = 0.0
            negative = min(abs(polarity), 1.0)
            neutral = 1.0 - negative
        else:
            positive = 0.0
            negative = 0.0
            neutral = 1.0

        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "score": polarity
        }

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def batch_analyze(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """
        Analyze sentiment for multiple texts efficiently.

        Uses batched inference for FinBERT, sequential for TextBlob.

        Args:
            texts: List of texts to analyze
            batch_size: Number of texts per batch (FinBERT only)

        Returns:
            List of sentiment dictionaries (same format as analyze_sentiment)
        """
        if self.backend == "finbert":
            return self._batch_analyze_finbert(texts, batch_size)
        else:
            return [self.analyze_sentiment(text) for text in texts]

    def _batch_analyze_finbert(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """Batch analyze using FinBERT with GPU-efficient batching."""
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)

            for probs in probabilities:
                positive = probs[0].item()
                negative = probs[1].item()
                neutral = probs[2].item()
                score = positive - negative

                results.append({
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                    "score": score
                })

        return results

    # ------------------------------------------------------------------
    # Keyword extraction & high-risk flagging
    # ------------------------------------------------------------------

    def extract_keywords(
        self,
        text: str,
        keywords: Optional[List[str]] = None
    ) -> List[str]:
        """
        Extract market shock keywords from text.

        Args:
            text: Text to search for keywords
            keywords: Custom keyword list. If None, uses default market shock keywords

        Returns:
            List of detected keywords (lowercase)
        """
        if keywords is None:
            keywords = self.market_shock_keywords

        text_lower = text.lower()
        return [kw.lower() for kw in keywords if kw.lower() in text_lower]

    def flag_high_risk(
        self,
        article: NewsArticle,
        sentiment_threshold: float = -0.6,
        keyword_threshold: int = 2
    ) -> bool:
        """
        Flag article as high-risk based on sentiment and keywords.

        An article is flagged as high-risk if:
        1. Sentiment score is below the threshold (default: -0.6), AND
        2. At least keyword_threshold market shock keywords are detected

        Args:
            article: NewsArticle object to evaluate
            sentiment_threshold: Minimum sentiment score for high-risk
            keyword_threshold: Minimum number of keywords for high-risk

        Returns:
            True if article is flagged as high-risk, False otherwise
        """
        # Analyze sentiment if not already computed
        if article.sentiment_score is None:
            combined_text = f"{article.title} {article.content}"
            sentiment = self.analyze_sentiment(combined_text)
            article.sentiment_score = sentiment["score"]

        # Extract keywords if not already extracted
        if not article.keywords:
            combined_text = f"{article.title} {article.content}"
            article.keywords = self.extract_keywords(combined_text)

        # Check both conditions for high-risk flagging
        is_negative_sentiment = article.sentiment_score < sentiment_threshold
        has_enough_keywords = len(article.keywords) >= keyword_threshold

        return is_negative_sentiment and has_enough_keywords

    # ------------------------------------------------------------------
    # Sentiment aggregation
    # ------------------------------------------------------------------

    def aggregate_sentiment(
        self,
        articles: List[NewsArticle],
        time_window: timedelta = timedelta(hours=24)
    ) -> float:
        """
        Aggregate sentiment scores over a time window.

        Computes a weighted average of sentiment scores for articles within
        the specified time window. More recent articles receive higher weights
        using exponential decay (half-life = 12 hours).

        Args:
            articles: List of NewsArticle objects with sentiment scores
            time_window: Time window for aggregation (default: 24 hours)

        Returns:
            Weighted average sentiment score (-1 to +1)
            Returns 0.0 if no articles or all articles lack sentiment scores
        """
        if not articles:
            return 0.0

        current_time = datetime.now(timezone.utc)

        # Filter articles within time window and with sentiment scores
        valid_articles = [
            article for article in articles
            if article.sentiment_score is not None
            and (current_time - article.published_at) <= time_window
        ]

        if not valid_articles:
            return 0.0

        # Compute weighted average with exponential decay
        total_weighted_score = 0.0
        total_weight = 0.0

        for article in valid_articles:
            time_diff = (current_time - article.published_at).total_seconds() / 3600
            # Half-life = 12 hours (weight halves every 12 hours)
            weight = np.exp(-time_diff / 12.0)

            total_weighted_score += article.sentiment_score * weight
            total_weight += weight

        if total_weight > 0:
            return total_weighted_score / total_weight
        else:
            return 0.0
