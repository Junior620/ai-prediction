"""
Sentiment Analysis for News Articles using TextBlob (lightweight alternative).
Analyzes market sentiment from cocoa-related news.
"""

import logging
from typing import Dict, List, Any, Optional

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Install with: pip install textblob")

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyzes sentiment of news articles using TextBlob (lightweight, no GPU needed).
    """
    
    def __init__(self, model_name: str = "textblob"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: Model name (currently only 'textblob' supported)
        """
        if not TEXTBLOB_AVAILABLE:
            logger.error("❌ TextBlob not installed")
            self.enabled = False
            return
        
        self.enabled = True
        logger.info(f"✅ Sentiment analyzer loaded (TextBlob)")
    
    def analyze_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment label and score
        """
        if not self.enabled or not text:
            return None
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            # Convert to label
            if polarity > 0.1:
                label = "POSITIVE"
            elif polarity < -0.1:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            
            # Convert polarity to confidence score (0 to 1)
            score = abs(polarity)
            
            return {
                "label": label,
                "score": score,
                "sentiment_value": polarity
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {str(e)}")
            return None
    
    def analyze_articles(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment of multiple news articles.
        
        Args:
            articles: List of article dictionaries with 'title' and 'description'
            
        Returns:
            Dictionary with aggregated sentiment metrics
        """
        if not self.enabled or not articles:
            return {
                "error": "Sentiment analyzer not enabled or no articles",
                "total_articles": 0
            }
        
        try:
            logger.info(f"Analyzing sentiment for {len(articles)} articles...")
            
            sentiments = []
            analyzed_articles = []
            
            for article in articles:
                # Combine title and description for analysis
                title = article.get("title", "")
                description = article.get("description", "")
                text = f"{title}. {description}"
                
                if not text.strip():
                    continue
                
                sentiment = self.analyze_text(text)
                
                if sentiment:
                    sentiments.append(sentiment)
                    analyzed_articles.append({
                        **article,
                        "sentiment": sentiment
                    })
            
            if not sentiments:
                return {
                    "error": "No sentiments analyzed",
                    "total_articles": len(articles)
                }
            
            # Calculate aggregated metrics
            positive_count = sum(1 for s in sentiments if s["label"] == "POSITIVE")
            negative_count = sum(1 for s in sentiments if s["label"] == "NEGATIVE")
            neutral_count = sum(1 for s in sentiments if s["label"] == "NEUTRAL")
            
            avg_score = sum(s["score"] for s in sentiments) / len(sentiments)
            avg_sentiment = sum(s["sentiment_value"] for s in sentiments) / len(sentiments)
            
            # Market sentiment indicator (-1 to 1)
            market_sentiment = avg_sentiment
            
            result = {
                "total_articles": len(articles),
                "analyzed_articles": len(sentiments),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "positive_ratio": positive_count / len(sentiments),
                "negative_ratio": negative_count / len(sentiments),
                "neutral_ratio": neutral_count / len(sentiments),
                "average_confidence": avg_score,
                "average_sentiment": avg_sentiment,
                "market_sentiment_score": market_sentiment,
                "market_sentiment_label": self._get_sentiment_label(market_sentiment),
                "articles_with_sentiment": analyzed_articles
            }
            
            logger.info(f"✅ Sentiment analysis complete: {result['market_sentiment_label']} ({market_sentiment:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Batch sentiment analysis failed: {str(e)}")
            return {
                "error": str(e),
                "total_articles": len(articles)
            }
    
    def _get_sentiment_label(self, score: float) -> str:
        """
        Convert sentiment score to human-readable label.
        
        Args:
            score: Sentiment score (-1 to 1)
            
        Returns:
            Sentiment label
        """
        if score > 0.5:
            return "Very Bullish"
        elif score > 0.2:
            return "Bullish"
        elif score > -0.2:
            return "Neutral"
        elif score > -0.5:
            return "Bearish"
        else:
            return "Very Bearish"
    
    def get_market_signal(self, sentiment_score: float) -> str:
        """
        Convert sentiment score to trading signal.
        
        Args:
            sentiment_score: Market sentiment score (-1 to 1)
            
        Returns:
            Trading signal: BUY, SELL, or HOLD
        """
        if sentiment_score > 0.3:
            return "BUY"
        elif sentiment_score < -0.3:
            return "SELL"
        else:
            return "HOLD"


if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Test articles
    test_articles = [
        {
            "title": "Cocoa prices surge to record highs",
            "description": "Cocoa futures reached unprecedented levels due to supply concerns in West Africa."
        },
        {
            "title": "Poor harvest threatens cocoa supply",
            "description": "Adverse weather conditions in Ghana and Ivory Coast are expected to reduce cocoa production significantly."
        },
        {
            "title": "Chocolate demand remains strong",
            "description": "Global chocolate consumption continues to grow, supporting cocoa prices."
        }
    ]
    
    print("=" * 80)
    print("SENTIMENT ANALYSIS TEST")
    print("=" * 80)
    print()
    
    results = analyzer.analyze_articles(test_articles)
    
    if "error" not in results:
        print(f"Total Articles: {results['total_articles']}")
        print(f"Analyzed: {results['analyzed_articles']}")
        print(f"Positive: {results['positive_count']} ({results['positive_ratio']:.1%})")
        print(f"Negative: {results['negative_count']} ({results['negative_ratio']:.1%})")
        print(f"Neutral: {results['neutral_count']} ({results['neutral_ratio']:.1%})")
        print(f"Market Sentiment: {results['market_sentiment_label']} ({results['market_sentiment_score']:.2f})")
        print(f"Trading Signal: {analyzer.get_market_signal(results['market_sentiment_score'])}")
        print()
        
        print("Individual Article Sentiments:")
        print("-" * 80)
        for article in results['articles_with_sentiment']:
            sentiment = article['sentiment']
            print(f"\n{article['title']}")
            print(f"  → {sentiment['label']} (score: {sentiment['sentiment_value']:.2f})")
    else:
        print(f"Error: {results['error']}")
