"""
Example usage of the NLPAnalyzer class for sentiment analysis.

This script demonstrates how to use the NLPAnalyzer to:
1. Analyze sentiment of individual news articles
2. Batch process multiple articles
3. Extract market shock keywords
4. Flag high-risk articles
5. Aggregate sentiment over time windows
"""

from datetime import datetime, timedelta
from src.nlp.nlp_analyzer import NLPAnalyzer
from src.models.data_models import NewsArticle


def example_single_sentiment_analysis():
    """Example: Analyze sentiment of a single article."""
    print("=" * 80)
    print("Example 1: Single Sentiment Analysis")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = NLPAnalyzer()
    
    # Sample article text
    text = """
    Cocoa prices surge to 10-year high as drought devastates West African crops.
    The severe El Niño weather pattern has caused widespread crop failures in
    Ghana and Côte d'Ivoire, the world's largest cocoa producers. Industry
    experts warn of potential supply shortages in the coming months.
    """
    
    # Analyze sentiment
    sentiment = analyzer.analyze_sentiment(text)
    
    print(f"Text: {text.strip()[:100]}...")
    print(f"\nSentiment Analysis Results:")
    print(f"  Positive: {sentiment['positive']:.3f}")
    print(f"  Negative: {sentiment['negative']:.3f}")
    print(f"  Neutral:  {sentiment['neutral']:.3f}")
    print(f"  Score:    {sentiment['score']:.3f} (range: -1 to +1)")
    print()


def example_batch_analysis():
    """Example: Batch analyze multiple articles."""
    print("=" * 80)
    print("Example 2: Batch Sentiment Analysis")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = NLPAnalyzer()
    
    # Sample articles
    articles = [
        "Cocoa harvest exceeds expectations in Brazil, prices stabilize.",
        "Export ban threatens cocoa supply chain, traders concerned.",
        "New disease-resistant cocoa varieties show promise in trials.",
        "Hedge funds increase short positions on cocoa futures.",
        "Weather conditions improve in West Africa, boosting crop outlook."
    ]
    
    # Batch analyze
    results = analyzer.batch_analyze(articles, batch_size=3)
    
    print(f"Analyzed {len(articles)} articles:\n")
    for i, (article, result) in enumerate(zip(articles, results), 1):
        print(f"{i}. {article[:60]}...")
        print(f"   Score: {result['score']:+.3f}")
    print()


def example_keyword_extraction():
    """Example: Extract market shock keywords."""
    print("=" * 80)
    print("Example 3: Keyword Extraction")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = NLPAnalyzer()
    
    # Sample article with multiple keywords
    text = """
    Swollen shoot disease and drought conditions create perfect storm for
    cocoa markets. Production cuts expected as farmers struggle with crop
    failures. Speculation intensifies as hedge funds bet on supply shortage.
    """
    
    # Extract keywords
    keywords = analyzer.extract_keywords(text)
    
    print(f"Text: {text.strip()}")
    print(f"\nDetected Market Shock Keywords:")
    for keyword in keywords:
        print(f"  - {keyword}")
    print(f"\nTotal keywords detected: {len(keywords)}")
    print()


def example_high_risk_flagging():
    """Example: Flag high-risk articles."""
    print("=" * 80)
    print("Example 4: High-Risk Article Flagging")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = NLPAnalyzer()
    
    # Create sample articles
    articles = [
        NewsArticle(
            id="1",
            source="reuters",
            title="Drought and disease devastate cocoa crops",
            content="Severe drought and swollen shoot disease have caused widespread "
                   "crop failures in major cocoa-producing regions. Export bans "
                   "are being considered as supply shortages worsen.",
            published_at=datetime.now(),
            url="http://example.com/article1"
        ),
        NewsArticle(
            id="2",
            source="bloomberg",
            title="Cocoa harvest meets expectations",
            content="This year's cocoa harvest has met industry expectations with "
                   "favorable weather conditions supporting healthy crop yields.",
            published_at=datetime.now(),
            url="http://example.com/article2"
        ),
        NewsArticle(
            id="3",
            source="reuters",
            title="Policy changes threaten cocoa exports",
            content="New export regulations and speculation about potential export "
                   "bans have created uncertainty in cocoa markets.",
            published_at=datetime.now(),
            url="http://example.com/article3"
        )
    ]
    
    # Flag high-risk articles
    print("Analyzing articles for high-risk indicators:\n")
    for article in articles:
        is_high_risk = analyzer.flag_high_risk(article)
        
        print(f"Article: {article.title}")
        print(f"  Sentiment Score: {article.sentiment_score:+.3f}")
        print(f"  Keywords: {', '.join(article.keywords) if article.keywords else 'None'}")
        print(f"  High Risk: {'YES' if is_high_risk else 'NO'}")
        print()


def example_sentiment_aggregation():
    """Example: Aggregate sentiment over time window."""
    print("=" * 80)
    print("Example 5: Sentiment Aggregation")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = NLPAnalyzer()
    
    # Create articles with different timestamps
    now = datetime.now()
    articles = [
        NewsArticle(
            id="1", source="reuters", title="Recent negative news",
            content="Drought threatens crops", published_at=now - timedelta(hours=2),
            url="http://example.com/1", sentiment_score=-0.7
        ),
        NewsArticle(
            id="2", source="bloomberg", title="Slightly older positive news",
            content="Harvest looks good", published_at=now - timedelta(hours=12),
            url="http://example.com/2", sentiment_score=0.5
        ),
        NewsArticle(
            id="3", source="reuters", title="Old neutral news",
            content="Market stable", published_at=now - timedelta(hours=20),
            url="http://example.com/3", sentiment_score=0.1
        ),
        NewsArticle(
            id="4", source="bloomberg", title="Very old news (outside window)",
            content="Prices rise", published_at=now - timedelta(hours=30),
            url="http://example.com/4", sentiment_score=0.8
        )
    ]
    
    # Aggregate sentiment over 24-hour window
    aggregated_score = analyzer.aggregate_sentiment(
        articles,
        time_window=timedelta(hours=24)
    )
    
    print("Articles in time window (24 hours):\n")
    for article in articles:
        hours_ago = (now - article.published_at).total_seconds() / 3600
        in_window = hours_ago <= 24
        print(f"  {article.title}")
        print(f"    Published: {hours_ago:.1f} hours ago")
        print(f"    Sentiment: {article.sentiment_score:+.3f}")
        print(f"    In window: {'YES' if in_window else 'NO'}")
        print()
    
    print(f"Aggregated Sentiment Score: {aggregated_score:+.3f}")
    print("(More recent articles weighted higher using exponential decay)")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 80)
    print("NLP ANALYZER EXAMPLES")
    print("*" * 80)
    print("\n")
    
    try:
        example_single_sentiment_analysis()
        example_batch_analysis()
        example_keyword_extraction()
        example_high_risk_flagging()
        example_sentiment_aggregation()
        
        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nNote: Make sure you have installed the required dependencies:")
        print("  pip install torch transformers")
        print("\nThe first run will download the FinBERT model (~400MB)")


if __name__ == "__main__":
    main()
