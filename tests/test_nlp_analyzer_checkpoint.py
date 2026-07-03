"""
Checkpoint test for NLPAnalyzer to verify basic functionality.

This is a minimal test to verify that the NLPAnalyzer can:
- Initialize properly
- Analyze sentiment of text
- Extract keywords
- Flag high-risk articles

Requirements: 3.3, 3.4, 3.5
"""

import pytest
from datetime import datetime
from src.nlp.nlp_analyzer import NLPAnalyzer
from src.models.data_models import NewsArticle


class TestNLPAnalyzerCheckpoint:
    """Checkpoint tests to verify NLPAnalyzer basic functionality."""
    
    def test_initialization(self):
        """Test that NLPAnalyzer can be initialized."""
        analyzer = NLPAnalyzer()
        
        assert analyzer is not None
        assert analyzer.model is not None
        assert analyzer.tokenizer is not None
        assert len(analyzer.market_shock_keywords) > 0
    
    def test_analyze_sentiment_basic(self):
        """Test basic sentiment analysis."""
        analyzer = NLPAnalyzer()
        
        # Test with a simple positive text
        positive_text = "Cocoa prices are stable and market conditions are favorable"
        result = analyzer.analyze_sentiment(positive_text)
        
        # Check structure
        assert "positive" in result
        assert "negative" in result
        assert "neutral" in result
        assert "score" in result
        
        # Check ranges
        assert 0 <= result["positive"] <= 1
        assert 0 <= result["negative"] <= 1
        assert 0 <= result["neutral"] <= 1
        assert -1 <= result["score"] <= 1
        
        # Check probabilities sum to ~1
        prob_sum = result["positive"] + result["negative"] + result["neutral"]
        assert abs(prob_sum - 1.0) < 0.01
    
    def test_analyze_sentiment_negative(self):
        """Test sentiment analysis with negative text."""
        analyzer = NLPAnalyzer()
        
        # Test with negative financial text
        negative_text = "Cocoa prices plunge amid supply concerns and market crisis"
        result = analyzer.analyze_sentiment(negative_text)
        
        # Should detect negative sentiment
        assert result["score"] < 0
        assert result["negative"] > result["positive"]
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        analyzer = NLPAnalyzer()
        
        # Text with multiple market shock keywords
        text = "Drought and swollen shoot disease threaten cocoa harvest in West Africa"
        keywords = analyzer.extract_keywords(text)
        
        # Should detect at least some keywords
        assert len(keywords) > 0
        assert "drought" in keywords
        assert "swollen shoot" in keywords
        assert "harvest" in keywords
    
    def test_flag_high_risk_positive_case(self):
        """Test high-risk flagging for a high-risk article."""
        analyzer = NLPAnalyzer()
        
        # Create a high-risk article (negative sentiment + multiple keywords)
        article = NewsArticle(
            id="test1",
            source="reuters",
            title="Crisis in Cocoa Markets",
            content="Severe drought and swollen shoot disease devastate cocoa crops, causing supply shortage and price surge",
            published_at=datetime.now(),
            url="http://example.com",
            keywords=[],
            sentiment_score=None
        )
        
        is_high_risk = analyzer.flag_high_risk(article)
        
        # Should be flagged as high-risk
        assert is_high_risk is True
        assert article.sentiment_score is not None
        assert article.sentiment_score < -0.6
        assert len(article.keywords) >= 2
    
    def test_flag_high_risk_negative_case(self):
        """Test high-risk flagging for a normal article."""
        analyzer = NLPAnalyzer()
        
        # Create a normal article (neutral/positive sentiment)
        article = NewsArticle(
            id="test2",
            source="bloomberg",
            title="Cocoa Market Update",
            content="Cocoa prices remain stable with normal trading conditions",
            published_at=datetime.now(),
            url="http://example.com",
            keywords=[],
            sentiment_score=None
        )
        
        is_high_risk = analyzer.flag_high_risk(article)
        
        # Should NOT be flagged as high-risk
        assert is_high_risk is False
    
    def test_batch_analyze(self):
        """Test batch sentiment analysis."""
        analyzer = NLPAnalyzer()
        
        texts = [
            "Cocoa prices are rising steadily",
            "Market conditions are deteriorating rapidly",
            "Stable trading continues in cocoa markets"
        ]
        
        results = analyzer.batch_analyze(texts, batch_size=2)
        
        # Should return results for all texts
        assert len(results) == 3
        
        # Each result should have the correct structure
        for result in results:
            assert "positive" in result
            assert "negative" in result
            assert "neutral" in result
            assert "score" in result
            assert -1 <= result["score"] <= 1
    
    def test_aggregate_sentiment(self):
        """Test sentiment aggregation."""
        analyzer = NLPAnalyzer()
        
        # Create articles with known sentiment scores
        articles = [
            NewsArticle(
                id="1",
                source="reuters",
                title="Title 1",
                content="Content 1",
                published_at=datetime.now(),
                url="http://example.com/1",
                keywords=[],
                sentiment_score=-0.5
            ),
            NewsArticle(
                id="2",
                source="bloomberg",
                title="Title 2",
                content="Content 2",
                published_at=datetime.now(),
                url="http://example.com/2",
                keywords=[],
                sentiment_score=0.3
            )
        ]
        
        avg_sentiment = analyzer.aggregate_sentiment(articles)
        
        # Should return a value between the two scores
        assert -0.5 <= avg_sentiment <= 0.3
        assert avg_sentiment != 0.0  # Should not be exactly zero


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
