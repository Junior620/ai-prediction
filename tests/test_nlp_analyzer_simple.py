"""
Simple checkpoint test for NLPAnalyzer to verify it can be imported and initialized.

This test verifies basic functionality without requiring model downloads.
"""

import pytest


def test_nlp_analyzer_import():
    """Test that NLPAnalyzer can be imported."""
    from src.nlp.nlp_analyzer import NLPAnalyzer
    assert NLPAnalyzer is not None


def test_nlp_analyzer_has_required_methods():
    """Test that NLPAnalyzer has all required methods."""
    from src.nlp.nlp_analyzer import NLPAnalyzer
    
    # Check that all required methods exist
    assert hasattr(NLPAnalyzer, 'analyze_sentiment')
    assert hasattr(NLPAnalyzer, 'batch_analyze')
    assert hasattr(NLPAnalyzer, 'extract_keywords')
    assert hasattr(NLPAnalyzer, 'flag_high_risk')
    assert hasattr(NLPAnalyzer, 'aggregate_sentiment')
    
    # Check that methods are callable
    assert callable(getattr(NLPAnalyzer, 'analyze_sentiment'))
    assert callable(getattr(NLPAnalyzer, 'batch_analyze'))
    assert callable(getattr(NLPAnalyzer, 'extract_keywords'))
    assert callable(getattr(NLPAnalyzer, 'flag_high_risk'))
    assert callable(getattr(NLPAnalyzer, 'aggregate_sentiment'))


def test_nlp_analyzer_keywords_defined():
    """Test that market shock keywords are defined."""
    from src.nlp.nlp_analyzer import NLPAnalyzer
    
    # Check that keywords are defined
    assert hasattr(NLPAnalyzer, 'MARKET_SHOCK_KEYWORDS')
    assert len(NLPAnalyzer.MARKET_SHOCK_KEYWORDS) > 0
    
    # Check some expected keywords
    keywords_lower = [k.lower() for k in NLPAnalyzer.MARKET_SHOCK_KEYWORDS]
    assert 'drought' in keywords_lower
    assert 'disease' in keywords_lower
    assert 'swollen shoot' in keywords_lower


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
