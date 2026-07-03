# NLP Analyzer Implementation Guide

## Overview

The `NLPAnalyzer` class provides sentiment analysis capabilities for cocoa market news using FinBERT, a BERT model pre-trained on financial text. This implementation supports the hybrid price prediction system by detecting market sentiment and potential market shocks from news articles.

## Architecture

### Model Selection: FinBERT

**Why FinBERT over VADER?**
- **Domain-specific training**: FinBERT is pre-trained on financial text (10k+ financial news articles)
- **Superior accuracy**: 81.5% accuracy on financial sentiment vs ~65% for VADER
- **Context understanding**: BERT-based architecture captures nuanced financial language
- **Proven performance**: Widely used in financial NLP applications

### Key Features

1. **Single & Batch Sentiment Analysis**: Efficient processing of individual or multiple articles
2. **Keyword Extraction**: Identifies market shock indicators (diseases, weather, policy, speculation)
3. **High-Risk Flagging**: Combines sentiment and keywords to flag potential market shocks
4. **Temporal Aggregation**: Weighted sentiment averaging with exponential decay for recency

## Class Interface

### Initialization

```python
from src.nlp.nlp_analyzer import NLPAnalyzer

# Initialize with default FinBERT model
analyzer = NLPAnalyzer()

# Or specify custom model and device
analyzer = NLPAnalyzer(
    model_name="ProsusAI/finbert",
    device="cuda"  # or "cpu"
)
```

**Parameters:**
- `model_name` (str): HuggingFace model identifier (default: "ProsusAI/finbert")
- `device` (str, optional): Computation device. Auto-selects CUDA if available

### Methods

#### 1. analyze_sentiment()

Analyzes sentiment of a single text.

```python
text = "Cocoa prices surge on supply concerns"
result = analyzer.analyze_sentiment(text)

# Returns:
# {
#     "positive": 0.15,
#     "negative": 0.70,
#     "neutral": 0.15,
#     "score": -0.55  # positive - negative
# }
```

**Parameters:**
- `text` (str): Text to analyze (article title + content recommended)

**Returns:**
- `dict`: Sentiment probabilities and normalized score (-1 to +1)

**Use Case:** Real-time sentiment analysis of breaking news

---

#### 2. batch_analyze()

Efficiently processes multiple texts using batching.

```python
texts = [
    "Article 1 content...",
    "Article 2 content...",
    "Article 3 content..."
]
results = analyzer.batch_analyze(texts, batch_size=32)

# Returns list of sentiment dicts (same format as analyze_sentiment)
```

**Parameters:**
- `texts` (List[str]): List of texts to analyze
- `batch_size` (int): Batch size for processing (default: 32)

**Returns:**
- `List[dict]`: List of sentiment dictionaries

**Use Case:** Processing historical news archives or daily news feeds

**Performance:** ~10x faster than sequential processing for large batches

---

#### 3. extract_keywords()

Extracts market shock keywords from text.

```python
text = "Drought and swollen shoot disease threaten cocoa harvest"
keywords = analyzer.extract_keywords(text)

# Returns: ['drought', 'swollen shoot', 'harvest']
```

**Parameters:**
- `text` (str): Text to search for keywords
- `keywords` (List[str], optional): Custom keyword list. Uses default if None

**Returns:**
- `List[str]`: Detected keywords (lowercase)

**Default Keywords Categories:**
- **Diseases**: swollen shoot, black pod, frosty pod
- **Weather**: el nino, la nina, drought, flood, hurricane
- **Policy**: export ban, export tax, quota, regulation
- **Speculation**: hedge fund, short squeeze
- **Production**: crop failure, harvest, yield, production cut
- **Supply**: supply shortage, supply disruption

**Use Case:** Identifying articles that mention specific risk factors

---

#### 4. flag_high_risk()

Flags articles as high-risk based on sentiment and keywords.

```python
from src.models.data_models import NewsArticle
from datetime import datetime

article = NewsArticle(
    id="1",
    source="reuters",
    title="Crisis in cocoa markets",
    content="Drought and disease devastate crops",
    published_at=datetime.now(),
    url="http://example.com"
)

is_high_risk = analyzer.flag_high_risk(
    article,
    sentiment_threshold=-0.6,
    keyword_threshold=2
)

# Returns: True (if sentiment < -0.6 AND keywords >= 2)
```

**Parameters:**
- `article` (NewsArticle): Article to evaluate
- `sentiment_threshold` (float): Minimum sentiment score (default: -0.6)
- `keyword_threshold` (int): Minimum keyword count (default: 2)

**Returns:**
- `bool`: True if high-risk conditions met

**High-Risk Criteria:**
1. Sentiment score < threshold (default: -0.6)
2. AND keyword count >= threshold (default: 2)

**Use Case:** Triggering alerts for potential market shocks

**Side Effects:** Updates `article.sentiment_score` and `article.keywords` if not set

---

#### 5. aggregate_sentiment()

Aggregates sentiment over a time window with recency weighting.

```python
from datetime import timedelta

articles = [...]  # List of NewsArticle objects with sentiment_score

avg_sentiment = analyzer.aggregate_sentiment(
    articles,
    time_window=timedelta(hours=24)
)

# Returns: -0.35 (weighted average, more recent articles weighted higher)
```

**Parameters:**
- `articles` (List[NewsArticle]): Articles with sentiment scores
- `time_window` (timedelta): Time window for aggregation (default: 24 hours)

**Returns:**
- `float`: Weighted average sentiment score (-1 to +1)

**Weighting Formula:**
```
weight = exp(-time_diff / half_life)
where half_life = 12 hours
```

**Use Case:** Computing overall market sentiment for prediction adjustment

---

## Integration with Price Predictor

The NLPAnalyzer integrates with the hybrid prediction system as follows:

```python
# 1. Collect recent news
recent_news = data_collector.collect_news_feed(
    sources=["reuters", "bloomberg"],
    keywords=["cocoa", "cacao"],
    hours_back=24
)

# 2. Analyze sentiment
for article in recent_news:
    sentiment = nlp_analyzer.analyze_sentiment(
        f"{article.title} {article.content}"
    )
    article.sentiment_score = sentiment["score"]
    article.keywords = nlp_analyzer.extract_keywords(article.content)
    article.is_high_risk = nlp_analyzer.flag_high_risk(article)

# 3. Aggregate sentiment
aggregated_sentiment = nlp_analyzer.aggregate_sentiment(recent_news)

# 4. Use in prediction
prediction = price_predictor.predict(
    horizons=[1, 7, 30],
    exog_features=econometric_data,
    recent_news=recent_news  # Includes sentiment scores
)

# 5. Adjust confidence interval if high-risk detected
if any(article.is_high_risk for article in recent_news):
    # Widen confidence interval by 50%
    prediction.confidence_interval = (
        prediction.price - 1.5 * interval_width,
        prediction.price + 1.5 * interval_width
    )
```

## Performance Considerations

### Model Loading

**First Run:**
- Downloads FinBERT model (~400MB) from HuggingFace
- Cached locally for subsequent runs
- Takes ~30 seconds on first initialization

**Subsequent Runs:**
- Loads from cache (~2 seconds)

### Inference Speed

**CPU:**
- Single article: ~100-200ms
- Batch (32 articles): ~2-3 seconds

**GPU (CUDA):**
- Single article: ~20-30ms
- Batch (32 articles): ~300-500ms

**Recommendations:**
- Use batch processing for multiple articles
- Use GPU if available for real-time applications
- Cache sentiment scores to avoid recomputation

### Memory Usage

- Model size: ~400MB
- Peak memory (CPU): ~1GB
- Peak memory (GPU): ~2GB

## Error Handling

```python
try:
    analyzer = NLPAnalyzer()
except Exception as e:
    # Handle model loading errors
    print(f"Failed to load FinBERT: {e}")
    # Fallback to simpler sentiment analysis or skip NLP

try:
    sentiment = analyzer.analyze_sentiment(text)
except Exception as e:
    # Handle inference errors
    print(f"Sentiment analysis failed: {e}")
    sentiment = {"score": 0.0}  # Neutral fallback
```

## Testing

### Unit Tests

```python
# tests/test_nlp_analyzer.py
import pytest
from src.nlp.nlp_analyzer import NLPAnalyzer
from src.models.data_models import NewsArticle
from datetime import datetime, timedelta

def test_analyze_sentiment():
    analyzer = NLPAnalyzer()
    result = analyzer.analyze_sentiment("Positive news about cocoa")
    
    assert "score" in result
    assert -1.0 <= result["score"] <= 1.0
    assert result["positive"] + result["negative"] + result["neutral"] == pytest.approx(1.0)

def test_batch_analyze():
    analyzer = NLPAnalyzer()
    texts = ["Text 1", "Text 2", "Text 3"]
    results = analyzer.batch_analyze(texts)
    
    assert len(results) == 3
    assert all("score" in r for r in results)

def test_extract_keywords():
    analyzer = NLPAnalyzer()
    text = "Drought and disease affect harvest"
    keywords = analyzer.extract_keywords(text)
    
    assert "drought" in keywords
    assert "disease" in keywords
    assert "harvest" in keywords

def test_flag_high_risk():
    analyzer = NLPAnalyzer()
    article = NewsArticle(
        id="1", source="reuters", title="Crisis",
        content="Drought and disease devastate crops",
        published_at=datetime.now(), url="http://example.com"
    )
    
    is_high_risk = analyzer.flag_high_risk(article)
    assert isinstance(is_high_risk, bool)

def test_aggregate_sentiment():
    analyzer = NLPAnalyzer()
    articles = [
        NewsArticle(
            id="1", source="reuters", title="Title",
            content="Content", published_at=datetime.now(),
            url="http://example.com", sentiment_score=-0.5
        )
    ]
    
    avg = analyzer.aggregate_sentiment(articles)
    assert -1.0 <= avg <= 1.0
```

## Configuration

Add to `config/config.yaml`:

```yaml
nlp:
  model_name: "ProsusAI/finbert"
  device: "cuda"  # or "cpu"
  batch_size: 32
  sentiment_threshold: -0.6
  keyword_threshold: 2
  aggregation_window_hours: 24
  half_life_hours: 12
```

## Dependencies

```
torch>=2.0.0
transformers>=4.37.0
numpy>=1.26.0
```

## Troubleshooting

### Issue: Model download fails

**Solution:**
```python
# Set HuggingFace cache directory
import os
os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'

analyzer = NLPAnalyzer()
```

### Issue: CUDA out of memory

**Solution:**
```python
# Reduce batch size or use CPU
analyzer = NLPAnalyzer(device="cpu")
results = analyzer.batch_analyze(texts, batch_size=8)
```

### Issue: Slow inference on CPU

**Solution:**
- Use batch processing
- Consider using quantized models
- Cache sentiment scores for historical articles

## Future Enhancements

1. **Fine-tuning**: Fine-tune FinBERT on cocoa-specific news corpus
2. **Multi-lingual**: Support French news sources (major cocoa markets)
3. **Entity Recognition**: Extract specific entities (countries, companies, diseases)
4. **Topic Modeling**: Cluster articles by topic for better aggregation
5. **Real-time Streaming**: Process news feeds in real-time with streaming API

## References

- [FinBERT Paper](https://arxiv.org/abs/1908.10063)
- [HuggingFace FinBERT](https://huggingface.co/ProsusAI/finbert)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
