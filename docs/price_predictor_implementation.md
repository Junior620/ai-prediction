# PricePredictor Implementation Documentation

## Overview

The `PricePredictor` class is the core component of the hybrid cocoa price forecasting system. It orchestrates the combination of three complementary models to generate accurate price predictions with confidence intervals.

## Architecture

### Hybrid Prediction Formula

The system uses an additive approach to combine predictions:

```
Final_Price = Baseline + Residual_Correction + Sentiment_Adjustment
```

Where:
- **Baseline**: Prophet's time series prediction (captures trend and seasonality)
- **Residual_Correction**: XGBoost's prediction of residuals (captures non-linear patterns using econometric features)
- **Sentiment_Adjustment**: FinBERT's sentiment-based adjustment (captures market sentiment from news)

### Components

1. **TimeSeriesModel (Prophet)**
   - Captures long-term trends and seasonal patterns
   - Provides baseline predictions with confidence intervals
   - Handles missing data and trend changes gracefully

2. **MLModel (XGBoost)**
   - Predicts residuals from the time series baseline
   - Uses econometric features: temperature, rainfall, stock levels, production, FX rates
   - Provides feature importance for interpretability

3. **NLPAnalyzer (FinBERT)**
   - Analyzes sentiment from financial news articles
   - Aggregates sentiment over time windows with exponential decay
   - Flags high-risk articles that may indicate market shocks

## Key Features

### Multi-Horizon Predictions

The predictor generates forecasts for multiple time horizons simultaneously:
- **Short-term**: 1 day ahead
- **Medium-term**: 7 days ahead
- **Long-term**: 30 days ahead

### Confidence Intervals

Confidence intervals are computed by:
1. Combining uncertainties from both time series and ML models
2. Using the formula: `combined_uncertainty = sqrt(ts_uncertainty² + ml_uncertainty²)`
3. Widening intervals by 50% when high-risk sentiment is detected (score < -0.6)

### Sentiment-Based Adjustments

Sentiment adjustments are proportional to the baseline price:
```python
sentiment_adjustment = sentiment_score * sentiment_weight * baseline
```

With default `sentiment_weight=0.1`, a sentiment score of -0.5 would reduce the price by 5%.

### Prediction Validation

All predictions are validated against historical price ranges to catch unrealistic values. Invalid predictions are clamped to the valid range.

## Implementation Details

### Class: `PricePredictor`

**Location**: `src/models/price_predictor.py`

**Key Methods**:

1. **`__init__(ts_model, ml_model, nlp_analyzer, sentiment_weight=0.1, model_version="1.0.0")`**
   - Initializes the predictor with trained models
   - Validates that models are fitted
   - Sets sentiment weight and version

2. **`predict(horizons, exog_features, recent_news, historical_range=None)`**
   - Main prediction method
   - Generates predictions for multiple horizons
   - Returns list of `Prediction` objects

3. **`combine_predictions(baseline, residual, sentiment_score, sentiment_weight)`**
   - Combines components using additive formula
   - Returns final predicted price

4. **`calculate_confidence_interval(prediction, ts_uncertainty, ml_uncertainty, sentiment_risk, confidence_level=0.95)`**
   - Computes combined confidence interval
   - Widens interval if sentiment risk detected
   - Returns (lower_bound, upper_bound)

5. **`validate_prediction(prediction, historical_range)`**
   - Validates prediction is within realistic bounds
   - Returns boolean indicating validity

6. **`get_model_info()`**
   - Returns dictionary with model metadata
   - Includes hyperparameters from all models

## Usage Example

```python
from src.models import TimeSeriesModel, MLModel, PricePredictor
from src.nlp import NLPAnalyzer

# Initialize and train models
ts_model = TimeSeriesModel()
ts_model.fit(historical_data)

residuals = ts_model.compute_residuals(historical_data)
ml_model = MLModel()
ml_model.fit(features, residuals)

nlp_analyzer = NLPAnalyzer()

# Create predictor
predictor = PricePredictor(
    ts_model=ts_model,
    ml_model=ml_model,
    nlp_analyzer=nlp_analyzer,
    sentiment_weight=0.1
)

# Generate predictions
predictions = predictor.predict(
    horizons=[1, 7, 30],
    exog_features=future_features,
    recent_news=news_articles
)

# Access results
for pred in predictions:
    print(f"Horizon {pred.horizon}d: ${pred.price:.2f}")
    print(f"CI: [{pred.confidence_interval[0]:.2f}, {pred.confidence_interval[1]:.2f}]")
    print(f"Components: {pred.components}")
```

## Testing

### Test Coverage

The implementation includes comprehensive unit tests covering:

1. **Initialization and Validation**
   - Fitted model requirements
   - Sentiment weight validation
   - Error handling

2. **Prediction Combination**
   - Basic combination logic
   - Positive/negative sentiment effects
   - Negative residual handling

3. **Confidence Interval Calculation**
   - Basic interval computation
   - Sentiment risk widening
   - Different confidence levels (90%, 95%, 99%)

4. **Prediction Validation**
   - Within-range validation
   - Out-of-range detection
   - Boundary cases

5. **Main Predict Method**
   - Single and multiple horizons
   - High-risk sentiment handling
   - Input validation
   - Error cases

### Running Tests

```bash
# Run all PricePredictor tests
python -m pytest tests/test_price_predictor.py -v

# Run with coverage
python -m pytest tests/test_price_predictor.py --cov=src.models.price_predictor
```

### Test Results

All 23 tests pass successfully:
- 4 initialization tests
- 4 combination tests
- 3 confidence interval tests
- 5 validation tests
- 6 predict method tests
- 1 model info test
- 1 repr test

## Performance Considerations

### Computational Complexity

- **Prophet prediction**: O(n) where n is the number of historical data points
- **XGBoost prediction**: O(k * d) where k is number of trees and d is tree depth
- **FinBERT sentiment**: O(m * l) where m is number of articles and l is text length
- **Overall**: Linear in the number of horizons

### Optimization Tips

1. **Batch Processing**: Use batch prediction for multiple horizons to minimize overhead
2. **Caching**: Cache Prophet forecasts if generating predictions for the same date multiple times
3. **Feature Precomputation**: Precompute econometric features before prediction
4. **Sentiment Aggregation**: Aggregate sentiment once and reuse for all horizons

## Integration with Other Components

### Upstream Dependencies

- `TimeSeriesModel`: Must be fitted on historical price data
- `MLModel`: Must be fitted on residuals with econometric features
- `NLPAnalyzer`: Requires FinBERT model loaded

### Downstream Consumers

- **API Service**: FastAPI endpoints use PricePredictor to serve predictions
- **Performance Monitor**: Tracks prediction accuracy over time
- **Model Manager**: Logs predictions and model versions to MLflow

## Configuration

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sentiment_weight` | 0.1 | Weight for sentiment adjustment (0.0-1.0) |
| `confidence_level` | 0.95 | Confidence level for intervals |
| `sentiment_risk_threshold` | -0.6 | Threshold for high-risk sentiment |
| `interval_widening_factor` | 1.5 | Factor to widen CI when risk detected |

### Tuning Guidelines

- **sentiment_weight**: Increase (0.15-0.2) if news has strong market impact; decrease (0.05) if market is less reactive
- **confidence_level**: Use 0.99 for conservative estimates, 0.90 for tighter bounds
- **sentiment_risk_threshold**: Lower (-0.7) for stricter risk detection, raise (-0.5) for more sensitivity

## Error Handling

The predictor handles several error scenarios:

1. **Unfitted Models**: Raises `ValueError` if models not trained
2. **Mismatched Inputs**: Raises `ValueError` if horizons and features have different lengths
3. **Invalid Horizons**: Raises `ValueError` for non-positive horizons
4. **Prophet Failure**: Raises `RuntimeError` with fallback suggestion
5. **XGBoost Failure**: Raises `RuntimeError` with fallback suggestion
6. **Sentiment Failure**: Logs warning and uses neutral sentiment (0.0)

## Logging

The predictor uses structured logging with the `loguru` library:

- **INFO**: Prediction generation, model initialization, results summary
- **DEBUG**: Detailed component values, intermediate calculations
- **WARNING**: Validation failures, out-of-range predictions, sentiment errors
- **ERROR**: Model failures, critical errors

## Future Enhancements

Potential improvements for future versions:

1. **Uncertainty Quantification**: Use quantile regression for better uncertainty estimates
2. **Ensemble Methods**: Add model averaging or stacking for improved accuracy
3. **Online Learning**: Implement incremental updates without full retraining
4. **Explainability**: Add SHAP values for feature importance in predictions
5. **Multi-Market**: Extend to predict multiple markets simultaneously
6. **Anomaly Detection**: Flag unusual predictions that may indicate data issues

## References

- **Prophet Documentation**: https://facebook.github.io/prophet/
- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **FinBERT Paper**: https://arxiv.org/abs/1908.10063
- **Requirements**: See `requirements.md` sections 7.1-7.5
- **Design**: See `design.md` section on Price Predictor

## Changelog

### Version 1.0.0 (2026-05-06)
- Initial implementation of PricePredictor class
- Multi-horizon prediction support
- Confidence interval calculation with sentiment-based widening
- Comprehensive unit tests (23 tests, 100% pass rate)
- Example script demonstrating usage
- Full documentation
