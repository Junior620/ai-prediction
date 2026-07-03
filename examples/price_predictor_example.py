"""
Example usage of the PricePredictor class.

This script demonstrates how to:
1. Load and prepare data
2. Train the time series and ML models
3. Initialize the PricePredictor
4. Generate predictions for multiple horizons
5. Interpret the results
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.models import TimeSeriesModel, MLModel, PricePredictor
from src.nlp import NLPAnalyzer
from src.models.data_models import NewsArticle


def generate_synthetic_data(n_days: int = 365) -> pd.DataFrame:
    """Generate synthetic cocoa price data for demonstration."""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate synthetic price data with trend and seasonality
    trend = np.linspace(2800, 3200, n_days)
    seasonality = 200 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    noise = np.random.normal(0, 50, n_days)
    prices = trend + seasonality + noise
    
    df = pd.DataFrame({
        'ds': dates,
        'y': prices
    })
    
    return df


def generate_synthetic_features(n_horizons: int) -> pd.DataFrame:
    """Generate synthetic econometric features for demonstration."""
    features = pd.DataFrame({
        'temperature': np.random.uniform(20, 30, n_horizons),
        'rainfall': np.random.uniform(5, 20, n_horizons),
        'stock_level': np.random.uniform(40000, 60000, n_horizons),
        'production': np.random.uniform(4000, 5000, n_horizons),
        'fx_rate_xaf_usd': np.random.uniform(0.0016, 0.0018, n_horizons),
        'fx_rate_gbp_usd': np.random.uniform(1.25, 1.30, n_horizons),
        'fx_rate_eur_usd': np.random.uniform(1.08, 1.12, n_horizons)
    })
    
    return features


def generate_synthetic_news() -> list:
    """Generate synthetic news articles for demonstration."""
    articles = [
        NewsArticle(
            id="1",
            source="reuters",
            title="Cocoa prices stable amid steady demand",
            content="Market conditions remain favorable with balanced supply and demand.",
            published_at=datetime.now() - timedelta(hours=2),
            url="http://example.com/article1",
            keywords=["demand", "supply"],
            sentiment_score=0.2,
            is_high_risk=False
        ),
        NewsArticle(
            id="2",
            source="bloomberg",
            title="Weather concerns in West Africa",
            content="Drought conditions may impact cocoa production in key regions.",
            published_at=datetime.now() - timedelta(hours=5),
            url="http://example.com/article2",
            keywords=["drought", "production"],
            sentiment_score=-0.4,
            is_high_risk=False
        ),
        NewsArticle(
            id="3",
            source="reuters",
            title="Strong harvest expected this season",
            content="Favorable weather and good crop management lead to optimistic forecasts.",
            published_at=datetime.now() - timedelta(hours=12),
            url="http://example.com/article3",
            keywords=["harvest", "forecast"],
            sentiment_score=0.6,
            is_high_risk=False
        )
    ]
    
    return articles


def main():
    """Main example workflow."""
    print("=" * 80)
    print("PricePredictor Example - Hybrid Cocoa Price Forecasting")
    print("=" * 80)
    print()
    
    # Step 1: Generate synthetic data
    print("Step 1: Generating synthetic historical data...")
    historical_data = generate_synthetic_data(n_days=365)
    print(f"  Generated {len(historical_data)} days of historical price data")
    print(f"  Price range: ${historical_data['y'].min():.2f} - ${historical_data['y'].max():.2f}")
    print()
    
    # Step 2: Train time series model (Prophet)
    print("Step 2: Training time series model (Prophet)...")
    ts_model = TimeSeriesModel(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    ts_model.fit(historical_data)
    print("  ✓ Time series model trained successfully")
    print()
    
    # Step 3: Compute residuals and train ML model (XGBoost)
    print("Step 3: Training ML model (XGBoost) on residuals...")
    
    # Compute residuals from time series model
    residuals = ts_model.compute_residuals(historical_data)
    
    # Generate synthetic features for training (in real scenario, use actual econometric data)
    train_features = generate_synthetic_features(len(residuals))
    
    # Train ML model
    ml_model = MLModel(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    ml_model.fit(train_features, residuals)
    print("  ✓ ML model trained successfully")
    print()
    
    # Step 4: Initialize NLP analyzer
    print("Step 4: Initializing NLP analyzer (FinBERT)...")
    nlp_analyzer = NLPAnalyzer()
    print("  ✓ NLP analyzer initialized")
    print()
    
    # Step 5: Create PricePredictor
    print("Step 5: Creating PricePredictor...")
    predictor = PricePredictor(
        ts_model=ts_model,
        ml_model=ml_model,
        nlp_analyzer=nlp_analyzer,
        sentiment_weight=0.1,
        model_version="1.0.0"
    )
    print("  ✓ PricePredictor created successfully")
    print()
    
    # Step 6: Generate predictions
    print("Step 6: Generating predictions for multiple horizons...")
    horizons = [1, 7, 30]
    
    # Generate features for prediction horizons
    prediction_features = generate_synthetic_features(len(horizons))
    
    # Generate synthetic news
    recent_news = generate_synthetic_news()
    
    # Make predictions
    predictions = predictor.predict(
        horizons=horizons,
        exog_features=prediction_features,
        recent_news=recent_news,
        historical_range=(historical_data['y'].min(), historical_data['y'].max())
    )
    print(f"  ✓ Generated {len(predictions)} predictions")
    print()
    
    # Step 7: Display results
    print("Step 7: Prediction Results")
    print("-" * 80)
    
    for pred in predictions:
        print(f"\nHorizon: {pred.horizon} day(s)")
        print(f"  Predicted Price: ${pred.price:.2f} USD/MT")
        print(f"  Confidence Interval (95%): [${pred.confidence_interval[0]:.2f}, ${pred.confidence_interval[1]:.2f}]")
        print(f"  Interval Width: ${pred.confidence_interval[1] - pred.confidence_interval[0]:.2f}")
        print(f"  Components:")
        print(f"    - Baseline (Prophet): ${pred.components['baseline']:.2f}")
        print(f"    - Residual (XGBoost): ${pred.components['residual']:.2f}")
        print(f"    - Sentiment Adjustment: ${pred.components['sentiment']:.2f}")
        print(f"  Model Version: {pred.model_version}")
        print(f"  Timestamp: {pred.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print()
    print("-" * 80)
    
    # Step 8: Model information
    print("\nStep 8: Model Information")
    print("-" * 80)
    model_info = predictor.get_model_info()
    print(f"Model Version: {model_info['model_version']}")
    print(f"Sentiment Weight: {model_info['sentiment_weight']}")
    print(f"\nTime Series Model Parameters:")
    for key, value in model_info['ts_model_params'].items():
        print(f"  {key}: {value}")
    print(f"\nML Model Parameters:")
    for key, value in model_info['ml_model_params'].items():
        print(f"  {key}: {value}")
    
    print()
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
