"""
Example usage of data models for the Cocoa Price Prediction System.

This script demonstrates how to create and validate instances of all data models.
"""

from datetime import datetime, timezone
from src.models.data_models import (
    PriceData,
    EconometricData,
    NewsArticle,
    Prediction,
    ModelMetrics,
    ValidationError,
)


def main():
    """Demonstrate usage of all data models."""
    
    # Example 1: PriceData
    print("=" * 60)
    print("Example 1: PriceData")
    print("=" * 60)
    price_data = PriceData(
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        market="ICE_London",
        price=3250.50,
        volume=1500.0,
        currency="USD"
    )
    print(f"Price: ${price_data.price:.2f}/MT on {price_data.market}")
    print(f"Volume: {price_data.volume:.0f} MT")
    print()
    
    # Example 2: EconometricData
    print("=" * 60)
    print("Example 2: EconometricData")
    print("=" * 60)
    econ_data = EconometricData(
        timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
        temperature=28.5,
        rainfall=12.3,
        stock_level=50000.0,
        production=10000.0,
        fx_rate_xaf_usd=0.0017,
        fx_rate_gbp_usd=1.27,
        fx_rate_eur_usd=1.09
    )
    print(f"Temperature: {econ_data.temperature}°C")
    print(f"Rainfall: {econ_data.rainfall}mm")
    print(f"Stock Level: {econ_data.stock_level:.0f} MT")
    print(f"FX Rates:")
    print(f"  XAF/USD: {econ_data.fx_rate_xaf_usd}")
    print(f"  GBP/USD: {econ_data.fx_rate_gbp_usd}")
    print(f"  EUR/USD: {econ_data.fx_rate_eur_usd}")
    print()
    
    # Example 3: NewsArticle
    print("=" * 60)
    print("Example 3: NewsArticle")
    print("=" * 60)
    article = NewsArticle(
        id="article_20240115_001",
        source="reuters",
        title="Cocoa prices surge on West African supply concerns",
        content="Cocoa futures rose sharply on Monday as concerns about crop disease...",
        published_at=datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc),
        url="https://reuters.com/markets/commodities/cocoa-prices-surge",
        keywords=["cocoa", "supply", "disease", "west africa"],
        sentiment_score=-0.45,
        is_high_risk=False
    )
    print(f"Title: {article.title}")
    print(f"Source: {article.source}")
    print(f"Sentiment: {article.sentiment_score:.2f}")
    print(f"Keywords: {', '.join(article.keywords)}")
    print()
    
    # Example 4: Prediction
    print("=" * 60)
    print("Example 4: Prediction")
    print("=" * 60)
    prediction = Prediction(
        horizon=7,
        price=3280.00,
        confidence_interval=(3150.00, 3410.00),
        confidence_level=0.95,
        timestamp=datetime.now(timezone.utc),
        model_version="v1.2.0",
        components={
            "baseline": 3200.00,
            "residual": 50.00,
            "sentiment": 30.00
        }
    )
    print(f"Horizon: {prediction.horizon} days")
    print(f"Predicted Price: ${prediction.price:.2f}/MT")
    print(f"95% CI: [${prediction.confidence_interval[0]:.2f}, ${prediction.confidence_interval[1]:.2f}]")
    print(f"Components:")
    for component, value in prediction.components.items():
        print(f"  - {component}: ${value:.2f}")
    print()
    
    # Example 5: ModelMetrics
    print("=" * 60)
    print("Example 5: ModelMetrics")
    print("=" * 60)
    metrics = ModelMetrics(
        rmse=145.30,
        mae=118.50,
        mape=0.038,
        directional_accuracy=0.78,
        coverage_rate=0.95,
        mean_interval_width=285.00,
        timestamp=datetime.now(timezone.utc),
        model_version="v1.2.0"
    )
    print(f"Model Version: {metrics.model_version}")
    print(f"RMSE: {metrics.rmse:.2f}")
    print(f"MAE: {metrics.mae:.2f}")
    print(f"MAPE: {metrics.mape:.1%}")
    print(f"Directional Accuracy: {metrics.directional_accuracy:.1%}")
    print(f"Coverage Rate: {metrics.coverage_rate:.1%}")
    print()
    
    # Example 6: ValidationError
    print("=" * 60)
    print("Example 6: ValidationError")
    print("=" * 60)
    error = ValidationError(
        field="price",
        value="15000.0",
        error_type="out_of_range",
        message="Price 15000.0 exceeds maximum allowed value of 10000.0",
        severity="ERROR"
    )
    print(f"Field: {error.field}")
    print(f"Error Type: {error.error_type}")
    print(f"Severity: {error.severity}")
    print(f"Message: {error.message}")
    print()
    
    # Example 7: Validation in action
    print("=" * 60)
    print("Example 7: Validation Error Handling")
    print("=" * 60)
    try:
        invalid_price = PriceData(
            timestamp=datetime.now(timezone.utc),
            market="ICE_London",
            price=15000.0,  # Invalid: exceeds maximum
            volume=1000.0,
            currency="USD"
        )
    except Exception as e:
        print(f"Validation failed as expected:")
        print(f"  {type(e).__name__}: {str(e)[:100]}...")
    print()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
