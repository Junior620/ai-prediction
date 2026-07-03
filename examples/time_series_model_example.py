"""
Example usage of TimeSeriesModel for cocoa price prediction.

This script demonstrates:
1. Creating synthetic cocoa price data
2. Fitting the Prophet-based time series model
3. Generating predictions
4. Computing residuals
5. Extracting trend and seasonal components
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.models.time_series_model import TimeSeriesModel


def create_synthetic_cocoa_data(days: int = 730) -> pd.DataFrame:
    """Create synthetic cocoa price data with trend and seasonality.
    
    Args:
        days: Number of days of historical data
        
    Returns:
        DataFrame with columns 'ds' (date) and 'y' (price)
    """
    # Generate dates
    start_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Create realistic cocoa price components
    base_price = 3000  # Base price around 3000 USD/MT
    
    # Upward trend (cocoa prices have been rising)
    trend = np.linspace(0, 500, days)
    
    # Yearly seasonality (harvest cycles)
    yearly_cycle = 300 * np.sin(2 * np.pi * np.arange(days) / 365)
    
    # Market volatility
    volatility = np.random.normal(0, 100, days)
    
    # Occasional market shocks
    shocks = np.zeros(days)
    shock_indices = np.random.choice(days, size=5, replace=False)
    shocks[shock_indices] = np.random.choice([-500, 500], size=5)
    
    # Combine components
    prices = base_price + trend + yearly_cycle + volatility + shocks
    
    # Create DataFrame in Prophet format
    df = pd.DataFrame({
        'ds': dates,
        'y': prices
    })
    
    return df


def main():
    """Demonstrate TimeSeriesModel usage."""
    
    print("=" * 70)
    print("TimeSeriesModel Example - Cocoa Price Prediction")
    print("=" * 70)
    print()
    
    # 1. Create synthetic data
    print("1. Creating synthetic cocoa price data (2 years)...")
    df = create_synthetic_cocoa_data(days=730)
    print(f"   Generated {len(df)} days of price data")
    print(f"   Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
    print(f"   Price range: ${df['y'].min():.2f} - ${df['y'].max():.2f} per MT")
    print()
    
    # 2. Split into train and validation
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    print(f"2. Splitting data:")
    print(f"   Training set: {len(train_df)} days")
    print(f"   Validation set: {len(val_df)} days")
    print()
    
    # 3. Initialize and fit the model
    print("3. Initializing TimeSeriesModel with Prophet...")
    model = TimeSeriesModel(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        changepoint_prior_scale=0.05
    )
    print(f"   Model: {model}")
    print()
    
    print("4. Fitting model to training data...")
    model.fit(train_df)
    print("   Model fitted successfully!")
    print()
    
    # 4. Generate predictions
    print("5. Generating predictions for next 30 days...")
    forecast = model.predict(periods=30, freq='D')
    
    # Get only future predictions
    future_forecast = forecast.tail(30)
    print(f"   Generated {len(future_forecast)} predictions")
    print(f"   Prediction range: ${future_forecast['yhat'].min():.2f} - ${future_forecast['yhat'].max():.2f}")
    print()
    print("   Sample predictions (first 5 days):")
    for idx, row in future_forecast.head(5).iterrows():
        print(f"     {row['ds'].date()}: ${row['yhat']:.2f} "
              f"(CI: ${row['yhat_lower']:.2f} - ${row['yhat_upper']:.2f})")
    print()
    
    # 5. Compute residuals on training data
    print("6. Computing residuals on training data...")
    residuals = model.compute_residuals(train_df)
    print(f"   Residual statistics:")
    print(f"     Mean: ${residuals.mean():.2f}")
    print(f"     Std Dev: ${residuals.std():.2f}")
    print(f"     Min: ${residuals.min():.2f}")
    print(f"     Max: ${residuals.max():.2f}")
    print()
    
    # 6. Extract components
    print("7. Extracting trend and seasonal components...")
    components = model.get_components()
    print(f"   Extracted components for {len(components)} data points")
    print(f"   Available components: {', '.join(components.columns)}")
    print()
    
    # 7. Evaluate on validation set
    print("8. Evaluating on validation set...")
    val_forecast = model.predict(periods=len(val_df))
    val_predictions = val_forecast.tail(len(val_df))
    
    # Calculate simple metrics
    actual = val_df['y'].values
    predicted = val_predictions['yhat'].values
    
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    print(f"   Validation Metrics:")
    print(f"     MAE:  ${mae:.2f}")
    print(f"     RMSE: ${rmse:.2f}")
    print(f"     MAPE: {mape:.2f}%")
    print()
    
    # 8. Show hyperparameters
    print("9. Model hyperparameters:")
    params = model.get_hyperparameters()
    for key, value in params.items():
        print(f"     {key}: {value}")
    print()
    
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
