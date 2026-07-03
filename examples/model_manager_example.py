"""
Example usage of ModelManager for MLflow model lifecycle management.

This example demonstrates:
1. Initializing ModelManager
2. Logging models with metrics and parameters
3. Registering models in the Model Registry
4. Loading models from the registry
5. Promoting models between stages
6. Rolling back to previous versions
7. Listing model versions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.models import ModelManager, TimeSeriesModel, MLModel


def example_model_manager_workflow():
    """Demonstrate complete ModelManager workflow."""
    
    print("=" * 80)
    print("ModelManager Example - MLflow Model Lifecycle Management")
    print("=" * 80)
    
    # Step 1: Initialize ModelManager
    print("\n1. Initializing ModelManager...")
    model_manager = ModelManager(
        tracking_uri="file:./mlruns",  # Local file-based tracking
        registry_uri="sqlite:///mlflow.db"  # Local SQLite registry
    )
    print(f"   {model_manager}")
    
    # Step 2: Train a sample Prophet model
    print("\n2. Training a sample Prophet model...")
    
    # Create synthetic training data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    prices = 3000 + 200 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + \
             np.random.normal(0, 50, len(dates))
    
    train_df = pd.DataFrame({
        'ds': dates,
        'y': prices
    })
    
    # Train Prophet model
    prophet_model = TimeSeriesModel(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    prophet_model.fit(train_df)
    
    # Generate predictions for evaluation
    forecast = prophet_model.predict(periods=30)
    
    # Calculate sample metrics
    metrics = {
        'rmse': 45.2,
        'mae': 32.1,
        'mape': 2.5,
        'r2': 0.85
    }
    
    params = prophet_model.get_hyperparameters()
    
    print(f"   Model trained with metrics: {metrics}")
    
    # Step 3: Log the model to MLflow
    print("\n3. Logging Prophet model to MLflow...")
    version_1 = model_manager.log_model(
        model=prophet_model.model,
        model_name="cocoa_prophet_baseline",
        model_type="prophet",
        metrics=metrics,
        params=params
    )
    print(f"   Model logged as version: {version_1}")
    
    # Step 4: Train and log an improved version
    print("\n4. Training and logging an improved model version...")
    
    # Simulate improved model with better metrics
    prophet_model_v2 = TimeSeriesModel(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        changepoint_prior_scale=0.03  # More conservative
    )
    prophet_model_v2.fit(train_df)
    
    improved_metrics = {
        'rmse': 38.5,  # Better than v1
        'mae': 28.3,
        'mape': 2.1,
        'r2': 0.88
    }
    
    params_v2 = prophet_model_v2.get_hyperparameters()
    
    version_2 = model_manager.log_model(
        model=prophet_model_v2.model,
        model_name="cocoa_prophet_baseline",
        model_type="prophet",
        metrics=improved_metrics,
        params=params_v2
    )
    print(f"   Improved model logged as version: {version_2}")
    print(f"   Metrics improved: RMSE {metrics['rmse']:.1f} -> {improved_metrics['rmse']:.1f}")
    
    # Step 5: List all model versions
    print("\n5. Listing all model versions...")
    versions = model_manager.list_model_versions(
        model_name="cocoa_prophet_baseline",
        max_results=5
    )
    print(f"   Found {len(versions)} versions:")
    for v in versions:
        print(f"     - Version {v.version}: stage={v.current_stage}")
    
    # Step 6: Promote version 2 to Staging
    print("\n6. Promoting version 2 to Staging...")
    model_manager.promote_model(
        model_name="cocoa_prophet_baseline",
        version=version_2,
        from_stage="None",
        to_stage="Staging"
    )
    print(f"   Version {version_2} promoted to Staging")
    
    # Step 7: Load model from Staging
    print("\n7. Loading model from Staging stage...")
    staging_model = model_manager.load_model(
        model_name="cocoa_prophet_baseline",
        stage="Staging"
    )
    print(f"   Model loaded successfully from Staging")
    
    # Step 8: Promote to Production after validation
    print("\n8. Promoting version 2 to Production...")
    model_manager.promote_model(
        model_name="cocoa_prophet_baseline",
        version=version_2,
        from_stage="Staging",
        to_stage="Production"
    )
    print(f"   Version {version_2} promoted to Production")
    
    # Step 9: Get model info
    print("\n9. Getting Production model info...")
    model_info = model_manager.get_model_info(
        model_name="cocoa_prophet_baseline",
        stage="Production"
    )
    print(f"   Production model version: {model_info['version']}")
    print(f"   Metrics: {model_info['metrics']}")
    print(f"   Parameters: {model_info['params']}")
    
    # Step 10: Simulate rollback scenario
    print("\n10. Simulating rollback to version 1...")
    model_manager.rollback_model(
        model_name="cocoa_prophet_baseline",
        to_version=version_1
    )
    print(f"   Rolled back to version {version_1}")
    
    # Step 11: Verify rollback
    print("\n11. Verifying rollback...")
    current_prod = model_manager.get_model_info(
        model_name="cocoa_prophet_baseline",
        stage="Production"
    )
    print(f"   Current Production version: {current_prod['version']}")
    print(f"   Rollback successful: {current_prod['version'] == version_1}")
    
    print("\n" + "=" * 80)
    print("ModelManager workflow completed successfully!")
    print("=" * 80)


def example_xgboost_model_logging():
    """Example of logging an XGBoost model."""
    
    print("\n" + "=" * 80)
    print("Example: Logging XGBoost Model")
    print("=" * 80)
    
    # Initialize ModelManager
    model_manager = ModelManager(
        tracking_uri="file:./mlruns",
        registry_uri="sqlite:///mlflow.db"
    )
    
    # Create synthetic residual data
    n_samples = 1000
    X = pd.DataFrame({
        'temperature': np.random.uniform(20, 35, n_samples),
        'rainfall': np.random.uniform(0, 200, n_samples),
        'stock_level': np.random.uniform(100000, 500000, n_samples),
        'fx_rate_xaf_usd': np.random.uniform(580, 620, n_samples),
        'sentiment_score': np.random.uniform(-1, 1, n_samples)
    })
    
    # Synthetic residuals
    y = 10 * X['temperature'] - 0.05 * X['rainfall'] + \
        0.0001 * X['stock_level'] + 2 * X['sentiment_score'] + \
        np.random.normal(0, 20, n_samples)
    
    # Train XGBoost model
    print("\n1. Training XGBoost model...")
    ml_model = MLModel(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    ml_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    # Compute metrics
    y_pred = ml_model.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
    mae = np.mean(np.abs(y_val - y_pred))
    
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': 0.75
    }
    
    params = ml_model.get_hyperparameters()
    
    print(f"   Model trained with RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    # Log model
    print("\n2. Logging XGBoost model to MLflow...")
    version = model_manager.log_model(
        model=ml_model.model,
        model_name="cocoa_xgboost_residual",
        model_type="xgboost",
        metrics=metrics,
        params=params
    )
    
    print(f"   XGBoost model logged as version: {version}")
    
    # Get feature importance and log as artifact
    print("\n3. Getting feature importance...")
    importance_df = ml_model.get_feature_importance()
    print("   Top 3 features:")
    for idx, row in importance_df.head(3).iterrows():
        print(f"     {row['feature']}: {row['importance']:.2f}")
    
    print("\n" + "=" * 80)
    print("XGBoost model logging completed!")
    print("=" * 80)


def example_model_comparison():
    """Example of comparing multiple model versions."""
    
    print("\n" + "=" * 80)
    print("Example: Comparing Model Versions")
    print("=" * 80)
    
    # Initialize ModelManager
    model_manager = ModelManager(
        tracking_uri="file:./mlruns",
        registry_uri="sqlite:///mlflow.db"
    )
    
    # List all versions
    print("\n1. Listing all versions of cocoa_prophet_baseline...")
    versions = model_manager.list_model_versions(
        model_name="cocoa_prophet_baseline",
        max_results=5
    )
    
    if len(versions) < 2:
        print("   Need at least 2 versions for comparison. Run example_model_manager_workflow() first.")
        return
    
    # Compare first two versions
    print(f"\n2. Comparing versions {versions[0].version} and {versions[1].version}...")
    
    info_v1 = model_manager.get_model_info(
        model_name="cocoa_prophet_baseline",
        version=versions[0].version
    )
    
    info_v2 = model_manager.get_model_info(
        model_name="cocoa_prophet_baseline",
        version=versions[1].version
    )
    
    print(f"\n   Version {info_v1['version']}:")
    print(f"     Stage: {info_v1['stage']}")
    print(f"     RMSE: {info_v1['metrics'].get('rmse', 'N/A')}")
    print(f"     MAE: {info_v1['metrics'].get('mae', 'N/A')}")
    
    print(f"\n   Version {info_v2['version']}:")
    print(f"     Stage: {info_v2['stage']}")
    print(f"     RMSE: {info_v2['metrics'].get('rmse', 'N/A')}")
    print(f"     MAE: {info_v2['metrics'].get('mae', 'N/A')}")
    
    # Determine better model
    rmse_v1 = float(info_v1['metrics'].get('rmse', float('inf')))
    rmse_v2 = float(info_v2['metrics'].get('rmse', float('inf')))
    
    if rmse_v1 < rmse_v2:
        print(f"\n   Version {info_v1['version']} has better RMSE")
    else:
        print(f"\n   Version {info_v2['version']} has better RMSE")
    
    print("\n" + "=" * 80)
    print("Model comparison completed!")
    print("=" * 80)


if __name__ == "__main__":
    # Run the main workflow example
    example_model_manager_workflow()
    
    # Run XGBoost logging example
    example_xgboost_model_logging()
    
    # Run model comparison example
    example_model_comparison()
