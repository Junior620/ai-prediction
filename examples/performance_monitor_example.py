"""
Example usage of the PerformanceMonitor class.

This example demonstrates:
1. Computing performance metrics from predictions
2. Tracking metrics to database
3. Detecting performance degradation
4. Comparing model versions
5. Triggering retraining alerts
"""

import numpy as np
from datetime import datetime
from unittest.mock import Mock

from src.monitoring.performance_monitor import PerformanceMonitor


def main():
    """Demonstrate PerformanceMonitor functionality."""
    
    print("=" * 80)
    print("PerformanceMonitor Example")
    print("=" * 80)
    
    # Create a mock Supabase client for demonstration
    # In production, use: from supabase import create_client
    mock_supabase = Mock()
    mock_table = Mock()
    mock_supabase.table.return_value = mock_table
    
    # Initialize PerformanceMonitor
    monitor = PerformanceMonitor(
        supabase_client=mock_supabase,
        degradation_threshold=0.15  # 15% degradation threshold
    )
    
    print("\n1. Computing Performance Metrics")
    print("-" * 80)
    
    # Simulate actual vs predicted prices
    y_true = np.array([3000.0, 3100.0, 3200.0, 3150.0, 3250.0, 3300.0, 3280.0, 3350.0])
    y_pred = np.array([2950.0, 3120.0, 3180.0, 3160.0, 3230.0, 3320.0, 3290.0, 3340.0])
    y_pred_lower = y_pred - 100.0
    y_pred_upper = y_pred + 100.0
    
    # Compute metrics
    metrics = monitor.compute_metrics(y_true, y_pred, y_pred_lower, y_pred_upper)
    
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MAPE: {metrics['mape']:.4f} ({metrics['mape']*100:.2f}%)")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    print(f"Coverage Rate: {metrics['coverage_rate']:.2%}")
    print(f"Mean Interval Width: {metrics['mean_interval_width']:.2f}")
    
    print("\n2. Tracking Performance to Database")
    print("-" * 80)
    
    # Mock database response
    mock_response = Mock()
    mock_response.data = [{"id": 1}]
    mock_table.insert.return_value.execute.return_value = mock_response
    
    # Track metrics
    timestamp = datetime.now()
    model_version = "v1.2.3"
    
    monitor.track_performance(metrics, timestamp, model_version)
    print(f"✓ Metrics tracked for model {model_version} at {timestamp}")
    
    print("\n3. Detecting Performance Degradation")
    print("-" * 80)
    
    # Baseline metrics (from initial model deployment)
    baseline_metrics = {
        "rmse": 40.0,
        "mae": 30.0,
        "mape": 0.010
    }
    
    # Current metrics (slightly worse)
    current_metrics_stable = {
        "rmse": 44.0,  # 10% increase (below 15% threshold)
        "mae": 32.0,   # 6.7% increase
        "mape": 0.011  # 10% increase
    }
    
    degraded = monitor.detect_degradation(current_metrics_stable, baseline_metrics)
    print(f"Stable performance - Degradation detected: {degraded}")
    
    # Current metrics (significantly worse)
    current_metrics_degraded = {
        "rmse": 50.0,  # 25% increase (above 15% threshold)
        "mae": 38.0,   # 26.7% increase
        "mape": 0.013  # 30% increase
    }
    
    degraded = monitor.detect_degradation(current_metrics_degraded, baseline_metrics)
    print(f"Degraded performance - Degradation detected: {degraded}")
    
    print("\n4. Comparing Model Versions")
    print("-" * 80)
    
    # Model A metrics (baseline model)
    model_a_metrics = {
        "rmse": 45.0,
        "mae": 35.0,
        "mape": 0.012,
        "directional_accuracy": 0.85,
        "coverage_rate": 0.90
    }
    
    # Model B metrics (improved model)
    model_b_metrics = {
        "rmse": 40.0,  # Better
        "mae": 30.0,   # Better
        "mape": 0.010, # Better
        "directional_accuracy": 0.88,  # Better
        "coverage_rate": 0.93  # Better
    }
    
    winner = monitor.compare_models(model_a_metrics, model_b_metrics)
    print(f"Model A vs Model B: Winner = {winner}")
    
    # Compare with mixed performance
    model_c_metrics = {
        "rmse": 42.0,  # Better than A
        "mae": 36.0,   # Worse than A
        "mape": 0.011, # Better than A
        "directional_accuracy": 0.84,  # Worse than A
        "coverage_rate": 0.92  # Better than A
    }
    
    winner = monitor.compare_models(model_a_metrics, model_c_metrics)
    print(f"Model A vs Model C: Winner = {winner}")
    
    print("\n5. Triggering Retraining Alert")
    print("-" * 80)
    
    # Trigger alert when degradation is detected
    if monitor.detect_degradation(current_metrics_degraded, baseline_metrics):
        monitor.trigger_retraining_alert(
            reason="performance_degradation",
            metrics=current_metrics_degraded
        )
        print("✓ Retraining alert triggered (check logs for CRITICAL alert)")
    
    print("\n6. Real-World Workflow Example")
    print("-" * 80)
    
    # Simulate a complete monitoring workflow
    print("\nScenario: Weekly model performance check")
    print("-" * 40)
    
    # Get baseline metrics (first deployment)
    baseline = {
        "rmse": 42.0,
        "mae": 32.0,
        "mape": 0.011,
        "directional_accuracy": 0.86,
        "coverage_rate": 0.91
    }
    print(f"Baseline RMSE: {baseline['rmse']:.2f}")
    
    # Week 1: Performance is stable
    week1_metrics = {
        "rmse": 43.5,
        "mae": 33.0,
        "mape": 0.0115,
        "directional_accuracy": 0.85,
        "coverage_rate": 0.90
    }
    print(f"\nWeek 1 RMSE: {week1_metrics['rmse']:.2f}")
    if monitor.detect_degradation(week1_metrics, baseline):
        print("  → Degradation detected! Triggering retraining...")
    else:
        print("  → Performance stable ✓")
    
    # Week 2: Performance degrades
    week2_metrics = {
        "rmse": 52.0,  # 23.8% increase
        "mae": 40.0,
        "mape": 0.014,
        "directional_accuracy": 0.80,
        "coverage_rate": 0.85
    }
    print(f"\nWeek 2 RMSE: {week2_metrics['rmse']:.2f}")
    if monitor.detect_degradation(week2_metrics, baseline):
        print("  → Degradation detected! Triggering retraining...")
        monitor.trigger_retraining_alert(
            reason="weekly_check_degradation",
            metrics=week2_metrics
        )
    else:
        print("  → Performance stable ✓")
    
    # Week 3: New model deployed
    week3_metrics = {
        "rmse": 38.0,  # Improved!
        "mae": 28.0,
        "mape": 0.009,
        "directional_accuracy": 0.89,
        "coverage_rate": 0.94
    }
    print(f"\nWeek 3 RMSE (new model): {week3_metrics['rmse']:.2f}")
    winner = monitor.compare_models(baseline, week3_metrics)
    if winner == "model_b":
        print("  → New model performs better! Promoting to production ✓")
    else:
        print("  → New model does not improve performance. Keeping current model.")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
