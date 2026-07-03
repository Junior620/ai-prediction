"""
Example usage of the RetrainingManager for automatic model retraining.

This script demonstrates:
1. Checking if retraining should be triggered
2. Executing manual retraining
3. Getting retraining status
4. Scheduling automatic retraining
"""

import logging
from datetime import datetime
from supabase import create_client

from src.models.retraining_manager import RetrainingManager
from src.models.model_manager import ModelManager
from src.monitoring.performance_monitor import PerformanceMonitor
from src.data_preprocessing.data_preprocessor import DataPreprocessor
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_check_retraining_trigger():
    """Example: Check if retraining should be triggered."""
    logger.info("=== Example 1: Check Retraining Trigger ===")
    
    # Initialize dependencies
    settings = get_settings()
    
    model_manager = ModelManager(
        tracking_uri=settings.mlflow_tracking_uri,
        registry_uri=settings.mlflow_registry_uri
    )
    
    performance_monitor = PerformanceMonitor()
    data_preprocessor = DataPreprocessor()
    
    # Initialize retraining manager
    retraining_manager = RetrainingManager(
        model_manager=model_manager,
        performance_monitor=performance_monitor,
        data_preprocessor=data_preprocessor,
        retraining_frequency_days=30,
        min_new_data_days=30,
        max_model_versions=5
    )
    
    # Check if retraining should be triggered
    should_retrain, reason = retraining_manager.should_trigger_retraining(
        model_name="cocoa_price_predictor"
    )
    
    logger.info(f"Should retrain: {should_retrain}")
    logger.info(f"Reason: {reason}")
    
    if should_retrain:
        logger.info("Retraining is recommended!")
    else:
        logger.info("No retraining needed at this time.")
    
    return should_retrain, reason


def example_manual_retraining():
    """Example: Execute manual model retraining."""
    logger.info("=== Example 2: Manual Retraining ===")
    
    # Initialize dependencies
    settings = get_settings()
    
    model_manager = ModelManager(
        tracking_uri=settings.mlflow_tracking_uri,
        registry_uri=settings.mlflow_registry_uri
    )
    
    performance_monitor = PerformanceMonitor()
    data_preprocessor = DataPreprocessor()
    
    # Initialize retraining manager
    retraining_manager = RetrainingManager(
        model_manager=model_manager,
        performance_monitor=performance_monitor,
        data_preprocessor=data_preprocessor
    )
    
    # Execute retraining with custom hyperparameters
    hyperparameters = {
        "prophet": {
            "seasonality_mode": "multiplicative",
            "yearly_seasonality": True,
            "weekly_seasonality": False,
            "changepoint_prior_scale": 0.05
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "reg:squarederror"
        }
    }
    
    logger.info("Starting model retraining...")
    success, message, new_version = retraining_manager.retrain_models(
        model_name="cocoa_price_predictor",
        validation_split=0.2,
        hyperparameters=hyperparameters
    )
    
    if success:
        logger.info(f"Retraining successful! New version: {new_version}")
        logger.info(f"Message: {message}")
    else:
        logger.error(f"Retraining failed: {message}")
    
    return success, message, new_version


def example_get_retraining_status():
    """Example: Get current retraining status."""
    logger.info("=== Example 3: Get Retraining Status ===")
    
    # Initialize dependencies
    settings = get_settings()
    
    model_manager = ModelManager(
        tracking_uri=settings.mlflow_tracking_uri,
        registry_uri=settings.mlflow_registry_uri
    )
    
    performance_monitor = PerformanceMonitor()
    data_preprocessor = DataPreprocessor()
    
    # Initialize retraining manager
    retraining_manager = RetrainingManager(
        model_manager=model_manager,
        performance_monitor=performance_monitor,
        data_preprocessor=data_preprocessor
    )
    
    # Get retraining status
    status = retraining_manager.get_retraining_status(
        model_name="cocoa_price_predictor"
    )
    
    logger.info("Retraining Status:")
    for key, value in status.items():
        logger.info(f"  {key}: {value}")
    
    return status


def example_automatic_retraining_scheduler():
    """Example: Automatic retraining scheduler (runs periodically)."""
    logger.info("=== Example 4: Automatic Retraining Scheduler ===")
    
    # Initialize dependencies
    settings = get_settings()
    
    model_manager = ModelManager(
        tracking_uri=settings.mlflow_tracking_uri,
        registry_uri=settings.mlflow_registry_uri
    )
    
    performance_monitor = PerformanceMonitor()
    data_preprocessor = DataPreprocessor()
    
    # Initialize retraining manager
    retraining_manager = RetrainingManager(
        model_manager=model_manager,
        performance_monitor=performance_monitor,
        data_preprocessor=data_preprocessor,
        retraining_frequency_days=30,
        min_new_data_days=30,
        max_model_versions=5
    )
    
    logger.info("Checking if automatic retraining should be triggered...")
    
    # Check trigger conditions
    should_retrain, reason = retraining_manager.should_trigger_retraining(
        model_name="cocoa_price_predictor"
    )
    
    if should_retrain:
        logger.info(f"Automatic retraining triggered: {reason}")
        
        # Execute retraining
        success, message, new_version = retraining_manager.retrain_models(
            model_name="cocoa_price_predictor",
            validation_split=0.2
        )
        
        if success:
            logger.info(f"Automatic retraining completed successfully!")
            logger.info(f"New model version: {new_version}")
            logger.info(f"Status: {message}")
        else:
            logger.error(f"Automatic retraining failed: {message}")
            # In production, send alert to administrators
    else:
        logger.info(f"No automatic retraining needed: {reason}")
    
    return should_retrain


def example_retraining_with_performance_degradation():
    """Example: Trigger retraining when performance degradation is detected."""
    logger.info("=== Example 5: Retraining on Performance Degradation ===")
    
    # Initialize dependencies
    settings = get_settings()
    
    model_manager = ModelManager(
        tracking_uri=settings.mlflow_tracking_uri,
        registry_uri=settings.mlflow_registry_uri
    )
    
    performance_monitor = PerformanceMonitor(
        degradation_threshold=0.15  # 15% degradation threshold
    )
    
    data_preprocessor = DataPreprocessor()
    
    # Initialize retraining manager
    retraining_manager = RetrainingManager(
        model_manager=model_manager,
        performance_monitor=performance_monitor,
        data_preprocessor=data_preprocessor
    )
    
    # Simulate current and baseline metrics
    current_metrics = {
        'rmse': 60.0,  # Degraded from baseline
        'mae': 50.0,
        'mape': 0.025,
        'directional_accuracy': 0.70,
        'coverage_rate': 0.93,
        'mean_interval_width': 220.0
    }
    
    baseline_metrics = {
        'rmse': 50.0,  # Original baseline
        'mae': 40.0,
        'mape': 0.02,
        'directional_accuracy': 0.75,
        'coverage_rate': 0.95,
        'mean_interval_width': 200.0
    }
    
    # Check for degradation
    degradation_detected = performance_monitor.detect_degradation(
        current_metrics=current_metrics,
        baseline_metrics=baseline_metrics
    )
    
    if degradation_detected:
        logger.warning("Performance degradation detected!")
        
        # Trigger retraining alert
        performance_monitor.trigger_retraining_alert(
            reason="performance_degradation",
            metrics=current_metrics
        )
        
        # Execute retraining
        logger.info("Triggering retraining due to performance degradation...")
        success, message, new_version = retraining_manager.retrain_models(
            model_name="cocoa_price_predictor"
        )
        
        if success:
            logger.info(f"Retraining completed! New version: {new_version}")
        else:
            logger.error(f"Retraining failed: {message}")
    else:
        logger.info("No performance degradation detected.")
    
    return degradation_detected


def main():
    """Run all examples."""
    logger.info("Starting RetrainingManager examples...")
    
    try:
        # Example 1: Check retraining trigger
        example_check_retraining_trigger()
        
        # Example 2: Get retraining status
        example_get_retraining_status()
        
        # Example 3: Automatic retraining scheduler
        # This would typically run as a cron job or scheduled task
        example_automatic_retraining_scheduler()
        
        # Example 4: Manual retraining (commented out to avoid actual retraining)
        # example_manual_retraining()
        
        # Example 5: Retraining on performance degradation
        # example_retraining_with_performance_degradation()
        
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
