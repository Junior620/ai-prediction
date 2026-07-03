"""
Performance Monitor for the Cocoa Price Prediction System.

This module provides comprehensive performance monitoring capabilities including:
- Computation of prediction accuracy metrics (RMSE, MAE, MAPE, directional accuracy, coverage rate)
- Tracking and persistence of performance metrics to database
- Detection of model performance degradation
- Triggering of retraining alerts
- Comparison of different model versions

Requirements addressed: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from supabase import Client, create_client

from src.models.data_models import ModelMetrics
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitors model performance and triggers retraining when degradation is detected.
    
    This class provides methods to:
    - Compute comprehensive performance metrics
    - Track metrics over time in the database
    - Detect performance degradation
    - Trigger alerts for retraining
    - Compare different model versions
    
    Attributes:
        supabase_client: Supabase client for database operations
        degradation_threshold: Threshold for detecting performance degradation (default: 0.15 = 15%)
    """
    
    def __init__(
        self,
        supabase_client: Optional[Client] = None,
        degradation_threshold: float = 0.15
    ):
        """
        Initialize the PerformanceMonitor.
        
        Args:
            supabase_client: Optional Supabase client. If None, creates a new client.
            degradation_threshold: Threshold for performance degradation detection (default: 0.15)
        
        Raises:
            ValueError: If degradation_threshold is not between 0 and 1
        """
        if not 0 < degradation_threshold < 1:
            raise ValueError(
                f"degradation_threshold must be between 0 and 1, got {degradation_threshold}"
            )
        
        self.degradation_threshold = degradation_threshold
        
        # Initialize Supabase client
        if supabase_client is None:
            settings = get_settings()
            self.supabase_client = create_client(
                settings.supabase_url,
                settings.supabase_key
            )
        else:
            self.supabase_client = supabase_client
        
        logger.info(
            f"PerformanceMonitor initialized with degradation_threshold={degradation_threshold}"
        )
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_lower: np.ndarray,
        y_pred_upper: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute comprehensive performance metrics for predictions.
        
        Computes the following metrics:
        - RMSE (Root Mean Squared Error): Measures average prediction error magnitude
        - MAE (Mean Absolute Error): Measures average absolute prediction error
        - MAPE (Mean Absolute Percentage Error): Measures percentage error
        - Directional Accuracy: Percentage of correct up/down predictions
        - Coverage Rate: Percentage of actual values within confidence intervals
        - Mean Interval Width: Average width of confidence intervals
        
        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values
            y_pred_lower: Array of lower confidence interval bounds
            y_pred_upper: Array of upper confidence interval bounds
        
        Returns:
            Dictionary containing all computed metrics
        
        Raises:
            ValueError: If input arrays have mismatched shapes or invalid values
        
        Requirements: 8.1
        """
        # Validate inputs
        if not (len(y_true) == len(y_pred) == len(y_pred_lower) == len(y_pred_upper)):
            raise ValueError(
                f"All input arrays must have the same length. Got: "
                f"y_true={len(y_true)}, y_pred={len(y_pred)}, "
                f"y_pred_lower={len(y_pred_lower)}, y_pred_upper={len(y_pred_upper)}"
            )
        
        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        # Convert to numpy arrays if not already
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        y_pred_lower = np.asarray(y_pred_lower, dtype=float)
        y_pred_upper = np.asarray(y_pred_upper, dtype=float)
        
        # Check for NaN or infinite values
        if np.any(~np.isfinite(y_true)) or np.any(~np.isfinite(y_pred)):
            raise ValueError("Input arrays contain NaN or infinite values")
        
        # Compute RMSE (Root Mean Squared Error)
        squared_errors = (y_true - y_pred) ** 2
        rmse = float(np.sqrt(np.mean(squared_errors)))
        
        # Compute MAE (Mean Absolute Error)
        absolute_errors = np.abs(y_true - y_pred)
        mae = float(np.mean(absolute_errors))
        
        # Compute MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero by filtering out zero values
        non_zero_mask = y_true != 0
        if np.sum(non_zero_mask) > 0:
            percentage_errors = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
            mape = float(np.mean(percentage_errors))
        else:
            mape = 0.0
            logger.warning("All y_true values are zero, MAPE set to 0.0")
        
        # Compute Directional Accuracy
        # Requires at least 2 observations to compute direction
        if len(y_true) >= 2:
            # Compute actual direction (up=1, down=0)
            actual_direction = np.diff(y_true) > 0
            # Compute predicted direction
            predicted_direction = np.diff(y_pred) > 0
            # Calculate accuracy
            directional_accuracy = float(np.mean(actual_direction == predicted_direction))
        else:
            directional_accuracy = 0.0
            logger.warning("Need at least 2 observations for directional accuracy, set to 0.0")
        
        # Compute Coverage Rate
        # Percentage of actual values within confidence intervals
        within_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
        coverage_rate = float(np.mean(within_interval))
        
        # Compute Mean Interval Width
        interval_widths = y_pred_upper - y_pred_lower
        mean_interval_width = float(np.mean(interval_widths))
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "directional_accuracy": directional_accuracy,
            "coverage_rate": coverage_rate,
            "mean_interval_width": mean_interval_width
        }
        
        logger.info(
            f"Computed metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}, "
            f"Directional Accuracy={directional_accuracy:.4f}, Coverage Rate={coverage_rate:.4f}"
        )
        
        return metrics
    
    def track_performance(
        self,
        metrics: Dict[str, float],
        timestamp: datetime,
        model_version: str
    ) -> None:
        """
        Store performance metrics in the database for tracking over time.
        
        Persists metrics to the model_metrics table in Supabase, enabling:
        - Historical performance tracking
        - Trend analysis
        - Model comparison
        - Degradation detection
        
        Args:
            metrics: Dictionary of computed metrics (from compute_metrics)
            timestamp: Timestamp when metrics were computed
            model_version: Version identifier of the model
        
        Raises:
            ValueError: If required metrics are missing
            Exception: If database insertion fails
        
        Requirements: 8.2
        """
        # Validate required metrics
        required_keys = {
            "rmse", "mae", "mape", "directional_accuracy",
            "coverage_rate", "mean_interval_width"
        }
        missing_keys = required_keys - set(metrics.keys())
        if missing_keys:
            raise ValueError(f"Missing required metrics: {missing_keys}")
        
        # Prepare data for insertion
        data = {
            "model_version": model_version,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "mape": metrics["mape"],
            "directional_accuracy": metrics["directional_accuracy"],
            "coverage_rate": metrics["coverage_rate"],
            "mean_interval_width": metrics["mean_interval_width"],
            "created_at": timestamp.isoformat()
        }
        
        try:
            # Insert metrics into database
            response = self.supabase_client.table("model_metrics").insert(data).execute()
            
            logger.info(
                f"Tracked performance metrics for model {model_version} at {timestamp}"
            )
            
        except Exception as e:
            logger.error(
                f"Failed to track performance metrics for model {model_version}: {str(e)}"
            )
            raise
    
    def detect_degradation(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        threshold: Optional[float] = None
    ) -> bool:
        """
        Detect if model performance has degraded beyond acceptable threshold.
        
        Compares current metrics against baseline metrics to detect degradation.
        Degradation is detected if any of the error metrics (RMSE, MAE, MAPE)
        have increased by more than the threshold percentage.
        
        Args:
            current_metrics: Current model performance metrics
            baseline_metrics: Baseline metrics to compare against
            threshold: Optional custom threshold (uses instance threshold if None)
        
        Returns:
            True if degradation detected, False otherwise
        
        Requirements: 8.3
        """
        if threshold is None:
            threshold = self.degradation_threshold
        
        # Validate required metrics
        required_keys = {"rmse", "mae", "mape"}
        if not required_keys.issubset(current_metrics.keys()):
            raise ValueError(f"current_metrics missing required keys: {required_keys}")
        if not required_keys.issubset(baseline_metrics.keys()):
            raise ValueError(f"baseline_metrics missing required keys: {required_keys}")
        
        # Check each error metric for degradation
        degraded_metrics = []
        
        for metric_name in ["rmse", "mae", "mape"]:
            current_value = current_metrics[metric_name]
            baseline_value = baseline_metrics[metric_name]
            
            # Avoid division by zero
            if baseline_value == 0:
                logger.warning(
                    f"Baseline {metric_name} is zero, skipping degradation check"
                )
                continue
            
            # Calculate percentage increase
            percentage_increase = (current_value - baseline_value) / baseline_value
            
            # Check if degradation exceeds threshold
            if percentage_increase > threshold:
                degraded_metrics.append({
                    "metric": metric_name,
                    "baseline": baseline_value,
                    "current": current_value,
                    "increase": percentage_increase
                })
                logger.warning(
                    f"Degradation detected in {metric_name}: "
                    f"baseline={baseline_value:.4f}, current={current_value:.4f}, "
                    f"increase={percentage_increase:.2%} (threshold={threshold:.2%})"
                )
        
        if degraded_metrics:
            logger.error(
                f"Model performance degradation detected in {len(degraded_metrics)} metrics"
            )
            return True
        else:
            logger.info("No performance degradation detected")
            return False
    
    def trigger_retraining_alert(
        self,
        reason: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Send alert notification for model retraining.
        
        Logs a CRITICAL alert when model retraining is needed. In a production
        environment, this would also send notifications via email, Slack, or
        other alerting systems.
        
        Args:
            reason: Reason for triggering retraining (e.g., "performance_degradation")
            metrics: Current performance metrics
        
        Requirements: 8.4
        """
        # Format metrics with proper handling of missing values
        rmse_str = f"{metrics['rmse']:.4f}" if 'rmse' in metrics else 'N/A'
        mae_str = f"{metrics['mae']:.4f}" if 'mae' in metrics else 'N/A'
        mape_str = f"{metrics['mape']:.4f}" if 'mape' in metrics else 'N/A'
        dir_acc_str = f"{metrics['directional_accuracy']:.4f}" if 'directional_accuracy' in metrics else 'N/A'
        cov_rate_str = f"{metrics['coverage_rate']:.4f}" if 'coverage_rate' in metrics else 'N/A'
        
        alert_message = (
            f"MODEL RETRAINING ALERT\n"
            f"Reason: {reason}\n"
            f"Current Metrics:\n"
            f"  - RMSE: {rmse_str}\n"
            f"  - MAE: {mae_str}\n"
            f"  - MAPE: {mape_str}\n"
            f"  - Directional Accuracy: {dir_acc_str}\n"
            f"  - Coverage Rate: {cov_rate_str}\n"
            f"Timestamp: {datetime.utcnow().isoformat()}"
        )
        
        # Log critical alert
        logger.critical(alert_message)
        
        # TODO: In production, integrate with alerting systems:
        # - Send email notification
        # - Post to Slack channel
        # - Create incident in PagerDuty
        # - Update monitoring dashboard
        
        logger.info(f"Retraining alert triggered: {reason}")
    
    def compare_models(
        self,
        model_a_metrics: Dict[str, float],
        model_b_metrics: Dict[str, float]
    ) -> str:
        """
        Compare two model versions and determine which performs better.
        
        Compares models based on multiple metrics with weighted scoring:
        - Error metrics (RMSE, MAE, MAPE): Lower is better (weight: 0.4 each)
        - Directional accuracy: Higher is better (weight: 0.3)
        - Coverage rate: Higher is better (weight: 0.2)
        
        Args:
            model_a_metrics: Performance metrics for model A
            model_b_metrics: Performance metrics for model B
        
        Returns:
            "model_a" if model A is better, "model_b" if model B is better,
            "tie" if performance is equal
        
        Requirements: 8.5
        """
        # Validate required metrics
        required_keys = {"rmse", "mae", "mape", "directional_accuracy", "coverage_rate"}
        if not required_keys.issubset(model_a_metrics.keys()):
            raise ValueError(f"model_a_metrics missing required keys: {required_keys}")
        if not required_keys.issubset(model_b_metrics.keys()):
            raise ValueError(f"model_b_metrics missing required keys: {required_keys}")
        
        # Initialize scores
        score_a = 0.0
        score_b = 0.0
        
        # Compare error metrics (lower is better)
        error_metrics = ["rmse", "mae", "mape"]
        error_weight = 0.4 / len(error_metrics)  # Total weight: 0.4
        
        for metric in error_metrics:
            value_a = model_a_metrics[metric]
            value_b = model_b_metrics[metric]
            
            if value_a < value_b:
                score_a += error_weight
            elif value_b < value_a:
                score_b += error_weight
            # If equal, no points awarded
        
        # Compare directional accuracy (higher is better)
        dir_acc_a = model_a_metrics["directional_accuracy"]
        dir_acc_b = model_b_metrics["directional_accuracy"]
        
        if dir_acc_a > dir_acc_b:
            score_a += 0.3
        elif dir_acc_b > dir_acc_a:
            score_b += 0.3
        
        # Compare coverage rate (higher is better)
        cov_rate_a = model_a_metrics["coverage_rate"]
        cov_rate_b = model_b_metrics["coverage_rate"]
        
        if cov_rate_a > cov_rate_b:
            score_a += 0.2
        elif cov_rate_b > cov_rate_a:
            score_b += 0.2
        
        # Determine winner
        if score_a > score_b:
            result = "model_a"
            logger.info(
                f"Model comparison: model_a wins (score_a={score_a:.2f}, score_b={score_b:.2f})"
            )
        elif score_b > score_a:
            result = "model_b"
            logger.info(
                f"Model comparison: model_b wins (score_a={score_a:.2f}, score_b={score_b:.2f})"
            )
        else:
            result = "tie"
            logger.info(
                f"Model comparison: tie (score_a={score_a:.2f}, score_b={score_b:.2f})"
            )
        
        return result
    
    def get_recent_metrics(
        self,
        model_version: str,
        limit: int = 10
    ) -> List[ModelMetrics]:
        """
        Retrieve recent performance metrics for a specific model version.
        
        Args:
            model_version: Version identifier of the model
            limit: Maximum number of recent metrics to retrieve
        
        Returns:
            List of ModelMetrics objects, ordered by timestamp (most recent first)
        
        Raises:
            Exception: If database query fails
        """
        try:
            response = (
                self.supabase_client
                .table("model_metrics")
                .select("*")
                .eq("model_version", model_version)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            
            metrics_list = []
            for row in response.data:
                metrics = ModelMetrics(
                    rmse=float(row["rmse"]),
                    mae=float(row["mae"]),
                    mape=float(row["mape"]),
                    directional_accuracy=float(row["directional_accuracy"]),
                    coverage_rate=float(row["coverage_rate"]),
                    mean_interval_width=float(row["mean_interval_width"]),
                    timestamp=datetime.fromisoformat(row["created_at"]),
                    model_version=row["model_version"]
                )
                metrics_list.append(metrics)
            
            logger.info(
                f"Retrieved {len(metrics_list)} recent metrics for model {model_version}"
            )
            
            return metrics_list
            
        except Exception as e:
            logger.error(
                f"Failed to retrieve metrics for model {model_version}: {str(e)}"
            )
            raise
    
    def get_baseline_metrics(
        self,
        model_version: str
    ) -> Optional[Dict[str, float]]:
        """
        Get baseline metrics for a model version (first recorded metrics).
        
        Args:
            model_version: Version identifier of the model
        
        Returns:
            Dictionary of baseline metrics, or None if no metrics found
        """
        try:
            response = (
                self.supabase_client
                .table("model_metrics")
                .select("*")
                .eq("model_version", model_version)
                .order("created_at", desc=False)
                .limit(1)
                .execute()
            )
            
            if not response.data:
                logger.warning(f"No baseline metrics found for model {model_version}")
                return None
            
            row = response.data[0]
            baseline = {
                "rmse": float(row["rmse"]),
                "mae": float(row["mae"]),
                "mape": float(row["mape"]),
                "directional_accuracy": float(row["directional_accuracy"]),
                "coverage_rate": float(row["coverage_rate"]),
                "mean_interval_width": float(row["mean_interval_width"])
            }
            
            logger.info(f"Retrieved baseline metrics for model {model_version}")
            return baseline
            
        except Exception as e:
            logger.error(
                f"Failed to retrieve baseline metrics for model {model_version}: {str(e)}"
            )
            raise
