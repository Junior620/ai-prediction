"""Configuration Manager for Cocoa Price Prediction System.

This module handles loading, validation, and access to system configuration.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Configuration validation error."""
    field: str
    value: Any
    error_type: str
    message: str
    severity: str


class ConfigManager:
    """Manages system configuration with validation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None
        self._validation_errors: List[ValidationError] = []
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return self._config
            
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse configuration file: {e}")
            raise
    
    def validate_config(self) -> tuple[bool, List[ValidationError]]:
        """Validate configuration parameters.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if self._config is None:
            self.load_config()
        
        self._validation_errors = []
        
        # Validate prediction settings
        self._validate_prediction_settings()
        
        # Validate model hyperparameters
        self._validate_prophet_config()
        self._validate_xgboost_config()
        self._validate_finbert_config()
        
        # Validate data collection settings
        self._validate_data_collection()
        
        # Validate preprocessing settings
        self._validate_preprocessing()
        
        # Validate monitoring settings
        self._validate_monitoring()
        
        # Validate retraining settings
        self._validate_retraining()
        
        # Validate API settings
        self._validate_api()
        
        is_valid = all(err.severity != "CRITICAL" for err in self._validation_errors)
        
        if not is_valid:
            logger.error(f"Configuration validation failed with {len(self._validation_errors)} errors")
            for err in self._validation_errors:
                logger.error(f"  {err.severity}: {err.field} - {err.message}")
        else:
            logger.info("Configuration validation passed")
        
        return is_valid, self._validation_errors
    
    def _validate_prediction_settings(self):
        """Validate prediction configuration."""
        pred = self._config.get("prediction", {})
        
        # Validate horizons
        horizons = pred.get("horizons", [])
        if not horizons:
            self._add_error("prediction.horizons", horizons, "missing", 
                          "Prediction horizons must be specified", "CRITICAL")
        elif not all(isinstance(h, int) and h > 0 for h in horizons):
            self._add_error("prediction.horizons", horizons, "invalid", 
                          "All horizons must be positive integers", "CRITICAL")
        
        # Validate confidence level
        conf_level = pred.get("confidence_level", 0.95)
        if not (0.8 <= conf_level <= 0.99):
            self._add_error("prediction.confidence_level", conf_level, "out_of_range",
                          "Confidence level must be between 0.80 and 0.99", "ERROR")
        
        # Validate sentiment weight
        sent_weight = pred.get("sentiment_weight", 0.1)
        if not (0 <= sent_weight <= 1):
            self._add_error("prediction.sentiment_weight", sent_weight, "out_of_range",
                          "Sentiment weight must be between 0 and 1", "ERROR")
        
        # Validate price bounds
        bounds = pred.get("price_bounds", {})
        min_price = bounds.get("min", 1000)
        max_price = bounds.get("max", 10000)
        if min_price >= max_price:
            self._add_error("prediction.price_bounds", bounds, "invalid",
                          "Min price must be less than max price", "CRITICAL")
    
    def _validate_prophet_config(self):
        """Validate Prophet model configuration."""
        prophet = self._config.get("prophet", {})
        
        # Validate seasonality mode
        seasonality_mode = prophet.get("seasonality_mode", "multiplicative")
        if seasonality_mode not in ["additive", "multiplicative"]:
            self._add_error("prophet.seasonality_mode", seasonality_mode, "invalid",
                          "Must be 'additive' or 'multiplicative'", "ERROR")
        
        # Validate changepoint_prior_scale
        cp_scale = prophet.get("changepoint_prior_scale", 0.05)
        if not (0.001 <= cp_scale <= 0.5):
            self._add_error("prophet.changepoint_prior_scale", cp_scale, "out_of_range",
                          "Must be between 0.001 and 0.5", "WARNING")
        
        # Validate interval_width
        interval_width = prophet.get("interval_width", 0.95)
        if not (0.8 <= interval_width <= 0.99):
            self._add_error("prophet.interval_width", interval_width, "out_of_range",
                          "Must be between 0.80 and 0.99", "WARNING")
    
    def _validate_xgboost_config(self):
        """Validate XGBoost model configuration."""
        xgb = self._config.get("xgboost", {})
        
        # Validate n_estimators
        n_est = xgb.get("n_estimators", 100)
        if not (10 <= n_est <= 1000):
            self._add_error("xgboost.n_estimators", n_est, "out_of_range",
                          "Must be between 10 and 1000", "WARNING")
        
        # Validate max_depth
        max_depth = xgb.get("max_depth", 6)
        if not (1 <= max_depth <= 20):
            self._add_error("xgboost.max_depth", max_depth, "out_of_range",
                          "Must be between 1 and 20", "WARNING")
        
        # Validate learning_rate
        lr = xgb.get("learning_rate", 0.1)
        if not (0.001 <= lr <= 1.0):
            self._add_error("xgboost.learning_rate", lr, "out_of_range",
                          "Must be between 0.001 and 1.0", "WARNING")
        
        # Validate subsample
        subsample = xgb.get("subsample", 0.8)
        if not (0.1 <= subsample <= 1.0):
            self._add_error("xgboost.subsample", subsample, "out_of_range",
                          "Must be between 0.1 and 1.0", "WARNING")
    
    def _validate_finbert_config(self):
        """Validate FinBERT NLP configuration."""
        finbert = self._config.get("finbert", {})
        
        # Validate sentiment_threshold
        threshold = finbert.get("sentiment_threshold", -0.6)
        if not (-1.0 <= threshold <= 1.0):
            self._add_error("finbert.sentiment_threshold", threshold, "out_of_range",
                          "Must be between -1.0 and 1.0", "WARNING")
        
        # Validate batch_size
        batch_size = finbert.get("batch_size", 32)
        if not (1 <= batch_size <= 128):
            self._add_error("finbert.batch_size", batch_size, "out_of_range",
                          "Must be between 1 and 128", "WARNING")
        
        # Validate device
        device = finbert.get("device", "auto")
        if device not in ["auto", "cpu", "cuda"]:
            self._add_error("finbert.device", device, "invalid",
                          "Must be 'auto', 'cpu', or 'cuda'", "WARNING")
    
    def _validate_data_collection(self):
        """Validate data collection configuration."""
        dc = self._config.get("data_collection", {})
        
        # Validate frequency
        freq = dc.get("frequency", "daily")
        if freq not in ["hourly", "daily", "weekly"]:
            self._add_error("data_collection.frequency", freq, "invalid",
                          "Must be 'hourly', 'daily', or 'weekly'", "ERROR")
        
        # Validate retry_attempts
        retries = dc.get("retry_attempts", 3)
        if not (1 <= retries <= 10):
            self._add_error("data_collection.retry_attempts", retries, "out_of_range",
                          "Must be between 1 and 10", "WARNING")
    
    def _validate_preprocessing(self):
        """Validate preprocessing configuration."""
        prep = self._config.get("preprocessing", {})
        
        # Validate train_val_split
        split = prep.get("train_val_split", {})
        val_size = split.get("validation_size", 0.2)
        if not (0.1 <= val_size <= 0.4):
            self._add_error("preprocessing.train_val_split.validation_size", val_size, 
                          "out_of_range", "Must be between 0.1 and 0.4", "WARNING")
        
        # Validate outlier detection
        outlier = prep.get("outlier_detection", {})
        method = outlier.get("method", "zscore")
        if method not in ["zscore", "iqr"]:
            self._add_error("preprocessing.outlier_detection.method", method, "invalid",
                          "Must be 'zscore' or 'iqr'", "WARNING")
    
    def _validate_monitoring(self):
        """Validate monitoring configuration."""
        mon = self._config.get("monitoring", {})
        
        # Validate degradation_threshold
        threshold = mon.get("degradation_threshold", 0.15)
        if not (0.05 <= threshold <= 0.5):
            self._add_error("monitoring.degradation_threshold", threshold, "out_of_range",
                          "Must be between 0.05 and 0.5", "WARNING")
    
    def _validate_retraining(self):
        """Validate retraining configuration."""
        retrain = self._config.get("retraining", {})
        
        # Validate frequency
        freq = retrain.get("frequency", "monthly")
        if freq not in ["daily", "weekly", "monthly"]:
            self._add_error("retraining.frequency", freq, "invalid",
                          "Must be 'daily', 'weekly', or 'monthly'", "WARNING")
        
        # Validate min_new_data_days
        min_days = retrain.get("min_new_data_days", 30)
        if not (7 <= min_days <= 365):
            self._add_error("retraining.min_new_data_days", min_days, "out_of_range",
                          "Must be between 7 and 365", "WARNING")
    
    def _validate_api(self):
        """Validate API configuration."""
        api = self._config.get("api", {})
        
        # Validate port
        port = api.get("port", 8000)
        if not (1024 <= port <= 65535):
            self._add_error("api.port", port, "out_of_range",
                          "Must be between 1024 and 65535", "ERROR")
        
        # Validate response_timeout
        timeout = api.get("response_timeout", 2)
        if not (0.5 <= timeout <= 30):
            self._add_error("api.response_timeout", timeout, "out_of_range",
                          "Must be between 0.5 and 30 seconds", "WARNING")
    
    def _add_error(self, field: str, value: Any, error_type: str, 
                   message: str, severity: str):
        """Add a validation error."""
        error = ValidationError(
            field=field,
            value=value,
            error_type=error_type,
            message=message,
            severity=severity
        )
        self._validation_errors.append(error)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., "prophet.seasonality_mode")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if self._config is None:
            self.load_config()
        
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.
        
        Args:
            section: Section name (e.g., "prophet", "xgboost")
            
        Returns:
            Configuration section dictionary
        """
        if self._config is None:
            self.load_config()
        
        return self._config.get(section, {})
    
    def reload(self):
        """Reload configuration from file."""
        self._config = None
        self._validation_errors = []
        self.load_config()
        self.validate_config()


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    global _config_manager
    _config_manager = ConfigManager(config_path)
    config = _config_manager.load_config()
    
    # Validate configuration
    is_valid, errors = _config_manager.validate_config()
    
    if not is_valid:
        critical_errors = [e for e in errors if e.severity == "CRITICAL"]
        if critical_errors:
            raise ValueError(f"Configuration has {len(critical_errors)} critical errors")
    
    return config


def get_config(key: str = None, default: Any = None) -> Any:
    """Get configuration value.
    
    Args:
        key: Configuration key (dot notation) or None for entire config
        default: Default value if key not found
        
    Returns:
        Configuration value or entire config
    """
    global _config_manager
    
    if _config_manager is None:
        load_config()
    
    if key is None:
        return _config_manager._config
    
    return _config_manager.get(key, default)


def validate_config() -> tuple[bool, List[ValidationError]]:
    """Validate current configuration.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    global _config_manager
    
    if _config_manager is None:
        load_config()
    
    return _config_manager.validate_config()
