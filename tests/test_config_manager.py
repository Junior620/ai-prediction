"""Tests for Configuration Manager."""

import pytest
import tempfile
from pathlib import Path
import yaml
from src.config.config_manager import ConfigManager, load_config, get_config, validate_config


@pytest.fixture
def valid_config():
    """Create a valid configuration dictionary."""
    return {
        "system": {
            "name": "Test System",
            "version": "1.0.0"
        },
        "prediction": {
            "horizons": [1, 7, 30],
            "confidence_level": 0.95,
            "sentiment_weight": 0.1,
            "price_bounds": {
                "min": 1000,
                "max": 10000
            }
        },
        "prophet": {
            "seasonality_mode": "multiplicative",
            "changepoint_prior_scale": 0.05,
            "interval_width": 0.95
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8
        },
        "finbert": {
            "sentiment_threshold": -0.6,
            "batch_size": 32,
            "device": "auto"
        },
        "data_collection": {
            "frequency": "daily",
            "retry_attempts": 3
        },
        "preprocessing": {
            "train_val_split": {
                "validation_size": 0.2
            },
            "outlier_detection": {
                "method": "zscore"
            }
        },
        "monitoring": {
            "degradation_threshold": 0.15
        },
        "retraining": {
            "frequency": "monthly",
            "min_new_data_days": 30
        },
        "api": {
            "port": 8000,
            "response_timeout": 2
        }
    }


@pytest.fixture
def config_file(valid_config):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_config, f)
        return f.name


def test_load_config_success(config_file):
    """Test successful configuration loading."""
    manager = ConfigManager(config_file)
    config = manager.load_config()
    
    assert config is not None
    assert config["system"]["name"] == "Test System"
    assert config["prediction"]["horizons"] == [1, 7, 30]


def test_load_config_file_not_found():
    """Test loading non-existent config file."""
    manager = ConfigManager("nonexistent.yaml")
    
    with pytest.raises(FileNotFoundError):
        manager.load_config()


def test_validate_config_valid(config_file):
    """Test validation of valid configuration."""
    manager = ConfigManager(config_file)
    manager.load_config()
    
    is_valid, errors = manager.validate_config()
    
    assert is_valid is True
    # May have warnings but no critical errors
    critical_errors = [e for e in errors if e.severity == "CRITICAL"]
    assert len(critical_errors) == 0


def test_validate_config_invalid_horizons(config_file):
    """Test validation with invalid prediction horizons."""
    manager = ConfigManager(config_file)
    config = manager.load_config()
    
    # Set invalid horizons
    config["prediction"]["horizons"] = []
    
    is_valid, errors = manager.validate_config()
    
    assert is_valid is False
    assert any(e.field == "prediction.horizons" for e in errors)


def test_validate_config_invalid_confidence_level(config_file):
    """Test validation with invalid confidence level."""
    manager = ConfigManager(config_file)
    config = manager.load_config()
    
    # Set invalid confidence level
    config["prediction"]["confidence_level"] = 1.5
    
    is_valid, errors = manager.validate_config()
    
    # Should have error for confidence level
    assert any(e.field == "prediction.confidence_level" for e in errors)


def test_validate_config_invalid_price_bounds(config_file):
    """Test validation with invalid price bounds."""
    manager = ConfigManager(config_file)
    config = manager.load_config()
    
    # Set invalid price bounds (min >= max)
    config["prediction"]["price_bounds"]["min"] = 10000
    config["prediction"]["price_bounds"]["max"] = 1000
    
    is_valid, errors = manager.validate_config()
    
    assert is_valid is False
    assert any(e.field == "prediction.price_bounds" for e in errors)


def test_validate_prophet_config(config_file):
    """Test Prophet configuration validation."""
    manager = ConfigManager(config_file)
    config = manager.load_config()
    
    # Set invalid seasonality mode
    config["prophet"]["seasonality_mode"] = "invalid"
    
    is_valid, errors = manager.validate_config()
    
    assert any(e.field == "prophet.seasonality_mode" for e in errors)


def test_validate_xgboost_config(config_file):
    """Test XGBoost configuration validation."""
    manager = ConfigManager(config_file)
    config = manager.load_config()
    
    # Set invalid learning rate
    config["xgboost"]["learning_rate"] = 2.0
    
    is_valid, errors = manager.validate_config()
    
    assert any(e.field == "xgboost.learning_rate" for e in errors)


def test_validate_finbert_config(config_file):
    """Test FinBERT configuration validation."""
    manager = ConfigManager(config_file)
    config = manager.load_config()
    
    # Set invalid device
    config["finbert"]["device"] = "invalid"
    
    is_valid, errors = manager.validate_config()
    
    assert any(e.field == "finbert.device" for e in errors)


def test_get_config_value(config_file):
    """Test getting configuration value by key."""
    manager = ConfigManager(config_file)
    manager.load_config()
    
    # Test dot notation
    value = manager.get("prophet.seasonality_mode")
    assert value == "multiplicative"
    
    # Test nested key
    value = manager.get("prediction.price_bounds.min")
    assert value == 1000
    
    # Test default value
    value = manager.get("nonexistent.key", default="default")
    assert value == "default"


def test_get_section(config_file):
    """Test getting entire configuration section."""
    manager = ConfigManager(config_file)
    manager.load_config()
    
    prophet_config = manager.get_section("prophet")
    
    assert prophet_config is not None
    assert prophet_config["seasonality_mode"] == "multiplicative"
    assert prophet_config["changepoint_prior_scale"] == 0.05


def test_reload_config(config_file):
    """Test reloading configuration."""
    manager = ConfigManager(config_file)
    manager.load_config()
    
    # Modify config
    manager._config["system"]["name"] = "Modified"
    
    # Reload
    manager.reload()
    
    # Should be back to original
    assert manager._config["system"]["name"] == "Test System"


def test_global_load_config(config_file):
    """Test global load_config function."""
    config = load_config(config_file)
    
    assert config is not None
    assert config["system"]["name"] == "Test System"


def test_global_get_config(config_file):
    """Test global get_config function."""
    load_config(config_file)
    
    # Get specific value
    value = get_config("prophet.seasonality_mode")
    assert value == "multiplicative"
    
    # Get entire config
    config = get_config()
    assert config is not None
    assert "system" in config


def test_global_validate_config(config_file):
    """Test global validate_config function."""
    load_config(config_file)
    
    is_valid, errors = validate_config()
    
    assert is_valid is True
    critical_errors = [e for e in errors if e.severity == "CRITICAL"]
    assert len(critical_errors) == 0


def test_validate_data_collection_frequency(config_file):
    """Test data collection frequency validation."""
    manager = ConfigManager(config_file)
    config = manager.load_config()
    
    # Set invalid frequency
    config["data_collection"]["frequency"] = "invalid"
    
    is_valid, errors = manager.validate_config()
    
    assert any(e.field == "data_collection.frequency" for e in errors)


def test_validate_preprocessing_validation_size(config_file):
    """Test preprocessing validation size validation."""
    manager = ConfigManager(config_file)
    config = manager.load_config()
    
    # Set invalid validation size
    config["preprocessing"]["train_val_split"]["validation_size"] = 0.9
    
    is_valid, errors = manager.validate_config()
    
    assert any(e.field == "preprocessing.train_val_split.validation_size" for e in errors)


def test_validate_api_port(config_file):
    """Test API port validation."""
    manager = ConfigManager(config_file)
    config = manager.load_config()
    
    # Set invalid port
    config["api"]["port"] = 100
    
    is_valid, errors = manager.validate_config()
    
    assert any(e.field == "api.port" for e in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
