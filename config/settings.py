"""
Configuration settings loader for the Cocoa Price Prediction System.
Loads configuration from config.yaml and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and config file."""
    
    # Database Configuration (Supabase)
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_key: str = Field(..., env="SUPABASE_KEY")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: str = Field(default="", env="REDIS_PASSWORD")
    redis_db: int = Field(default=0)
    
    # MLflow Configuration
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        env="MLFLOW_TRACKING_URI"
    )
    mlflow_registry_uri: str = Field(
        default="http://localhost:5000",
        env="MLFLOW_REGISTRY_URI"
    )
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    tradingview_webhook_secret: Optional[str] = Field(
        default=None, env="TRADINGVIEW_WEBHOOK_SECRET"
    )
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=60)
    # Comma-separated browser origins allowed for CORS (WS / local debug)
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        env="CORS_ORIGINS",
    )
    
    # Alert System Configuration
    alert_email_enabled: bool = Field(default=False, env="ALERT_EMAIL_ENABLED")
    alert_email_to: Optional[str] = Field(default=None, env="ALERT_EMAIL_TO")
    alert_email_from: Optional[str] = Field(default=None, env="ALERT_EMAIL_FROM")
    alert_webhook_url: Optional[str] = Field(default=None, env="ALERT_WEBHOOK_URL")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests_per_minute: int = Field(default=60)
    
    # API Keys for Data Sources
    ice_london_api_url: Optional[str] = Field(default=None, env="ICE_LONDON_API_URL")
    ice_london_api_key: Optional[str] = Field(default=None, env="ICE_LONDON_API_KEY")
    ice_ny_api_url: Optional[str] = Field(default=None, env="ICE_NY_API_URL")
    ice_ny_api_key: Optional[str] = Field(default=None, env="ICE_NY_API_KEY")
    weather_api_url: Optional[str] = Field(default=None, env="WEATHER_API_URL")
    weather_api_key: Optional[str] = Field(default=None, env="WEATHER_API_KEY")
    icco_api_url: Optional[str] = Field(default=None, env="ICCO_API_URL")
    icco_api_key: Optional[str] = Field(default=None, env="ICCO_API_KEY")
    fx_rates_api_url: Optional[str] = Field(default=None, env="FX_RATES_API_URL")
    fx_rates_api_key: Optional[str] = Field(default=None, env="FX_RATES_API_KEY")
    reuters_api_url: Optional[str] = Field(default=None, env="REUTERS_API_URL")
    reuters_api_key: Optional[str] = Field(default=None, env="REUTERS_API_KEY")
    bloomberg_api_url: Optional[str] = Field(default=None, env="BLOOMBERG_API_URL")
    bloomberg_api_key: Optional[str] = Field(default=None, env="BLOOMBERG_API_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = 'ignore'  # Ignore extra fields from .env


class ConfigLoader:
    """Loads and manages configuration from YAML file."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to config.yaml file. Defaults to config/config.yaml
        """
        if config_path is None:
            config_path = PROJECT_ROOT / "config" / "config.yaml"
        
        self.config_path = config_path
        self._config: Optional[Dict[str, Any]] = None
        self._settings: Optional[Settings] = None
    
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file with environment variable substitution.
        
        Returns:
            Dictionary containing all configuration values
        """
        if self._config is not None:
            return self._config
        
        with open(self.config_path, 'r') as f:
            config_text = f.read()
        
        # Substitute environment variables
        config_text = self._substitute_env_vars(config_text)
        
        self._config = yaml.safe_load(config_text)
        return self._config
    
    def _substitute_env_vars(self, text: str) -> str:
        """
        Substitute environment variables in format ${VAR_NAME} or ${VAR_NAME:default}.
        
        Args:
            text: Text containing environment variable placeholders
            
        Returns:
            Text with environment variables substituted
        """
        import re
        
        def replace_env_var(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
                return os.getenv(var_name, default_value)
            else:
                return os.getenv(var_expr, match.group(0))
        
        return re.sub(r'\$\{([^}]+)\}', replace_env_var, text)
    
    @property
    def settings(self) -> Settings:
        """Get Pydantic settings instance."""
        if self._settings is None:
            self._settings = Settings()
        return self._settings
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.
        
        Args:
            key_path: Dot-separated path to config value (e.g., "database.supabase_url")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get("prophet.seasonality_mode")
            "multiplicative"
        """
        if self._config is None:
            self.load()
        
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if self._config is None:
            self.load()
        
        # Validate required settings
        try:
            self.settings
        except Exception as e:
            errors.append(f"Settings validation failed: {str(e)}")
        
        # Validate prediction horizons
        horizons = self.get("prediction.horizons", [])
        if not horizons or not all(isinstance(h, int) and h > 0 for h in horizons):
            errors.append("prediction.horizons must be a list of positive integers")
        
        # Validate confidence level
        confidence = self.get("prediction.confidence_level", 0)
        if not (0 < confidence < 1):
            errors.append("prediction.confidence_level must be between 0 and 1")
        
        # Validate model hyperparameters
        prophet_changepoint = self.get("prophet.changepoint_prior_scale", 0)
        if not (0 < prophet_changepoint <= 1):
            errors.append("prophet.changepoint_prior_scale must be between 0 and 1")
        
        xgb_learning_rate = self.get("xgboost.learning_rate", 0)
        if not (0 < xgb_learning_rate <= 1):
            errors.append("xgboost.learning_rate must be between 0 and 1")
        
        # Validate cache TTLs
        prediction_ttl = self.get("cache.prediction_ttl", 0)
        if prediction_ttl <= 0:
            errors.append("cache.prediction_ttl must be positive")
        
        return errors


# Global configuration instance
config_loader = ConfigLoader()

# Convenience function to get settings
def get_settings() -> Settings:
    """Get application settings instance."""
    return config_loader.settings


# Convenience function to get config value
def get_config(key_path: str, default: Any = None) -> Any:
    """Get configuration value by key path."""
    return config_loader.get(key_path, default)
