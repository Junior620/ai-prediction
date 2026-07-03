"""
Data models for the Cocoa Price Prediction System.

This module defines Pydantic models for all core data structures used throughout
the system, including price data, econometric data, news articles, predictions,
model metrics, and validation errors.

All models include field validation to ensure data quality and realistic bounds.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, ConfigDict


class PriceData(BaseModel):
    """
    Historical price data point from commodity exchanges.
    
    Attributes:
        timestamp: UTC timestamp of the price observation
        market: Market identifier (e.g., "ICE_London", "ICE_NY")
        price: Price in USD per metric ton (valid range: 1000-10000)
        volume: Trading volume in metric tons
        currency: Currency code (e.g., "USD", "GBP")
    """
    
    model_config = ConfigDict(frozen=False)
    
    timestamp: datetime = Field(..., description="UTC timestamp of price observation")
    market: str = Field(..., description="Market identifier (ICE_London, ICE_NY)")
    price: float = Field(..., ge=1000.0, le=10000.0, description="Price in USD/MT")
    volume: float = Field(..., ge=0.0, description="Trading volume in metric tons")
    currency: str = Field(..., min_length=3, max_length=3, description="Currency code")
    
    @field_validator("market")
    @classmethod
    def validate_market(cls, v: str) -> str:
        """Validate market identifier."""
        valid_markets = ["ICE_London", "ICE_NY"]
        if v not in valid_markets:
            raise ValueError(f"Market must be one of {valid_markets}, got {v}")
        return v
    
    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code."""
        valid_currencies = ["USD", "GBP", "EUR"]
        if v.upper() not in valid_currencies:
            raise ValueError(f"Currency must be one of {valid_currencies}, got {v}")
        return v.upper()


class EconometricData(BaseModel):
    """
    Econometric data point including weather, stocks, production, and FX rates.
    
    Attributes:
        timestamp: UTC timestamp of the observation
        temperature: Temperature in Celsius (valid range: -10 to 50°C)
        rainfall: Rainfall in mm/day (valid range: 0 to 500mm)
        stock_level: Stock level in metric tons (must be non-negative)
        production: Production volume in metric tons (must be non-negative)
        fx_rate_xaf_usd: XAF to USD exchange rate (must be positive)
        fx_rate_gbp_usd: GBP to USD exchange rate (must be positive)
        fx_rate_eur_usd: EUR to USD exchange rate (must be positive)
    """
    
    model_config = ConfigDict(frozen=False)
    
    timestamp: datetime = Field(..., description="UTC timestamp of observation")
    temperature: Optional[float] = Field(
        None, ge=-10.0, le=50.0, description="Temperature in Celsius"
    )
    rainfall: Optional[float] = Field(
        None, ge=0.0, le=500.0, description="Rainfall in mm/day"
    )
    stock_level: Optional[float] = Field(
        None, ge=0.0, description="Stock level in metric tons"
    )
    production: Optional[float] = Field(
        None, ge=0.0, description="Production volume in metric tons"
    )
    fx_rate_xaf_usd: Optional[float] = Field(
        None, gt=0.0, description="XAF to USD exchange rate"
    )
    fx_rate_gbp_usd: Optional[float] = Field(
        None, gt=0.0, description="GBP to USD exchange rate"
    )
    fx_rate_eur_usd: Optional[float] = Field(
        None, gt=0.0, description="EUR to USD exchange rate"
    )


class NewsArticle(BaseModel):
    """
    News article with metadata and sentiment analysis.
    
    Attributes:
        id: Unique identifier for the article (int or str)
        source: News source (any string)
        title: Article title
        content: Full article content
        published_at: Publication timestamp
        url: Article URL
        keywords: List of extracted keywords
        sentiment_score: Sentiment score from -1 (negative) to +1 (positive)
        is_high_risk: Flag indicating potential market shock risk
    """
    
    model_config = ConfigDict(frozen=False)
    
    id: int | str = Field(..., description="Unique article identifier")
    source: str = Field(..., description="News source")
    title: str = Field(..., min_length=1, description="Article title")
    content: str = Field(default="", description="Full article content")
    published_at: datetime = Field(..., description="Publication timestamp")
    url: str = Field(..., description="Article URL")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    sentiment_score: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Sentiment score (-1 to +1)"
    )
    is_high_risk: Optional[bool] = Field(
        None, description="Flag for potential market shock"
    )
    
    @field_validator("id")
    @classmethod
    def validate_id(cls, v: int | str) -> str:
        """Convert ID to string for consistency."""
        return str(v)
    
    @field_validator("keywords", mode="before")
    @classmethod
    def validate_keywords(cls, v):
        """Convert None to empty list."""
        return v if v is not None else []


class Prediction(BaseModel):
    """
    Price prediction with confidence interval and model components.
    
    Attributes:
        horizon: Prediction horizon in days (e.g., 1, 7, 30)
        price: Predicted price in USD per metric ton
        confidence_interval: Tuple of (lower_bound, upper_bound)
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        timestamp: Timestamp when prediction was made
        model_version: Version identifier of the model
        components: Dictionary of prediction components (baseline, residual, sentiment)
    """
    
    model_config = ConfigDict(frozen=False)
    
    horizon: int = Field(..., ge=1, description="Prediction horizon in days")
    price: float = Field(..., ge=500.0, le=15000.0, description="Predicted price in USD/MT or USD/T")
    confidence_interval: Tuple[float, float] = Field(
        ..., description="Confidence interval (lower, upper)"
    )
    confidence_level: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence level (e.g., 0.95)"
    )
    timestamp: datetime = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version identifier")
    components: Dict[str, Any] = Field(
        default_factory=dict,
        description="Prediction components (baseline, residual, sentiment, ensemble_weights, garch, etc.)"
    )
    
    @field_validator("confidence_interval")
    @classmethod
    def validate_confidence_interval(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """Validate that confidence interval bounds are ordered correctly."""
        lower, upper = v
        if lower >= upper:
            raise ValueError(
                f"Lower bound must be less than upper bound, got ({lower}, {upper})"
            )
        if lower < 1000.0 or upper > 10000.0:
            raise ValueError(
                f"Confidence interval bounds must be within [1000, 10000], got ({lower}, {upper})"
            )
        return v
    
    @field_validator("components")
    @classmethod
    def validate_components(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that expected components are present."""
        expected_keys = {"baseline", "residual", "sentiment"}
        if v and not expected_keys.issubset(v.keys()):
            missing = expected_keys - v.keys()
            raise ValueError(f"Missing required components: {missing}")
        return v


class ModelMetrics(BaseModel):
    """
    Model performance metrics for evaluation and monitoring.
    
    Attributes:
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        mape: Mean Absolute Percentage Error
        directional_accuracy: Percentage of correct up/down predictions (0-1)
        coverage_rate: Percentage of actual values within confidence interval (0-1)
        mean_interval_width: Average width of confidence intervals
        timestamp: Timestamp when metrics were computed
        model_version: Version identifier of the model
    """
    
    model_config = ConfigDict(frozen=False)
    
    rmse: float = Field(..., ge=0.0, description="Root Mean Squared Error")
    mae: float = Field(..., ge=0.0, description="Mean Absolute Error")
    mape: float = Field(..., ge=0.0, description="Mean Absolute Percentage Error")
    directional_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Directional accuracy (0-1)"
    )
    coverage_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Coverage rate (0-1)"
    )
    mean_interval_width: float = Field(
        ..., ge=0.0, description="Mean confidence interval width"
    )
    timestamp: datetime = Field(..., description="Metrics computation timestamp")
    model_version: str = Field(..., description="Model version identifier")


class ValidationError(BaseModel):
    """
    Data validation error with severity classification.
    
    Attributes:
        field: Name of the field that failed validation
        value: The invalid value (as string representation)
        error_type: Type of validation error
        message: Human-readable error message
        severity: Error severity level
    """
    
    model_config = ConfigDict(frozen=False)
    
    field: str = Field(..., description="Field name that failed validation")
    value: Optional[str] = Field(None, description="Invalid value (string representation)")
    error_type: str = Field(..., description="Validation error type")
    message: str = Field(..., min_length=1, description="Error message")
    severity: str = Field(..., description="Error severity level")
    
    @field_validator("error_type")
    @classmethod
    def validate_error_type(cls, v: str) -> str:
        """Validate error type."""
        valid_types = ["out_of_range", "missing", "duplicate", "invalid_format", "constraint_violation"]
        if v not in valid_types:
            raise ValueError(f"Error type must be one of {valid_types}, got {v}")
        return v
    
    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity level."""
        valid_severities = ["INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_severities:
            raise ValueError(f"Severity must be one of {valid_severities}, got {v}")
        return v.upper()
