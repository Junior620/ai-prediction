"""Data models and ML models for the Cocoa Price Prediction System."""

from .data_models import (
    PriceData,
    EconometricData,
    NewsArticle,
    Prediction,
    ModelMetrics,
    ValidationError,
)

# Lazy imports to avoid mlflow dependency issues in testing
def __getattr__(name):
    if name == "TimeSeriesModel":
        from .time_series_model import TimeSeriesModel
        return TimeSeriesModel
    elif name == "MLModel":
        from .ml_model import MLModel
        return MLModel
    elif name == "PricePredictor":
        from .price_predictor import PricePredictor
        return PricePredictor
    elif name == "ModelManager":
        from .model_manager import ModelManager
        return ModelManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PriceData",
    "EconometricData",
    "NewsArticle",
    "Prediction",
    "ModelMetrics",
    "ValidationError",
    "TimeSeriesModel",
    "MLModel",
    "PricePredictor",
    "ModelManager",
]
