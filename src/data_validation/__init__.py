"""
Data validation module for the Cocoa Price Prediction System.

This module provides data validation functionality including:
- Price data validation with realistic bounds checking
- Econometric data validation with range verification
- Outlier detection using statistical methods
- Market shock detection for significant price changes
"""

from src.data_validation.data_validator import DataValidator

__all__ = ["DataValidator"]
