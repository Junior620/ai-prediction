"""
Main entry point for the FastAPI application.
This module imports the FastAPI app from app.py for uvicorn to run.
"""

from src.api.app import app

__all__ = ["app"]
