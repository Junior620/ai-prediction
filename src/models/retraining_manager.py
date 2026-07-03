"""
Retraining Manager for the Cocoa Price Prediction System.

This module implements automatic model retraining capabilities including:
- Monthly automatic retraining schedule
- Trigger-based retraining after 30 days of new data
- Validation of retrained models on recent validation set
- Retention logic to keep current model if new model performs worse
- Model version history management (maintains 5 most recent versions)

Requirements addressed: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from supabase import Client, create_client
from loguru import logger

from src.models.time_series_model import TimeSeriesModel
from src.models.ml_model import MLModel
from src.nlp.nlp_analyzer import NLPAnalyzer
from src.models.price_predictor import PricePredictor
from src.models.model_manager import ModelManager
from src.monitoring.performance_monitor import PerformanceMonitor
from src.data_preprocessing.data_preprocessor import DataPreprocessor
from src.models.data_models import ModelMetrics
from config.settings import get_settings


class RetrainingManager:
    """
    Manages automatic model retraining and validation.
    
    This class orchestrates the complete retraining workflow:
    1. Monitors data availability and triggers retraining when needed
    2. Collects and prepares training data
    3. Trains new model versions (Prophet, XGBoost, FinBERT)
    4. Validates new models against recent validation data
    5. Compares new model performance with current production model
    6. Promotes new model to production if it performs better
    7. Maintains history of recent model versions
    
    Attributes:
        model_manager: MLflow model manager for versioning
        performance_monitor: Performance monitoring and comparison
        data_preprocessor: Data preprocessing utilities
        supabase_client: Database client for data access
        retraining_frequency_days: Days between automatic retraining (default: 30)
        min_new_data_days: Minimum days of new data to trigger retraining (default: 30)
        max_model_versions: Maximum model versions to retain (default: 5)
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        performance_monitor: PerformanceMonitor,
        data_preprocessor: DataPreprocessor,
        supabase_client: Optional[Client] = None,
        retraining_frequency_days: int = 30,
        min_new_data_days: int = 30,
        max_model_versions: int = 5
    ):
        """
        Initialize RetrainingManager.
        
        Args:
            model_manager: ModelManager instance for model versioning
            performance_monitor: PerformanceMonitor for validation
            data_preprocessor: DataPreprocessor for data preparation
            supabase_client: Optional Supabase client (creates new if None)
            retraining_frequency_days: Days between automatic retraining
            min_new_data_days: Minimum days of new data to trigger retraining
            max_model_versions: Maximum model versions to retain
        
        Raises:
            ValueError: If parameters are invalid
        """
        if retraining_frequency_days <= 0:
            raise ValueError(
                f"retraining_frequency_days must be positive, got {retraining_frequency_days}"
            )
        if min_new_data_days <= 0:
            raise ValueError(
                f"min_new_data_days must be positive, got {min_new_data_days}"
            )
        if max_model_versions < 1:
            raise ValueError(
                f"max_model_versions must be at least 1, got {max_model_versions}"
            )
        
        self.model_manager = model_manager
        self.performance_monitor = performance_monitor
        self.data_preprocessor = data_preprocessor
        self.retraining_frequency_days = retraining_frequency_days
        self.min_new_data_days = min_new_data_days
        self.max_model_versions = max_model_versions
        
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
            f"RetrainingManager initialized with retraining_frequency={retraining_frequency_days}d, "
            f"min_new_data={min_new_data_days}d, max_versions={max_model_versions}"
        )
    
    def should_trigger_retraining(
        self,
        model_name: str = "cocoa_price_predictor"
    ) -> Tuple[bool, str]:
        """
        Determine if automatic retraining should be triggered.
        
        Checks two conditions:
        1. Time-based: Has it been at least retraining_frequency_days since last training?
        2. Data-based: Is there at least min_new_data_days of new data available?
        
        Args:
            model_name: Name of the model to check
        
        Returns:
            Tuple of (should_retrain: bool, reason: str)
        
        Requirements: 9.1, 9.2
        """
        logger.info(f"Checking if retraining should be triggered for {model_name}")
        
        try:
            # Get the most recent model version
            versions = self.model_manager.list_model_versions(
                model_name=model_name,
                max_results=1
            )
            
            if not versions:
                logger.info("No existing model found, retraining should be triggered")
                return True, "no_existing_model"
            
            latest_version = versions[0]
            last_training_time = datetime.fromtimestamp(
                latest_version.creation_timestamp / 1000  # MLflow uses milliseconds
            )
            
            # Check time-based condition
            days_since_training = (datetime.now() - last_training_time).days
            logger.info(f"Days since last training: {days_since_training}")
            
            if days_since_training >= self.retraining_frequency_days:
                logger.info(
                    f"Time-based retraining triggered: {days_since_training} days >= "
                    f"{self.retraining_frequency_days} days"
                )
                return True, f"time_based_{days_since_training}_days"
            
            # Check data-based condition
            new_data_days = self._count_new_data_days(last_training_time)
            logger.info(f"Days of new data available: {new_data_days}")
            
            if new_data_days >= self.min_new_data_days:
                logger.info(
                    f"Data-based retraining triggered: {new_data_days} days >= "
                    f"{self.min_new_data_days} days"
                )
                return True, f"data_based_{new_data_days}_days"
            
            logger.info("No retraining trigger conditions met")
            return False, "no_trigger"
        
        except Exception as e:
            logger.error(f"Error checking retraining trigger: {e}")
            # In case of error, don't trigger retraining
            return False, f"error_{str(e)}"
    
    def _count_new_data_days(self, since_date: datetime) -> int:
        """
        Count days of new price data available since a given date.
        
        Args:
            since_date: Date to count from
        
        Returns:
            Number of days with new data
        """
        try:
            # Query price_data table for data after since_date
            response = (
                self.supabase_client
                .table("price_data")
                .select("timestamp")
                .gte("timestamp", since_date.isoformat())
                .execute()
            )
            
            if not response.data:
                return 0
            
            # Count unique dates
            timestamps = [
                datetime.fromisoformat(row["timestamp"]).date()
                for row in response.data
            ]
            unique_dates = set(timestamps)
            
            return len(unique_dates)
        
        except Exception as e:
            logger.error(f"Error counting new data days: {e}")
            return 0
    
    def retrain_models(
        self,
        model_name: str = "cocoa_price_predictor",
        validation_split: float = 0.2,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Execute complete model retraining workflow.
        
        This method:
        1. Fetches latest training data from database
        2. Preprocesses and splits data (train/validation)
        3. Trains new Prophet model
        4. Trains new XGBoost model on residuals
        5. Creates new PricePredictor with trained models
        6. Validates new model on recent validation set
        7. Compares with current production model
        8. Promotes to production if better, otherwise retains current
        9. Logs all models to MLflow
        10. Cleans up old model versions
        
        Args:
            model_name: Base name for the model
            validation_split: Fraction of data for validation (default: 0.2)
            hyperparameters: Optional dict of hyperparameters to override defaults
        
        Returns:
            Tuple of (success: bool, message: str, new_version: Optional[str])
        
        Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
        """
        logger.info(f"Starting retraining workflow for {model_name}")
        
        try:
            # Step 1: Fetch training data
            logger.info("Step 1: Fetching training data")
            price_data, econometric_data = self._fetch_training_data()
            
            if price_data.empty:
                logger.error("No price data available for training")
                return False, "no_data_available", None
            
            logger.info(
                f"Fetched {len(price_data)} price records and "
                f"{len(econometric_data)} econometric records"
            )
            
            # Step 2: Preprocess and split data
            logger.info("Step 2: Preprocessing and splitting data")
            train_data, val_data = self._prepare_training_data(
                price_data,
                econometric_data,
                validation_split
            )
            
            logger.info(
                f"Training set: {len(train_data)} records, "
                f"Validation set: {len(val_data)} records"
            )
            
            # Step 3: Train Prophet model
            logger.info("Step 3: Training Prophet time series model")
            prophet_model = self._train_prophet_model(
                train_data,
                hyperparameters.get("prophet", {}) if hyperparameters else {}
            )
            
            # Step 4: Compute residuals and train XGBoost
            logger.info("Step 4: Computing residuals and training XGBoost model")
            xgboost_model = self._train_xgboost_model(
                prophet_model,
                train_data,
                econometric_data,
                hyperparameters.get("xgboost", {}) if hyperparameters else {}
            )
            
            # Step 5: Initialize NLP analyzer (no retraining needed for FinBERT)
            logger.info("Step 5: Initializing NLP analyzer")
            nlp_analyzer = NLPAnalyzer()
            
            # Step 6: Create new PricePredictor
            logger.info("Step 6: Creating new PricePredictor")
            new_version = self._generate_version_string()
            new_predictor = PricePredictor(
                ts_model=prophet_model,
                ml_model=xgboost_model,
                nlp_analyzer=nlp_analyzer,
                model_version=new_version
            )
            
            # Step 7: Validate new model
            logger.info("Step 7: Validating new model on validation set")
            new_metrics = self._validate_model(
                new_predictor,
                val_data,
                econometric_data
            )
            
            logger.info(
                f"New model metrics: RMSE={new_metrics['rmse']:.4f}, "
                f"MAE={new_metrics['mae']:.4f}, MAPE={new_metrics['mape']:.4f}"
            )
            
            # Step 8: Compare with current production model
            logger.info("Step 8: Comparing with current production model")
            should_promote = self._should_promote_model(
                model_name,
                new_metrics,
                val_data,
                econometric_data
            )
            
            # Step 9: Log models to MLflow
            logger.info("Step 9: Logging models to MLflow")
            self._log_models_to_mlflow(
                prophet_model,
                xgboost_model,
                new_version,
                new_metrics,
                hyperparameters or {}
            )
            
            # Step 10: Promote or retain
            if should_promote:
                logger.info("Step 10: Promoting new model to production")
                self.model_manager.promote_model(
                    model_name=f"{model_name}_prophet",
                    version=new_version,
                    from_stage="None",
                    to_stage="Production"
                )
                self.model_manager.promote_model(
                    model_name=f"{model_name}_xgboost",
                    version=new_version,
                    from_stage="None",
                    to_stage="Production"
                )
                
                message = f"New model version {new_version} promoted to production"
                logger.info(message)
            else:
                logger.info("Step 10: Retaining current production model")
                message = f"New model version {new_version} not promoted (current model performs better)"
                logger.info(message)
            
            # Step 11: Clean up old versions
            logger.info("Step 11: Cleaning up old model versions")
            self._cleanup_old_versions(model_name)
            
            logger.info("Retraining workflow completed successfully")
            return True, message, new_version
        
        except Exception as e:
            logger.error(f"Retraining workflow failed: {e}", exc_info=True)
            return False, f"retraining_failed: {str(e)}", None
    
    def _fetch_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch all available training data from database.
        
        Returns:
            Tuple of (price_data, econometric_data) DataFrames
        """
        try:
            # Fetch price data
            price_response = (
                self.supabase_client
                .table("price_data")
                .select("*")
                .order("timestamp", desc=False)
                .execute()
            )
            
            price_data = pd.DataFrame(price_response.data)
            
            if not price_data.empty:
                price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
                price_data = price_data.sort_values('timestamp')
            
            # Fetch econometric data
            econ_response = (
                self.supabase_client
                .table("econometric_data")
                .select("*")
                .order("timestamp", desc=False)
                .execute()
            )
            
            econometric_data = pd.DataFrame(econ_response.data)
            
            if not econometric_data.empty:
                econometric_data['timestamp'] = pd.to_datetime(econometric_data['timestamp'])
                econometric_data = econometric_data.sort_values('timestamp')
            
            return price_data, econometric_data
        
        except Exception as e:
            logger.error(f"Error fetching training data: {e}")
            raise
    
    def _prepare_training_data(
        self,
        price_data: pd.DataFrame,
        econometric_data: pd.DataFrame,
        validation_split: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess and split data into training and validation sets.
        
        Args:
            price_data: Raw price data
            econometric_data: Raw econometric data
            validation_split: Fraction for validation
        
        Returns:
            Tuple of (train_data, val_data)
        """
        # Merge price and econometric data
        merged_data = pd.merge(
            price_data,
            econometric_data,
            on='timestamp',
            how='left'
        )
        
        # Handle missing values
        merged_data = self.data_preprocessor.handle_missing_values(
            merged_data,
            strategy={
                'price': 'forward_fill',
                'temperature': 'interpolate',
                'rainfall': 'interpolate',
                'stock_level': 'forward_fill',
                'production': 'forward_fill',
                'fx_rate_xaf_usd': 'forward_fill',
                'fx_rate_gbp_usd': 'forward_fill',
                'fx_rate_eur_usd': 'forward_fill'
            }
        )
        
        # Create train/val split (chronological)
        train_data, val_data = self.data_preprocessor.create_train_val_split(
            merged_data,
            val_size=validation_split,
            shuffle=False
        )
        
        return train_data, val_data
    
    def _train_prophet_model(
        self,
        train_data: pd.DataFrame,
        hyperparameters: Dict[str, Any]
    ) -> TimeSeriesModel:
        """
        Train Prophet time series model.
        
        Args:
            train_data: Training data
            hyperparameters: Prophet hyperparameters
        
        Returns:
            Trained TimeSeriesModel
        """
        # Prepare data in Prophet format
        prophet_df = pd.DataFrame({
            'ds': train_data['timestamp'],
            'y': train_data['price']
        })
        
        # Initialize model with hyperparameters
        model = TimeSeriesModel(
            seasonality_mode=hyperparameters.get('seasonality_mode', 'multiplicative'),
            yearly_seasonality=hyperparameters.get('yearly_seasonality', True),
            weekly_seasonality=hyperparameters.get('weekly_seasonality', False),
            changepoint_prior_scale=hyperparameters.get('changepoint_prior_scale', 0.05)
        )
        
        # Fit model
        model.fit(prophet_df)
        
        return model
    
    def _train_xgboost_model(
        self,
        prophet_model: TimeSeriesModel,
        train_data: pd.DataFrame,
        econometric_data: pd.DataFrame,
        hyperparameters: Dict[str, Any]
    ) -> MLModel:
        """
        Train XGBoost model on Prophet residuals.
        
        Args:
            prophet_model: Trained Prophet model
            train_data: Training data
            econometric_data: Econometric features
            hyperparameters: XGBoost hyperparameters
        
        Returns:
            Trained MLModel
        """
        # Compute residuals
        prophet_df = pd.DataFrame({
            'ds': train_data['timestamp'],
            'y': train_data['price']
        })
        residuals = prophet_model.compute_residuals(prophet_df)
        
        # Prepare features
        feature_cols = [
            'temperature', 'rainfall', 'stock_level', 'production',
            'fx_rate_xaf_usd', 'fx_rate_gbp_usd', 'fx_rate_eur_usd'
        ]
        
        X = train_data[feature_cols].copy()
        y = residuals
        
        # Initialize model with hyperparameters
        model = MLModel(
            n_estimators=hyperparameters.get('n_estimators', 100),
            max_depth=hyperparameters.get('max_depth', 6),
            learning_rate=hyperparameters.get('learning_rate', 0.1),
            objective=hyperparameters.get('objective', 'reg:squarederror')
        )
        
        # Fit model
        model.fit(X, y)
        
        return model
    
    def _validate_model(
        self,
        predictor: PricePredictor,
        val_data: pd.DataFrame,
        econometric_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Validate model on validation set and compute metrics.
        
        Args:
            predictor: PricePredictor to validate
            val_data: Validation data
            econometric_data: Econometric features
        
        Returns:
            Dictionary of validation metrics
        
        Requirements: 9.3
        """
        # Prepare features for validation
        feature_cols = [
            'temperature', 'rainfall', 'stock_level', 'production',
            'fx_rate_xaf_usd', 'fx_rate_gbp_usd', 'fx_rate_eur_usd'
        ]
        
        exog_features = val_data[feature_cols].copy()
        
        # Generate predictions for 1-day horizon
        predictions = []
        for i in range(len(val_data)):
            try:
                pred = predictor.predict(
                    horizons=[1],
                    exog_features=exog_features.iloc[i:i+1],
                    recent_news=[],  # No news for validation
                    historical_range=(1000.0, 10000.0)
                )
                predictions.append(pred[0])
            except Exception as e:
                logger.warning(f"Prediction failed for validation row {i}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No successful predictions on validation set")
        
        # Extract predictions and confidence intervals
        y_true = val_data['price'].values[:len(predictions)]
        y_pred = np.array([p.price for p in predictions])
        y_pred_lower = np.array([p.confidence_interval[0] for p in predictions])
        y_pred_upper = np.array([p.confidence_interval[1] for p in predictions])
        
        # Compute metrics
        metrics = self.performance_monitor.compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_lower=y_pred_lower,
            y_pred_upper=y_pred_upper
        )
        
        return metrics
    
    def _should_promote_model(
        self,
        model_name: str,
        new_metrics: Dict[str, float],
        val_data: pd.DataFrame,
        econometric_data: pd.DataFrame
    ) -> bool:
        """
        Determine if new model should be promoted to production.
        
        Compares new model with current production model. New model is promoted
        only if it performs better.
        
        Args:
            model_name: Base model name
            new_metrics: Metrics for new model
            val_data: Validation data
            econometric_data: Econometric features
        
        Returns:
            True if new model should be promoted, False otherwise
        
        Requirements: 9.4
        """
        try:
            # Try to load current production model
            current_prophet = self.model_manager.load_model(
                model_name=f"{model_name}_prophet",
                stage="Production"
            )
            current_xgboost = self.model_manager.load_model(
                model_name=f"{model_name}_xgboost",
                stage="Production"
            )
            
            # Create predictor with current models
            nlp_analyzer = NLPAnalyzer()
            current_predictor = PricePredictor(
                ts_model=current_prophet,
                ml_model=current_xgboost,
                nlp_analyzer=nlp_analyzer,
                model_version="current"
            )
            
            # Validate current model
            current_metrics = self._validate_model(
                current_predictor,
                val_data,
                econometric_data
            )
            
            logger.info(
                f"Current model metrics: RMSE={current_metrics['rmse']:.4f}, "
                f"MAE={current_metrics['mae']:.4f}, MAPE={current_metrics['mape']:.4f}"
            )
            
            # Compare models
            comparison_result = self.performance_monitor.compare_models(
                model_a_metrics=new_metrics,
                model_b_metrics=current_metrics
            )
            
            if comparison_result == "model_a":
                logger.info("New model performs better than current model")
                return True
            else:
                logger.info("Current model performs better or equal to new model")
                return False
        
        except Exception as e:
            # If no current production model exists, promote new model
            logger.info(f"No current production model found ({e}), promoting new model")
            return True
    
    def _log_models_to_mlflow(
        self,
        prophet_model: TimeSeriesModel,
        xgboost_model: MLModel,
        version: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any]
    ) -> None:
        """
        Log trained models to MLflow.
        
        Args:
            prophet_model: Trained Prophet model
            xgboost_model: Trained XGBoost model
            version: Version string
            metrics: Validation metrics
            hyperparameters: Hyperparameters used
        """
        # Log Prophet model
        self.model_manager.log_model(
            model=prophet_model.model,
            model_name="cocoa_price_predictor_prophet",
            model_type="prophet",
            metrics=metrics,
            params=hyperparameters.get("prophet", {}),
            artifacts={}
        )
        
        # Log XGBoost model
        self.model_manager.log_model(
            model=xgboost_model.model,
            model_name="cocoa_price_predictor_xgboost",
            model_type="xgboost",
            metrics=metrics,
            params=hyperparameters.get("xgboost", {}),
            artifacts={}
        )
        
        logger.info(f"Models logged to MLflow with version {version}")
    
    def _generate_version_string(self) -> str:
        """
        Generate version string for new model.
        
        Returns:
            Version string in format "YYYY.MM.DD.HHMM"
        """
        now = datetime.now()
        return now.strftime("%Y.%m.%d.%H%M")
    
    def _cleanup_old_versions(
        self,
        model_name: str
    ) -> None:
        """
        Clean up old model versions, keeping only the most recent ones.
        
        Maintains history of max_model_versions most recent versions by
        archiving older versions.
        
        Args:
            model_name: Base model name
        
        Requirements: 9.5
        """
        try:
            for model_type in ["prophet", "xgboost"]:
                full_model_name = f"{model_name}_{model_type}"
                
                # Get all versions
                versions = self.model_manager.list_model_versions(
                    model_name=full_model_name,
                    max_results=100  # Get all versions
                )
                
                if len(versions) <= self.max_model_versions:
                    logger.info(
                        f"Model {full_model_name} has {len(versions)} versions, "
                        f"no cleanup needed"
                    )
                    continue
                
                # Archive old versions (keep most recent max_model_versions)
                versions_to_archive = versions[self.max_model_versions:]
                
                for version in versions_to_archive:
                    if version.current_stage != "Archived":
                        logger.info(
                            f"Archiving old version {version.version} of {full_model_name}"
                        )
                        self.model_manager.client.transition_model_version_stage(
                            name=full_model_name,
                            version=version.version,
                            stage="Archived"
                        )
                
                logger.info(
                    f"Cleaned up {len(versions_to_archive)} old versions of {full_model_name}"
                )
        
        except Exception as e:
            logger.error(f"Error cleaning up old versions: {e}")
            # Don't fail the retraining workflow if cleanup fails
    
    def get_retraining_status(
        self,
        model_name: str = "cocoa_price_predictor"
    ) -> Dict[str, Any]:
        """
        Get current retraining status and information.
        
        Args:
            model_name: Model name to check
        
        Returns:
            Dictionary with retraining status information
        """
        try:
            # Get latest model version
            versions = self.model_manager.list_model_versions(
                model_name=f"{model_name}_prophet",
                max_results=1
            )
            
            if not versions:
                return {
                    "has_model": False,
                    "should_retrain": True,
                    "reason": "no_existing_model"
                }
            
            latest_version = versions[0]
            last_training_time = datetime.fromtimestamp(
                latest_version.creation_timestamp / 1000
            )
            
            days_since_training = (datetime.now() - last_training_time).days
            new_data_days = self._count_new_data_days(last_training_time)
            should_retrain, reason = self.should_trigger_retraining(model_name)
            
            return {
                "has_model": True,
                "latest_version": latest_version.version,
                "last_training_time": last_training_time.isoformat(),
                "days_since_training": days_since_training,
                "new_data_days": new_data_days,
                "should_retrain": should_retrain,
                "reason": reason,
                "retraining_frequency_days": self.retraining_frequency_days,
                "min_new_data_days": self.min_new_data_days
            }
        
        except Exception as e:
            logger.error(f"Error getting retraining status: {e}")
            return {
                "error": str(e)
            }
    
    def __repr__(self) -> str:
        """String representation of the RetrainingManager."""
        return (
            f"RetrainingManager(retraining_frequency={self.retraining_frequency_days}d, "
            f"min_new_data={self.min_new_data_days}d, "
            f"max_versions={self.max_model_versions})"
        )
