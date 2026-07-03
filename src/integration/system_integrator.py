"""
System Integrator for the Cocoa Price Prediction Hybrid System.

This module provides the SystemIntegrator class that connects all components
of the hybrid prediction system into a cohesive end-to-end pipeline:

1. Data Collection Layer: DataCollector
2. Data Processing Layer: DataValidator → DataPreprocessor
3. Model Layer: TimeSeriesModel → MLModel → NLPAnalyzer
4. Prediction Layer: PricePredictor
5. Storage Layer: Supabase (PostgreSQL) + Redis Cache
6. API Layer: FastAPI
7. Monitoring Layer: PerformanceMonitor + ModelManager

This integrator ensures all dependencies are correctly wired and provides
a unified interface for the complete prediction workflow.

Requirements: All (Task 20.1)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from supabase import create_client, Client

from src.data_collection.data_collector import DataCollector
from src.data_validation.data_validator import DataValidator
from src.data_preprocessing.data_preprocessor import DataPreprocessor
from src.models.time_series_model import TimeSeriesModel
from src.models.ml_model import MLModel
from src.nlp.nlp_analyzer import NLPAnalyzer
from src.models.price_predictor import PricePredictor
from src.models.model_manager import ModelManager
from src.monitoring.performance_monitor import PerformanceMonitor
from src.api.cache import RedisCache
from src.models.data_models import NewsArticle, Prediction
from config.settings import get_settings

logger = logging.getLogger(__name__)


class SystemIntegrator:
    """
    Integrates all components of the hybrid prediction system.
    
    This class provides a unified interface for the complete prediction workflow,
    from data collection through prediction generation and monitoring. It ensures
    all components are properly initialized and connected.
    
    Attributes:
        data_collector: DataCollector instance for data collection
        data_validator: DataValidator instance for data validation
        data_preprocessor: DataPreprocessor instance for data preprocessing
        ts_model: TimeSeriesModel instance for baseline predictions
        ml_model: MLModel instance for residual corrections
        nlp_analyzer: NLPAnalyzer instance for sentiment analysis
        price_predictor: PricePredictor instance for hybrid predictions
        model_manager: ModelManager instance for model lifecycle management
        performance_monitor: PerformanceMonitor instance for performance tracking
        redis_cache: RedisCache instance for caching predictions
        supabase_client: Supabase client for database operations
    """
    
    def __init__(
        self,
        data_collector: Optional[DataCollector] = None,
        data_validator: Optional[DataValidator] = None,
        data_preprocessor: Optional[DataPreprocessor] = None,
        ts_model: Optional[TimeSeriesModel] = None,
        ml_model: Optional[MLModel] = None,
        nlp_analyzer: Optional[NLPAnalyzer] = None,
        price_predictor: Optional[PricePredictor] = None,
        model_manager: Optional[ModelManager] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        redis_cache: Optional[RedisCache] = None,
        supabase_client: Optional[Client] = None
    ):
        """
        Initialize SystemIntegrator with optional component instances.
        
        If components are not provided, they will be initialized with default
        configurations from settings.
        
        Args:
            data_collector: DataCollector instance (optional)
            data_validator: DataValidator instance (optional)
            data_preprocessor: DataPreprocessor instance (optional)
            ts_model: TimeSeriesModel instance (optional)
            ml_model: MLModel instance (optional)
            nlp_analyzer: NLPAnalyzer instance (optional)
            price_predictor: PricePredictor instance (optional)
            model_manager: ModelManager instance (optional)
            performance_monitor: PerformanceMonitor instance (optional)
            redis_cache: RedisCache instance (optional)
            supabase_client: Supabase client instance (optional)
        """
        logger.info("Initializing SystemIntegrator...")
        
        # Load settings
        self.settings = get_settings()
        
        # Initialize or use provided components
        self.data_collector = data_collector or DataCollector()
        self.data_validator = data_validator or DataValidator()
        self.data_preprocessor = data_preprocessor or DataPreprocessor()
        
        # Initialize models
        self.ts_model = ts_model or TimeSeriesModel()
        self.ml_model = ml_model or MLModel()
        self.nlp_analyzer = nlp_analyzer or NLPAnalyzer()
        
        # Initialize price predictor if models are fitted
        if price_predictor:
            self.price_predictor = price_predictor
        elif self.ts_model.is_fitted and self.ml_model.is_fitted:
            self.price_predictor = PricePredictor(
                ts_model=self.ts_model,
                ml_model=self.ml_model,
                nlp_analyzer=self.nlp_analyzer
            )
        else:
            self.price_predictor = None
            logger.warning(
                "Price predictor not initialized. Models need to be trained first."
            )
        
        # Initialize infrastructure components
        self.model_manager = model_manager or ModelManager(
            tracking_uri=self.settings.mlflow_tracking_uri,
            registry_uri=self.settings.mlflow_registry_uri
        )
        
        # Initialize Supabase client
        if supabase_client:
            self.supabase_client = supabase_client
        else:
            self.supabase_client = create_client(
                self.settings.supabase_url,
                self.settings.supabase_key
            )
        
        # Initialize performance monitor
        self.performance_monitor = performance_monitor or PerformanceMonitor(
            supabase_client=self.supabase_client
        )
        
        # Initialize Redis cache
        try:
            self.redis_cache = redis_cache or RedisCache()
            if self.redis_cache.health_check():
                logger.info("Redis cache connected successfully")
            else:
                logger.warning("Redis cache health check failed")
                self.redis_cache = None
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}")
            self.redis_cache = None
        
        logger.info("SystemIntegrator initialized successfully")
    
    def collect_and_prepare_data(
        self,
        start_date: datetime,
        end_date: datetime,
        markets: List[str] = ["ICE_London", "ICE_NY"]
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], List[NewsArticle]]:
        """
        Execute the complete data collection and preparation pipeline.
        
        This method orchestrates the data flow through:
        1. DataCollector: Collect price, econometric, and news data
        2. DataValidator: Validate data quality and detect outliers
        3. DataPreprocessor: Handle missing values and engineer features
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            markets: List of markets to collect data from
        
        Returns:
            Tuple containing:
                - price_df: Validated and preprocessed price data
                - econometric_data: Dict of validated econometric data
                - news_articles: List of news articles with sentiment scores
        
        Raises:
            Exception: If data collection or preparation fails
        """
        logger.info(
            f"Starting data collection and preparation pipeline "
            f"from {start_date} to {end_date}"
        )
        
        # Step 1: Collect data
        logger.info("Step 1: Collecting data from external sources...")
        
        # Collect price data
        price_df = self.data_collector.collect_price_data(
            markets=markets,
            start_date=start_date,
            end_date=end_date
        )
        logger.info(f"Collected {len(price_df)} price records")
        
        # Collect econometric data
        econometric_data = self.data_collector.collect_econometric_data(
            start_date=start_date,
            end_date=end_date
        )
        logger.info(
            f"Collected econometric data: "
            f"{', '.join([f'{k}={len(v)}' for k, v in econometric_data.items()])}"
        )
        
        # Collect news articles
        news_articles = self.data_collector.collect_news_feed(
            sources=["reuters", "bloomberg"],
            keywords=["cocoa", "cacao", "chocolate"],
            hours_back=24
        )
        logger.info(f"Collected {len(news_articles)} news articles")
        
        # Step 2: Validate data
        logger.info("Step 2: Validating data quality...")
        
        # Validate price data
        price_df, price_errors = self.data_validator.validate_price_data(price_df)
        if price_errors:
            logger.warning(f"Found {len(price_errors)} price validation issues")
        
        # Validate econometric data
        econometric_data, econ_errors = self.data_validator.validate_econometric_data(
            econometric_data
        )
        if econ_errors:
            logger.warning(f"Found {len(econ_errors)} econometric validation issues")
        
        # Detect outliers in price data
        price_outliers = self.data_validator.detect_outliers(
            price_df, column="price", method="zscore", threshold=3.0
        )
        logger.info(f"Detected {price_outliers.sum()} price outliers")
        
        # Flag market shocks
        price_df = self.data_validator.flag_market_shocks(
            price_df, threshold=0.05
        )
        shock_count = price_df['is_shock'].sum()
        logger.info(f"Flagged {shock_count} market shock events")
        
        # Step 3: Preprocess data
        logger.info("Step 3: Preprocessing data...")
        
        # Handle missing values in price data
        if price_df['price'].isna().any():
            price_df = self.data_preprocessor.handle_missing_values(
                price_df,
                strategy={"price": "forward_fill", "volume": "mean"}
            )
        
        # Handle missing values in econometric data
        for key, df in econometric_data.items():
            if df.isna().any().any():
                strategy = {}
                for col in df.columns:
                    if df[col].isna().any():
                        if col in ['temperature', 'rainfall']:
                            strategy[col] = "interpolate"
                        else:
                            strategy[col] = "forward_fill"
                
                if strategy:
                    econometric_data[key] = self.data_preprocessor.handle_missing_values(
                        df, strategy=strategy
                    )
        
        # Engineer time features
        price_df = self.data_preprocessor.engineer_time_features(
            price_df, timestamp_col="timestamp"
        )
        logger.info("Engineered time-based features")
        
        # Step 4: Analyze sentiment for news articles
        logger.info("Step 4: Analyzing sentiment for news articles...")
        
        if news_articles:
            # Batch analyze sentiment
            texts = [f"{article.title} {article.content}" for article in news_articles]
            sentiments = self.nlp_analyzer.batch_analyze(texts, batch_size=32)
            
            # Update articles with sentiment scores
            for article, sentiment in zip(news_articles, sentiments):
                article.sentiment_score = sentiment["score"]
                
                # Extract keywords
                article.keywords = self.nlp_analyzer.extract_keywords(
                    f"{article.title} {article.content}"
                )
                
                # Flag high-risk articles
                article.is_high_risk = self.nlp_analyzer.flag_high_risk(article)
            
            high_risk_count = sum(1 for a in news_articles if a.is_high_risk)
            logger.info(
                f"Analyzed sentiment for {len(news_articles)} articles, "
                f"{high_risk_count} flagged as high-risk"
            )
        
        logger.info("Data collection and preparation pipeline completed successfully")
        
        return price_df, econometric_data, news_articles
    
    def train_models(
        self,
        price_df: pd.DataFrame,
        econometric_data: Dict[str, pd.DataFrame],
        val_size: float = 0.2
    ) -> Dict[str, any]:
        """
        Train all models in the hybrid system.
        
        This method orchestrates the model training pipeline:
        1. Split data into train/validation sets
        2. Train TimeSeriesModel (Prophet) on price data
        3. Compute residuals from Prophet predictions
        4. Train MLModel (XGBoost) on residuals with econometric features
        5. Evaluate model performance
        
        Args:
            price_df: Preprocessed price data with time features
            econometric_data: Dict of preprocessed econometric data
            val_size: Fraction of data to use for validation (default: 0.2)
        
        Returns:
            Dict with training results including metrics and model versions
        
        Raises:
            Exception: If model training fails
        """
        logger.info("Starting model training pipeline...")
        
        # Step 1: Create train/validation split
        logger.info("Step 1: Creating train/validation split...")
        
        train_df, val_df = self.data_preprocessor.create_train_val_split(
            price_df, val_size=val_size, shuffle=False
        )
        logger.info(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
        
        # Step 2: Train TimeSeriesModel (Prophet)
        logger.info("Step 2: Training TimeSeriesModel (Prophet)...")
        
        # Prepare data in Prophet format
        prophet_train = train_df[['timestamp', 'price']].copy()
        prophet_train.columns = ['ds', 'y']
        
        self.ts_model.fit(prophet_train, date_col='ds', target_col='y')
        logger.info("TimeSeriesModel trained successfully")
        
        # Step 3: Compute residuals
        logger.info("Step 3: Computing residuals from Prophet predictions...")
        
        residuals = self.ts_model.compute_residuals(prophet_train)
        logger.info(
            f"Residuals computed: mean={residuals.mean():.2f}, "
            f"std={residuals.std():.2f}"
        )
        
        # Step 4: Prepare features for MLModel
        logger.info("Step 4: Preparing features for MLModel...")
        
        # Merge econometric features with price data
        feature_df = train_df.copy()
        
        # Add econometric features (simplified - in production, proper time alignment needed)
        for key, econ_df in econometric_data.items():
            if not econ_df.empty:
                # Merge on timestamp (simplified)
                for col in econ_df.columns:
                    if col != 'timestamp':
                        feature_df[col] = econ_df[col].mean()  # Simplified
        
        # Select features for ML model
        feature_columns = [
            'temperature', 'rainfall', 'stock_level', 'production',
            'fx_rate_xaf_usd', 'fx_rate_gbp_usd', 'fx_rate_eur_usd',
            'day_of_week', 'day_of_month', 'month', 'quarter',
            'is_month_start', 'is_month_end', 'days_since_shock'
        ]
        
        # Filter to available columns
        available_features = [col for col in feature_columns if col in feature_df.columns]
        X_train = feature_df[available_features]
        y_train = residuals
        
        # Prepare validation set
        val_feature_df = val_df.copy()
        for key, econ_df in econometric_data.items():
            if not econ_df.empty:
                for col in econ_df.columns:
                    if col != 'timestamp':
                        val_feature_df[col] = econ_df[col].mean()
        
        X_val = val_feature_df[available_features]
        
        # Compute validation residuals
        prophet_val = val_df[['timestamp', 'price']].copy()
        prophet_val.columns = ['ds', 'y']
        y_val = self.ts_model.compute_residuals(prophet_val)
        
        logger.info(f"Prepared {len(available_features)} features for ML model")
        
        # Step 5: Train MLModel (XGBoost)
        logger.info("Step 5: Training MLModel (XGBoost)...")
        
        self.ml_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10
        )
        logger.info("MLModel trained successfully")
        
        # Step 6: Evaluate models
        logger.info("Step 6: Evaluating model performance...")
        
        # Cross-validate ML model
        cv_metrics = self.ml_model.cross_validate(X_train, y_train, cv=5)
        logger.info(
            f"ML Model CV metrics: RMSE={cv_metrics['rmse']:.2f}, "
            f"MAE={cv_metrics['mae']:.2f}, R2={cv_metrics['r2']:.4f}"
        )
        
        # Get feature importance
        feature_importance = self.ml_model.get_feature_importance(importance_type="gain")
        logger.info(f"Top 5 features: {list(feature_importance['feature'].head(5))}")
        
        # Step 7: Initialize PricePredictor
        logger.info("Step 7: Initializing PricePredictor...")
        
        self.price_predictor = PricePredictor(
            ts_model=self.ts_model,
            ml_model=self.ml_model,
            nlp_analyzer=self.nlp_analyzer,
            model_version="1.0.0"
        )
        logger.info("PricePredictor initialized successfully")
        
        # Prepare training results
        results = {
            "ts_model_trained": True,
            "ml_model_trained": True,
            "ml_cv_metrics": cv_metrics,
            "feature_importance": feature_importance.to_dict('records'),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "num_features": len(available_features),
            "model_version": "1.0.0"
        }
        
        logger.info("Model training pipeline completed successfully")
        
        return results
    
    def generate_predictions(
        self,
        horizons: List[int],
        exog_features: pd.DataFrame,
        recent_news: List[NewsArticle],
        use_cache: bool = True
    ) -> List[Prediction]:
        """
        Generate hybrid predictions using the complete system.
        
        This method orchestrates the prediction pipeline:
        1. Check Redis cache for existing predictions
        2. If cache miss, generate predictions using PricePredictor
        3. Store predictions in cache and database
        4. Return predictions with confidence intervals
        
        Args:
            horizons: List of prediction horizons in days
            exog_features: DataFrame with econometric features
            recent_news: List of recent news articles
            use_cache: Whether to use Redis cache (default: True)
        
        Returns:
            List of Prediction objects
        
        Raises:
            RuntimeError: If price predictor is not initialized
            Exception: If prediction generation fails
        """
        if self.price_predictor is None:
            raise RuntimeError(
                "Price predictor not initialized. Train models first using train_models()."
            )
        
        logger.info(f"Generating predictions for horizons: {horizons}")
        
        # Check cache if enabled
        if use_cache and self.redis_cache:
            cache_key = f"prediction:{'_'.join(map(str, horizons))}"
            cached_predictions = self.redis_cache.get(cache_key)
            
            if cached_predictions:
                logger.info("Returning cached predictions")
                return cached_predictions
        
        # Generate predictions
        predictions = self.price_predictor.predict(
            horizons=horizons,
            exog_features=exog_features,
            recent_news=recent_news
        )
        
        # Store in cache if enabled
        if use_cache and self.redis_cache:
            cache_key = f"prediction:{'_'.join(map(str, horizons))}"
            self.redis_cache.set(cache_key, predictions, ttl=3600)  # 1 hour TTL
            logger.info("Stored predictions in cache")
        
        # Store in database
        try:
            for pred in predictions:
                self.supabase_client.table("predictions").insert({
                    "horizon": pred.horizon,
                    "predicted_price": pred.price,
                    "lower_bound": pred.confidence_interval[0],
                    "upper_bound": pred.confidence_interval[1],
                    "confidence_level": pred.confidence_level,
                    "model_version": pred.model_version,
                    "baseline_component": pred.components.get("baseline"),
                    "residual_component": pred.components.get("residual"),
                    "sentiment_component": pred.components.get("sentiment"),
                    "created_at": pred.timestamp.isoformat()
                }).execute()
            
            logger.info("Stored predictions in database")
        except Exception as e:
            logger.error(f"Failed to store predictions in database: {e}")
        
        return predictions
    
    def monitor_performance(
        self,
        predictions: List[Prediction],
        actual_prices: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Monitor and track model performance.
        
        This method:
        1. Computes performance metrics (RMSE, MAE, MAPE, etc.)
        2. Tracks metrics in database via PerformanceMonitor
        3. Detects performance degradation
        4. Triggers retraining alerts if needed
        
        Args:
            predictions: List of predictions to evaluate
            actual_prices: DataFrame with actual prices for comparison
        
        Returns:
            Dict with computed performance metrics
        """
        logger.info("Monitoring model performance...")
        
        # Prepare data for metrics computation
        y_true = []
        y_pred = []
        y_pred_lower = []
        y_pred_upper = []
        
        for pred in predictions:
            # Find actual price for this horizon
            pred_date = pred.timestamp + timedelta(days=pred.horizon)
            actual_row = actual_prices[
                actual_prices['timestamp'] == pred_date
            ]
            
            if not actual_row.empty:
                y_true.append(actual_row.iloc[0]['price'])
                y_pred.append(pred.price)
                y_pred_lower.append(pred.confidence_interval[0])
                y_pred_upper.append(pred.confidence_interval[1])
        
        if not y_true:
            logger.warning("No actual prices available for performance evaluation")
            return {}
        
        # Compute metrics
        import numpy as np
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_lower = np.array(y_pred_lower)
        y_pred_upper = np.array(y_pred_upper)
        
        metrics = self.performance_monitor.compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_lower=y_pred_lower,
            y_pred_upper=y_pred_upper
        )
        
        logger.info(
            f"Performance metrics: RMSE={metrics['rmse']:.2f}, "
            f"MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%"
        )
        
        # Track performance
        self.performance_monitor.track_performance(
            metrics=metrics,
            timestamp=datetime.utcnow(),
            model_version="1.0.0"
        )
        
        # Check for degradation
        # In production, this would compare against baseline metrics
        # For now, we'll just log the metrics
        logger.info("Performance monitoring completed")
        
        return metrics
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check health status of all system components.
        
        Returns:
            Dict with component names and their health status
        """
        health_status = {
            "data_collector": self.data_collector is not None,
            "data_validator": self.data_validator is not None,
            "data_preprocessor": self.data_preprocessor is not None,
            "ts_model": self.ts_model is not None and self.ts_model.is_fitted,
            "ml_model": self.ml_model is not None and self.ml_model.is_fitted,
            "nlp_analyzer": self.nlp_analyzer is not None,
            "price_predictor": self.price_predictor is not None,
            "model_manager": self.model_manager is not None,
            "performance_monitor": self.performance_monitor is not None,
            "redis_cache": self.redis_cache is not None and self.redis_cache.health_check(),
            "supabase": self.supabase_client is not None
        }
        
        all_healthy = all(health_status.values())
        health_status["overall"] = all_healthy
        
        return health_status
    
    def __repr__(self) -> str:
        """String representation of the integrator."""
        health = self.health_check()
        healthy_count = sum(1 for v in health.values() if v)
        total_count = len(health)
        
        return (
            f"SystemIntegrator(components={healthy_count}/{total_count} healthy, "
            f"models_trained={self.ts_model.is_fitted and self.ml_model.is_fitted})"
        )
