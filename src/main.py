"""Main integration module for Cocoa Price Prediction System.

This module integrates all components:
- Data Collection → Data Validation → Data Preprocessing
- Time Series Model (Prophet) → ML Model (XGBoost)
- NLP Analyzer (FinBERT) → Price Predictor
- Performance Monitor → Model Manager
- API → Cache
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

from src.config.config_manager import load_config, get_config
from src.data_collection.multi_source_collector import MultiSourceCollector
from src.data_collection.supabase_saver import SupabaseSaver
from src.data_validation.data_validator import DataValidator
from src.data_preprocessing.data_preprocessor import DataPreprocessor
from src.models.time_series_model import TimeSeriesModel
from src.models.ml_model import MLModel
from src.nlp.sentiment_analyzer import SentimentAnalyzer
from src.models.price_predictor import PricePredictor
from src.monitoring.performance_monitor import PerformanceMonitor
from src.integration.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CocoaPredictionSystem:
    """Main system integrating all components."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the prediction system.
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing Cocoa Price Prediction System...")
        
        # Load configuration
        self.config = load_config(config_path)
        logger.info("Configuration loaded and validated")
        
        # Initialize components
        self._initialize_components()
        
        logger.info("System initialization complete")
    
    def _initialize_components(self):
        """Initialize all system components."""
        # Data collection and storage
        self.data_collector = MultiSourceCollector()
        self.data_saver = SupabaseSaver()
        
        # Data validation and preprocessing
        self.data_validator = DataValidator()
        self.data_preprocessor = DataPreprocessor()
        
        # Models
        prophet_config = get_config("prophet")
        self.ts_model = TimeSeriesModel(
            seasonality_mode=prophet_config.get("seasonality_mode", "multiplicative"),
            yearly_seasonality=prophet_config.get("yearly_seasonality", True),
            weekly_seasonality=prophet_config.get("weekly_seasonality", False),
            changepoint_prior_scale=prophet_config.get("changepoint_prior_scale", 0.05)
        )
        
        xgb_config = get_config("xgboost")
        self.ml_model = MLModel(
            n_estimators=xgb_config.get("n_estimators", 100),
            max_depth=xgb_config.get("max_depth", 6),
            learning_rate=xgb_config.get("learning_rate", 0.1)
        )
        
        # NLP analyzer
        self.nlp_analyzer = SentimentAnalyzer()
        
        # Price predictor
        self.price_predictor = PricePredictor(
            ts_model=self.ts_model,
            ml_model=self.ml_model,
            nlp_analyzer=self.nlp_analyzer
        )
        
        # Monitoring and management
        self.performance_monitor = PerformanceMonitor()
        
        mlflow_config = get_config("mlflow")
        self.model_manager = ModelManager(
            tracking_uri=mlflow_config.get("tracking_uri", "http://localhost:5000"),
            registry_uri=mlflow_config.get("registry_uri", "http://localhost:5000")
        )
        
        logger.info("All components initialized")
    
    def collect_and_store_data(self) -> Dict[str, Any]:
        """Collect data from all sources and store in database.
        
        Returns:
            Dictionary with collection results
        """
        logger.info("Starting data collection...")
        
        # Collect data
        data = self.data_collector.collect_all()
        
        # Analyze sentiment
        if data.get("news"):
            logger.info("Analyzing news sentiment...")
            sentiment_results = self.nlp_analyzer.analyze_batch(
                [article["title"] + " " + article.get("description", "") 
                 for article in data["news"][:10]]
            )
            
            # Add sentiment to news articles
            for i, article in enumerate(data["news"][:10]):
                if i < len(sentiment_results):
                    article["sentiment_score"] = sentiment_results[i]["score"]
        
        # Save to database
        logger.info("Saving data to database...")
        save_results = self.data_saver.save_all(data)
        
        logger.info(f"Data collection complete: {save_results}")
        return save_results
    
    def prepare_training_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for model training.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            Tuple of (training_data, validation_data)
        """
        logger.info("Preparing training data...")
        
        # Fetch historical data from database
        price_data = self.data_saver.fetch_price_data(start_date, end_date)
        
        if price_data.empty:
            raise ValueError("No price data available for training")
        
        logger.info(f"Fetched {len(price_data)} price data points")
        
        # Validate data
        validated_data, errors = self.data_validator.validate_price_data(price_data)
        
        if errors:
            logger.warning(f"Found {len(errors)} validation errors")
            for error in errors[:5]:  # Log first 5 errors
                logger.warning(f"  {error.severity}: {error.message}")
        
        # Preprocess data
        processed_data = self.data_preprocessor.handle_missing_values(
            validated_data,
            strategy=get_config("preprocessing.missing_values")
        )
        
        # Create train/val split
        train_data, val_data = self.data_preprocessor.create_train_val_split(
            processed_data,
            val_size=get_config("preprocessing.train_val_split.validation_size", 0.2)
        )
        
        logger.info(f"Training data: {len(train_data)} points")
        logger.info(f"Validation data: {len(val_data)} points")
        
        return train_data, val_data
    
    def train_models(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Train all models.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("Starting model training...")
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_train = train_data[['date', 'price']].copy()
        prophet_train.columns = ['ds', 'y']
        
        prophet_val = val_data[['date', 'price']].copy()
        prophet_val.columns = ['ds', 'y']
        
        # Train Prophet model
        logger.info("Training Prophet model...")
        self.ts_model.fit(prophet_train)
        
        # Generate predictions and compute residuals
        prophet_predictions = self.ts_model.predict(periods=len(prophet_val))
        residuals = self.ts_model.compute_residuals(prophet_val)
        
        # Prepare features for XGBoost
        logger.info("Preparing features for XGBoost...")
        X_train = self.data_preprocessor.engineer_time_features(train_data)
        X_val = self.data_preprocessor.engineer_time_features(val_data)
        
        # Train XGBoost on residuals
        logger.info("Training XGBoost model...")
        y_train_residuals = train_data['price'] - self.ts_model.predict(periods=len(train_data))['yhat'].values
        
        self.ml_model.fit(
            X_train,
            y_train_residuals,
            eval_set=[(X_val, residuals)],
            early_stopping_rounds=get_config("xgboost.early_stopping_rounds", 10)
        )
        
        # Compute metrics
        logger.info("Computing performance metrics...")
        
        # Get hybrid predictions
        xgb_residuals = self.ml_model.predict(X_val)
        hybrid_predictions = prophet_predictions['yhat'].values + xgb_residuals
        
        metrics = self.performance_monitor.compute_metrics(
            y_true=val_data['price'].values,
            y_pred=hybrid_predictions,
            y_pred_lower=prophet_predictions['yhat_lower'].values + xgb_residuals,
            y_pred_upper=prophet_predictions['yhat_upper'].values + xgb_residuals
        )
        
        logger.info(f"Training complete. Metrics: {metrics}")
        
        # Log models to MLflow
        logger.info("Logging models to MLflow...")
        
        prophet_version = self.model_manager.log_model(
            model=self.ts_model.model,
            model_name=get_config("mlflow.model_names.prophet", "cocoa_prophet_model"),
            model_type="prophet",
            metrics=metrics,
            params=get_config("prophet"),
            artifacts={}
        )
        
        xgb_version = self.model_manager.log_model(
            model=self.ml_model.model,
            model_name=get_config("mlflow.model_names.xgboost", "cocoa_xgboost_model"),
            model_type="xgboost",
            metrics=metrics,
            params=get_config("xgboost"),
            artifacts={}
        )
        
        return {
            "metrics": metrics,
            "prophet_version": prophet_version,
            "xgboost_version": xgb_version,
            "training_samples": len(train_data),
            "validation_samples": len(val_data)
        }
    
    def generate_prediction(
        self,
        horizons: Optional[List[int]] = None,
        include_sentiment: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate price predictions.
        
        Args:
            horizons: List of prediction horizons in days (default from config)
            include_sentiment: Whether to include sentiment analysis
            
        Returns:
            List of prediction dictionaries
        """
        if horizons is None:
            horizons = get_config("prediction.horizons", [1, 7, 30])
        
        logger.info(f"Generating predictions for horizons: {horizons}")
        
        # Collect recent data for features
        recent_data = self.data_saver.fetch_price_data(
            start_date=datetime.now() - timedelta(days=90),
            end_date=datetime.now()
        )
        
        # Prepare features
        exog_features = self.data_preprocessor.engineer_time_features(recent_data)
        
        # Get recent news for sentiment
        recent_news = []
        if include_sentiment:
            news_data = self.data_collector.collect_all().get("news", [])
            recent_news = news_data[:10]  # Last 10 articles
        
        # Generate predictions
        predictions = self.price_predictor.predict(
            horizons=horizons,
            exog_features=exog_features,
            recent_news=recent_news
        )
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline: collect → train → predict.
        
        Returns:
            Dictionary with pipeline results
        """
        logger.info("="*80)
        logger.info("Running Full Pipeline")
        logger.info("="*80)
        
        results = {}
        
        # Step 1: Collect and store data
        logger.info("\nStep 1: Data Collection")
        logger.info("-"*80)
        collection_results = self.collect_and_store_data()
        results["data_collection"] = collection_results
        
        # Step 2: Prepare training data
        logger.info("\nStep 2: Data Preparation")
        logger.info("-"*80)
        train_data, val_data = self.prepare_training_data()
        results["data_preparation"] = {
            "training_samples": len(train_data),
            "validation_samples": len(val_data)
        }
        
        # Step 3: Train models
        logger.info("\nStep 3: Model Training")
        logger.info("-"*80)
        training_results = self.train_models(train_data, val_data)
        results["training"] = training_results
        
        # Step 4: Generate predictions
        logger.info("\nStep 4: Prediction Generation")
        logger.info("-"*80)
        predictions = self.generate_prediction()
        results["predictions"] = predictions
        
        logger.info("\n" + "="*80)
        logger.info("Pipeline Complete")
        logger.info("="*80)
        
        return results


def main():
    """Main entry point."""
    try:
        # Initialize system
        system = CocoaPredictionSystem()
        
        # Run full pipeline
        results = system.run_full_pipeline()
        
        # Print summary
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        print(f"\nData Collection:")
        print(f"  Sources collected: {len(results['data_collection'])}")
        
        print(f"\nData Preparation:")
        print(f"  Training samples: {results['data_preparation']['training_samples']}")
        print(f"  Validation samples: {results['data_preparation']['validation_samples']}")
        
        print(f"\nModel Training:")
        metrics = results['training']['metrics']
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  MAPE: {metrics['mape']:.2%}")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
        
        print(f"\nPredictions Generated:")
        for pred in results['predictions']:
            print(f"  {pred['horizon']}-day: ${pred['price']:.2f} "
                  f"[{pred['confidence_interval'][0]:.2f} - {pred['confidence_interval'][1]:.2f}]")
        
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
