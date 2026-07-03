"""
ML Model using XGBoost for residual prediction in cocoa price forecasting.

This module implements the machine learning component that predicts residuals
from the time series model using econometric features and sentiment scores.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
from loguru import logger


class MLModel:
    """XGBoost-based residual prediction model.
    
    This model predicts residuals from the time series baseline using
    econometric features (weather, stocks, production, FX rates) and
    sentiment scores from news analysis.
    
    Attributes:
        model: XGBoost regressor instance
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Boosting learning rate
        objective: Loss function to optimize
        is_fitted: Whether the model has been trained
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        objective: str = "reg:squarederror"
    ):
        """Initialize XGBoost regressor.
        
        Args:
            n_estimators: Number of boosting rounds. Higher values may improve
                performance but increase training time and risk overfitting.
                Default 100 is a good starting point.
            max_depth: Maximum depth of trees. Controls model complexity.
                Lower values (3-6) prevent overfitting, higher values (6-10)
                allow more complex patterns. Default 6 balances both.
            learning_rate: Step size shrinkage to prevent overfitting.
                Lower values (0.01-0.1) are more conservative and may need
                more n_estimators. Default 0.1 is standard.
            objective: Loss function. 'reg:squarederror' for MSE minimization,
                'reg:squaredlogerror' for RMSLE, 'reg:pseudohubererror' for
                robustness to outliers.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.is_fitted = False
        
        # Initialize XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=objective,
            random_state=42,  # For reproducibility
            n_jobs=-1  # Use all CPU cores
        )
        
        logger.info(
            f"MLModel initialized with n_estimators={n_estimators}, "
            f"max_depth={max_depth}, learning_rate={learning_rate}, "
            f"objective={objective}"
        )
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[Tuple]] = None,
        early_stopping_rounds: int = 10
    ) -> None:
        """Train XGBoost model on residuals.
        
        The model learns to predict residuals (errors from the time series model)
        using econometric features and sentiment scores. Early stopping prevents
        overfitting by monitoring validation performance.
        
        Args:
            X: Feature matrix with econometric data and sentiment scores.
                Expected columns may include: temperature, rainfall, stock_level,
                production, fx_rate_*, sentiment_score, etc.
            y: Target variable (residuals from time series model)
            eval_set: List of (X_val, y_val) tuples for validation during training.
                Used for early stopping. If None, no early stopping is applied.
            early_stopping_rounds: Number of rounds with no improvement before
                stopping. Only used if eval_set is provided.
        
        Raises:
            ValueError: If X and y have mismatched lengths or if X is empty
        """
        # Validate input
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length. Got X: {len(X)}, y: {len(y)}"
            )
        
        if len(X) == 0:
            raise ValueError("Cannot fit model with empty dataset")
        
        if X.shape[1] == 0:
            raise ValueError("Feature matrix X must have at least one column")
        
        logger.info(f"Fitting MLModel on {len(X)} samples with {X.shape[1]} features")
        logger.info(f"Feature columns: {list(X.columns)}")
        logger.info(
            f"Target statistics: mean={y.mean():.2f}, std={y.std():.2f}, "
            f"min={y.min():.2f}, max={y.max():.2f}"
        )
        
        # Prepare training arguments
        fit_params = {}
        
        if eval_set is not None:
            # Validate eval_set
            if not isinstance(eval_set, list):
                raise ValueError("eval_set must be a list of (X, y) tuples")
            
            for i, (X_val, y_val) in enumerate(eval_set):
                if len(X_val) != len(y_val):
                    raise ValueError(
                        f"eval_set[{i}]: X and y must have same length. "
                        f"Got X: {len(X_val)}, y: {len(y_val)}"
                    )
            
            # For XGBoost sklearn API, we need to set early_stopping_rounds
            # as a parameter during initialization, not during fit
            # So we'll create a new model with early stopping enabled
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective=self.objective,
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=early_stopping_rounds
            )
            
            fit_params['eval_set'] = eval_set
            fit_params['verbose'] = False  # Suppress XGBoost output
            
            logger.info(
                f"Using early stopping with {early_stopping_rounds} rounds "
                f"and {len(eval_set)} validation set(s)"
            )
        
        # Fit the model
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        
        # Log training results
        if eval_set is not None and hasattr(self.model, 'best_iteration'):
            logger.info(
                f"Training completed. Best iteration: {self.model.best_iteration} "
                f"(out of {self.n_estimators})"
            )
        else:
            logger.info(f"Training completed using all {self.n_estimators} iterations")
        
        # Compute training metrics
        y_pred = self.model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        train_mae = mean_absolute_error(y, y_pred)
        
        logger.info(f"Training RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
    
    def predict(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """Predict residual corrections.
        
        Given econometric features and sentiment scores, predict the residual
        correction that should be added to the time series baseline prediction.
        
        Args:
            X: Feature matrix with same columns as used during training
        
        Returns:
            Array of predicted residual values
        
        Raises:
            RuntimeError: If model hasn't been fitted yet
            ValueError: If X has wrong number of features
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before making predictions. Call fit() first."
            )
        
        if X.shape[1] != self.model.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained with "
                f"{self.model.n_features_in_} features"
            )
        
        logger.info(f"Generating predictions for {len(X)} samples")
        
        # Generate predictions
        predictions = self.model.predict(X)
        
        logger.info(
            f"Predictions: mean={predictions.mean():.2f}, "
            f"std={predictions.std():.2f}, "
            f"min={predictions.min():.2f}, "
            f"max={predictions.max():.2f}"
        )
        
        return predictions
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = "neg_root_mean_squared_error"
    ) -> Dict[str, float]:
        """Perform time series cross-validation.
        
        Uses TimeSeriesSplit to respect temporal order of data. Each fold uses
        past data for training and future data for validation, simulating
        real-world forecasting scenarios.
        
        Args:
            X: Feature matrix
            y: Target variable (residuals)
            cv: Number of cross-validation folds. Default 5 provides good
                balance between computation time and reliability.
            scoring: Scoring metric (not used directly, metrics computed manually)
        
        Returns:
            Dictionary with metrics:
                - rmse: Root Mean Squared Error
                - mae: Mean Absolute Error
                - mape: Mean Absolute Percentage Error
                - r2: R-squared score
        
        Raises:
            ValueError: If X and y have mismatched lengths or cv is invalid
        """
        # Validate input
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length. Got X: {len(X)}, y: {len(y)}"
            )
        
        if cv < 2:
            raise ValueError(f"cv must be at least 2, got {cv}")
        
        if len(X) < cv:
            raise ValueError(
                f"Cannot perform {cv}-fold CV with only {len(X)} samples"
            )
        
        logger.info(
            f"Performing {cv}-fold time series cross-validation on {len(X)} samples"
        )
        
        # Initialize TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Store metrics for each fold
        rmse_scores = []
        mae_scores = []
        mape_scores = []
        r2_scores = []
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            logger.info(
                f"Fold {fold}/{cv}: train_size={len(train_idx)}, "
                f"val_size={len(val_idx)}"
            )
            
            # Create and train a fresh model for this fold
            fold_model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective=self.objective,
                random_state=42,
                n_jobs=-1
            )
            
            fold_model.fit(X_train, y_train, verbose=False)
            
            # Predict on validation set
            y_pred = fold_model.predict(X_val)
            
            # Compute metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            
            # MAPE: handle zero values to avoid division by zero
            # Use epsilon to avoid inf when y_val is zero
            epsilon = 1e-10
            mape = np.mean(np.abs((y_val - y_pred) / (np.abs(y_val) + epsilon))) * 100
            
            r2 = r2_score(y_val, y_pred)
            
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            mape_scores.append(mape)
            r2_scores.append(r2)
            
            logger.info(
                f"Fold {fold} metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, "
                f"MAPE={mape:.2f}%, R2={r2:.4f}"
            )
        
        # Compute average metrics
        avg_metrics = {
            'rmse': float(np.mean(rmse_scores)),
            'mae': float(np.mean(mae_scores)),
            'mape': float(np.mean(mape_scores)),
            'r2': float(np.mean(r2_scores))
        }
        
        logger.info(
            f"Cross-validation results: RMSE={avg_metrics['rmse']:.2f}, "
            f"MAE={avg_metrics['mae']:.2f}, MAPE={avg_metrics['mape']:.2f}%, "
            f"R2={avg_metrics['r2']:.4f}"
        )
        
        return avg_metrics
    
    def get_feature_importance(
        self,
        importance_type: str = "gain"
    ) -> pd.DataFrame:
        """Get feature importance scores.
        
        Feature importance helps understand which features contribute most to
        residual predictions. This is crucial for model interpretability and
        feature engineering.
        
        Args:
            importance_type: Type of importance metric:
                - 'gain': Average gain across all splits using the feature
                - 'weight': Number of times feature appears in trees
                - 'cover': Average coverage (samples affected) by the feature
                - 'total_gain': Total gain across all splits
                - 'total_cover': Total coverage across all splits
        
        Returns:
            DataFrame with columns [feature, importance], sorted by importance
            in descending order
        
        Raises:
            RuntimeError: If model hasn't been fitted yet
            ValueError: If importance_type is invalid
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before extracting feature importance. "
                "Call fit() first."
            )
        
        valid_types = ['gain', 'weight', 'cover', 'total_gain', 'total_cover']
        if importance_type not in valid_types:
            raise ValueError(
                f"importance_type must be one of {valid_types}, "
                f"got '{importance_type}'"
            )
        
        logger.info(f"Extracting feature importance (type: {importance_type})")
        
        # Get importance scores from XGBoost
        importance_dict = self.model.get_booster().get_score(
            importance_type=importance_type
        )
        
        if not importance_dict:
            logger.warning("No feature importance scores available")
            return pd.DataFrame(columns=['feature', 'importance'])
        
        # Convert to DataFrame
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': score}
            for feature, score in importance_dict.items()
        ])
        
        # Sort by importance (descending)
        importance_df = importance_df.sort_values(
            'importance', ascending=False
        ).reset_index(drop=True)
        
        logger.info(
            f"Top 5 features: {list(importance_df['feature'].head(5))}"
        )
        
        return importance_df
    
    def get_hyperparameters(self) -> Dict[str, any]:
        """Get the current hyperparameters of the model.
        
        Returns:
            Dictionary with hyperparameter names and values
        """
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'objective': self.objective
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"MLModel(n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"learning_rate={self.learning_rate}, "
            f"objective='{self.objective}', "
            f"status='{status}')"
        )
