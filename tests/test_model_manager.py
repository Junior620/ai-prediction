"""
Unit tests for ModelManager class.

Tests cover:
- Initialization
- Model logging with metrics and parameters
- Model registration in the registry
- Model loading from registry
- Model promotion between stages
- Model rollback to previous versions
- Listing model versions
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.models import ModelManager, TimeSeriesModel, MLModel


@pytest.fixture
def temp_mlflow_dir():
    """Create temporary directory for MLflow tracking."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    try:
        shutil.rmtree(temp_dir)
    except OSError:
        # Windows may still lock MLflow DB/files briefly after the test
        pass


@pytest.fixture
def model_manager(temp_mlflow_dir):
    """Create ModelManager instance with temporary storage."""
    import mlflow

    mlruns_dir = Path(temp_mlflow_dir) / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = mlruns_dir.as_uri()
    registry_uri = f"sqlite:///{Path(temp_mlflow_dir) / 'mlflow.db'}"

    manager = ModelManager(
        tracking_uri=tracking_uri,
        registry_uri=registry_uri
    )
    # Fresh file stores have no Default experiment (id 0); set one explicitly.
    mlflow.set_experiment("test_experiment")

    return manager


@pytest.fixture
def sample_prophet_model():
    """Create a trained Prophet model for testing."""
    # Create synthetic data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    prices = 3000 + 200 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + \
             np.random.normal(0, 50, len(dates))
    
    df = pd.DataFrame({
        'ds': dates,
        'y': prices
    })
    
    # Train model
    model = TimeSeriesModel(
        seasonality_mode='multiplicative',
        yearly_seasonality=True
    )
    model.fit(df)
    
    return model


@pytest.fixture
def sample_xgboost_model():
    """Create a trained XGBoost model for testing."""
    # Create synthetic data
    n_samples = 500
    X = pd.DataFrame({
        'temperature': np.random.uniform(20, 35, n_samples),
        'rainfall': np.random.uniform(0, 200, n_samples),
        'sentiment_score': np.random.uniform(-1, 1, n_samples)
    })
    
    y = pd.Series(
        10 * X['temperature'] - 0.05 * X['rainfall'] + 
        2 * X['sentiment_score'] + np.random.normal(0, 20, n_samples)
    )
    
    # Train model
    model = MLModel(n_estimators=50, max_depth=4)
    model.fit(X, y)
    if not hasattr(model.model, "_estimator_type"):
        model.model._estimator_type = "regressor"
    
    return model


class TestModelManagerInitialization:
    """Test ModelManager initialization."""
    
    def test_initialization_with_valid_uris(self, temp_mlflow_dir):
        """Test initialization with valid URIs."""
        import mlflow

        mlruns_dir = Path(temp_mlflow_dir) / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        tracking_uri = mlruns_dir.as_uri()
        registry_uri = f"sqlite:///{Path(temp_mlflow_dir) / 'mlflow.db'}"

        manager = ModelManager(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri
        )
        mlflow.set_experiment("test_experiment")

        assert manager.tracking_uri == tracking_uri
        assert manager.registry_uri == registry_uri
        assert manager.client is not None
    
    def test_repr(self, model_manager):
        """Test string representation."""
        repr_str = repr(model_manager)
        
        assert "ModelManager" in repr_str
        assert "tracking_uri" in repr_str
        assert "registry_uri" in repr_str


class TestModelLogging:
    """Test model logging functionality."""
    
    def test_log_prophet_model(self, model_manager, sample_prophet_model):
        """Test logging a Prophet model."""
        metrics = {
            'rmse': 45.2,
            'mae': 32.1,
            'mape': 2.5
        }
        
        params = sample_prophet_model.get_hyperparameters()
        
        version = model_manager.log_model(
            model=sample_prophet_model.model,
            model_name="test_prophet",
            model_type="prophet",
            metrics=metrics,
            params=params
        )
        
        assert version is not None
        assert int(version) >= 1
    
    def test_log_xgboost_model(self, model_manager, sample_xgboost_model):
        """Test logging an XGBoost model."""
        metrics = {
            'rmse': 25.3,
            'mae': 18.7,
            'r2': 0.82
        }
        
        params = sample_xgboost_model.get_hyperparameters()
        # Newer sklearn/mlflow expect this attribute on estimators
        if not hasattr(sample_xgboost_model.model, "_estimator_type"):
            sample_xgboost_model.model._estimator_type = "regressor"
        
        version = model_manager.log_model(
            model=sample_xgboost_model.model,
            model_name="test_xgboost",
            model_type="xgboost",
            metrics=metrics,
            params=params
        )
        
        assert version is not None
        assert int(version) >= 1
    
    def test_log_model_with_invalid_type(self, model_manager, sample_prophet_model):
        """Test logging with invalid model type."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            model_manager.log_model(
                model=sample_prophet_model.model,
                model_name="test_invalid",
                model_type="invalid_type",
                metrics={'rmse': 45.2},
                params={}
            )
    
    def test_log_multiple_versions(self, model_manager, sample_prophet_model):
        """Test logging multiple versions of the same model."""
        metrics_v1 = {'rmse': 45.2, 'mae': 32.1}
        metrics_v2 = {'rmse': 38.5, 'mae': 28.3}
        
        params = sample_prophet_model.get_hyperparameters()
        
        version_1 = model_manager.log_model(
            model=sample_prophet_model.model,
            model_name="test_multi_version",
            model_type="prophet",
            metrics=metrics_v1,
            params=params
        )
        
        version_2 = model_manager.log_model(
            model=sample_prophet_model.model,
            model_name="test_multi_version",
            model_type="prophet",
            metrics=metrics_v2,
            params=params
        )
        
        assert version_1 != version_2
        assert int(version_2) > int(version_1)


class TestModelRegistration:
    """Test model registration functionality."""
    
    def test_register_model_with_staging(self, model_manager, sample_prophet_model):
        """Test registering a model in Staging stage."""
        # First log the model
        metrics = {'rmse': 45.2}
        params = sample_prophet_model.get_hyperparameters()
        
        version = model_manager.log_model(
            model=sample_prophet_model.model,
            model_name="test_register_staging",
            model_type="prophet",
            metrics=metrics,
            params=params
        )
        
        # Model is automatically registered, verify it exists
        versions = model_manager.list_model_versions(
            model_name="test_register_staging",
            max_results=1
        )
        
        assert len(versions) >= 1
        assert versions[0].version == version
    
    def test_register_model_with_invalid_stage(self, model_manager):
        """Test registering with invalid stage."""
        with pytest.raises(ValueError, match="stage must be one of"):
            model_manager.register_model(
                model_uri="runs:/fake_run_id/model",
                model_name="test_invalid_stage",
                stage="InvalidStage"
            )


class TestModelLoading:
    """Test model loading functionality."""
    
    def test_load_model_from_production(self, model_manager, sample_prophet_model):
        """Test loading a model from Production stage."""
        # Log and promote model to Production
        metrics = {'rmse': 45.2}
        params = sample_prophet_model.get_hyperparameters()
        
        version = model_manager.log_model(
            model=sample_prophet_model.model,
            model_name="test_load_prod",
            model_type="prophet",
            metrics=metrics,
            params=params
        )
        
        # Promote to Production
        model_manager.promote_model(
            model_name="test_load_prod",
            version=version,
            from_stage="None",
            to_stage="Production"
        )
        
        # Load from Production
        loaded_model = model_manager.load_model(
            model_name="test_load_prod",
            stage="Production"
        )
        
        assert loaded_model is not None
    
    def test_load_model_with_invalid_stage(self, model_manager):
        """Test loading with invalid stage."""
        with pytest.raises(ValueError, match="stage must be one of"):
            model_manager.load_model(
                model_name="test_model",
                stage="InvalidStage"
            )
    
    def test_load_nonexistent_model(self, model_manager):
        """Test loading a model that doesn't exist."""
        with pytest.raises(ValueError, match="No model found"):
            model_manager.load_model(
                model_name="nonexistent_model",
                stage="Production"
            )


class TestModelPromotion:
    """Test model promotion functionality."""
    
    def test_promote_staging_to_production(self, model_manager, sample_prophet_model):
        """Test promoting a model from Staging to Production."""
        # Log model
        metrics = {'rmse': 45.2}
        params = sample_prophet_model.get_hyperparameters()
        
        version = model_manager.log_model(
            model=sample_prophet_model.model,
            model_name="test_promote",
            model_type="prophet",
            metrics=metrics,
            params=params
        )
        
        # Promote to Staging first
        model_manager.promote_model(
            model_name="test_promote",
            version=version,
            from_stage="None",
            to_stage="Staging"
        )
        
        # Then promote to Production
        model_manager.promote_model(
            model_name="test_promote",
            version=version,
            from_stage="Staging",
            to_stage="Production"
        )
        
        # Verify it's in Production
        info = model_manager.get_model_info(
            model_name="test_promote",
            stage="Production"
        )
        
        assert info['version'] == version
        assert info['stage'] == "Production"
    
    def test_promote_with_invalid_stages(self, model_manager):
        """Test promotion with invalid stages."""
        with pytest.raises(ValueError, match="Stages must be one of"):
            model_manager.promote_model(
                model_name="test_model",
                version="1",
                from_stage="InvalidStage",
                to_stage="Production"
            )
    
    def test_promote_nonexistent_version(self, model_manager):
        """Test promoting a version that doesn't exist."""
        with pytest.raises(ValueError, match="Model version .* not found"):
            model_manager.promote_model(
                model_name="nonexistent_model",
                version="999",
                from_stage="Staging",
                to_stage="Production"
            )
    
    def test_promote_archives_existing_production(self, model_manager, sample_prophet_model):
        """Test that promoting to Production archives existing Production model."""
        metrics = {'rmse': 45.2}
        params = sample_prophet_model.get_hyperparameters()
        
        # Log and promote first version
        version_1 = model_manager.log_model(
            model=sample_prophet_model.model,
            model_name="test_archive",
            model_type="prophet",
            metrics=metrics,
            params=params
        )
        
        model_manager.promote_model(
            model_name="test_archive",
            version=version_1,
            from_stage="None",
            to_stage="Production"
        )
        
        # Log and promote second version
        version_2 = model_manager.log_model(
            model=sample_prophet_model.model,
            model_name="test_archive",
            model_type="prophet",
            metrics={'rmse': 38.5},  # Better metrics
            params=params
        )
        
        model_manager.promote_model(
            model_name="test_archive",
            version=version_2,
            from_stage="None",
            to_stage="Production"
        )
        
        # Verify version 2 is in Production
        info = model_manager.get_model_info(
            model_name="test_archive",
            stage="Production"
        )
        
        assert info['version'] == version_2


class TestModelRollback:
    """Test model rollback functionality."""
    
    def test_rollback_to_previous_version(self, model_manager, sample_prophet_model):
        """Test rolling back to a previous model version."""
        metrics = {'rmse': 45.2}
        params = sample_prophet_model.get_hyperparameters()
        
        # Log version 1 and promote to Production
        version_1 = model_manager.log_model(
            model=sample_prophet_model.model,
            model_name="test_rollback",
            model_type="prophet",
            metrics=metrics,
            params=params
        )
        
        model_manager.promote_model(
            model_name="test_rollback",
            version=version_1,
            from_stage="None",
            to_stage="Production"
        )
        
        # Log version 2 and promote to Production
        version_2 = model_manager.log_model(
            model=sample_prophet_model.model,
            model_name="test_rollback",
            model_type="prophet",
            metrics={'rmse': 55.0},  # Worse metrics
            params=params
        )
        
        model_manager.promote_model(
            model_name="test_rollback",
            version=version_2,
            from_stage="None",
            to_stage="Production"
        )
        
        # Rollback to version 1
        model_manager.rollback_model(
            model_name="test_rollback",
            to_version=version_1
        )
        
        # Verify version 1 is back in Production
        info = model_manager.get_model_info(
            model_name="test_rollback",
            stage="Production"
        )
        
        assert info['version'] == version_1
    
    def test_rollback_to_nonexistent_version(self, model_manager):
        """Test rollback to a version that doesn't exist."""
        with pytest.raises(ValueError, match="Model version .* not found"):
            model_manager.rollback_model(
                model_name="nonexistent_model",
                to_version="999"
            )


class TestListModelVersions:
    """Test listing model versions functionality."""
    
    def test_list_versions(self, model_manager, sample_prophet_model):
        """Test listing model versions."""
        metrics = {'rmse': 45.2}
        params = sample_prophet_model.get_hyperparameters()
        
        # Log multiple versions
        versions_logged = []
        for i in range(3):
            version = model_manager.log_model(
                model=sample_prophet_model.model,
                model_name="test_list_versions",
                model_type="prophet",
                metrics={'rmse': 45.2 - i},
                params=params
            )
            versions_logged.append(version)
        
        # List versions
        versions = model_manager.list_model_versions(
            model_name="test_list_versions",
            max_results=5
        )
        
        assert len(versions) == 3
        # Versions should be sorted in descending order
        assert versions[0].version == versions_logged[-1]
    
    def test_list_versions_with_limit(self, model_manager, sample_prophet_model):
        """Test listing versions with max_results limit."""
        metrics = {'rmse': 45.2}
        params = sample_prophet_model.get_hyperparameters()
        
        # Log 5 versions
        for i in range(5):
            model_manager.log_model(
                model=sample_prophet_model.model,
                model_name="test_list_limit",
                model_type="prophet",
                metrics={'rmse': 45.2 - i},
                params=params
            )
        
        # List only 3 versions
        versions = model_manager.list_model_versions(
            model_name="test_list_limit",
            max_results=3
        )
        
        assert len(versions) == 3
    
    def test_list_versions_nonexistent_model(self, model_manager):
        """Test listing versions for a model that doesn't exist returns empty."""
        versions = model_manager.list_model_versions(
            model_name="nonexistent_model",
            max_results=5
        )
        assert versions == []


class TestGetModelInfo:
    """Test getting model information."""
    
    def test_get_model_info_by_version(self, model_manager, sample_prophet_model):
        """Test getting model info by version number."""
        metrics = {'rmse': 45.2, 'mae': 32.1}
        params = sample_prophet_model.get_hyperparameters()
        
        version = model_manager.log_model(
            model=sample_prophet_model.model,
            model_name="test_get_info",
            model_type="prophet",
            metrics=metrics,
            params=params
        )
        
        info = model_manager.get_model_info(
            model_name="test_get_info",
            version=version
        )
        
        assert info['version'] == version
        assert info['name'] == "test_get_info"
        assert 'rmse' in info['metrics']
        assert float(info['metrics']['rmse']) == 45.2
    
    def test_get_model_info_by_stage(self, model_manager, sample_prophet_model):
        """Test getting model info by stage."""
        metrics = {'rmse': 45.2}
        params = sample_prophet_model.get_hyperparameters()
        
        version = model_manager.log_model(
            model=sample_prophet_model.model,
            model_name="test_get_info_stage",
            model_type="prophet",
            metrics=metrics,
            params=params
        )
        
        # Promote to Production
        model_manager.promote_model(
            model_name="test_get_info_stage",
            version=version,
            from_stage="None",
            to_stage="Production"
        )
        
        info = model_manager.get_model_info(
            model_name="test_get_info_stage",
            stage="Production"
        )
        
        assert info['version'] == version
        assert info['stage'] == "Production"
    
    def test_get_model_info_without_version_or_stage(self, model_manager):
        """Test getting model info without specifying version or stage."""
        with pytest.raises(ValueError, match="Must provide either version or stage"):
            model_manager.get_model_info(
                model_name="test_model"
            )
    
    def test_get_model_info_nonexistent(self, model_manager):
        """Test getting info for a model that doesn't exist."""
        with pytest.raises(ValueError, match="Could not retrieve model info"):
            model_manager.get_model_info(
                model_name="nonexistent_model",
                version="1"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
