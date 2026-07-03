"""
Model Manager using MLflow for model lifecycle management.

This module implements model versioning, deployment, and rollback capabilities
using MLflow's tracking and model registry features.
"""

from typing import Any, Dict, List, Optional
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from loguru import logger
import os


class ModelManager:
    """Manages model lifecycle using MLflow.
    
    This class provides a unified interface for:
    - Logging models with metrics and parameters
    - Registering models in the Model Registry
    - Loading models from the registry
    - Promoting models between stages (Staging -> Production)
    - Rolling back to previous model versions
    - Listing model version history
    
    Attributes:
        tracking_uri: MLflow tracking server URI
        registry_uri: MLflow model registry URI
        client: MLflow client instance
    """
    
    def __init__(self, tracking_uri: str, registry_uri: str):
        """Initialize MLflow client.
        
        Args:
            tracking_uri: URI of the MLflow tracking server.
                Examples:
                - Local: "file:///path/to/mlruns"
                - Remote: "http://mlflow-server:5000"
                - Databricks: "databricks"
            registry_uri: URI of the MLflow model registry.
                Can be same as tracking_uri or separate.
                Examples:
                - Local: "sqlite:///path/to/mlflow.db"
                - Remote: "postgresql://user:pass@host:5432/mlflow"
        """
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        
        # Set MLflow URIs
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)
        
        # Initialize MLflow client
        self.client = MlflowClient(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri
        )
        
        logger.info(
            f"ModelManager initialized with tracking_uri={tracking_uri}, "
            f"registry_uri={registry_uri}"
        )
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        artifacts: Optional[Dict[str, str]] = None
    ) -> str:
        """Log model to MLflow with metrics and parameters.
        
        This method creates a new MLflow run, logs the model along with its
        performance metrics, hyperparameters, and any additional artifacts.
        
        Args:
            model: The trained model object (Prophet, XGBoost, or FinBERT)
            model_name: Name for the model (e.g., "cocoa_prophet_baseline")
            model_type: Type of model - one of ["prophet", "xgboost", "finbert"]
            metrics: Dictionary of performance metrics to log.
                Examples: {"rmse": 45.2, "mae": 32.1, "mape": 2.5}
            params: Dictionary of model hyperparameters to log.
                Examples: {"n_estimators": 100, "max_depth": 6}
            artifacts: Optional dictionary of artifact paths to log.
                Keys are artifact names, values are local file paths.
                Examples: {"feature_importance": "plots/importance.png"}
        
        Returns:
            Model version string (e.g., "1", "2", "3")
        
        Raises:
            ValueError: If model_type is not recognized
        """
        # Validate model_type
        valid_types = ["prophet", "xgboost", "finbert"]
        if model_type not in valid_types:
            raise ValueError(
                f"model_type must be one of {valid_types}, got '{model_type}'"
            )
        
        logger.info(
            f"Logging {model_type} model '{model_name}' with "
            f"{len(metrics)} metrics and {len(params)} parameters"
        )
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_{model_type}") as run:
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model type as a tag
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("model_name", model_name)
            
            # Log the model based on type
            if model_type == "prophet":
                # Prophet models are logged using mlflow.prophet
                mlflow.prophet.log_model(
                    pr_model=model,
                    artifact_path="model"
                )
            elif model_type == "xgboost":
                # XGBoost models are logged using mlflow.xgboost
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="model"
                )
            elif model_type == "finbert":
                # FinBERT (transformers) models are logged using mlflow.transformers
                # Note: This requires the model to be in HuggingFace format
                mlflow.transformers.log_model(
                    transformers_model=model,
                    artifact_path="model"
                )
            
            # Log additional artifacts if provided
            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    if os.path.exists(artifact_path):
                        mlflow.log_artifact(artifact_path, artifact_name)
                        logger.info(f"Logged artifact: {artifact_name}")
                    else:
                        logger.warning(
                            f"Artifact path does not exist: {artifact_path}"
                        )
            
            # Get run ID for model URI
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"
            
            logger.info(
                f"Model logged successfully. Run ID: {run_id}, "
                f"Model URI: {model_uri}"
            )
        
        # Register the model (this returns the version)
        model_version = self.register_model(
            model_uri=model_uri,
            model_name=model_name,
            stage="None"  # Start in no stage, will be promoted later
        )
        
        return model_version.version
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        stage: str = "Staging"
    ) -> ModelVersion:
        """Register model in MLflow Model Registry.
        
        This method registers a logged model in the Model Registry, making it
        available for deployment and version management.
        
        Args:
            model_uri: URI of the logged model.
                Format: "runs:/<run_id>/model" or "models:/<name>/<version>"
            model_name: Name to register the model under.
                This is the registry name, can be different from logging name.
            stage: Initial stage for the model. One of:
                - "None": No stage (default for new models)
                - "Staging": For models being tested
                - "Production": For deployed models
                - "Archived": For deprecated models
        
        Returns:
            ModelVersion object containing version info
        
        Raises:
            ValueError: If stage is invalid
        """
        # Validate stage
        valid_stages = ["None", "Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ValueError(
                f"stage must be one of {valid_stages}, got '{stage}'"
            )
        
        logger.info(
            f"Registering model '{model_name}' from URI: {model_uri} "
            f"with stage: {stage}"
        )
        
        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        # Transition to the specified stage if not "None"
        if stage != "None":
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
            logger.info(
                f"Model version {model_version.version} transitioned to {stage}"
            )
        
        logger.info(
            f"Model registered successfully. Name: {model_name}, "
            f"Version: {model_version.version}"
        )
        
        return model_version
    
    def load_model(
        self,
        model_name: str,
        stage: str = "Production"
    ) -> Any:
        """Load model from registry by stage.
        
        This method loads a model from the Model Registry based on its stage.
        Typically used to load the current production model for inference.
        
        Args:
            model_name: Name of the registered model
            stage: Stage to load from. One of:
                - "Staging": Load the staging model
                - "Production": Load the production model (default)
                - "Archived": Load an archived model
        
        Returns:
            The loaded model object (Prophet, XGBoost, or FinBERT)
        
        Raises:
            ValueError: If stage is invalid or no model exists in that stage
        """
        # Validate stage
        valid_stages = ["Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ValueError(
                f"stage must be one of {valid_stages}, got '{stage}'"
            )
        
        logger.info(f"Loading model '{model_name}' from stage: {stage}")
        
        # Construct model URI
        model_uri = f"models:/{model_name}/{stage}"
        
        try:
            # Load the model using MLflow's generic load_model
            # This automatically detects the model flavor (prophet, xgboost, etc.)
            model = mlflow.pyfunc.load_model(model_uri)
            
            logger.info(
                f"Model loaded successfully from {stage} stage. "
                f"Model URI: {model_uri}"
            )
            
            return model
        except Exception as e:
            logger.error(
                f"Failed to load model '{model_name}' from {stage} stage: {e}"
            )
            raise ValueError(
                f"No model found for '{model_name}' in {stage} stage"
            ) from e
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        from_stage: str = "Staging",
        to_stage: str = "Production"
    ) -> None:
        """Promote model from one stage to another.
        
        This method transitions a model version from one stage to another,
        typically from Staging to Production after validation.
        
        Args:
            model_name: Name of the registered model
            version: Version number to promote (e.g., "1", "2", "3")
            from_stage: Current stage of the model (for validation)
            to_stage: Target stage to promote to
        
        Raises:
            ValueError: If stages are invalid or model is not in from_stage
        """
        # Validate stages
        valid_stages = ["None", "Staging", "Production", "Archived"]
        if from_stage not in valid_stages or to_stage not in valid_stages:
            raise ValueError(
                f"Stages must be one of {valid_stages}. "
                f"Got from_stage='{from_stage}', to_stage='{to_stage}'"
            )
        
        logger.info(
            f"Promoting model '{model_name}' version {version} "
            f"from {from_stage} to {to_stage}"
        )
        
        # Get model version details
        try:
            model_version = self.client.get_model_version(
                name=model_name,
                version=version
            )
        except Exception as e:
            logger.error(
                f"Failed to get model version {version} for '{model_name}': {e}"
            )
            raise ValueError(
                f"Model version {version} not found for '{model_name}'"
            ) from e
        
        # Verify current stage matches from_stage
        if model_version.current_stage != from_stage:
            logger.warning(
                f"Model version {version} is in stage '{model_version.current_stage}', "
                f"not '{from_stage}' as expected. Proceeding with promotion anyway."
            )
        
        # Archive any existing models in the target stage
        if to_stage == "Production":
            # Get all versions in Production stage
            production_versions = self.client.get_latest_versions(
                name=model_name,
                stages=["Production"]
            )
            
            for prod_version in production_versions:
                logger.info(
                    f"Archiving existing Production model version "
                    f"{prod_version.version}"
                )
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=prod_version.version,
                    stage="Archived"
                )
        
        # Transition to new stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=to_stage
        )
        
        logger.info(
            f"Model version {version} successfully promoted to {to_stage}"
        )
    
    def rollback_model(
        self,
        model_name: str,
        to_version: str
    ) -> None:
        """Rollback to a previous model version.
        
        This method promotes a previous model version back to Production,
        effectively rolling back to a known good state. The current Production
        model is archived.
        
        Args:
            model_name: Name of the registered model
            to_version: Version number to rollback to (e.g., "1", "2", "3")
        
        Raises:
            ValueError: If version doesn't exist
        """
        logger.info(
            f"Rolling back model '{model_name}' to version {to_version}"
        )
        
        # Verify the target version exists
        try:
            target_version = self.client.get_model_version(
                name=model_name,
                version=to_version
            )
        except Exception as e:
            logger.error(
                f"Failed to get model version {to_version} for '{model_name}': {e}"
            )
            raise ValueError(
                f"Model version {to_version} not found for '{model_name}'"
            ) from e
        
        # Get current Production version(s)
        production_versions = self.client.get_latest_versions(
            name=model_name,
            stages=["Production"]
        )
        
        # Archive current Production version(s)
        for prod_version in production_versions:
            logger.info(
                f"Archiving current Production model version "
                f"{prod_version.version}"
            )
            self.client.transition_model_version_stage(
                name=model_name,
                version=prod_version.version,
                stage="Archived"
            )
        
        # Promote target version to Production
        self.client.transition_model_version_stage(
            name=model_name,
            version=to_version,
            stage="Production"
        )
        
        logger.info(
            f"Successfully rolled back to version {to_version}. "
            f"Previous stage: {target_version.current_stage}"
        )
    
    def list_model_versions(
        self,
        model_name: str,
        max_results: int = 5
    ) -> List[ModelVersion]:
        """List recent model versions.
        
        This method retrieves the most recent versions of a registered model,
        useful for viewing version history and selecting versions for rollback.
        
        Args:
            model_name: Name of the registered model
            max_results: Maximum number of versions to return (default: 5)
        
        Returns:
            List of ModelVersion objects, sorted by version number (descending)
        
        Raises:
            ValueError: If model doesn't exist
        """
        logger.info(
            f"Listing up to {max_results} versions for model '{model_name}'"
        )
        
        try:
            # Search for all versions of the model
            versions = self.client.search_model_versions(
                filter_string=f"name='{model_name}'"
            )
            
            if not versions:
                logger.warning(f"No versions found for model '{model_name}'")
                return []
            
            # Sort by version number (descending)
            versions_sorted = sorted(
                versions,
                key=lambda v: int(v.version),
                reverse=True
            )
            
            # Limit to max_results
            result = versions_sorted[:max_results]
            
            logger.info(
                f"Found {len(versions)} total versions, returning {len(result)}"
            )
            
            # Log version details
            for version in result:
                logger.info(
                    f"  Version {version.version}: "
                    f"stage={version.current_stage}, "
                    f"created={version.creation_timestamp}"
                )
            
            return result
        
        except Exception as e:
            logger.error(
                f"Failed to list versions for model '{model_name}': {e}"
            )
            raise ValueError(
                f"Model '{model_name}' not found in registry"
            ) from e
    
    def get_model_info(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed information about a model version.
        
        Args:
            model_name: Name of the registered model
            version: Specific version to get info for (optional)
            stage: Stage to get info for (optional, used if version not provided)
        
        Returns:
            Dictionary with model information including metrics, params, and metadata
        
        Raises:
            ValueError: If neither version nor stage is provided, or model not found
        """
        if version is None and stage is None:
            raise ValueError("Must provide either version or stage")
        
        logger.info(
            f"Getting info for model '{model_name}' "
            f"(version={version}, stage={stage})"
        )
        
        try:
            if version:
                # Get specific version
                model_version = self.client.get_model_version(
                    name=model_name,
                    version=version
                )
            else:
                # Get latest version in stage
                versions = self.client.get_latest_versions(
                    name=model_name,
                    stages=[stage]
                )
                if not versions:
                    raise ValueError(
                        f"No model found in {stage} stage for '{model_name}'"
                    )
                model_version = versions[0]
            
            # Get run details to fetch metrics and params
            run = self.client.get_run(model_version.run_id)
            
            info = {
                'name': model_version.name,
                'version': model_version.version,
                'stage': model_version.current_stage,
                'run_id': model_version.run_id,
                'creation_timestamp': model_version.creation_timestamp,
                'last_updated_timestamp': model_version.last_updated_timestamp,
                'description': model_version.description,
                'metrics': run.data.metrics,
                'params': run.data.params,
                'tags': run.data.tags
            }
            
            logger.info(
                f"Retrieved info for model version {info['version']} "
                f"in stage {info['stage']}"
            )
            
            return info
        
        except Exception as e:
            logger.error(
                f"Failed to get model info for '{model_name}': {e}"
            )
            raise ValueError(
                f"Could not retrieve model info for '{model_name}'"
            ) from e
    
    def __repr__(self) -> str:
        """String representation of the ModelManager."""
        return (
            f"ModelManager(tracking_uri='{self.tracking_uri}', "
            f"registry_uri='{self.registry_uri}')"
        )
