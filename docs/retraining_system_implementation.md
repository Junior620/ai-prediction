# Retraining System Implementation

## Overview

The Retraining System provides automatic model retraining capabilities for the Cocoa Price Prediction System. It ensures models stay accurate as market conditions evolve by periodically retraining with new data and validating performance before deployment.

## Architecture

### Components

1. **RetrainingManager**: Orchestrates the complete retraining workflow
2. **ModelManager**: Handles model versioning and deployment via MLflow
3. **PerformanceMonitor**: Validates and compares model performance
4. **DataPreprocessor**: Prepares training data

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                   Retraining Workflow                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ Check Triggers   │
                  │ - Time-based     │
                  │ - Data-based     │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Fetch Training   │
                  │ Data from DB     │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Preprocess &     │
                  │ Split Data       │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Train Prophet    │
                  │ Model            │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Train XGBoost    │
                  │ on Residuals     │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Validate on      │
                  │ Validation Set   │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Compare with     │
                  │ Current Model    │
                  └────────┬─────────┘
                           │
                  ┌────────┴─────────┐
                  │                  │
                  ▼                  ▼
         ┌────────────────┐  ┌────────────────┐
         │ New Model      │  │ Retain Current │
         │ Better         │  │ Model          │
         └───────┬────────┘  └────────────────┘
                 │
                 ▼
         ┌────────────────┐
         │ Promote to     │
         │ Production     │
         └───────┬────────┘
                 │
                 ▼
         ┌────────────────┐
         │ Cleanup Old    │
         │ Versions       │
         └────────────────┘
```

## Features

### 1. Automatic Retraining Triggers

#### Time-Based Trigger
- **Requirement**: 9.1 - Retrain at least monthly
- **Implementation**: Checks if `retraining_frequency_days` (default: 30) have passed since last training
- **Configuration**: Adjustable via `retraining_frequency_days` parameter

#### Data-Based Trigger
- **Requirement**: 9.2 - Trigger when 30 days of new data available
- **Implementation**: Counts unique dates in `price_data` table since last training
- **Configuration**: Adjustable via `min_new_data_days` parameter

### 2. Model Validation

**Requirement**: 9.3 - Validate retrained models on recent validation set

The system:
1. Splits data chronologically (80% train, 20% validation)
2. Trains new models on training set
3. Generates predictions on validation set
4. Computes comprehensive metrics:
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - MAPE (Mean Absolute Percentage Error)
   - Directional Accuracy
   - Coverage Rate
   - Mean Interval Width

### 3. Model Comparison and Retention

**Requirement**: 9.4 - Retain current model if new model performs worse

The system:
1. Loads current production model
2. Validates both models on same validation set
3. Compares using weighted scoring:
   - Error metrics (RMSE, MAE, MAPE): 40% weight
   - Directional accuracy: 30% weight
   - Coverage rate: 20% weight
4. Promotes new model only if it performs better
5. Retains current model otherwise

### 4. Version History Management

**Requirement**: 9.5 - Maintain history of 5 most recent versions

The system:
1. Logs all models to MLflow with version numbers
2. Tracks model metadata (metrics, parameters, timestamps)
3. Automatically archives versions beyond the limit
4. Maintains configurable number of recent versions (default: 5)

## Usage

### Basic Usage

```python
from src.models.retraining_manager import RetrainingManager
from src.models.model_manager import ModelManager
from src.monitoring.performance_monitor import PerformanceMonitor
from src.data_preprocessing.data_preprocessor import DataPreprocessor

# Initialize dependencies
model_manager = ModelManager(
    tracking_uri="http://localhost:5000",
    registry_uri="sqlite:///mlflow.db"
)

performance_monitor = PerformanceMonitor()
data_preprocessor = DataPreprocessor()

# Initialize retraining manager
retraining_manager = RetrainingManager(
    model_manager=model_manager,
    performance_monitor=performance_monitor,
    data_preprocessor=data_preprocessor,
    retraining_frequency_days=30,
    min_new_data_days=30,
    max_model_versions=5
)

# Check if retraining should be triggered
should_retrain, reason = retraining_manager.should_trigger_retraining()

if should_retrain:
    # Execute retraining
    success, message, new_version = retraining_manager.retrain_models()
    
    if success:
        print(f"Retraining successful! New version: {new_version}")
    else:
        print(f"Retraining failed: {message}")
```

### Automatic Retraining Scheduler

```python
import schedule
import time

def automatic_retraining_job():
    """Job to check and execute retraining if needed."""
    should_retrain, reason = retraining_manager.should_trigger_retraining()
    
    if should_retrain:
        print(f"Triggering retraining: {reason}")
        success, message, new_version = retraining_manager.retrain_models()
        
        if success:
            print(f"Retraining completed: {new_version}")
        else:
            print(f"Retraining failed: {message}")

# Schedule to run daily at 2 AM
schedule.every().day.at("02:00").do(automatic_retraining_job)

while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

### Manual Retraining with Custom Hyperparameters

```python
# Define custom hyperparameters
hyperparameters = {
    "prophet": {
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": True,
        "weekly_seasonality": False,
        "changepoint_prior_scale": 0.05
    },
    "xgboost": {
        "n_estimators": 150,
        "max_depth": 8,
        "learning_rate": 0.05,
        "objective": "reg:squarederror"
    }
}

# Execute retraining
success, message, new_version = retraining_manager.retrain_models(
    model_name="cocoa_price_predictor",
    validation_split=0.2,
    hyperparameters=hyperparameters
)
```

### Get Retraining Status

```python
# Get current retraining status
status = retraining_manager.get_retraining_status()

print(f"Has model: {status['has_model']}")
print(f"Latest version: {status.get('latest_version', 'N/A')}")
print(f"Days since training: {status.get('days_since_training', 'N/A')}")
print(f"New data days: {status.get('new_data_days', 'N/A')}")
print(f"Should retrain: {status['should_retrain']}")
print(f"Reason: {status['reason']}")
```

## Configuration

### Environment Variables

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_REGISTRY_URI=sqlite:///mlflow.db

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-key
```

### Configuration File (config.yaml)

```yaml
retraining:
  frequency_days: 30
  min_new_data_days: 30
  max_model_versions: 5
  validation_split: 0.2
  
  hyperparameters:
    prophet:
      seasonality_mode: multiplicative
      yearly_seasonality: true
      weekly_seasonality: false
      changepoint_prior_scale: 0.05
    
    xgboost:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
      objective: reg:squarederror
```

## API Integration

### FastAPI Endpoint for Manual Retraining

```python
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter()

class RetrainingRequest(BaseModel):
    model_name: str = "cocoa_price_predictor"
    validation_split: float = 0.2
    hyperparameters: dict = None

@router.post("/api/v1/retrain")
async def trigger_retraining(
    request: RetrainingRequest,
    # Add authentication dependency here
):
    """Trigger manual model retraining."""
    try:
        success, message, new_version = retraining_manager.retrain_models(
            model_name=request.model_name,
            validation_split=request.validation_split,
            hyperparameters=request.hyperparameters
        )
        
        return {
            "success": success,
            "message": message,
            "new_version": new_version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/v1/retraining/status")
async def get_retraining_status():
    """Get current retraining status."""
    try:
        status = retraining_manager.get_retraining_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Monitoring and Alerts

### Performance Degradation Detection

```python
# Check for performance degradation
current_metrics = performance_monitor.compute_metrics(
    y_true=actual_prices,
    y_pred=predicted_prices,
    y_pred_lower=lower_bounds,
    y_pred_upper=upper_bounds
)

baseline_metrics = performance_monitor.get_baseline_metrics(
    model_version="current_version"
)

degradation_detected = performance_monitor.detect_degradation(
    current_metrics=current_metrics,
    baseline_metrics=baseline_metrics,
    threshold=0.15  # 15% degradation threshold
)

if degradation_detected:
    # Trigger retraining alert
    performance_monitor.trigger_retraining_alert(
        reason="performance_degradation",
        metrics=current_metrics
    )
    
    # Execute retraining
    retraining_manager.retrain_models()
```

## Testing

### Unit Tests

Run the comprehensive test suite:

```bash
pytest tests/test_retraining_manager.py -v
```

Test coverage includes:
- Initialization validation
- Trigger logic (time-based and data-based)
- Model validation
- Model comparison and promotion
- Version history management
- Error handling

### Integration Tests

```python
# Test complete retraining workflow
def test_complete_retraining_workflow():
    # Setup test data
    # Execute retraining
    # Verify new model is logged
    # Verify promotion logic
    # Verify old versions are archived
    pass
```

## Best Practices

1. **Schedule Regular Checks**: Run retraining checks daily, but actual retraining only when triggered
2. **Monitor Performance**: Track model metrics continuously to detect degradation early
3. **Validate Before Promotion**: Always validate new models on recent data before promoting
4. **Maintain Version History**: Keep at least 5 recent versions for rollback capability
5. **Log Everything**: Use comprehensive logging for debugging and auditing
6. **Handle Failures Gracefully**: Implement fallback mechanisms if retraining fails
7. **Test Hyperparameters**: Experiment with different hyperparameters during retraining
8. **Alert on Failures**: Send notifications when retraining fails or degradation is detected

## Troubleshooting

### Common Issues

1. **No data available for training**
   - Check database connectivity
   - Verify data collection is running
   - Check date ranges in queries

2. **Retraining fails during model training**
   - Check data quality and completeness
   - Verify hyperparameters are valid
   - Check for sufficient training data

3. **New model not promoted**
   - Review validation metrics comparison
   - Check if new model actually performs better
   - Verify comparison logic is working correctly

4. **Version cleanup not working**
   - Check MLflow connectivity
   - Verify model registry permissions
   - Check version count logic

## Performance Considerations

- **Training Time**: Retraining can take 10-30 minutes depending on data size
- **Database Load**: Fetching training data can be intensive, consider off-peak scheduling
- **MLflow Storage**: Monitor storage usage as model versions accumulate
- **Validation Overhead**: Validation adds ~20% to total retraining time

## Future Enhancements

1. **Hyperparameter Tuning**: Automatic hyperparameter optimization during retraining
2. **A/B Testing**: Deploy new models to subset of users before full promotion
3. **Ensemble Models**: Combine multiple model versions for improved predictions
4. **Distributed Training**: Scale training across multiple machines for large datasets
5. **Real-time Monitoring**: Dashboard for tracking retraining status and metrics
6. **Automated Rollback**: Automatic rollback if promoted model underperforms

## References

- Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
- Design Document: Section on Model Retraining
- MLflow Documentation: https://mlflow.org/docs/latest/model-registry.html
- Prophet Documentation: https://facebook.github.io/prophet/
- XGBoost Documentation: https://xgboost.readthedocs.io/
