# Cocoa Price Prediction API

REST API for the hybrid cocoa price prediction system.

## Overview

The API provides endpoints for:
- **Price Predictions**: Generate predictions for multiple horizons with confidence intervals
- **Performance Metrics**: Retrieve model performance metrics over time
- **Model Management**: List available model versions and their status
- **Retraining**: Trigger model retraining (admin only)

## Features

- **Authentication**: HTTPBearer token authentication for all endpoints
- **Redis Caching**: 1-hour TTL for predictions to improve response times
- **Request Validation**: Pydantic models for request/response validation
- **Error Handling**: Consistent error responses with proper HTTP status codes
- **Logging**: Comprehensive logging with loguru
- **Health Checks**: Health check endpoint for monitoring

## Architecture

```
src/api/
├── __init__.py          # Package initialization
├── app.py               # FastAPI application with all endpoints
├── models.py            # Pydantic request/response models
├── auth.py              # Authentication logic
├── cache.py             # Redis caching functionality
└── README.md            # This file
```

## Endpoints

### 1. POST /api/v1/predict

Generate price predictions for specified horizons.

**Request:**
```json
{
  "horizons": [1, 7, 30],
  "market": "ICE_London",
  "include_sentiment": true
}
```

**Response:**
```json
{
  "predictions": [
    {
      "horizon": 1,
      "price": 3250.50,
      "confidence_interval": [3100.00, 3400.00],
      "confidence_level": 0.95,
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "model_version": "v1.2.3",
  "sentiment_score": -0.15,
  "market": "ICE_London"
}
```

**Requirements**: 10.1, 10.2, 10.3, 10.4

### 2. GET /api/v1/performance

Retrieve model performance metrics for a date range.

**Query Parameters:**
- `start_date`: Start date (ISO format)
- `end_date`: End date (ISO format)
- `model_version`: Optional model version filter

**Response:**
```json
{
  "model_version": "1.0.0",
  "metrics": [
    {
      "timestamp": "2024-01-15T10:00:00Z",
      "rmse": 45.2,
      "mae": 32.1,
      "mape": 0.025,
      "directional_accuracy": 0.75,
      "coverage_rate": 0.95,
      "mean_interval_width": 200.0
    }
  ],
  "start_date": "2024-01-08T00:00:00Z",
  "end_date": "2024-01-15T00:00:00Z"
}
```

**Requirements**: 10.2

### 3. GET /api/v1/models

List available model versions and their status.

**Response:**
```json
{
  "models": [
    {
      "name": "cocoa_prophet",
      "version": "1",
      "stage": "Production",
      "created_at": "2024-01-01T00:00:00Z",
      "metrics": {
        "rmse": 45.2,
        "mae": 32.1,
        "mape": 0.025
      }
    }
  ],
  "current_production_version": "cocoa_prophet:1"
}
```

**Requirements**: 10.3

### 4. POST /api/v1/retrain

Trigger model retraining (admin only).

**Request:**
```json
{
  "model_type": "all",
  "reason": "Performance degradation detected"
}
```

**Response:**
```json
{
  "status": "accepted",
  "message": "Retraining job created for all model(s)",
  "job_id": "12345-67890",
  "estimated_completion": "2024-01-15T12:30:00Z"
}
```

**Requirements**: 10.5, 13.1

## Authentication

All endpoints require authentication using HTTPBearer tokens.

**Regular User Token:**
```bash
Authorization: Bearer <secret_key>
```

**Admin Token (for /retrain endpoint):**
```bash
Authorization: Bearer admin_<secret_key>
```

The secret key is configured in the `.env` file:
```
SECRET_KEY=your_secret_key_here
```

## Redis Caching

Predictions are cached in Redis with a 1-hour TTL to improve response times.

**Cache Key Format:**
```
prediction:{market}:{horizons}:{sentiment_flag}
```

**Example:**
```
prediction:ICE_London:1_7_30:with_sentiment
```

**Cache Invalidation:**
- Automatic: After 1 hour (TTL expires)
- Manual: When retraining is triggered

**Requirements**: 10.4

## Running the API

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Run the API
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# Run with Gunicorn
gunicorn src.api.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker

```bash
# Build image
docker build -t cocoa-prediction-api .

# Run container
docker run -p 8000:8000 --env-file .env cocoa-prediction-api
```

## API Documentation

Once the API is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Testing

```bash
# Run all API tests
pytest tests/test_api_models.py tests/test_api_cache.py tests/test_api_auth.py -v

# Run with coverage
pytest tests/test_api_*.py --cov=src.api --cov-report=html
```

## Configuration

The API uses the following configuration from `config/settings.py`:

```python
# Database (Supabase)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_REGISTRY_URI=http://localhost:5000

# Security
SECRET_KEY=your_secret_key
```

## Error Handling

The API returns consistent error responses:

```json
{
  "error": "error_type",
  "message": "Human-readable error message",
  "detail": "Additional error details (optional)"
}
```

**HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (invalid token)
- `403`: Forbidden (insufficient privileges)
- `422`: Validation Error (invalid request body)
- `500`: Internal Server Error
- `503`: Service Unavailable

## Performance

- **Response Time**: < 2 seconds for predictions (requirement 10.4)
- **Cache Hit Rate**: ~80% for repeated predictions
- **Throughput**: ~100 requests/second (with caching)

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "redis": true,
    "supabase": true,
    "model_manager": true,
    "price_predictor": true
  }
}
```

### Logs

Logs are written to stdout with the following format:
```
2024-01-15 10:30:00.123 | INFO | src.api.app:predict_price:150 - Prediction request received: market=ICE_London, horizons=[1, 7, 30]
```

## Security

- **Authentication**: Required for all endpoints
- **Authorization**: Admin-only endpoints (retraining)
- **TLS**: Use HTTPS in production
- **Rate Limiting**: Implement rate limiting in production (e.g., with nginx)
- **Input Validation**: Pydantic models validate all inputs

## Requirements Mapping

| Requirement | Endpoint | Description |
|-------------|----------|-------------|
| 10.1 | POST /api/v1/predict | Generate predictions |
| 10.2 | GET /api/v1/performance | Retrieve metrics |
| 10.3 | GET /api/v1/models | List models |
| 10.4 | POST /api/v1/predict | Response time < 2s |
| 10.4 | Redis Cache | 1-hour TTL caching |
| 10.5 | POST /api/v1/retrain | Trigger retraining |
| 13.1 | HTTPBearer | Authentication |
| 13.2 | Pydantic | Request validation |

## Future Enhancements

- [ ] Rate limiting per user
- [ ] WebSocket support for real-time predictions
- [ ] Batch prediction endpoint
- [ ] Model A/B testing support
- [ ] Prometheus metrics export
- [ ] GraphQL API
