# Redis Configuration Guide

## Overview

Redis is used for caching predictions, model metadata, and rate limiting in the Cocoa Price Prediction System.

## Cache Structure

### 1. Prediction Cache (TTL: 1 hour)
```
Key: prediction:{market}:{horizon}
Value: JSON(Prediction)
Example: prediction:ICE_London:7
```

### 2. Model Metadata Cache (TTL: 24 hours)
```
Key: model:metadata:{version}
Value: JSON(ModelMetadata)
Example: model:metadata:v1.2.3
```

### 3. Sentiment Aggregation Cache (TTL: 1 hour)
```
Key: sentiment:aggregate:{timestamp}
Value: float
Example: sentiment:aggregate:2024-01-15T10:00:00Z
```

### 4. Rate Limiting (TTL: 1 minute)
```
Key: ratelimit:{user_id}:{endpoint}
Value: int (request count)
Example: ratelimit:user123:/api/v1/predict
```

## Installation

### Local Development

#### Using Docker (Recommended)
```bash
docker run -d \
  --name redis-cocoa \
  -p 6379:6379 \
  redis:7-alpine
```

#### Using Package Manager

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**macOS:**
```bash
brew install redis
brew services start redis
```

**Windows:**
Download from https://github.com/microsoftarchive/redis/releases

### Production Deployment

For production, consider using managed Redis services:
- **AWS ElastiCache for Redis**
- **Azure Cache for Redis**
- **Google Cloud Memorystore**
- **Redis Cloud**

## Configuration

Update the `.env` file with your Redis connection details:

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-secure-password
```

## Testing Connection

```bash
# Test Redis connection
redis-cli ping
# Expected output: PONG

# Check Redis info
redis-cli info server
```

## Python Client Usage

```python
import redis
from config.settings import settings

# Create Redis client
redis_client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    password=settings.redis_password,
    db=settings.redis_db,
    decode_responses=True
)

# Test connection
redis_client.ping()
```

## Cache Invalidation Strategy

- **Predictions**: Automatically expire after 1 hour
- **Model Metadata**: Invalidate when new model is deployed
- **Sentiment**: Automatically expire after 1 hour
- **Rate Limits**: Reset every minute

## Monitoring

Monitor Redis performance:
```bash
# Monitor commands in real-time
redis-cli monitor

# Check memory usage
redis-cli info memory

# Check connected clients
redis-cli client list
```

## Security Best Practices

1. **Use strong passwords** for Redis authentication
2. **Bind to localhost** in development: `bind 127.0.0.1`
3. **Enable TLS** in production
4. **Disable dangerous commands**: `rename-command FLUSHDB ""`
5. **Set maxmemory policy**: `maxmemory-policy allkeys-lru`

## Troubleshooting

### Connection Refused
```bash
# Check if Redis is running
sudo systemctl status redis-server

# Check port availability
netstat -an | grep 6379
```

### Memory Issues
```bash
# Check memory usage
redis-cli info memory

# Clear all keys (development only!)
redis-cli FLUSHALL
```

### Performance Issues
```bash
# Check slow queries
redis-cli slowlog get 10

# Monitor latency
redis-cli --latency
```
