# Security Configuration Guide

This document provides guidance on configuring security features for the Cocoa Price Prediction System in production environments.

**Implements Requirements 13.1, 13.2, 13.3, 13.4, 13.5**

## Table of Contents

1. [Authentication (JWT)](#authentication-jwt)
2. [Data Encryption at Rest](#data-encryption-at-rest)
3. [TLS 1.3 Configuration](#tls-13-configuration)
4. [Access Logging](#access-logging)
5. [Rate Limiting](#rate-limiting)
6. [Security Best Practices](#security-best-practices)

---

## Authentication (JWT)

### Overview

The system uses JWT (JSON Web Tokens) for authentication. All API endpoints require a valid JWT token in the `Authorization` header.

**Requirement 13.1**: Authentication for all API requests

### Configuration

Set the following environment variables in your `.env` file:

```bash
# Secret key for JWT signing (generate with: openssl rand -hex 32)
SECRET_KEY=your-secret-key-here-generate-with-openssl-rand-hex-32

# JWT algorithm (default: HS256)
JWT_ALGORITHM=HS256

# Token expiration time in minutes (default: 60)
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

### Generating Tokens

To generate a JWT token for a user:

```python
from src.api.auth import create_user_token

# Create token for regular user
user_token = create_user_token(user_id="trader123", role="user")

# Create token for admin user
admin_token = create_user_token(user_id="admin001", role="admin")
```

### Using Tokens

Include the token in the `Authorization` header of API requests:

```bash
curl -X POST https://api.example.com/api/v1/predict \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"horizons": [1, 7, 30], "market": "ICE_London"}'
```

### Token Structure

JWT tokens contain the following claims:

- `sub`: User identifier (subject)
- `role`: User role (`user` or `admin`)
- `exp`: Expiration timestamp
- `iat`: Issued at timestamp
- `type`: Token type (`access`)

### Token Validation

The system validates:

1. **Signature**: Verifies token was signed with the correct SECRET_KEY
2. **Expiration**: Ensures token has not expired
3. **Structure**: Validates required claims are present
4. **Type**: Confirms token is an access token

Failed authentication attempts are logged with WARNING level (Requirement 13.5).

---

## Data Encryption at Rest

### Supabase PostgreSQL Encryption

**Requirement 13.2**: Encrypt stored data at rest

Supabase provides **automatic encryption at rest** for all PostgreSQL databases using AES-256 encryption. This is enabled by default and requires no additional configuration.

### What is Encrypted

The following data is automatically encrypted at rest:

- Historical price data (`price_data` table)
- Econometric data (`econometric_data` table)
- News articles (`news_articles` table)
- Predictions log (`predictions` table)
- Model performance metrics (`model_metrics` table)
- Validation errors (`validation_errors` table)

### Encryption Details

- **Algorithm**: AES-256
- **Key Management**: Managed by Supabase infrastructure
- **Scope**: All database files, backups, and snapshots
- **Compliance**: SOC 2 Type II, ISO 27001

### Verification

To verify encryption is enabled:

1. Log into your Supabase dashboard
2. Navigate to Settings → Database
3. Confirm "Encryption at Rest" is enabled (default)

### Additional Security Measures

For enhanced security, consider:

1. **Column-level encryption** for highly sensitive fields (e.g., API keys)
2. **Application-level encryption** before storing data
3. **Regular security audits** of database access patterns

---

## TLS 1.3 Configuration

### Overview

**Requirement 13.3**: Encrypt data in transit using TLS 1.3 or higher

All API communications should use TLS 1.3 for encryption in transit. This protects data from interception and tampering.

### Production Deployment with Nginx

#### 1. Install Nginx

```bash
sudo apt update
sudo apt install nginx
```

#### 2. Obtain SSL Certificate

Use Let's Encrypt for free SSL certificates:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com
```

#### 3. Configure Nginx for TLS 1.3

Create or edit `/etc/nginx/sites-available/cocoa-api`:

```nginx
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    # SSL Certificate
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    # TLS 1.3 Configuration
    ssl_protocols TLSv1.3;
    ssl_prefer_server_ciphers off;
    
    # Modern cipher suite for TLS 1.3
    ssl_ciphers 'TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256';
    
    # SSL Session Configuration
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;

    # HSTS (HTTP Strict Transport Security)
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;

    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;

    # Proxy to FastAPI application
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

#### 4. Enable Configuration

```bash
sudo ln -s /etc/nginx/sites-available/cocoa-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Production Deployment with Docker

If using Docker, configure TLS termination at the load balancer or reverse proxy level (e.g., AWS ALB, Azure Application Gateway, Google Cloud Load Balancer).

### Verification

Test TLS 1.3 configuration:

```bash
# Check TLS version
openssl s_client -connect api.yourdomain.com:443 -tls1_3

# Test with curl
curl -v --tlsv1.3 https://api.yourdomain.com/health
```

Use SSL Labs to verify configuration:
https://www.ssllabs.com/ssltest/

---

## Access Logging

### Overview

**Requirement 13.4**: Log all access attempts with user_id and timestamp

The system logs all API requests with structured logging including:

- Request ID (UUID)
- User ID (from JWT token)
- HTTP method and path
- Client IP address
- Timestamp (ISO 8601 format)
- Response status code
- Request duration (milliseconds)

### Log Files

Logs are written to:

- `logs/api_YYYY-MM-DD.log` - All requests (INFO level)
- `logs/api_errors_YYYY-MM-DD.log` - Errors only (ERROR level)

### Log Format

Logs are written in JSON format for easy parsing:

```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "level": "INFO",
  "message": "Request completed",
  "extra": {
    "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "user_id": "trader123",
    "method": "POST",
    "path": "/api/v1/predict",
    "status_code": 200,
    "duration_ms": 245.67,
    "client_ip": "192.168.1.100"
  }
}
```

### Log Rotation

Logs are automatically rotated:

- **Daily rotation**: New log file created at midnight
- **Retention**: 30 days for general logs, 90 days for error logs
- **Compression**: Older logs are automatically compressed

### Accessing Logs

View recent requests:

```bash
tail -f logs/api_$(date +%Y-%m-%d).log | jq
```

Search for specific user:

```bash
grep "trader123" logs/api_*.log | jq
```

Count requests by user:

```bash
cat logs/api_*.log | jq -r '.extra.user_id' | sort | uniq -c | sort -rn
```

### Unauthorized Access Attempts

**Requirement 13.5**: Block and log unauthorized access attempts

Failed authentication attempts are logged with WARNING level:

```json
{
  "timestamp": "2024-01-15T10:35:12.456789",
  "level": "WARNING",
  "message": "Unauthorized access attempt with invalid token",
  "extra": {
    "timestamp": "2024-01-15T10:35:12.456789"
  }
}
```

The system automatically blocks requests with invalid tokens and returns HTTP 401.

---

## Rate Limiting

### Overview

Rate limiting prevents abuse and ensures fair resource allocation. The system implements per-user rate limiting.

### Configuration

Set in `config/settings.py`:

```python
rate_limit_enabled: bool = True
rate_limit_requests_per_minute: int = 60  # 60 requests per minute per user
```

### Implementation

Rate limiting is enforced at the API gateway level using Redis:

```python
# Redis key format
ratelimit:{user_id}:{endpoint} -> request_count (TTL: 60 seconds)
```

### Exceeded Rate Limit

When rate limit is exceeded, the API returns:

```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Please try again later.",
  "retry_after": 30
}
```

HTTP Status: `429 Too Many Requests`

---

## Security Best Practices

### 1. Secret Key Management

- **Generate strong keys**: Use `openssl rand -hex 32`
- **Never commit secrets**: Add `.env` to `.gitignore`
- **Rotate keys regularly**: Update SECRET_KEY every 90 days
- **Use environment variables**: Never hardcode secrets in code

### 2. Token Management

- **Short expiration**: Keep token lifetime under 1 hour
- **Refresh tokens**: Implement refresh token mechanism for long sessions
- **Revocation**: Maintain token blacklist for compromised tokens
- **Secure storage**: Store tokens in httpOnly cookies or secure storage

### 3. Network Security

- **Firewall rules**: Restrict access to API ports
- **VPC/Private networks**: Deploy in isolated network segments
- **DDoS protection**: Use CloudFlare or AWS Shield
- **IP whitelisting**: Restrict admin endpoints to known IPs

### 4. Database Security

- **Least privilege**: Grant minimal required permissions
- **Connection pooling**: Use connection pools to prevent exhaustion
- **Prepared statements**: Always use parameterized queries
- **Audit logs**: Enable PostgreSQL audit logging

### 5. Monitoring and Alerts

- **Failed auth attempts**: Alert on >10 failed attempts per minute
- **Unusual patterns**: Monitor for abnormal request patterns
- **Error rates**: Alert on elevated error rates
- **Performance**: Track response times and throughput

### 6. Compliance

- **GDPR**: Implement data retention and deletion policies
- **SOC 2**: Follow security controls for data handling
- **PCI DSS**: If handling payment data, ensure compliance
- **Regular audits**: Conduct security audits quarterly

### 7. Incident Response

- **Incident plan**: Document response procedures
- **Contact list**: Maintain updated contact information
- **Backup strategy**: Regular backups with encryption
- **Recovery testing**: Test disaster recovery procedures

---

## Security Checklist

Before deploying to production:

- [ ] Generate strong SECRET_KEY and store securely
- [ ] Configure TLS 1.3 with valid SSL certificate
- [ ] Enable Supabase encryption at rest (default)
- [ ] Configure access logging with user_id tracking
- [ ] Enable rate limiting
- [ ] Set up security monitoring and alerts
- [ ] Configure firewall rules
- [ ] Implement token refresh mechanism
- [ ] Set up automated backups
- [ ] Document incident response procedures
- [ ] Conduct security audit
- [ ] Train team on security practices

---

## Support

For security concerns or questions:

- **Email**: security@cocoatrading.com
- **Documentation**: https://docs.cocoatrading.com/security
- **Emergency**: Contact system administrator immediately

---

**Last Updated**: 2024-01-15  
**Version**: 1.0.0  
**Maintained by**: DevOps Team
