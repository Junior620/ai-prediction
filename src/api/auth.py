"""
Authentication module for the FastAPI application.

This module provides JWT-based authentication for securing API endpoints.
Implements Requirements 13.1, 13.4, 13.5.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from loguru import logger

from config.settings import get_settings

# Initialize HTTPBearer security scheme
security = HTTPBearer()

# JWT Configuration
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour
ADMIN_ROLES = ["admin", "superuser"]


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Payload data to encode in the token (should include 'sub' for user_id)
        expires_delta: Optional custom expiration time
    
    Returns:
        Encoded JWT token string
    
    Example:
        >>> token = create_access_token({"sub": "user123", "role": "trader"})
    """
    settings = get_settings()
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    logger.debug(f"Created access token for user: {data.get('sub')}")
    
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded token payload
    
    Raises:
        HTTPException: If token is invalid, expired, or malformed
    """
    settings = get_settings()
    
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[ALGORITHM]
        )
        
        # Validate token type
        if payload.get("type") != "access":
            logger.warning("Invalid token type")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Validate required fields
        if "sub" not in payload:
            logger.warning("Token missing 'sub' field")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return payload
        
    except JWTError as e:
        logger.warning(f"JWT validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    """
    Verify the JWT bearer token from the request.
    
    Validates:
    - Token signature using SECRET_KEY
    - Token expiration
    - Token structure and required fields
    
    Implements Requirement 13.1: Authentication for all API requests
    Implements Requirement 13.4: Log access attempts with user_id
    
    Args:
        credentials: HTTP authorization credentials from the request
    
    Returns:
        User identifier (sub claim) if token is valid
    
    Raises:
        HTTPException: If token is invalid, expired, or missing
    """
    token = credentials.credentials
    
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        
        logger.info(f"Authentication successful for user: {user_id}")
        return user_id
        
    except HTTPException:
        # Log unauthorized access attempt (Requirement 13.5)
        logger.warning(
            f"Unauthorized access attempt with invalid token",
            extra={"timestamp": datetime.utcnow().isoformat()}
        )
        raise


def verify_admin_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    """
    Verify the bearer token has admin privileges.
    
    This is used for admin-only endpoints like retraining.
    Checks both token validity and user role.
    
    Implements Requirement 13.1: Authentication with role-based access
    Implements Requirement 13.5: Block unauthorized access attempts
    
    Args:
        credentials: HTTP authorization credentials from the request
    
    Returns:
        Admin user identifier if token is valid and user has admin role
    
    Raises:
        HTTPException: If token is invalid or user lacks admin privileges
    """
    token = credentials.credentials
    
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        user_role = payload.get("role", "user")
        
        # Check if user has admin role
        if user_role not in ADMIN_ROLES:
            logger.warning(
                f"Non-admin user {user_id} attempted to access admin endpoint",
                extra={
                    "user_id": user_id,
                    "role": user_role,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required",
            )
        
        logger.info(f"Admin authentication successful for user: {user_id}")
        return user_id
        
    except HTTPException:
        # Log unauthorized admin access attempt (Requirement 13.5)
        logger.warning(
            f"Unauthorized admin access attempt",
            extra={"timestamp": datetime.utcnow().isoformat()}
        )
        raise


def create_user_token(
    user_id: str,
    role: str = "user",
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Convenience function to create a token for a user.
    
    Args:
        user_id: User identifier
        role: User role (default: "user", can be "admin")
        expires_delta: Optional custom expiration (default: 1 hour)
    
    Returns:
        JWT token string
    """
    return create_access_token(
        data={"sub": user_id, "role": role},
        expires_delta=expires_delta,
    )


def create_service_token(
    user_id: str = "service@dashboard",
    role: str = "user",
    days: int = 365,
) -> str:
    """Create a long-lived service token for batch scripts and the Next.js BFF."""
    return create_access_token(
        data={"sub": user_id, "role": role},
        expires_delta=timedelta(days=days),
    )
