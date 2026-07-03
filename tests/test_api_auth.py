"""
Tests for authentication functionality.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.api.auth import verify_token, verify_admin_token


class TestAuthentication:
    """Tests for authentication functions."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings."""
        with patch('src.api.auth.get_settings') as mock:
            settings = Mock()
            settings.secret_key = "test_secret_key"
            mock.return_value = settings
            yield settings
    
    def test_verify_token_success(self, mock_settings):
        """Test successful token verification."""
        from src.api.auth import create_user_token
        
        # Create a valid JWT token
        token = create_user_token("authenticated_user", role="user")
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token
        )
        
        result = verify_token(credentials)
        
        assert result == "authenticated_user"
    
    def test_verify_token_invalid(self, mock_settings):
        """Test token verification with invalid token."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="invalid_token"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            verify_token(credentials)
        
        assert exc_info.value.status_code == 401
        assert "Could not validate credentials" in exc_info.value.detail
    
    def test_verify_admin_token_success(self, mock_settings):
        """Test successful admin token verification."""
        from src.api.auth import create_user_token
        
        # Create a valid admin JWT token
        token = create_user_token("admin_user", role="admin")
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token
        )
        
        result = verify_admin_token(credentials)
        
        assert result == "admin_user"
    
    def test_verify_admin_token_with_secret_key(self, mock_settings):
        """Test admin token verification with admin role."""
        from src.api.auth import create_user_token
        
        # Create a valid admin JWT token
        token = create_user_token("admin_user", role="admin")
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token
        )
        
        result = verify_admin_token(credentials)
        
        assert result == "admin_user"
    
    def test_verify_admin_token_non_admin(self, mock_settings):
        """Test admin token verification with non-admin token."""
        from src.api.auth import create_user_token
        
        # Create a regular user token (not admin)
        token = create_user_token("regular_user", role="user")
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token
        )
        
        with pytest.raises(HTTPException) as exc_info:
            verify_admin_token(credentials)
        
        assert exc_info.value.status_code == 403
        assert "Admin privileges required" in exc_info.value.detail
    
    def test_verify_admin_token_invalid(self, mock_settings):
        """Test admin token verification with invalid token."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="invalid_token"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            verify_admin_token(credentials)
        
        # Should fail at token verification step (401)
        assert exc_info.value.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
