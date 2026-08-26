"""
Tests for Alert System.

Tests Requirement 12.4: Alert system for CRITICAL errors
"""

import pytest
from unittest.mock import Mock, patch
import requests
from loguru import logger

from src.monitoring.alert_system import (
    AlertSystem,
    AlertSeverity,
    AlertType,
    get_alert_system
)


@pytest.fixture
def loguru_caplog():
    """Capture loguru messages (stdlib caplog does not see loguru)."""
    messages = []

    def sink(message):
        record = message.record
        messages.append({
            "level": record["level"].name,
            "message": record["message"],
            "extra": dict(record["extra"]),
        })

    handler_id = logger.add(sink, level="DEBUG")
    try:
        yield messages
    finally:
        logger.remove(handler_id)


class TestAlertSystem:
    """Test alert system functionality."""
    
    def test_alert_system_initialization(self):
        """Test alert system initializes correctly."""
        alert_system = AlertSystem()
        
        assert alert_system is not None
        assert hasattr(alert_system, 'settings')
        assert hasattr(alert_system, 'email_enabled')
        assert hasattr(alert_system, 'webhook_enabled')
    
    def test_send_alert_logs_message(self, loguru_caplog):
        """Test that alerts are always logged."""
        alert_system = AlertSystem()
        
        result = alert_system.send_alert(
            severity=AlertSeverity.INFO,
            alert_type=AlertType.SYSTEM_ERROR,
            message="Test alert message",
            details={"key": "value"}
        )
        
        assert result is True
        assert any("ALERT: Test alert message" in m["message"] for m in loguru_caplog)
    
    def test_send_critical_alert_logs_at_critical_level(self, loguru_caplog):
        """Test CRITICAL alerts are logged at CRITICAL level."""
        alert_system = AlertSystem()
        
        alert_system.send_alert(
            severity=AlertSeverity.CRITICAL,
            alert_type=AlertType.MODEL_FAILURE,
            message="Critical model failure",
            details={"model": "XGBoost"}
        )
        
        critical = [m for m in loguru_caplog if m["level"] == "CRITICAL"]
        assert len(critical) > 0
        assert any("Critical model failure" in m["message"] for m in critical)
    
    def test_send_error_alert_logs_at_error_level(self, loguru_caplog):
        """Test ERROR alerts are logged at ERROR level."""
        alert_system = AlertSystem()
        
        alert_system.send_alert(
            severity=AlertSeverity.ERROR,
            alert_type=AlertType.DATA_SOURCE_FAILURE,
            message="Data source unavailable",
            details={"source": "Weather API"}
        )
        
        error_records = [m for m in loguru_caplog if m["level"] == "ERROR"]
        assert len(error_records) > 0
    
    def test_send_warning_alert_logs_at_warning_level(self, loguru_caplog):
        """Test WARNING alerts are logged at WARNING level."""
        alert_system = AlertSystem()
        
        alert_system.send_alert(
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            message="Performance degraded",
            details={"rmse": 150.5}
        )
        
        warning_records = [m for m in loguru_caplog if m["level"] == "WARNING"]
        assert len(warning_records) > 0
    
    def test_alert_payload_structure(self, loguru_caplog):
        """Test alert payload contains required fields."""
        alert_system = AlertSystem()
        
        alert_system.send_alert(
            severity=AlertSeverity.ERROR,
            alert_type=AlertType.PREDICTION_ERROR,
            message="Prediction failed",
            details={"horizon": 7, "error": "Model not fitted"},
            context={"user_id": "trader123", "request_id": "abc-123"}
        )
        
        log_records = [m for m in loguru_caplog if "Prediction failed" in m["message"]]
        assert len(log_records) > 0
    
    @patch('requests.post')
    def test_send_webhook_alert_success(self, mock_post):
        """Test sending webhook alert successfully."""
        # Mock successful webhook response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Create alert system with webhook enabled
        with patch('src.monitoring.alert_system.get_settings') as mock_settings:
            mock_settings.return_value.alert_webhook_url = "https://hooks.example.com/webhook"
            mock_settings.return_value.alert_email_enabled = False
            
            alert_system = AlertSystem()
            
            result = alert_system.send_alert(
                severity=AlertSeverity.CRITICAL,
                alert_type=AlertType.MODEL_FAILURE,
                message="Model failed",
                details={"model": "XGBoost"}
            )
        
        assert result is True
        assert mock_post.called
        
        # Verify webhook payload
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://hooks.example.com/webhook"
        payload = call_args[1]['json']
        assert 'text' in payload
        assert 'blocks' in payload
    
    @patch('requests.post')
    def test_send_webhook_alert_failure(self, mock_post, loguru_caplog):
        """Test webhook alert handles failures gracefully."""
        # Mock failed webhook response
        mock_post.side_effect = requests.RequestException("Connection failed")
        
        with patch('src.monitoring.alert_system.get_settings') as mock_settings:
            mock_settings.return_value.alert_webhook_url = "https://hooks.example.com/webhook"
            mock_settings.return_value.alert_email_enabled = False
            
            alert_system = AlertSystem()
            
            result = alert_system.send_alert(
                severity=AlertSeverity.ERROR,
                alert_type=AlertType.SYSTEM_ERROR,
                message="System error",
                details={}
            )
        
        # Should return False but not crash
        assert result is False
        assert any("Failed to send webhook alert" in m["message"] for m in loguru_caplog)
    
    def test_send_model_failure_alert(self, loguru_caplog):
        """Test convenience method for model failure alerts."""
        alert_system = AlertSystem()
        
        result = alert_system.send_model_failure_alert(
            model_name="XGBoost",
            error_message="Model not fitted",
            context={"user_id": "trader123"}
        )
        
        assert result is True
        assert any("XGBoost" in m["message"] for m in loguru_caplog)
    
    def test_send_data_source_failure_alert(self, loguru_caplog):
        """Test convenience method for data source failure alerts."""
        alert_system = AlertSystem()
        
        result = alert_system.send_data_source_failure_alert(
            source_name="Weather API",
            error_message="Connection timeout",
            context={}
        )
        
        assert result is True
        assert any("Weather API" in m["message"] for m in loguru_caplog)
    
    def test_send_performance_degradation_alert(self, loguru_caplog):
        """Test convenience method for performance degradation alerts."""
        alert_system = AlertSystem()
        
        result = alert_system.send_performance_degradation_alert(
            model_version="v1.2.3",
            metrics={"rmse": 150.5, "mae": 120.3},
            context={}
        )
        
        assert result is True
        assert any("v1.2.3" in m["message"] for m in loguru_caplog)
    
    def test_format_email_body(self):
        """Test email body formatting."""
        alert_system = AlertSystem()
        
        alert_payload = {
            "timestamp": "2024-01-15T10:30:00",
            "severity": "CRITICAL",
            "alert_type": "model_failure",
            "message": "Model failed",
            "details": {"model": "XGBoost"},
            "context": {"user_id": "trader123"}
        }
        
        body = alert_system._format_email_body(alert_payload)
        
        assert "CRITICAL" in body
        assert "model_failure" in body
        assert "Model failed" in body
        assert "XGBoost" in body
        assert "trader123" in body
    
    def test_format_webhook_payload(self):
        """Test webhook payload formatting (Slack-compatible)."""
        alert_system = AlertSystem()
        
        alert_payload = {
            "timestamp": "2024-01-15T10:30:00",
            "severity": "ERROR",
            "alert_type": "data_source_failure",
            "message": "Data source unavailable",
            "details": {"source": "Weather API", "error": "Timeout"},
            "context": {}
        }
        
        payload = alert_system._format_webhook_payload(alert_payload)
        
        assert "text" in payload
        assert "blocks" in payload
        assert "ERROR" in payload["text"]
        
        # Check blocks structure
        blocks = payload["blocks"]
        assert len(blocks) >= 3  # header, fields, message
        assert blocks[0]["type"] == "header"
        assert blocks[1]["type"] == "section"
    
    def test_get_alert_system_singleton(self):
        """Test get_alert_system returns singleton instance."""
        # Reset singleton
        import src.monitoring.alert_system as alert_module
        alert_module._alert_system = None
        
        system1 = get_alert_system()
        system2 = get_alert_system()
        
        assert system1 is system2
    
    def test_alert_severity_enum(self):
        """Test AlertSeverity enum values."""
        assert AlertSeverity.INFO.value == "INFO"
        assert AlertSeverity.WARNING.value == "WARNING"
        assert AlertSeverity.ERROR.value == "ERROR"
        assert AlertSeverity.CRITICAL.value == "CRITICAL"
    
    def test_alert_type_enum(self):
        """Test AlertType enum values."""
        assert AlertType.MODEL_FAILURE.value == "model_failure"
        assert AlertType.DATA_SOURCE_FAILURE.value == "data_source_failure"
        assert AlertType.PERFORMANCE_DEGRADATION.value == "performance_degradation"
        assert AlertType.AUTHENTICATION_BREACH.value == "authentication_breach"
        assert AlertType.SYSTEM_ERROR.value == "system_error"
        assert AlertType.PREDICTION_ERROR.value == "prediction_error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
