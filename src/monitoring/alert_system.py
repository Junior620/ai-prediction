"""
Alert System for critical errors and system events.

This module implements the alert notification system for CRITICAL errors
and important system events. Supports multiple alert channels:
- Email notifications
- Webhook notifications (Slack, Teams, etc.)
- Logging-based alerts

Implements Requirement 12.4: Alert system for CRITICAL errors
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import requests
from loguru import logger

from config.settings import get_settings


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Types of alerts."""
    MODEL_FAILURE = "model_failure"
    DATA_SOURCE_FAILURE = "data_source_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    AUTHENTICATION_BREACH = "authentication_breach"
    SYSTEM_ERROR = "system_error"
    PREDICTION_ERROR = "prediction_error"


class AlertSystem:
    """
    Manages alert notifications for critical system events.
    
    Supports multiple notification channels:
    - Email (via SMTP or email service API)
    - Webhooks (Slack, Microsoft Teams, custom endpoints)
    - Structured logging (always enabled as fallback)
    
    Attributes:
        settings: Application settings
        email_enabled: Whether email alerts are enabled
        webhook_enabled: Whether webhook alerts are enabled
    """
    
    def __init__(self):
        """Initialize alert system with configuration from settings."""
        self.settings = get_settings()
        self.email_enabled = self.settings.alert_email_enabled
        self.webhook_enabled = bool(self.settings.alert_webhook_url)
        
        logger.info(
            f"Alert system initialized: "
            f"email={self.email_enabled}, webhook={self.webhook_enabled}"
        )
    
    def send_alert(
        self,
        severity: AlertSeverity,
        alert_type: AlertType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send an alert through configured channels.
        
        Args:
            severity: Alert severity level
            alert_type: Type of alert
            message: Human-readable alert message
            details: Additional details about the alert
            context: Contextual information (user_id, request_id, etc.)
        
        Returns:
            True if alert was sent successfully through at least one channel
        
        Example:
            >>> alert_system = AlertSystem()
            >>> alert_system.send_alert(
            ...     severity=AlertSeverity.CRITICAL,
            ...     alert_type=AlertType.MODEL_FAILURE,
            ...     message="XGBoost model failed to generate predictions",
            ...     details={"error": "Model not fitted", "horizon": 7},
            ...     context={"user_id": "trader123", "request_id": "abc-123"}
            ... )
        """
        if details is None:
            details = {}
        if context is None:
            context = {}
        
        # Build alert payload
        alert_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity.value,
            "alert_type": alert_type.value,
            "message": message,
            "details": details,
            "context": context
        }
        
        # Always log the alert
        self._log_alert(alert_payload)
        
        success = True
        
        # Send through additional channels based on severity
        if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            # Send email for ERROR and CRITICAL alerts
            if self.email_enabled:
                email_success = self._send_email_alert(alert_payload)
                success = success and email_success
            
            # Send webhook for ERROR and CRITICAL alerts
            if self.webhook_enabled:
                webhook_success = self._send_webhook_alert(alert_payload)
                success = success and webhook_success
        
        return success
    
    def _log_alert(self, alert_payload: Dict[str, Any]) -> None:
        """
        Log alert using structured logging.
        
        This is always executed as a fallback notification method.
        
        Args:
            alert_payload: Alert data to log
        """
        severity = alert_payload["severity"]
        message = alert_payload["message"]
        
        # Map severity to logger level
        if severity == "CRITICAL":
            logger.critical(f"ALERT: {message}", extra=alert_payload)
        elif severity == "ERROR":
            logger.error(f"ALERT: {message}", extra=alert_payload)
        elif severity == "WARNING":
            logger.warning(f"ALERT: {message}", extra=alert_payload)
        else:
            logger.info(f"ALERT: {message}", extra=alert_payload)
    
    def _send_email_alert(self, alert_payload: Dict[str, Any]) -> bool:
        """
        Send alert via email.
        
        Note: This is a placeholder implementation. In production, you would:
        - Use an SMTP library (smtplib) to send emails
        - Or use an email service API (SendGrid, AWS SES, etc.)
        
        Args:
            alert_payload: Alert data to send
        
        Returns:
            True if email was sent successfully
        """
        try:
            email_to = self.settings.alert_email_to
            email_from = self.settings.alert_email_from
            
            if not email_to or not email_from:
                logger.warning("Email alert configuration incomplete")
                return False
            
            # Build email content
            subject = f"[{alert_payload['severity']}] {alert_payload['alert_type']}"
            body = self._format_email_body(alert_payload)
            
            # TODO: Implement actual email sending
            # For now, just log that we would send an email
            logger.info(
                f"Email alert would be sent to {email_to}",
                extra={
                    "subject": subject,
                    "to": email_to,
                    "from": email_from
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _send_webhook_alert(self, alert_payload: Dict[str, Any]) -> bool:
        """
        Send alert via webhook (Slack, Teams, custom endpoint).
        
        Args:
            alert_payload: Alert data to send
        
        Returns:
            True if webhook was called successfully
        """
        try:
            webhook_url = self.settings.alert_webhook_url
            
            if not webhook_url:
                logger.warning("Webhook URL not configured")
                return False
            
            # Format payload for webhook
            webhook_payload = self._format_webhook_payload(alert_payload)
            
            # Send POST request to webhook
            response = requests.post(
                webhook_url,
                json=webhook_payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            
            logger.info(
                f"Webhook alert sent successfully",
                extra={"webhook_url": webhook_url, "status_code": response.status_code}
            )
            
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending webhook alert: {e}")
            return False
    
    def _format_email_body(self, alert_payload: Dict[str, Any]) -> str:
        """
        Format alert payload as email body.
        
        Args:
            alert_payload: Alert data
        
        Returns:
            Formatted email body text
        """
        body = f"""
Cocoa Price Prediction System Alert

Severity: {alert_payload['severity']}
Alert Type: {alert_payload['alert_type']}
Timestamp: {alert_payload['timestamp']}

Message:
{alert_payload['message']}

Details:
{json.dumps(alert_payload['details'], indent=2)}

Context:
{json.dumps(alert_payload['context'], indent=2)}

---
This is an automated alert from the Cocoa Price Prediction System.
"""
        return body
    
    def _format_webhook_payload(self, alert_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format alert payload for webhook (Slack-compatible format).
        
        Args:
            alert_payload: Alert data
        
        Returns:
            Webhook-compatible payload
        """
        # Slack-compatible format (also works with many other webhook services)
        severity_emoji = {
            "INFO": ":information_source:",
            "WARNING": ":warning:",
            "ERROR": ":x:",
            "CRITICAL": ":rotating_light:"
        }
        
        emoji = severity_emoji.get(alert_payload['severity'], ":bell:")
        
        payload = {
            "text": f"{emoji} *{alert_payload['severity']}* Alert",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} {alert_payload['severity']} Alert"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Type:*\n{alert_payload['alert_type']}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:*\n{alert_payload['timestamp']}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Message:*\n{alert_payload['message']}"
                    }
                }
            ]
        }
        
        # Add details if present
        if alert_payload['details']:
            details_text = "\n".join(
                f"• {k}: {v}" for k, v in alert_payload['details'].items()
            )
            payload["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Details:*\n{details_text}"
                }
            })
        
        return payload
    
    def send_model_failure_alert(
        self,
        model_name: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Convenience method to send model failure alert.
        
        Args:
            model_name: Name of the failed model
            error_message: Error message
            context: Additional context
        
        Returns:
            True if alert was sent successfully
        """
        return self.send_alert(
            severity=AlertSeverity.CRITICAL,
            alert_type=AlertType.MODEL_FAILURE,
            message=f"Model {model_name} failed to generate predictions",
            details={"model": model_name, "error": error_message},
            context=context or {}
        )
    
    def send_data_source_failure_alert(
        self,
        source_name: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Convenience method to send data source failure alert.
        
        Args:
            source_name: Name of the failed data source
            error_message: Error message
            context: Additional context
        
        Returns:
            True if alert was sent successfully
        """
        return self.send_alert(
            severity=AlertSeverity.ERROR,
            alert_type=AlertType.DATA_SOURCE_FAILURE,
            message=f"Data source {source_name} is unavailable",
            details={"source": source_name, "error": error_message},
            context=context or {}
        )
    
    def send_performance_degradation_alert(
        self,
        model_version: str,
        metrics: Dict[str, float],
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Convenience method to send performance degradation alert.
        
        Args:
            model_version: Version of the degraded model
            metrics: Current performance metrics
            context: Additional context
        
        Returns:
            True if alert was sent successfully
        """
        return self.send_alert(
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            message=f"Model {model_version} performance has degraded",
            details={"model_version": model_version, "metrics": metrics},
            context=context or {}
        )


# Global alert system instance
_alert_system: Optional[AlertSystem] = None


def get_alert_system() -> AlertSystem:
    """
    Get or create the global alert system instance.
    
    Returns:
        AlertSystem instance
    """
    global _alert_system
    if _alert_system is None:
        _alert_system = AlertSystem()
    return _alert_system
