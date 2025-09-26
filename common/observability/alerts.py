"""
Alert management and notification system for trading platform observability.
Integrates with Sentry alerts and custom notification channels.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod

import sentry_sdk
from app_settings.settings import get_settings


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SENTRY = "sentry"


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    description: str
    severity: AlertSeverity
    metric: str
    condition: str  # e.g., ">" for greater than
    threshold: float
    time_window_minutes: int = 5
    min_occurrences: int = 1
    channels: List[AlertChannel] = None
    tags: Dict[str, str] = None
    enabled: bool = True
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = [AlertChannel.SENTRY]
        if self.tags is None:
            self.tags = {}


@dataclass
class Alert:
    """An active or resolved alert."""
    rule_name: str
    severity: AlertSeverity
    title: str
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime
    context: Dict[str, Any] = None
    tags: Dict[str, str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.tags is None:
            self.tags = {}


class AlertNotifier(ABC):
    """Base class for alert notification channels."""
    
    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert notification. Returns True if successful."""
        pass
    
    @abstractmethod
    async def send_resolution(self, alert: Alert) -> bool:
        """Send alert resolution notification. Returns True if successful."""
        pass


class SentryAlertNotifier(AlertNotifier):
    """Sentry integration for alert notifications."""
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to Sentry as an issue."""
        try:
            with sentry_sdk.configure_scope() as scope:
                # Set alert context
                scope.set_context("alert", {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "metric_value": alert.metric_value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp.isoformat()
                })
                
                # Set tags
                scope.set_tag("alert.severity", alert.severity.value)
                scope.set_tag("alert.rule", alert.rule_name)
                
                for key, value in alert.tags.items():
                    scope.set_tag(f"alert.{key}", value)
                
                # Capture as error for high/critical, message for others
                if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                    sentry_sdk.capture_exception(Exception(alert.message))
                else:
                    sentry_sdk.capture_message(alert.message, level="warning")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert to Sentry: {e}")
            return False
    
    async def send_resolution(self, alert: Alert) -> bool:
        """Send alert resolution to Sentry."""
        try:
            with sentry_sdk.configure_scope() as scope:
                scope.set_context("alert_resolution", {
                    "rule_name": alert.rule_name,
                    "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
                })
                
                scope.set_tag("alert.status", "resolved")
                scope.set_tag("alert.rule", alert.rule_name)
                
                sentry_sdk.capture_message(
                    f"Alert resolved: {alert.title}",
                    level="info"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send resolution to Sentry: {e}")
            return False


class EmailAlertNotifier(AlertNotifier):
    """Email notification for alerts."""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            # Email implementation would go here
            # For now, just log the alert
            logger.info(f"EMAIL ALERT: {alert.title} - {alert.message}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    async def send_resolution(self, alert: Alert) -> bool:
        """Send resolution notification via email."""
        try:
            logger.info(f"EMAIL RESOLUTION: Alert resolved - {alert.title}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email resolution: {e}")
            return False


class TradingAlertManager:
    """
    Main alert management system for the trading platform.
    Monitors metrics and triggers notifications based on defined rules.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notifiers: Dict[AlertChannel, AlertNotifier] = {}
        
        self._setup_notifiers()
        self._setup_default_rules()
    
    def _setup_notifiers(self):
        """Initialize alert notifiers based on configuration."""
        # Always include Sentry notifier
        self.notifiers[AlertChannel.SENTRY] = SentryAlertNotifier()
        
        # Add email notifier if configured
        if self.settings.observability.alert_email_recipients:
            email_config = {
                "recipients": self.settings.observability.alert_email_recipients,
                # Add SMTP configuration from settings if available
            }
            self.notifiers[AlertChannel.EMAIL] = EmailAlertNotifier(email_config)
    
    def _setup_default_rules(self):
        """Setup default alert rules for trading operations."""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                description="Error rate exceeds threshold",
                severity=AlertSeverity.HIGH,
                metric="error_rate",
                condition=">",
                threshold=0.1,  # 10% error rate
                time_window_minutes=5,
                channels=[AlertChannel.SENTRY, AlertChannel.EMAIL],
                tags={"category": "reliability"}
            ),
            AlertRule(
                name="critical_error_rate", 
                description="Critical error rate threshold exceeded",
                severity=AlertSeverity.CRITICAL,
                metric="error_rate",
                condition=">",
                threshold=0.25,  # 25% error rate
                time_window_minutes=2,
                channels=[AlertChannel.SENTRY, AlertChannel.EMAIL],
                tags={"category": "reliability"}
            ),
            AlertRule(
                name="slow_response_time",
                description="Response time is too slow",
                severity=AlertSeverity.MEDIUM,
                metric="response_time_p95",
                condition=">",
                threshold=5000.0,  # 5 seconds
                time_window_minutes=10,
                channels=[AlertChannel.SENTRY],
                tags={"category": "performance"}
            ),
            AlertRule(
                name="ib_api_failures",
                description="IB API failure rate is high",
                severity=AlertSeverity.HIGH,
                metric="ib_api_error_rate",
                condition=">",
                threshold=0.2,  # 20% failure rate
                time_window_minutes=5,
                channels=[AlertChannel.SENTRY, AlertChannel.EMAIL],
                tags={"category": "external_api", "provider": "interactive_brokers"}
            ),
            AlertRule(
                name="celery_task_failures",
                description="High Celery task failure rate",
                severity=AlertSeverity.MEDIUM,
                metric="celery_error_rate",
                condition=">",
                threshold=0.15,  # 15% failure rate
                time_window_minutes=10,
                channels=[AlertChannel.SENTRY],
                tags={"category": "background_jobs"}
            ),
            AlertRule(
                name="data_processing_lag",
                description="Data processing is lagging behind",
                severity=AlertSeverity.MEDIUM,
                metric="processing_lag_minutes",
                condition=">",
                threshold=30.0,  # 30 minutes lag
                time_window_minutes=15,
                channels=[AlertChannel.SENTRY],
                tags={"category": "data_pipeline"}
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name} ({rule.severity.value})")
    
    def remove_rule(self, rule_name: str) -> None:
        """Remove an alert rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def enable_rule(self, rule_name: str) -> None:
        """Enable an alert rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            logger.info(f"Enabled alert rule: {rule_name}")
    
    def disable_rule(self, rule_name: str) -> None:
        """Disable an alert rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            logger.info(f"Disabled alert rule: {rule_name}")
    
    async def evaluate_rules(self, metrics: Dict[str, float]) -> List[Alert]:
        """
        Evaluate alert rules against current metrics.
        
        Args:
            metrics: Dictionary of metric name to value
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            if rule.metric not in metrics:
                continue
            
            metric_value = metrics[rule.metric]
            
            # Evaluate condition
            triggered = self._evaluate_condition(
                metric_value, 
                rule.condition, 
                rule.threshold
            )
            
            if triggered:
                # Check if this is a new alert or existing one
                alert_key = f"{rule_name}:{rule.metric}"
                
                if alert_key not in self.active_alerts:
                    # Create new alert
                    alert = Alert(
                        rule_name=rule.name,
                        severity=rule.severity,
                        title=f"Trading Alert: {rule.description}",
                        message=f"{rule.description}. Current value: {metric_value:.2f}, Threshold: {rule.threshold:.2f}",
                        metric_value=metric_value,
                        threshold=rule.threshold,
                        timestamp=datetime.now(),
                        tags=rule.tags
                    )
                    
                    self.active_alerts[alert_key] = alert
                    self.alert_history.append(alert)
                    triggered_alerts.append(alert)
                    
                    # Send notifications
                    await self._send_alert_notifications(alert, rule)
                    
                    logger.warning(f"Alert triggered: {alert.title}")
            
            else:
                # Check if we need to resolve an existing alert
                alert_key = f"{rule_name}:{rule.metric}"
                if alert_key in self.active_alerts:
                    alert = self.active_alerts[alert_key]
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    
                    # Send resolution notifications
                    await self._send_resolution_notifications(alert, rule)
                    
                    del self.active_alerts[alert_key]
                    logger.info(f"Alert resolved: {alert.title}")
        
        return triggered_alerts
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate a metric condition."""
        if condition == ">":
            return value > threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<":
            return value < threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 0.001  # Float equality with tolerance
        elif condition == "!=":
            return abs(value - threshold) >= 0.001
        else:
            logger.error(f"Unknown condition: {condition}")
            return False
    
    async def _send_alert_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send alert notifications to configured channels."""
        for channel in rule.channels:
            if channel in self.notifiers:
                try:
                    success = await self.notifiers[channel].send_alert(alert)
                    if success:
                        logger.info(f"Alert notification sent via {channel.value}")
                    else:
                        logger.error(f"Failed to send alert via {channel.value}")
                except Exception as e:
                    logger.error(f"Error sending alert via {channel.value}: {e}")
    
    async def _send_resolution_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send alert resolution notifications."""
        for channel in rule.channels:
            if channel in self.notifiers:
                try:
                    success = await self.notifiers[channel].send_resolution(alert)
                    if success:
                        logger.info(f"Resolution notification sent via {channel.value}")
                    else:
                        logger.error(f"Failed to send resolution via {channel.value}")
                except Exception as e:
                    logger.error(f"Error sending resolution via {channel.value}: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history 
            if alert.timestamp > cutoff
        ]
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of alerts for the specified time period."""
        alerts = self.get_alert_history(hours)
        
        summary = {
            "total_alerts": len(alerts),
            "active_alerts": len(self.active_alerts),
            "resolved_alerts": len([a for a in alerts if a.resolved]),
            "by_severity": {},
            "by_rule": {},
            "recent_alerts": alerts[-10:] if alerts else []  # Last 10 alerts
        }
        
        # Count by severity
        for severity in AlertSeverity:
            summary["by_severity"][severity.value] = len([
                a for a in alerts if a.severity == severity
            ])
        
        # Count by rule
        for alert in alerts:
            rule_name = alert.rule_name
            if rule_name not in summary["by_rule"]:
                summary["by_rule"][rule_name] = 0
            summary["by_rule"][rule_name] += 1
        
        return summary


# Global alert manager instance
_alert_manager = None


def get_alert_manager() -> TradingAlertManager:
    """Get the global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = TradingAlertManager()
    return _alert_manager


async def trigger_custom_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.MEDIUM,
    context: Dict[str, Any] = None,
    tags: Dict[str, str] = None
) -> None:
    """
    Trigger a custom alert outside of the normal rule evaluation.
    
    Args:
        title: Alert title
        message: Alert message
        severity: Alert severity
        context: Additional context data
        tags: Alert tags
    """
    alert = Alert(
        rule_name="custom_alert",
        severity=severity,
        title=title,
        message=message,
        metric_value=0.0,
        threshold=0.0,
        timestamp=datetime.now(),
        context=context or {},
        tags=tags or {}
    )
    
    manager = get_alert_manager()
    
    # Send to Sentry notifier directly
    if AlertChannel.SENTRY in manager.notifiers:
        await manager.notifiers[AlertChannel.SENTRY].send_alert(alert)
    
    # Add to history
    manager.alert_history.append(alert)
    
    logger.warning(f"Custom alert triggered: {title}")