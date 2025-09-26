"""
Core observability initialization and configuration for Sentry integration.
"""

import logging
import socket
from typing import Any, Dict, Optional

import sentry_sdk
from sentry_sdk.hub import Hub
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.httpx import HttpxIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from app_settings.settings import get_settings


logger = logging.getLogger(__name__)


def init_observability(service_name: str, service_version: str = "1.0.0") -> None:
    """
    Initialize comprehensive observability with Sentry integration.
    
    Args:
        service_name: Name of the service (e.g., 'equity-api', 'orchestrator')
        service_version: Version of the service
    """
    settings = get_settings()
    obs_config = settings.observability
    
    # Skip initialization if Sentry DSN not configured
    if not obs_config.sentry_dsn:
        logger.info("Sentry DSN not configured, skipping observability initialization")
        return
    
    # Determine environment from settings
    environment = obs_config.sentry_environment or settings.env.value
    
    # Configure integrations based on feature flags
    integrations = []
    
    # Always add logging integration for error capture
    if obs_config.enable_error_tracking:
        integrations.append(
            LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR
            )
        )
    
    # Add FastAPI integration for web services
    if obs_config.enable_performance_monitoring:
        try:
            # Try newer FastAPI integration first
            integrations.append(FastApiIntegration(auto_integration=True))
        except TypeError:
            # Fallback for older Sentry versions
            integrations.append(FastApiIntegration())
        integrations.append(HttpxIntegration())
    
    # Add Celery integration for async tasks
    if obs_config.enable_celery_monitoring:
        integrations.append(CeleryIntegration())
    
    # Initialize Sentry SDK
    sentry_sdk.init(
        dsn=obs_config.sentry_dsn,
        environment=environment,
        release=obs_config.sentry_release or service_version,
        sample_rate=obs_config.sentry_sample_rate,
        traces_sample_rate=obs_config.sentry_traces_sample_rate if obs_config.enable_performance_monitoring else 0.0,
        profiles_sample_rate=obs_config.sentry_profiles_sample_rate if obs_config.enable_profiling else 0.0,
        max_breadcrumbs=obs_config.sentry_max_breadcrumbs,
        attach_stacktrace=obs_config.sentry_attach_stacktrace,
        send_default_pii=obs_config.sentry_send_default_pii,
        debug=obs_config.sentry_debug,
        integrations=integrations,
        before_send=_before_send_filter,
        before_send_transaction=_before_send_transaction_filter,
    )
    
    # Set service-level tags and context
    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("service", service_name)
        scope.set_tag("version", service_version)
        scope.set_tag("environment", environment)
        scope.set_tag("hostname", socket.gethostname())
        
        # Set trading platform context
        scope.set_context("trading_platform", {
            "service": service_name,
            "version": service_version,
            "data_path": str(settings.paths.data_root),
            "ib_host": settings.ib.host,
            "ib_port": settings.ib.port,
        })
    
    logger.info(
        f"Observability initialized for {service_name} v{service_version} "
        f"(env: {environment}, dsn configured: {bool(obs_config.sentry_dsn)})"
    )


def _before_send_filter(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filter events before sending to Sentry.
    Remove sensitive information and filter out noise.
    """
    # Filter out known non-critical exceptions
    if 'exc_info' in hint:
        exc_type, exc_value, tb = hint['exc_info']
        
        # Skip certain exception types that are expected
        if exc_type.__name__ in [
            'KeyboardInterrupt',
            'ConnectionError',  # IB API connection issues are common
            'TimeoutError',     # Timeout errors are handled gracefully
        ]:
            return None
    
    # Remove sensitive data from request headers
    if 'request' in event and 'headers' in event['request']:
        sensitive_headers = ['authorization', 'cookie', 'x-api-key']
        for header in sensitive_headers:
            if header in event['request']['headers']:
                event['request']['headers'][header] = '[Filtered]'
    
    # Scrub sensitive data from extra context
    if 'extra' in event:
        sensitive_keys = ['password', 'token', 'key', 'secret', 'dsn']
        for key in list(event['extra'].keys()):
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                event['extra'][key] = '[Filtered]'
    
    return event


def _before_send_transaction_filter(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filter performance transactions before sending to Sentry.
    Remove low-value transactions and sensitive data.
    """
    # Skip health check endpoints
    if event.get('transaction') in ['/health', '/healthcheck', '/ping']:
        return None
    
    # Skip very fast transactions (less than 10ms) for internal endpoints
    if (event.get('start_timestamp', 0) and 
        event.get('timestamp', 0) and
        (event['timestamp'] - event['start_timestamp']) < 0.01):
        if event.get('transaction', '').startswith('/internal/'):
            return None
    
    return event


def get_sentry_hub() -> Hub:
    """Get the current Sentry hub for advanced operations."""
    return sentry_sdk.Hub.current


def capture_trading_event(
    message: str,
    level: str = "info",
    extra: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None
) -> None:
    """
    Capture a trading-specific event with context.
    
    Args:
        message: Event message
        level: Severity level (debug, info, warning, error, fatal)
        extra: Additional context data
        tags: Event tags
    """
    with sentry_sdk.configure_scope() as scope:
        # Add trading-specific tags
        if tags:
            for key, value in tags.items():
                scope.set_tag(f"trading.{key}", value)
        
        # Add extra context
        if extra:
            scope.set_context("trading_event", extra)
    
    sentry_sdk.capture_message(message, level=level)


def capture_trading_exception(
    exception: Exception,
    extra: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None
) -> None:
    """
    Capture a trading-specific exception with context.
    
    Args:
        exception: The exception to capture
        extra: Additional context data
        tags: Exception tags
    """
    with sentry_sdk.configure_scope() as scope:
        # Add trading-specific tags
        if tags:
            for key, value in tags.items():
                scope.set_tag(f"trading.{key}", value)
        
        # Add extra context
        if extra:
            scope.set_context("trading_exception", extra)
    
    sentry_sdk.capture_exception(exception)


def set_user_context(user_id: str, email: Optional[str] = None, username: Optional[str] = None) -> None:
    """Set user context for tracking."""
    sentry_sdk.set_user({
        "id": user_id,
        "email": email,
        "username": username,
    })


def set_trading_context(
    symbol: Optional[str] = None,
    operation: Optional[str] = None,
    data_source: Optional[str] = None,
    **kwargs
) -> None:
    """
    Set trading-specific context for all subsequent events.
    
    Args:
        symbol: Trading symbol (e.g., 'AAPL')
        operation: Trading operation (e.g., 'bars_fetch', 'option_chain')
        data_source: Data source (e.g., 'ib_api', 'nasdaq')
        **kwargs: Additional context
    """
    context = {}
    
    if symbol:
        context["symbol"] = symbol
    if operation:
        context["operation"] = operation
    if data_source:
        context["data_source"] = data_source
    
    context.update(kwargs)
    
    with sentry_sdk.configure_scope() as scope:
        scope.set_context("trading", context)
        
        # Also set as tags for easier filtering
        if symbol:
            scope.set_tag("trading.symbol", symbol)
        if operation:
            scope.set_tag("trading.operation", operation)
        if data_source:
            scope.set_tag("trading.data_source", data_source)