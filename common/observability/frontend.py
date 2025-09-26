"""
Frontend observability integration for React and Next.js applications.
Provides Sentry configuration and JavaScript snippets for client-side monitoring.
"""

from typing import Dict, Any, Optional
from app_settings.settings import get_settings


def generate_sentry_config_js(
    release: Optional[str] = None,
    environment: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate JavaScript configuration for Sentry in frontend applications.
    
    Args:
        release: Optional release version override
        environment: Optional environment override  
        user_context: Optional user context for tracking
        
    Returns:
        JavaScript code string for Sentry initialization
    """
    settings = get_settings()
    obs_config = settings.observability
    
    # Use DSN but remove sensitive info for client-side
    dsn = obs_config.sentry_dsn
    if not dsn:
        return "// Sentry DSN not configured - observability disabled"
    
    # Extract just the public parts of the DSN
    # Format: https://public_key@organization.ingest.sentry.io/project_id
    if "@" in dsn:
        public_dsn = dsn  # For client-side, we can use the full DSN
    else:
        return "// Invalid Sentry DSN format"
    
    # Configuration values
    config = {
        "dsn": public_dsn,
        "environment": environment or obs_config.sentry_environment or settings.env.value,
        "release": release or obs_config.sentry_release or "1.0.0",
        "tracesSampleRate": obs_config.sentry_traces_sample_rate if obs_config.enable_performance_monitoring else 0,
        "debug": obs_config.sentry_debug and settings.env.value == "dev",
        "maxBreadcrumbs": obs_config.sentry_max_breadcrumbs,
        "attachStacktrace": obs_config.sentry_attach_stacktrace,
        "sendDefaultPii": obs_config.sentry_send_default_pii,
        "integrations": []
    }
    
    # Add integrations based on feature flags
    integrations = []
    
    if obs_config.enable_performance_monitoring:
        integrations.append("new Sentry.BrowserTracing()")
    
    if obs_config.enable_error_tracking:
        integrations.append("new Sentry.Replay({ sessionSampleRate: 0.1, errorSampleRate: 1.0 })")
    
    # Build JavaScript code
    js_code = f"""
// Sentry Configuration for Trading Platform Frontend
import * as Sentry from "@sentry/react";

Sentry.init({{
  dsn: "{config['dsn']}",
  environment: "{config['environment']}",
  release: "{config['release']}",
  tracesSampleRate: {config['tracesSampleRate']},
  debug: {str(config['debug']).lower()},
  maxBreadcrumbs: {config['maxBreadcrumbs']},
  attachStacktrace: {str(config['attachStacktrace']).lower()},
  sendDefaultPii: {str(config['sendDefaultPii']).lower()},
  
  integrations: [
    {', '.join(integrations) if integrations else ''}
  ],
  
  // Custom error filtering
  beforeSend(event, hint) {{
    // Filter out development errors
    if (event.environment === 'dev' && event.level === 'warning') {{
      return null;
    }}
    
    // Add trading context to all events
    if (event.contexts) {{
      event.contexts.trading = {{
        platform: 'frontend',
        userAgent: navigator.userAgent,
        url: window.location.href,
        timestamp: new Date().toISOString()
      }};
    }}
    
    return event;
  }},
  
  // Custom performance transaction filtering  
  beforeSendTransaction(transaction) {{
    // Skip very fast transactions
    if (transaction.start_timestamp && transaction.timestamp) {{
      const duration = transaction.timestamp - transaction.start_timestamp;
      if (duration < 0.001) {{ // Less than 1ms
        return null;
      }}
    }}
    
    return transaction;
  }}
}});
"""
    
    # Add user context if provided
    if user_context:
        js_code += f"""
// Set user context
Sentry.setUser({user_context});
"""
    
    # Add trading-specific helper functions
    js_code += """
// Trading Platform Helper Functions
window.TradingObservability = {
  // Capture trading-specific events
  captureEvent(message, level = 'info', extra = {}, tags = {}) {
    Sentry.addBreadcrumb({
      message: message,
      level: level,
      category: 'trading',
      data: extra
    });
    
    Sentry.withScope((scope) => {
      // Add trading tags
      Object.keys(tags).forEach(key => {
        scope.setTag(`trading.${key}`, tags[key]);
      });
      
      // Add extra data
      scope.setContext('trading_event', extra);
      
      Sentry.captureMessage(message, level);
    });
  },
  
  // Capture API errors with request/response context
  captureApiError(error, request = {}, response = {}) {
    Sentry.withScope((scope) => {
      scope.setTag('error_type', 'api_error');
      scope.setContext('api_request', {
        url: request.url,
        method: request.method,
        headers: request.headers ? Object.keys(request.headers) : [],
        timestamp: new Date().toISOString()
      });
      
      if (response.status) {
        scope.setContext('api_response', {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers ? Object.keys(response.headers) : []
        });
      }
      
      Sentry.captureException(error);
    });
  },
  
  // Track symbol-specific operations
  trackSymbolOperation(symbol, operation, duration = null, success = true, metadata = {}) {
    Sentry.addBreadcrumb({
      message: `${operation} for ${symbol}`,
      level: success ? 'info' : 'error', 
      category: 'trading.symbol',
      data: {
        symbol: symbol,
        operation: operation,
        duration: duration,
        success: success,
        ...metadata
      }
    });
    
    // Send as custom metric if available, otherwise as events
    try {
      if (Sentry.metrics && Sentry.metrics.increment) {
        Sentry.metrics.increment('trading.operations.total', 1, {
          tags: {
            symbol: symbol,
            operation: operation,
            success: success.toString()
          }
        });
        
        if (duration !== null) {
          Sentry.metrics.distribution('trading.operations.duration', duration, {
            unit: 'millisecond',
            tags: {
              symbol: symbol,
              operation: operation
            }
          });
        }
      } else {
        // Fallback to events
        Sentry.withScope((scope) => {
          scope.setTag('symbol', symbol);
          scope.setTag('operation', operation);
          scope.setTag('success', success.toString());
          scope.setContext('trading_metrics', {
            symbol: symbol,
            operation: operation,
            duration: duration,
            success: success,
            timestamp: new Date().toISOString()
          });
          Sentry.captureMessage(`Trading operation: ${operation} for ${symbol}`, 'info');
        });
      }
    } catch (error) {
      console.warn('Failed to send trading metrics:', error);
    }
  },
  
  // Set current trading context
  setTradingContext(symbol = null, operation = null, correlationId = null) {
    Sentry.withScope((scope) => {
      if (symbol) scope.setTag('trading.symbol', symbol);
      if (operation) scope.setTag('trading.operation', operation);
      if (correlationId) scope.setTag('correlation_id', correlationId);
      
      scope.setContext('trading', {
        symbol: symbol,
        operation: operation,
        correlationId: correlationId,
        timestamp: new Date().toISOString()
      });
    });
  }
};

// React Error Boundary integration
export function createTradingErrorBoundary() {
  return Sentry.withErrorBoundary;
}

// React Router integration
export function createSentryBrowserRouter() {
  return Sentry.wrapCreateBrowserRouter;
}

// Performance monitoring for React components
export function withTradingPerformance(component, componentName) {
  return Sentry.withProfiler(component, { name: componentName });
}
"""
    
    return js_code


def generate_nextjs_config() -> str:
    """
    Generate Next.js specific Sentry configuration.
    
    Returns:
        Next.js configuration code
    """
    settings = get_settings()
    obs_config = settings.observability
    
    if not obs_config.sentry_dsn:
        return "// Sentry DSN not configured - observability disabled"
    
    return f"""
// next.config.js - Sentry configuration
const {{ withSentryConfig }} = require('@sentry/nextjs');

const nextConfig = {{
  // Your existing Next.js configuration
  experimental: {{
    serverComponentsExternalPackages: ['@sentry/nextjs']
  }}
}};

const sentryWebpackPluginOptions = {{
  // Sentry Webpack Plugin options
  silent: true,
  org: process.env.SENTRY_ORG,
  project: process.env.SENTRY_PROJECT,
  authToken: process.env.SENTRY_AUTH_TOKEN,
  
  // Upload source maps in production only
  include: ['.next'],
  ignore: ['node_modules'],
  dryRun: process.env.NODE_ENV !== 'production'
}};

module.exports = withSentryConfig(nextConfig, sentryWebpackPluginOptions);

// sentry.client.config.js
import * as Sentry from '@sentry/nextjs';

Sentry.init({{
  dsn: '{obs_config.sentry_dsn}',
  environment: process.env.NODE_ENV,
  tracesSampleRate: {obs_config.sentry_traces_sample_rate},
  debug: process.env.NODE_ENV === 'development',
  
  integrations: [
    new Sentry.BrowserTracing({{
      routingInstrumentation: Sentry.nextRouterInstrumentation(require('next/router'))
    }})
  ],
  
  beforeSend(event, hint) {{
    // Custom filtering for Next.js
    if (event.exception) {{
      const error = hint.originalException;
      
      // Filter out Next.js hydration warnings in development
      if (process.env.NODE_ENV === 'development' && 
          error.message?.includes('Hydration')) {{
        return null;
      }}
    }}
    
    return event;
  }}
}});

// sentry.server.config.js  
import * as Sentry from '@sentry/nextjs';

Sentry.init({{
  dsn: '{obs_config.sentry_dsn}',
  environment: process.env.NODE_ENV,
  tracesSampleRate: {obs_config.sentry_traces_sample_rate},
  
  // Performance monitoring for API routes
  integrations: [
    new Sentry.Integrations.Http({{ tracing: true }})
  ],
  
  beforeSend(event) {{
    // Server-side filtering
    if (event.request && event.request.url) {{
      // Skip health check endpoints
      if (event.request.url.includes('/health')) {{
        return null;
      }}
    }}
    
    return event;
  }}
}});
"""


def generate_react_hook() -> str:
    """
    Generate React hook for trading operations monitoring.
    
    Returns:
        React hook code
    """
    return """
// hooks/useTradingObservability.js
import { useCallback, useEffect, useRef } from 'react';
import * as Sentry from '@sentry/react';

export function useTradingObservability() {
  const operationRef = useRef(null);
  
  const startOperation = useCallback((operation, symbol = null, metadata = {}) => {
    const transaction = Sentry.startTransaction({
      name: `trading.${operation}`,
      op: 'trading.operation',
      tags: {
        'trading.operation': operation,
        'trading.symbol': symbol
      },
      data: metadata
    });
    
    operationRef.current = {
      transaction,
      operation,
      symbol,
      startTime: Date.now()
    };
    
    // Set context for all subsequent operations
    window.TradingObservability?.setTradingContext(symbol, operation);
    
    return transaction;
  }, []);
  
  const finishOperation = useCallback((success = true, metadata = {}) => {
    if (!operationRef.current) return;
    
    const { transaction, operation, symbol, startTime } = operationRef.current;
    const duration = Date.now() - startTime;
    
    // Track the operation
    window.TradingObservability?.trackSymbolOperation(
      symbol,
      operation,
      duration,
      success,
      metadata
    );
    
    // Finish the transaction
    transaction.setStatus(success ? 'ok' : 'internal_error');
    transaction.finish();
    
    operationRef.current = null;
  }, []);
  
  const captureError = useCallback((error, context = {}) => {
    if (operationRef.current) {
      context.operation = operationRef.current.operation;
      context.symbol = operationRef.current.symbol;
    }
    
    Sentry.withScope(scope => {
      scope.setContext('trading_error', context);
      Sentry.captureException(error);
    });
  }, []);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (operationRef.current?.transaction) {
        operationRef.current.transaction.setStatus('cancelled');
        operationRef.current.transaction.finish();
      }
    };
  }, []);
  
  return {
    startOperation,
    finishOperation,
    captureError,
    captureEvent: window.TradingObservability?.captureEvent,
    setTradingContext: window.TradingObservability?.setTradingContext
  };
}

// Example usage in a component
export function useSymbolData(symbol) {
  const { startOperation, finishOperation, captureError } = useTradingObservability();
  
  const fetchSymbolData = useCallback(async () => {
    const transaction = startOperation('symbol_data_fetch', symbol);
    
    try {
      const response = await fetch(`/api/equity/${symbol}/bars`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const data = await response.json();
      finishOperation(true, { records: data.rows });
      
      return data;
    } catch (error) {
      captureError(error, { symbol, endpoint: 'equity_bars' });
      finishOperation(false, { error: error.message });
      throw error;
    }
  }, [symbol, startOperation, finishOperation, captureError]);
  
  return { fetchSymbolData };
}
"""