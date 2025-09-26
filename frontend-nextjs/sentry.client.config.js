import * as Sentry from "@sentry/nextjs";

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  environment: process.env.NODE_ENV || 'development',
  release: process.env.NEXT_PUBLIC_SENTRY_RELEASE || '1.0.0',
  tracesSampleRate: parseFloat(process.env.NEXT_PUBLIC_SENTRY_TRACES_SAMPLE_RATE || '0.1'),
  debug: process.env.NODE_ENV === 'development',
  maxBreadcrumbs: 50,
  attachStacktrace: true,
  sendDefaultPii: false,

  integrations: [
    Sentry.replayIntegration({
      // Session Replay for debugging user interactions
      sessionSampleRate: 0.1,
      errorSampleRate: 1.0,
    }),
  ],

  // Custom error filtering
  beforeSend(event, hint) {
    // Filter out development errors
    if (event.environment === 'development' && event.level === 'warning') {
      return null;
    }

    // Add trading context to all events
    if (event.contexts) {
      event.contexts.trading = {
        platform: 'frontend',
        userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown',
        url: typeof window !== 'undefined' ? window.location.href : 'unknown',
        timestamp: new Date().toISOString()
      };
    }

    return event;
  },

  // Custom performance transaction filtering
  beforeSendTransaction(transaction) {
    // Skip very fast transactions
    if (transaction.start_timestamp && transaction.timestamp) {
      const duration = transaction.timestamp - transaction.start_timestamp;
      if (duration < 0.001) { // Less than 1ms
        return null;
      }
    }

    return transaction;
  }
});

// Trading Platform Helper Functions
if (typeof window !== 'undefined') {
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
}