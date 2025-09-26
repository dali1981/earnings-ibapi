'use client';

import React from 'react';
import * as Sentry from '@sentry/react';

// Error Boundary for catching React errors
const TradingErrorBoundary = Sentry.withErrorBoundary(
  ({ children }: { children: React.ReactNode }) => children,
  {
    fallback: ({ error, resetError }) => (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="max-w-md w-full mx-4">
          <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-6 text-center">
            <div className="w-12 h-12 mx-auto mb-4 bg-destructive/20 rounded-full flex items-center justify-center">
              <svg
                className="w-6 h-6 text-destructive"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 18.5C3.498 20.333 4.46 22 6 22z"
                />
              </svg>
            </div>
            <h2 className="text-xl font-semibold text-foreground mb-2">
              Something went wrong
            </h2>
            <p className="text-muted-foreground mb-4">
              An error occurred in the trading platform. The error has been reported automatically.
            </p>
            <div className="space-y-2">
              <button
                onClick={resetError}
                className="w-full bg-primary text-primary-foreground hover:bg-primary/90 px-4 py-2 rounded-md font-medium transition-colors"
              >
                Try again
              </button>
              <button
                onClick={() => window.location.reload()}
                className="w-full bg-secondary text-secondary-foreground hover:bg-secondary/90 px-4 py-2 rounded-md font-medium transition-colors"
              >
                Reload page
              </button>
            </div>
            {process.env.NODE_ENV === 'development' && (
              <details className="mt-4 text-left">
                <summary className="cursor-pointer text-sm text-muted-foreground hover:text-foreground">
                  Error Details (Development)
                </summary>
                <pre className="mt-2 text-xs bg-muted p-2 rounded overflow-auto text-muted-foreground">
                  {error?.toString()}
                </pre>
              </details>
            )}
          </div>
        </div>
      </div>
    ),
    beforeCapture: (scope, error, errorInfo) => {
      scope.setTag('error_boundary', 'trading_platform');
      scope.setContext('react_error_info', errorInfo);
      scope.setLevel('error');
    }
  }
);

// Provider component that wraps the app with Sentry error boundary
export function SentryProvider({ children }: { children: React.ReactNode }) {
  return <TradingErrorBoundary>{children}</TradingErrorBoundary>;
}

// HOC for wrapping individual components with error boundaries
export function withTradingErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  componentName?: string
) {
  return Sentry.withErrorBoundary(Component, {
    fallback: ({ error, resetError }) => (
      <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 text-center">
        <p className="text-sm text-destructive mb-2">
          Error in {componentName || Component.displayName || Component.name || 'component'}
        </p>
        <button
          onClick={resetError}
          className="text-xs bg-destructive text-destructive-foreground px-2 py-1 rounded"
        >
          Retry
        </button>
        {process.env.NODE_ENV === 'development' && (
          <details className="mt-2 text-left">
            <summary className="cursor-pointer text-xs text-muted-foreground">
              Error Details
            </summary>
            <pre className="mt-1 text-xs bg-muted p-2 rounded overflow-auto">
              {error?.toString()}
            </pre>
          </details>
        )}
      </div>
    ),
    beforeCapture: (scope, error) => {
      scope.setTag('component', componentName || Component.displayName || Component.name || 'unknown');
      scope.setTag('error_boundary', 'component_level');
    }
  });
}

// Performance profiling wrapper for components
export function withTradingProfiler<P extends object>(
  Component: React.ComponentType<P>,
  componentName?: string
) {
  return Sentry.withProfiler(Component, {
    name: componentName || Component.displayName || Component.name || 'TradingComponent'
  });
}