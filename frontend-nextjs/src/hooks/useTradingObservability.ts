import { useCallback, useEffect, useRef } from 'react';
import * as Sentry from '@sentry/react';

interface OperationContext {
  transaction: ReturnType<typeof Sentry.startTransaction>;
  operation: string;
  symbol: string | null;
  startTime: number;
}

export function useTradingObservability() {
  const operationRef = useRef<OperationContext | null>(null);

  const startOperation = useCallback((operation: string, symbol: string | null = null, metadata: Record<string, any> = {}) => {
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
    if (typeof window !== 'undefined' && window.TradingObservability) {
      window.TradingObservability.setTradingContext(symbol, operation);
    }

    return transaction;
  }, []);

  const finishOperation = useCallback((success: boolean = true, metadata: Record<string, any> = {}) => {
    if (!operationRef.current) return;

    const { transaction, operation, symbol, startTime } = operationRef.current;
    const duration = Date.now() - startTime;

    // Track the operation
    if (typeof window !== 'undefined' && window.TradingObservability) {
      window.TradingObservability.trackSymbolOperation(
        symbol,
        operation,
        duration,
        success,
        metadata
      );
    }

    // Finish the transaction
    transaction.setStatus(success ? 'ok' : 'internal_error');
    transaction.finish();

    operationRef.current = null;
  }, []);

  const captureError = useCallback((error: Error, context: Record<string, any> = {}) => {
    if (operationRef.current) {
      context.operation = operationRef.current.operation;
      context.symbol = operationRef.current.symbol;
    }

    Sentry.withScope(scope => {
      scope.setContext('trading_error', context);
      Sentry.captureException(error);
    });
  }, []);

  const captureApiError = useCallback((error: Error, request: RequestInit & { url?: string } = {}, response: Response | null = null) => {
    if (typeof window !== 'undefined' && window.TradingObservability) {
      window.TradingObservability.captureApiError(error, request, response);
    } else {
      captureError(error, { request, response: response ? { status: response.status, statusText: response.statusText } : null });
    }
  }, [captureError]);

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
    captureApiError,
    captureEvent: typeof window !== 'undefined' && window.TradingObservability ? window.TradingObservability.captureEvent : undefined,
    setTradingContext: typeof window !== 'undefined' && window.TradingObservability ? window.TradingObservability.setTradingContext : undefined
  };
}

// Hook for tracking API operations with symbols
export function useSymbolOperation(symbol: string) {
  const { startOperation, finishOperation, captureApiError } = useTradingObservability();

  const executeWithTracking = useCallback(async <T>(
    operation: string,
    apiCall: () => Promise<T>,
    metadata: Record<string, any> = {}
  ): Promise<T> => {
    const transaction = startOperation(operation, symbol, metadata);

    try {
      const result = await apiCall();
      finishOperation(true, {
        ...metadata,
        resultType: typeof result,
        hasData: !!result
      });
      return result;
    } catch (error) {
      const apiError = error as Error & { status?: number; response?: Response };
      captureApiError(
        apiError,
        { url: `${operation} for ${symbol}`, method: 'GET' },
        apiError.response || null
      );
      finishOperation(false, {
        ...metadata,
        error: apiError.message,
        status: apiError.status
      });
      throw error;
    }
  }, [symbol, startOperation, finishOperation, captureApiError]);

  return { executeWithTracking };
}

// Global types for TradingObservability
declare global {
  interface Window {
    TradingObservability?: {
      captureEvent: (message: string, level?: string, extra?: Record<string, any>, tags?: Record<string, string>) => void;
      captureApiError: (error: Error, request?: any, response?: any) => void;
      trackSymbolOperation: (symbol: string | null, operation: string, duration: number | null, success: boolean, metadata?: Record<string, any>) => void;
      setTradingContext: (symbol?: string | null, operation?: string | null, correlationId?: string | null) => void;
    };
  }
}