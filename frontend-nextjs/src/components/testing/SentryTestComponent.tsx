'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useTradingObservability } from '@/hooks/useTradingObservability';
import * as Sentry from '@sentry/react';

export function SentryTestComponent() {
  const [testResults, setTestResults] = useState<string[]>([]);
  const { startOperation, finishOperation, captureError, captureApiError } = useTradingObservability();

  const addResult = (message: string) => {
    setTestResults(prev => [...prev, `${new Date().toLocaleTimeString()}: ${message}`]);
  };

  const testBasicError = () => {
    try {
      throw new Error('Test error for Sentry verification');
    } catch (error) {
      captureError(error as Error, { test: 'basic_error', component: 'SentryTestComponent' });
      addResult('✅ Basic error captured');
    }
  };

  const testApiError = () => {
    const mockApiError = new Error('API request failed');
    const mockRequest = {
      url: '/api/equity/AAPL/bars',
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    };
    const mockResponse = {
      status: 500,
      statusText: 'Internal Server Error'
    } as Response;

    captureApiError(mockApiError, mockRequest, mockResponse);
    addResult('✅ API error captured with context');
  };

  const testTradingOperation = () => {
    const transaction = startOperation('test_symbol_fetch', 'AAPL', { test: true });

    // Simulate some work
    setTimeout(() => {
      finishOperation(true, { rows: 100, duration: 250 });
      addResult('✅ Trading operation tracked successfully');
    }, 100);

    addResult('🔄 Started trading operation tracking...');
  };

  const testTradingEvent = () => {
    if (typeof window !== 'undefined' && window.TradingObservability) {
      window.TradingObservability.captureEvent(
        'Test trading event from frontend',
        'info',
        { symbol: 'AAPL', operation: 'test_event', test: true },
        { component: 'SentryTestComponent', test_type: 'trading_event' }
      );
      addResult('✅ Trading event captured');
    } else {
      addResult('❌ TradingObservability not available');
    }
  };

  const testBreadcrumb = () => {
    Sentry.addBreadcrumb({
      message: 'Test breadcrumb from trading platform',
      category: 'testing',
      level: 'info',
      data: {
        component: 'SentryTestComponent',
        timestamp: new Date().toISOString()
      }
    });
    addResult('✅ Breadcrumb added');
  };

  const testPerformanceTransaction = () => {
    const transaction = Sentry.startTransaction({
      name: 'test_frontend_operation',
      op: 'test.operation'
    });

    // Simulate work
    setTimeout(() => {
      transaction.setStatus('ok');
      transaction.finish();
      addResult('✅ Performance transaction completed');
    }, 200);

    addResult('🔄 Started performance transaction...');
  };

  const testSetContext = () => {
    Sentry.setContext('test_context', {
      test_mode: true,
      component: 'SentryTestComponent',
      timestamp: new Date().toISOString(),
      user_agent: navigator.userAgent
    });

    if (typeof window !== 'undefined' && window.TradingObservability) {
      window.TradingObservability.setTradingContext('TEST', 'sentry_verification', 'test-correlation-123');
    }

    addResult('✅ Context and trading context set');
  };

  const clearResults = () => {
    setTestResults([]);
  };

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>Sentry Integration Test Panel</CardTitle>
        <p className="text-sm text-muted-foreground">
          Test various Sentry features and trading observability functions.
          Check your Sentry dashboard to verify events are being captured.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-2">
          <Button onClick={testBasicError} variant="outline" size="sm">
            Test Basic Error
          </Button>
          <Button onClick={testApiError} variant="outline" size="sm">
            Test API Error
          </Button>
          <Button onClick={testTradingOperation} variant="outline" size="sm">
            Test Trading Operation
          </Button>
          <Button onClick={testTradingEvent} variant="outline" size="sm">
            Test Trading Event
          </Button>
          <Button onClick={testBreadcrumb} variant="outline" size="sm">
            Test Breadcrumb
          </Button>
          <Button onClick={testPerformanceTransaction} variant="outline" size="sm">
            Test Performance
          </Button>
          <Button onClick={testSetContext} variant="outline" size="sm">
            Test Context
          </Button>
          <Button onClick={clearResults} variant="secondary" size="sm">
            Clear Results
          </Button>
        </div>

        {testResults.length > 0 && (
          <div className="bg-muted rounded-lg p-3">
            <h4 className="text-sm font-medium mb-2">Test Results:</h4>
            <div className="space-y-1 text-xs font-mono">
              {testResults.map((result, index) => (
                <div key={index} className="text-muted-foreground">
                  {result}
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-3 text-sm">
          <p className="font-medium text-blue-900 dark:text-blue-100 mb-1">
            📝 Instructions:
          </p>
          <ul className="text-blue-800 dark:text-blue-200 space-y-1 text-xs">
            <li>• Set up your Sentry DSN in <code>.env.local</code></li>
            <li>• Run tests and check your Sentry dashboard</li>
            <li>• Look for errors, transactions, and breadcrumbs</li>
            <li>• Verify trading-specific tags and context are present</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}