// Base API Client with error handling
export interface ApiResponse<T> {
  data: T;
  response: Response;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public response: Response,
    public data?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function fetchApi<T>(
  url: string,
  init?: RequestInit
): Promise<ApiResponse<T>> {
  const startTime = Date.now();

  try {
    const response = await fetch(url, init);
    const contentType = response.headers.get('content-type') || '';

    let body: any;
    try {
      if (contentType.includes('application/json')) {
        body = await response.json();
      } else {
        body = await response.text();
      }
    } catch {
      body = null;
    }

    if (!response.ok) {
      const message = body && typeof body === 'object' && body.error
        ? body.error
        : `${response.status} ${response.statusText}`;

      const apiError = new ApiError(
        typeof message === 'string' ? message : JSON.stringify(message),
        response.status,
        response,
        body
      );

      // Capture API error with Sentry
      if (typeof window !== 'undefined' && window.TradingObservability) {
        window.TradingObservability.captureApiError(
          apiError,
          { url, method: init?.method || 'GET', headers: init?.headers },
          response
        );
      }

      throw apiError;
    }

    // Track successful API call
    const duration = Date.now() - startTime;
    if (typeof window !== 'undefined' && window.TradingObservability) {
      // Extract operation and symbol from URL for better tracking
      const urlObj = new URL(url, window.location.origin);
      const pathParts = urlObj.pathname.split('/').filter(Boolean);
      const operation = pathParts.length > 1 ? `${pathParts[0]}_${pathParts[pathParts.length - 1]}` : pathParts[0] || 'api_call';
      const symbol = pathParts.find(part => /^[A-Z]{1,5}$/.test(part)) || null; // Simple symbol detection

      window.TradingObservability.trackSymbolOperation(
        symbol,
        operation,
        duration,
        true,
        {
          url: urlObj.pathname,
          method: init?.method || 'GET',
          status: response.status,
          contentType
        }
      );
    }

    return { data: body, response };
  } catch (error) {
    const duration = Date.now() - startTime;

    // Capture network/fetch errors
    if (typeof window !== 'undefined' && window.TradingObservability && !(error instanceof ApiError)) {
      window.TradingObservability.captureApiError(
        error as Error,
        { url, method: init?.method || 'GET', headers: init?.headers },
        null
      );

      // Track failed API call
      const urlObj = new URL(url, window.location.origin);
      const pathParts = urlObj.pathname.split('/').filter(Boolean);
      const operation = pathParts.length > 1 ? `${pathParts[0]}_${pathParts[pathParts.length - 1]}` : pathParts[0] || 'api_call';
      const symbol = pathParts.find(part => /^[A-Z]{1,5}$/.test(part)) || null;

      window.TradingObservability.trackSymbolOperation(
        symbol,
        operation,
        duration,
        false,
        {
          url: urlObj.pathname,
          method: init?.method || 'GET',
          error: (error as Error).message
        }
      );
    }

    throw error;
  }
}

// Base URLs from environment
export const API_BASES = {
  calendar: process.env.NEXT_PUBLIC_CALENDAR_API_URL || 'http://localhost:9001',
  equity: process.env.NEXT_PUBLIC_EQUITY_API_URL || 'http://localhost:9002',
  chains: process.env.NEXT_PUBLIC_CHAINS_API_URL || 'http://localhost:9004',
  orchestrator: process.env.NEXT_PUBLIC_ORCHESTRATOR_API_URL || 'http://localhost:9005',
} as const;

// JSON headers
export const JSON_HEADERS = {
  'Content-Type': 'application/json',
} as const;