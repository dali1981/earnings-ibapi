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
    
    throw new ApiError(
      typeof message === 'string' ? message : JSON.stringify(message),
      response.status,
      response,
      body
    );
  }

  return { data: body, response };
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