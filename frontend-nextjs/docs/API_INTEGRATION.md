# API Integration Guide

## Overview

The frontend integrates with multiple backend microservices through a centralized API client architecture.

## API Client Architecture

### Base Configuration
```typescript
// Environment-aware API base URLs
const API_BASES = {
  calendar: process.env.NODE_ENV === 'production' 
    ? 'https://your-domain.com:9001' 
    : 'http://localhost:9001',
  equity: process.env.NODE_ENV === 'production'
    ? 'https://your-domain.com:9002'
    : 'http://localhost:9002',
  // ... other services
};
```

### Centralized Client
```typescript
// src/lib/api/index.ts
export const api = {
  calendar: calendarApi,
  equity: equityApi,
  chains: chainsApi,
  orchestrator: orchestratorApi,
} as const;
```

## Service Endpoints

### Calendar API (Port 9001)
```typescript
// Get earnings events
GET /calendar/events?start_date=2024-01-01&end_date=2024-12-31&symbols=AAPL,MSFT

// Fetch new earnings data
POST /calendar/fetch
```

### Equity API (Port 9002)
```typescript
// Get available dates for symbol
GET /equity/{symbol}/dates?bar_size=1%20day

// Get historical bars
GET /equity/{symbol}/bars?start=2024-01-01&end=2024-01-31&bar_size=1%20day
```

### Option Chain API (Port 9004)
```typescript
// Browse option chains
GET /chains/{symbol}?as_of=2024-01-01

// Get available dates
GET /chains/{symbol}/dates
```

### Orchestrator API (Port 9005)
```typescript
// Contract management
POST /orchestrator/contracts/search
POST /orchestrator/contracts/details
POST /orchestrator/contracts/setup

// Historical data fetching
POST /orchestrator/historical/bars-by-symbols

// Option contract operations
POST /orchestrator/option-contracts/select
POST /orchestrator/option-contracts/enrich
POST /orchestrator/option-contracts/save
GET  /orchestrator/option-contracts/availability
GET  /orchestrator/option-contracts/expiries
```

## React Query Integration

### Query Patterns
```typescript
// Automatic caching and refetching
const { data, isLoading, error } = useQuery({
  queryKey: ['calendar-events', startDate, endDate],
  queryFn: () => api.calendar.getEvents(startDate, endDate),
  staleTime: 5 * 60 * 1000, // 5 minutes
});
```

### Mutation Patterns
```typescript
// Optimistic updates with error handling
const mutation = useMutation({
  mutationFn: api.orchestrator.fetchHistoricalBars,
  onSuccess: () => {
    queryClient.invalidateQueries(['historical-data']);
    toast.success('Data fetched successfully');
  },
  onError: (error) => {
    toast.error(error.message);
  },
});
```

## Error Handling

### API Client Level
```typescript
// Centralized error handling
export async function fetchApi<T>(url: string, options?: RequestInit): Promise<ApiResponse<T>> {
  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
}
```

### Component Level
```typescript
// User-friendly error display
if (error) {
  toast.error(error.message || 'An unexpected error occurred');
  return <div>Error loading data...</div>;
}
```

## Type Safety

### API Response Types
```typescript
// src/types/api.ts
export interface CalendarEvent {
  symbol: string;
  company_name: string;
  earnings_date: string;
  time: string;
  eps_estimate: number | null;
  revenue_estimate: number | null;
}

export interface ApiResponse<T> {
  data: T;
  status: number;
}
```

### Request Types
```typescript
export interface HistoricalDataRequest {
  symbols: string[];
  end_date_time: string;
  duration_str: string;
  bar_size_setting: string;
  repository_type?: string;
}
```

## Environment Configuration

### Development
```bash
# .env.local
NEXT_PUBLIC_API_BASE_URL=http://localhost:9000
NEXT_PUBLIC_CALENDAR_API_URL=http://localhost:9001
NEXT_PUBLIC_EQUITY_API_URL=http://localhost:9002
# ... other service URLs
```

### Production
```bash
# Production environment variables
NEXT_PUBLIC_API_BASE_URL=https://your-domain.com
NEXT_PUBLIC_CALENDAR_API_URL=https://your-domain.com:9001
# ... configure all service URLs
```

## Best Practices

### 1. Use React Query for All API Calls
- Automatic caching and background updates
- Built-in loading and error states
- Optimistic updates for mutations

### 2. Centralize API Logic
- Keep all endpoint definitions in `/lib/api/`
- Use typed interfaces for all requests/responses
- Handle errors consistently across services

### 3. Environment-Aware Configuration
- Use environment variables for all API URLs
- Support both development and production configurations
- Fail gracefully when services are unavailable

### 4. User Experience
- Show loading states during API calls
- Display user-friendly error messages
- Use toast notifications for feedback
- Cache data appropriately to minimize requests

## Troubleshooting

### Common Issues

**CORS Errors**: Ensure backend services include proper CORS headers
```typescript
// Backend CORS configuration needed
headers: {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization'
}
```

**Network Timeouts**: Adjust request timeouts for long-running operations
```typescript
const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), 30000);
```

**Type Errors**: Keep TypeScript definitions synchronized with backend APIs
- Update `/types/api.ts` when backend schemas change
- Use generic types for flexible API responses

### Development Tools
- Use browser DevTools Network tab to inspect API calls
- React Query DevTools for cache inspection
- Console logging for debugging API client behavior