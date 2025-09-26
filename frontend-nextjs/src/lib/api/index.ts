// Import APIs
import { calendarApi } from './calendar';
import { equityApi } from './equity';
import { chainsApi } from './chains';
import { orchestratorApi } from './orchestrator';

// Re-export individual APIs
export { calendarApi } from './calendar';
export { equityApi } from './equity';
export { chainsApi } from './chains';
export { orchestratorApi } from './orchestrator';
export { fetchApi, ApiError, API_BASES, JSON_HEADERS } from './client';
export type { ApiResponse } from './client';

// Consolidated API object for easy imports
export const api = {
  calendar: calendarApi,
  equity: equityApi,
  chains: chainsApi,
  orchestrator: orchestratorApi,
} as const;