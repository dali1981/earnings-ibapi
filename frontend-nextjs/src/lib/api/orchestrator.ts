import { fetchApi, API_BASES, JSON_HEADERS } from './client';
import type { OrchestratorResult, HistoricalDataRequest, ContractSearchRequest, ContractSelectRequest, OptionContract } from '@/types/api';

export const orchestratorApi = {
  /**
   * Search for contract descriptions
   */
  async searchContracts(payload: ContractSearchRequest) {
    const url = `${API_BASES.orchestrator}/orchestrator/contracts/search`;
    return fetchApi<OrchestratorResult>(url, {
      method: 'POST',
      headers: JSON_HEADERS,
      body: JSON.stringify(payload),
    });
  },

  /**
   * Fetch detailed contract information
   */
  async getContractDetails(payload: { symbols: string[] }) {
    const url = `${API_BASES.orchestrator}/orchestrator/contracts/details`;
    return fetchApi<OrchestratorResult>(url, {
      method: 'POST',
      headers: JSON_HEADERS,
      body: JSON.stringify(payload),
    });
  },

  /**
   * Setup contracts (search + details)
   */
  async setupContracts(payload: ContractSearchRequest) {
    const url = `${API_BASES.orchestrator}/orchestrator/contracts/setup`;
    return fetchApi<OrchestratorResult>(url, {
      method: 'POST',
      headers: JSON_HEADERS,
      body: JSON.stringify(payload),
    });
  },

  /**
   * Fetch historical bars via orchestrator
   */
  async fetchHistoricalBars(payload: HistoricalDataRequest) {
    const url = `${API_BASES.orchestrator}/orchestrator/historical/bars-by-symbols`;
    return fetchApi<OrchestratorResult>(url, {
      method: 'POST',
      headers: JSON_HEADERS,
      body: JSON.stringify(payload),
    });
  },

  /**
   * Option contracts operations
   */
  optionContracts: {
    /**
     * Check data availability for option contracts
     */
    async availability(symbol: string, expiry?: string) {
      const params = new URLSearchParams({ symbol });
      if (expiry) params.append('expiry', expiry);
      const url = `${API_BASES.orchestrator}/orchestrator/option-contracts/availability?${params}`;
      return fetchApi(url);
    },

    /**
     * Get available expiries for a symbol
     */
    async expiries(symbol: string) {
      const params = new URLSearchParams({ symbol });
      const url = `${API_BASES.orchestrator}/orchestrator/option-contracts/expiries?${params}`;
      return fetchApi(url);
    },

    /**
     * Select option contracts
     */
    async select(payload: ContractSelectRequest) {
      const url = `${API_BASES.orchestrator}/orchestrator/option-contracts/select`;
      return fetchApi<OrchestratorResult>(url, {
        method: 'POST',
        headers: JSON_HEADERS,
        body: JSON.stringify(payload),
      });
    },

    /**
     * Get status of contract selection task
     */
    async selectStatus(taskId: string) {
      const url = `${API_BASES.orchestrator}/orchestrator/option-contracts/select/status/${taskId}`;
      return fetchApi(url);
    },

    /**
     * Enrich contracts with additional data
     */
    async enrich(payload: { contracts: OptionContract[] }) {
      const url = `${API_BASES.orchestrator}/orchestrator/option-contracts/enrich`;
      return fetchApi<OrchestratorResult>(url, {
        method: 'POST',
        headers: JSON_HEADERS,
        body: JSON.stringify(payload),
      });
    },

    /**
     * Save contracts to repository
     */
    async save(payload: { contracts: OptionContract[] }) {
      const url = `${API_BASES.orchestrator}/orchestrator/option-contracts/save`;
      return fetchApi<OrchestratorResult>(url, {
        method: 'POST',
        headers: JSON_HEADERS,
        body: JSON.stringify(payload),
      });
    },

    /**
     * Fetch historical option bar data
     */
    async historical(payload: { 
      contracts: OptionContract[];
      end_date_time: string;
      duration_str: string;
      bar_size_setting: string;
      timeout_seconds?: number;
    }) {
      const url = `${API_BASES.orchestrator}/orchestrator/option-contracts/historical`;
      return fetchApi<OrchestratorResult>(url, {
        method: 'POST',
        headers: JSON_HEADERS,
        body: JSON.stringify(payload),
      });
    },

    /**
     * Browse saved option bar data
     */
    async browseBars(params: {
      symbol: string;
      expiry?: string;
      right?: string;
      strike?: string;
      bar_size?: string;
      start_date?: string;
      end_date?: string;
      limit?: number;
    }) {
      const queryParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== '') {
          queryParams.append(key, String(value));
        }
      });
      const url = `${API_BASES.orchestrator}/orchestrator/option-contracts/browse-bars?${queryParams}`;
      return fetchApi(url);
    },
  },
};