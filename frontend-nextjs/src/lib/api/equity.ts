import { fetchApi, API_BASES } from './client';
import type { EquityBarsResponse, EquityDateRange } from '@/types/api';

export const equityApi = {
  /**
   * Get historical equity bars from repository
   */
  async getBars(symbol: string, start?: string, end?: string, barSize?: string) {
    const params = new URLSearchParams();
    if (start) params.append('start', start);
    if (end) params.append('end', end);
    if (barSize) params.append('bar_size', barSize);

    const url = `${API_BASES.equity}/equity/${encodeURIComponent(symbol)}/bars?${params.toString()}`;
    return fetchApi<EquityBarsResponse>(url);
  },

  /**
   * Get available dates for a symbol
   */
  async getDates(symbol: string, barSize: string = '1 day') {
    const url = `${API_BASES.equity}/equity/${encodeURIComponent(symbol)}/dates?bar_size=${encodeURIComponent(barSize)}`;
    return fetchApi<EquityDateRange>(url);
  },
};