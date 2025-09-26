import { fetchApi, API_BASES, JSON_HEADERS } from './client';
import type { CalendarResponse } from '@/types/api';

export const calendarApi = {
  /**
   * Get upcoming earnings from local repository
   */
  async getUpcoming(days: number, symbols?: string) {
    const url = `${API_BASES.calendar}/calendar/upcoming?days=${encodeURIComponent(days)}` +
      (symbols ? `&symbols=${encodeURIComponent(symbols)}` : '');
    return fetchApi<CalendarResponse>(url);
  },

  /**
   * Fetch upcoming earnings from external source and persist
   */
  async fetchUpcoming(days: number, source: string, symbols?: string) {
    const params = new URLSearchParams({ 
      days: String(days), 
      source 
    });
    if (symbols) {
      params.append('symbols', symbols);
    }

    const url = `${API_BASES.calendar}/calendar/fetch-upcoming?${params.toString()}`;
    return fetchApi<CalendarResponse>(url, {
      method: 'POST',
      headers: JSON_HEADERS,
    });
  },
};