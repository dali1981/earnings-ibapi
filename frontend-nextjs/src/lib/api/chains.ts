import { fetchApi, API_BASES } from './client';
import type { OptionChain } from '@/types/api';

export const chainsApi = {
  /**
   * Get latest option chain snapshot
   */
  async getLatest(symbol: string) {
    const url = `${API_BASES.chains}/chains/${encodeURIComponent(symbol)}/latest`;
    return fetchApi<OptionChain>(url);
  },

  /**
   * Get available as-of dates for option chains
   */
  async getAsOfs(symbol: string) {
    const url = `${API_BASES.chains}/chains/${encodeURIComponent(symbol)}/as-ofs`;
    return fetchApi<string[]>(url);
  },

  /**
   * Get option chain summary for specific date
   */
  async getSummary(symbol: string, asOf: string) {
    const url = `${API_BASES.chains}/chains/${encodeURIComponent(symbol)}/${encodeURIComponent(asOf)}/summary`;
    return fetchApi<any>(url);
  },

  /**
   * Get available expiries for option chains
   */
  async getExpiries(symbol: string, asOf: string) {
    const url = `${API_BASES.chains}/chains/${encodeURIComponent(symbol)}/${encodeURIComponent(asOf)}/expiries`;
    return fetchApi<string[]>(url);
  },

  /**
   * Get available strikes for option chains  
   */
  async getStrikes(symbol: string, asOf: string) {
    const url = `${API_BASES.chains}/chains/${encodeURIComponent(symbol)}/${encodeURIComponent(asOf)}/strikes`;
    return fetchApi<number[]>(url);
  },

  /**
   * Get option chain parameters
   */
  async getParams(symbol: string, asOf: string) {
    const url = `${API_BASES.chains}/chains/${encodeURIComponent(symbol)}/${encodeURIComponent(asOf)}/params`;
    return fetchApi<any>(url);
  },
};