// API Response Types
export interface CalendarEvent {
  symbol: string;
  company_name: string;
  earnings_date: string;
  time: string;
  eps_estimate: number | null;
  revenue_estimate: number | null;
  source: string;
}

export interface CalendarResponse {
  events: CalendarEvent[];
}

export interface EquityBar {
  symbol: string;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface EquityBarsResponse {
  symbol: string;
  bar_size: string;
  start: string;
  end: string;
  rows: number;
  sample: EquityBar[];
}

export interface EquityDateRange {
  dates: string[];
}

export interface OptionChain {
  symbol: string;
  as_of: string;
  calls: any[];
  puts: any[];
}

export interface OrchestratorResult {
  processed: number;
  saved: number;
  skipped?: number;
  details?: any;
  job_id?: string;
  monitor_url?: string;
}

// Request Types
export interface HistoricalDataRequest {
  symbols: string[];
  end_date_time: string;
  duration_str: string;
  bar_size_setting: string;
  what_to_show?: string;
  use_rth?: number;
  repository_type?: string;
  timeout_seconds?: number;
}

export interface ContractSearchRequest {
  symbols: string[];
  sec_types: string[];
  currency: string;
  exchanges: string[];
}

export interface ContractSelectRequest {
  symbol: string;
  snapshot_date?: string;
  expiry_filter?: string;
  rights: string[];
  strikes_per_side?: number;
  max_contracts?: number;
  selection_mode: string;
  expiry_count?: number;
  spot_price?: number;
}

export interface OptionContract {
  symbol: string;
  expiry: string;
  strike: number;
  right: string;
  conId?: number;
  localSymbol?: string;
  exchange?: string;
  multiplier?: number;
}