/**
 * Lightweight frontend API client for Trading Services.
 *
 * Purpose
 * - Centralize all backend HTTP calls behind a small, discoverable surface.
 * - Expose a global `window.TradingAPI` for reuse in controllers and console.
 * - Normalize responses to `{ data, res }` and raise errors on non-OK.
 *
 * Configuration
 * - Reads service bases from `window.__BASES__` injected by the template.
 * - Falls back to localhost defaults for local development.
 *
 * Examples (in DevTools)
 *   TradingAPI.calendar.getUpcoming(7, 'AAPL,MSFT').then(({data}) => console.log(data))
 *   TradingAPI.equity.bars('AAPL', '2024-01-01', '2024-03-01', '1 day').then(({data}) => console.log(data))
 */

(function () {
  const DEFAULT_BASES = {
    calendar: 'http://localhost:9001',
    equity: 'http://localhost:9002',
    chains: 'http://localhost:9004',
    orchestrator: 'http://localhost:9005',
  };

  const BASES = (window.__BASES__ && typeof window.__BASES__ === 'object')
    ? window.__BASES__
    : DEFAULT_BASES;

  /**
   * Fetch helper returning `{ data, res }` and throwing on non-OK.
   * Attempts JSON parse; falls back to text; null on failure.
   * @param {string} url
   * @param {RequestInit} [init]
   * @returns {Promise<{data:any,res:Response}>}
   */
  async function fetchJson(url, init) {
    const res = await fetch(url, init);
    const contentType = res.headers.get('content-type') || '';
    let body;
    try {
      if (contentType.includes('application/json')) {
        body = await res.json();
      } else {
        body = await res.text();
      }
    } catch (_) {
      body = null;
    }
    if (!res.ok) {
      const msg = body && typeof body === 'object' && body.error
        ? body.error
        : `${res.status} ${res.statusText}`;
      const err = new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
      err.status = res.status;
      err.response = res;
      err.data = body;
      throw err;
    }
    return { data: body, res };
  }

  const json = (v) => JSON.stringify(v);
  const hdrs = { 'content-type': 'application/json' };

  /** Public API exposed as `window.TradingAPI`. */
  const api = {
    calendar: {
      /**
       * Read upcoming earnings from local repository.
       * @param {number|string} days
       * @param {string=} symbols - comma-separated (optional)
       */
      getUpcoming(days, symbols) {
        const url = `${BASES.calendar}/calendar/upcoming?days=${encodeURIComponent(days)}` +
          (symbols ? `&symbols=${encodeURIComponent(symbols)}` : '');
        return fetchJson(url);
      },
      /**
       * Fetch from external source(s) and persist locally.
       * @param {number|string} days
       * @param {string} source - 'nasdaq' | 'yahoo' | 'all'
       * @param {string=} symbols - required for 'yahoo'
       */
      fetchUpcoming(days, source, symbols) {
        const qs = new URLSearchParams({ days: String(days), source });
        if (symbols) qs.append('symbols', symbols);
        const url = `${BASES.calendar}/calendar/fetch-upcoming?${qs.toString()}`;
        return fetchJson(url, { method: 'POST' });
      },
    },
    orchestrator: {
      /** @param {{symbols:string[],sec_types:string[],currency:string,exchanges:string[]}} payload */
      search(payload) {
        return fetchJson(`${BASES.orchestrator}/orchestrator/contracts/search`, {
          method: 'POST', headers: hdrs, body: json(payload),
        });
      },
      /** @param {{symbols:string[]}} payload */
      details(payload) {
        return fetchJson(`${BASES.orchestrator}/orchestrator/contracts/details`, {
          method: 'POST', headers: hdrs, body: json(payload),
        });
      },
      /** @param {{symbols:string[],sec_types:string[],currency:string,exchanges:string[]}} payload */
      setup(payload) {
        return fetchJson(`${BASES.orchestrator}/orchestrator/contracts/setup`, {
          method: 'POST', headers: hdrs, body: json(payload),
        });
      },
    },
    equity: {
      /**
       * Get historical equity bars.
       * @param {string} symbol
       * @param {string} start - YYYY-MM-DD
       * @param {string} end - YYYY-MM-DD
       * @param {string} barSize - e.g. '1 day'
       */
      bars(symbol, start, end, barSize) {
        const url = `${BASES.equity}/equity/${encodeURIComponent(symbol)}/bars` +
          `?start=${encodeURIComponent(start || '')}` +
          `&end=${encodeURIComponent(end || '')}` +
          `&bar_size=${encodeURIComponent(barSize || '')}`;
        return fetchJson(url);
      },
    },
    chains: {
      /** Get latest option chain snapshot for a symbol. */
      latest(symbol) {
        const url = `${BASES.chains}/chains/${encodeURIComponent(symbol)}/latest`;
        return fetchJson(url);
      },
    },
    // Expose resolved bases for quick inspection
    _bases: BASES,
  };

  // Attach to window for console/devtools usage
  window.TradingAPI = api;
})();
