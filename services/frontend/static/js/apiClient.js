// Lightweight frontend API client for Trading Services
// Exposes a global `window.TradingAPI` with typed-ish methods.
// Usage examples (in browser console):
//   TradingAPI.calendar.getUpcoming(7, 'AAPL,MSFT').then(({data}) => console.log(data))
//   TradingAPI.equity.bars('AAPL', '2024-01-01', '2024-03-01', '1 day').then(console.log)
//
// This module is intentionally side‑effect free (no event listeners).
// It reads service bases from `window.__BASES__` that the template injects.

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

  const api = {
    calendar: {
      getUpcoming(days, symbols) {
        const url = `${BASES.calendar}/calendar/get-upcoming?days=${encodeURIComponent(days)}` +
          (symbols ? `&symbols=${encodeURIComponent(symbols)}` : '');
        return fetchJson(url);
      },
      fetchUpcoming(days, source, symbols) {
        const qs = new URLSearchParams({ days: String(days), source });
        if (symbols) qs.append('symbols', symbols);
        const url = `${BASES.calendar}/calendar/fetch-upcoming?${qs.toString()}`;
        return fetchJson(url, { method: 'POST' });
      },
    },
    orchestrator: {
      search(payload) {
        return fetchJson(`${BASES.orchestrator}/orchestrator/contracts/search`, {
          method: 'POST', headers: hdrs, body: json(payload),
        });
      },
      details(payload) {
        return fetchJson(`${BASES.orchestrator}/orchestrator/contracts/details`, {
          method: 'POST', headers: hdrs, body: json(payload),
        });
      },
      setup(payload) {
        return fetchJson(`${BASES.orchestrator}/orchestrator/contracts/setup`, {
          method: 'POST', headers: hdrs, body: json(payload),
        });
      },
    },
    equity: {
      bars(symbol, start, end, barSize) {
        const url = `${BASES.equity}/equity/${encodeURIComponent(symbol)}/bars` +
          `?start=${encodeURIComponent(start || '')}` +
          `&end=${encodeURIComponent(end || '')}` +
          `&bar_size=${encodeURIComponent(barSize || '')}`;
        return fetchJson(url);
      },
    },
    chains: {
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

