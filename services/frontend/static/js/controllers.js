/**
 * Controllers: map UI actions to TradingAPI calls.
 *
 * Responsibilities
 * - Read values from the DOM (forms/inputs).
 * - Call `window.TradingAPI` methods.
 * - Write results back to the DOM (pre tags/outputs).
 * - Perform light validation and header meta rendering.
 *
 * Exports
 * - `window.TradingControllers` object keyed by action names.
 *   Used by `bootstrap.js` to wire `[data-action]` elements automatically.
 */

(function () {
  const $ = (id) => document.getElementById(id);
  const parseList = (input) => (input || '').split(',').map(s => s.trim()).filter(Boolean);
  let lastCalendarData = null;

  function calendarViewMode() {
    return $('calendarViewTable')?.checked ? 'table' : 'json';
  }

  function renderJSON(outEl, data) {
    if (outEl) outEl.textContent = JSON.stringify(data ?? null, null, 2);
  }

  function renderCalendarTable(outEl, data) {
    if (!outEl) return;
    if (!Array.isArray(data) || data.length === 0) {
      outEl.innerHTML = '<div>No events</div>';
      return;
    }
    const cols = ['symbol','company_name','earnings_date','time','eps_estimate','revenue_estimate','source'];
    const thead = '<tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr>';
    const rows = data.map(e => '<tr>' + cols.map(c => {
      const v = (e && Object.prototype.hasOwnProperty.call(e, c)) ? e[c] : '';
      return `<td>${(v ?? '')}</td>`;
    }).join('') + '</tr>').join('');
    outEl.innerHTML = `<table class="tbl">${thead}${rows}</table>`;
  }

  function renderCalendar(data) {
    lastCalendarData = data;
    const mode = calendarViewMode();
    const pre = $('calendarOut');
    const tbl = $('calendarTable');
    if (mode === 'table') {
      if (pre) pre.style.display = 'none';
      if (tbl) { tbl.style.display = ''; renderCalendarTable(tbl, data); }
    } else {
      if (tbl) tbl.style.display = 'none';
      if (pre) { pre.style.display = ''; renderJSON(pre, data); }
    }
  }

  /**
   * Render calendar source/collection metadata from response headers.
   * @param {Response} res
   */
  function setCalendarMeta(res) {
    if (!res || !res.headers) return;
    const src = res.headers.get('X-Calendar-Source') || res.headers.get('x-calendar-source') || '-';
    const col = res.headers.get('X-Calendar-Collection-Date') || res.headers.get('x-calendar-collection-date') || '-';
    const el = $('calendarMeta');
    if (el) el.textContent = `Last collection: ${col} | Source: ${src}`;
  }

  /** Handle repo-only calendar read. */
  async function calendarGet() {
    const days = $('daysGet')?.value;
    const symbols = $('symbolsGet')?.value;
    const pre = $('calendarOut');
    const tbl = $('calendarTable');
    if (tbl) tbl.style.display = 'none';
    if (pre) { pre.style.display = ''; pre.textContent = 'Loading...'; }
    const { data, res } = await window.TradingAPI.calendar.getUpcoming(days, symbols);
    setCalendarMeta(res);
    renderCalendar(data);
  }

  /** Handle external calendar fetch + persist. */
  async function calendarFetch() {
    const days = $('daysFetch')?.value;
    const source = $('sourceFetch')?.value;
    const symbols = $('symbolsFetch')?.value;
    const pre = $('calendarOut');
    const tbl = $('calendarTable');
    if (source === 'yahoo' && !symbols) {
      alert('Yahoo requires symbols (comma-separated).');
      return;
    }
    if (tbl) tbl.style.display = 'none';
    if (pre) { pre.style.display = ''; pre.textContent = 'Fetching...'; }
    const { data, res } = await window.TradingAPI.calendar.fetchUpcoming(days, source, symbols);
    setCalendarMeta(res);
    renderCalendar(data);
  }

  /** Handle equity bars read from repository. */
  async function equityRead() {
    console.log('equityRead called');
    const s = $('eqSymbol')?.value;
    const st = $('eqStart')?.value;
    const en = $('eqEnd')?.value;
    const bar = $('eqBar')?.value;
    const status = $('equityStatus');
    const out = $('equityOut');
    
    if (status) status.textContent = 'Reading from repository...';
    out.textContent = 'Loading...';
    
    try {
      const { data } = await window.TradingAPI.equity.bars(s, st, en, bar);
      out.textContent = JSON.stringify(data, null, 2);
      if (status) status.textContent = `Read ${data.rows || 0} bars from repository`;
    } catch (err) {
      out.textContent = `Error: ${err.message}`;
      if (status) status.textContent = 'Read failed';
    }
  }

  /** Handle equity bars fetch from TWS via orchestrator. */
  async function equityFetchTWS() {
    console.log('equityFetchTWS called');
    const symbols = $('eqSymbolTWS')?.value;
    const duration = $('eqDuration')?.value;
    const barSize = $('eqBarTWS')?.value;
    const endDate = $('eqEndTWS')?.value;
    const status = $('equityStatus');
    const out = $('equityOut');
    
    console.log('Form values:', { symbols, duration, barSize, endDate });
    
    if (!symbols) {
      alert('Please enter symbol(s)');
      return;
    }
    
    // Format end date time for IB API
    const endDateTime = endDate 
      ? `${endDate.replace(/-/g, '')} 23:59:59`
      : new Date().toISOString().slice(0, 10).replace(/-/g, '') + ' 23:59:59';
    
    if (status) status.textContent = 'Fetching from TWS...';
    out.textContent = 'Dispatching request to orchestrator...';
    try { console.log('[equity.fetchTWS] base:', window.TradingAPI?._bases?.orchestrator); } catch (_) {}
    if (status && window.TradingAPI?._bases?.orchestrator) {
      status.textContent = `${status.textContent} → ${window.TradingAPI._bases.orchestrator}/orchestrator/historical/bars-by-symbols`;
    }
    
    try {
      const { data } = await window.TradingAPI.orchestrator.fetchHistoricalBars({
        symbols: symbols.split(',').map(s => s.trim()),
        end_date_time: endDateTime,
        duration_str: duration,
        bar_size_setting: barSize,
        repository_type: 'equity'
      });
      
      out.textContent = JSON.stringify(data, null, 2);
      if (status) {
        status.textContent = `Fetching ${data.processed || 0} symbol(s) from TWS. ` +
          `Job ID: ${data.details?.group_id || data.details?.chain_id || 'N/A'}`;
      }
    } catch (err) {
      out.textContent = `Error: ${err.message}`;
      if (status) status.textContent = 'Fetch failed';
    }
  }

  /** Handle option chain latest snapshot. */
  async function chainLatest() {
    const s = $('chSymbol')?.value;
    const { data } = await window.TradingAPI.chains.latest(s);
    $('chainOut').textContent = JSON.stringify(data, null, 2);
  }

  /**
   * Orchestrator actions: 'search' | 'details' | 'setup'.
   * @param {'search'|'details'|'setup'} kind
   */
  async function orc(kind) {
    const symbols = parseList($('orcSymbols')?.value).map(s => s.toUpperCase());
    const sec_types = parseList($('orcSecTypes')?.value).map(s => s.toUpperCase());
    const currency = ($('orcCurrency')?.value || '').trim().toUpperCase() || 'USD';
    const exchanges = parseList($('orcExchanges')?.value).map(s => s.toUpperCase());
    const payload = { symbols, sec_types, currency, exchanges };
    let res;
    if (kind === 'search') res = await window.TradingAPI.orchestrator.search(payload);
    else if (kind === 'details') res = await window.TradingAPI.orchestrator.details({ symbols });
    else res = await window.TradingAPI.orchestrator.setup(payload);
    const { data } = res;
    $('orchestratorOut').textContent = JSON.stringify(data, null, 2);
  }

  window.TradingControllers = {
    'calendar.get': calendarGet,
    'calendar.fetch': calendarFetch,
    'equity.read': equityRead,
    'equity.fetchTWS': equityFetchTWS,
    'chains.latest': chainLatest,
    'orc.search': () => orc('search'),
    'orc.details': () => orc('details'),
    'orc.setup': () => orc('setup'),
  };

  // Re-render calendar on view toggle changes
  document.addEventListener('DOMContentLoaded', () => {
    const t = $('calendarViewTable');
    if (t) t.addEventListener('change', () => {
      if (lastCalendarData !== null) renderCalendar(lastCalendarData);
    });
    // Redundant explicit binding in case bootstrap misses wiring
    const fetchForm = $('equityFetchForm');
    if (fetchForm && window.TradingControllers?.['equity.fetchTWS']) {
      fetchForm.addEventListener('submit', (e) => {
        e.preventDefault();
        Promise.resolve(window.TradingControllers['equity.fetchTWS']()).catch(err => {
          console.error('equity.fetchTWS failed:', err);
          alert(err?.message || String(err));
        });
      });
    }
  });
})();
