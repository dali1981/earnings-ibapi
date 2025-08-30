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

  /** Handle equity bars fetch. */
  async function equityFetch() {
    const s = $('eqSymbol')?.value;
    const st = $('eqStart')?.value;
    const en = $('eqEnd')?.value;
    const bar = $('eqBar')?.value;
    const { data } = await window.TradingAPI.equity.bars(s, st, en, bar);
    $('equityOut').textContent = JSON.stringify(data, null, 2);
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
    'equity.fetch': equityFetch,
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
  });
})();
