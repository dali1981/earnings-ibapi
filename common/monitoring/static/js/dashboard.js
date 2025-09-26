(() => {
  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));

  const state = {
    activeTab: 'overview',
    timers: [],
  };

  const fmtTime = (ts) => {
    try {
      const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
      return d.toLocaleString();
    } catch { return String(ts); }
  };

  const copy = (text) => navigator.clipboard?.writeText(text).catch(()=>{});

  async function fetchJSON(url) {
    const res = await fetch(url, { headers: { 'Accept': 'application/json' }});
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  function setStatus(msg) { $('#statusMessage').textContent = msg; }

  // Tabs
  function initTabs() {
    $$('.tab').forEach(btn => {
      btn.addEventListener('click', () => {
        $$('.tab').forEach(x => x.classList.remove('active'));
        $$('.panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const id = btn.getAttribute('data-tab');
        state.activeTab = id;
        $('#' + id).classList.add('active');
        refreshActive();
      });
    });
  }

  function clearTimers() { state.timers.forEach(t => clearInterval(t)); state.timers = []; }

  function scheduleAutoRefresh(fn) {
    const auto = $('#autoRefresh').checked;
    const intervalSec = parseInt($('#refreshInterval').value, 10) || 10;
    if (!auto) return;
    const tid = setInterval(fn, intervalSec * 1000);
    state.timers.push(tid);
  }

  async function loadOverview() {
    try {
      const data = await fetchJSON('/api/status');
      renderOverview(data);
      setStatus('Overview updated');
    } catch (e) { setStatus('Failed to load status'); }
  }

  function renderOverview(data) {
    const cards = $('#statusCards');
    cards.innerHTML = '';
    const items = [
      { title: 'Symbols Processed', key: 'symbols_processed' },
      { title: 'Successful Symbols', key: 'successful_symbols' },
      { title: 'Failed Symbols', key: 'failed_symbols' },
      { title: 'Earnings Events', key: 'earnings_events_processed' },
      { title: 'Circuit Breakers', key: 'circuit_breakers_active' },
    ];
    (items).forEach(({title, key}) => {
      const val = data?.[key];
      const cls = key === 'failed_symbols' ? 'err' : key === 'successful_symbols' ? 'ok' : '';
      const div = document.createElement('div');
      div.className = `card ${cls}`;
      div.innerHTML = `<div class="title">${title}</div><div class="value">${val ?? '—'}</div>`;
      cards.appendChild(div);
    });
  }

  async function loadEvents() {
    const topic = $('#evTopic').value.trim();
    const etype = $('#evType').value.trim();
    const hours = parseInt($('#evHours').value, 10) || 24;
    const agg = $('#evAggregate').checked;
    const corrFilter = $('#evCorrelation').value.trim();
    const qs = new URLSearchParams();
    if (topic) qs.set('topic', topic);
    if (etype) qs.set('type', etype);
    if (hours) qs.set('hours', String(hours));
    if (agg) qs.set('aggregate', '1');
    try {
      const data = await fetchJSON('/events' + (qs.toString() ? ('?' + qs.toString()) : ''));
      renderEvents(data, agg, corrFilter);
      setStatus('Events updated');
    } catch (e) { setStatus('Failed to load events'); }
  }

  function renderEvents(data, aggregate, corrFilter) {
    const summary = $('#eventsSummary');
    const tbody = $('#eventsTable tbody');
    tbody.innerHTML = '';
    summary.textContent = '';

    if (aggregate) {
      const arr = data?.aggregate || [];
      summary.textContent = `Aggregated counts (${arr.length} groups)`;
      arr.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>—</td>
          <td>${row.topic}</td>
          <td>${row.type}</td>
          <td>—</td>
          <td>${row.count} events</td>`;
        tbody.appendChild(tr);
      });
      return;
    }

    let events = data?.events || [];
    if (corrFilter) {
      events = events.filter(e => (e.payload?.correlation_id || '').includes(corrFilter));
    }
    summary.textContent = `Events (${events.length})`;

    events.slice().reverse().forEach(ev => {
      const tr = document.createElement('tr');
      const corr = ev.payload?.correlation_id || '';
      const payloadStr = JSON.stringify(ev.payload || {}, null, 0);
      tr.innerHTML = `
        <td>${fmtTime(ev.ts)}</td>
        <td><span class="pill">${ev.topic}</span></td>
        <td>${ev.type}</td>
        <td>${corr ? `<span class="pill">${corr}</span><button class="copy-btn" title="Copy" data-copy="${corr}">Copy</button>` : '—'}</td>
        <td class="payload">${payloadStr}</td>`;
      tbody.appendChild(tr);
    });

    $$('#eventsTable .copy-btn').forEach(btn => {
      btn.addEventListener('click', () => copy(btn.getAttribute('data-copy')));
    });
  }

  async function loadAlerts() {
    const hours = parseInt($('#alertHours').value, 10) || 24;
    try {
      const data = await fetchJSON('/api/alerts?hours=' + hours);
      renderAlerts(data);
      setStatus('Alerts updated');
    } catch (e) { setStatus('Failed to load alerts'); }
  }

  function renderAlerts(alerts) {
    const sevFilter = $('#alertSeverity').value;
    const tbody = $('#alertsTable tbody');
    tbody.innerHTML = '';
    (alerts || [])
      .filter(a => !sevFilter || a.severity === sevFilter)
      .slice().reverse()
      .forEach(a => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${fmtTime(a.timestamp)}</td>
          <td>${a.alert_type}</td>
          <td>${a.severity}</td>
          <td>${a.message}</td>`;
        tbody.appendChild(tr);
      });
  }

  async function loadMetrics() {
    const hours = parseInt($('#metricsHours').value, 10) || 24;
    try {
      const data = await fetchJSON('/api/metrics?hours=' + hours);
      renderMetrics(data);
      setStatus('Metrics updated');
    } catch (e) { setStatus('Failed to load metrics'); }
  }

  function renderMetrics(list) {
    const tbody = $('#metricsTable tbody');
    tbody.innerHTML = '';
    (list || []).slice().reverse().forEach(m => {
      const rate = (m.successful_symbols / Math.max(1, m.symbols_processed)) * 100;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${fmtTime(m.timestamp)}</td>
        <td>${m.symbols_processed}</td>
        <td>${m.successful_symbols}</td>
        <td>${m.failed_symbols}</td>
        <td>${(m.total_duration_seconds/60).toFixed(1)}</td>
        <td>${rate.toFixed(1)}%</td>`;
      tbody.appendChild(tr);
    });
  }

  async function loadReport() {
    const days = parseInt($('#reportDays').value, 10) || 7;
    try {
      const data = await fetchJSON('/api/report?days=' + days);
      $('#reportJson').textContent = JSON.stringify(data, null, 2);
      setStatus('Report generated');
    } catch (e) { setStatus('Failed to generate report'); }
  }

  function refreshActive() {
    clearTimers();
    const tab = state.activeTab;
    if (tab === 'overview') { loadOverview(); scheduleAutoRefresh(loadOverview); }
    if (tab === 'events')   { loadEvents();   scheduleAutoRefresh(loadEvents); }
    if (tab === 'alerts')   { loadAlerts();   scheduleAutoRefresh(loadAlerts); }
    if (tab === 'metrics')  { loadMetrics();  scheduleAutoRefresh(loadMetrics); }
    if (tab === 'report')   { /* generate on demand only */ }
    if (tab === 'jobs')     { /* manual triggers */ }
    if (tab === 'config')   { loadConfig(); }
  }

  function initFilters() {
    $('#evApply').addEventListener('click', loadEvents);
    $('#alertApply').addEventListener('click', loadAlerts);
    $('#metricsApply').addEventListener('click', loadMetrics);
    $('#reportApply').addEventListener('click', loadReport);
    $('#jobSetupBtn').addEventListener('click', runSetupSymbol);
    $('#jobDailyBtn').addEventListener('click', runDailyUpdate);
    $('#autoRefresh').addEventListener('change', refreshActive);
    $('#refreshInterval').addEventListener('change', refreshActive);
  }

  async function runSetupSymbol() {
    const symbol = $('#jobSymbol').value.trim();
    const lookback_days = parseInt($('#jobLookback').value, 10) || 365;
    const job_id = $('#jobCorrelation').value.trim();
    if (!symbol) { setStatus('Symbol required'); return; }
    try {
      const res = await fetch('/api/jobs/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'setup_new_symbol', symbol, lookback_days, job_id })
      });
      const data = await res.json();
      $('#jobsResult').textContent = JSON.stringify(data, null, 2);
      setStatus('Setup job started');
    } catch (e) { setStatus('Failed to start setup job'); }
  }

  async function runDailyUpdate() {
    const raw = $('#jobSymbolsList').value.trim();
    const symbols = raw.split(/[\s,]+/).filter(Boolean);
    const job_id = $('#jobCorrelation').value.trim();
    if (!symbols.length) { setStatus('At least one symbol required'); return; }
    try {
      const res = await fetch('/api/jobs/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'daily_update', symbols, job_id })
      });
      const data = await res.json();
      $('#jobsResult').textContent = JSON.stringify(data, null, 2);
      setStatus('Daily update started');
    } catch (e) { setStatus('Failed to start daily update'); }
  }

  async function loadConfig() {
    try {
      const cfg = await fetchJSON('/api/config');
      const cards = $('#configCards');
      cards.innerHTML = '';
      const kv = [
        ['PROJECT_ROOT', cfg.PROJECT_ROOT],
        ['DATA_ROOT', cfg.DATA_ROOT],
        ['CACHE_PATH', cfg.CACHE_PATH],
        ['LOGS_PATH', cfg.LOGS_PATH],
        ['IB_HOST', cfg.IB_HOST],
        ['IB_PORT', cfg.IB_PORT],
      ];
      kv.forEach(([k,v]) => {
        const div = document.createElement('div');
        div.className = 'card';
        div.innerHTML = `<div class="title">${k}</div><div class="value">${v ?? '—'}</div>`;
        cards.appendChild(div);
      });
      $('#portfolioYaml').textContent = cfg.portfolio_yaml || 'No portfolio.yaml found';
      setStatus('Config loaded');
    } catch (e) { setStatus('Failed to load config'); }
  }

  // Init
  document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initFilters();
    refreshActive();
  });
})();
