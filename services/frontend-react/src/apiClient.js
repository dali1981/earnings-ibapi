// Resolve service bases in this order:
// 1) window.__BASES__ injected at runtime (if provided by host page)
// 2) Vite env vars (VITE_*_BASE) for builds/dev
// 3) Dev proxy paths under /svc/* as a fallback
const injected = (typeof window !== 'undefined' && window.__BASES__) || null
const envBases = (typeof import.meta !== 'undefined' && import.meta.env) ? {
  calendar: import.meta.env.VITE_CALENDAR_BASE,
  equity: import.meta.env.VITE_EQUITY_BASE,
  chains: import.meta.env.VITE_CHAINS_BASE,
  orchestrator: import.meta.env.VITE_ORCHESTRATOR_BASE,
} : {}
const proxyBases = {
  calendar: '/svc/calendar',
  equity: '/svc/equity',
  chains: '/svc/chains',
  orchestrator: '/svc/orchestrator',
}
const BASES = {
  calendar: (injected && injected.calendar) || envBases.calendar || proxyBases.calendar,
  equity: (injected && injected.equity) || envBases.equity || proxyBases.equity,
  chains: (injected && injected.chains) || envBases.chains || proxyBases.chains,
  orchestrator: (injected && injected.orchestrator) || envBases.orchestrator || proxyBases.orchestrator,
}

async function fetchJson(url, init) {
  const res = await fetch(url, init)
  const ct = res.headers.get('content-type') || ''
  let body = null
  try { body = ct.includes('application/json') ? await res.json() : await res.text() } catch {}
  if (!res.ok) {
    const msg = body && body.error ? body.error : `${res.status} ${res.statusText}`
    const err = new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
    err.status = res.status; err.data = body; err.response = res
    throw err
  }
  return { data: body, res }
}

export const api = {
  calendar: {
    getUpcoming(days, symbols) {
      const url = `${BASES.calendar}/calendar/upcoming?days=${encodeURIComponent(days)}` + (symbols?`&symbols=${encodeURIComponent(symbols)}`:'')
      return fetchJson(url)
    },
    fetchUpcoming(days, source, symbols) {
      const qs = new URLSearchParams({ days: String(days), source })
      if (symbols) qs.append('symbols', symbols)
      return fetchJson(`${BASES.calendar}/calendar/fetch-upcoming?${qs.toString()}`, { method: 'POST' })
    },
  },
  equity: {
    bars(sym, start, end, bar) {
      const url = `${BASES.equity}/equity/${encodeURIComponent(sym)}/bars?start=${encodeURIComponent(start||'')}&end=${encodeURIComponent(end||'')}&bar_size=${encodeURIComponent(bar||'')}`
      return fetchJson(url)
    },
    dates(sym, bar) {
      const url = `${BASES.equity}/equity/${encodeURIComponent(sym)}/dates?bar_size=${encodeURIComponent(bar||'1 day')}`
      return fetchJson(url)
    },
  },
  chains: {
    latest(sym) { return fetchJson(`${BASES.chains}/chains/${encodeURIComponent(sym)}/latest`) },
    asOfs(sym) { return fetchJson(`${BASES.chains}/chains/${encodeURIComponent(sym)}/as-ofs`) },
    summary(sym, asOf) { return fetchJson(`${BASES.chains}/chains/${encodeURIComponent(sym)}/${encodeURIComponent(asOf)}/summary`) },
    expiries(sym, asOf) { return fetchJson(`${BASES.chains}/chains/${encodeURIComponent(sym)}/${encodeURIComponent(asOf)}/expiries`) },
    strikes(sym, asOf) { return fetchJson(`${BASES.chains}/chains/${encodeURIComponent(sym)}/${encodeURIComponent(asOf)}/strikes`) },
    params(sym, asOf) { return fetchJson(`${BASES.chains}/chains/${encodeURIComponent(sym)}/${encodeURIComponent(asOf)}/params`) },
  },
  orchestrator: {
    search(payload) { return fetchJson(`${BASES.orchestrator}/orchestrator/contracts/search`, { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload) }) },
    details(payload) { return fetchJson(`${BASES.orchestrator}/orchestrator/contracts/details`, { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload) }) },
    setup(payload) { return fetchJson(`${BASES.orchestrator}/orchestrator/contracts/setup`, { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload) }) },
    fetchHistoricalBars(payload) { return fetchJson(`${BASES.orchestrator}/orchestrator/historical/bars-by-symbols`, { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload) }) },
    fetchOptionChainSnapshots(payload) { return fetchJson(`${BASES.orchestrator}/orchestrator/option-chains/snapshots`, { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload) }) },
    optionContracts: {
      select(payload) { return fetchJson(`${BASES.orchestrator}/option-contracts/select`, { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload) }) },
      selectStatus(taskId) { return fetchJson(`${BASES.orchestrator}/option-contracts/result/${encodeURIComponent(taskId)}`) },
      enrich(payload) { return fetchJson(`${BASES.orchestrator}/option-contracts/enrich`, { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload) }) },
      save(payload) { return fetchJson(`${BASES.orchestrator}/option-contracts/save`, { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload) }) },
      historical(payload) { return fetchJson(`${BASES.orchestrator}/option-contracts/historical`, { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload) }) },
      fullflow(payload) { return fetchJson(`${BASES.orchestrator}/option-contracts/fullflow`, { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload) }) },
      browseBars(params) {
        const qs = new URLSearchParams()
        Object.entries(params||{}).forEach(([k,v])=>{ if (v!==undefined && v!==null && v!=='') qs.append(k, String(v)) })
        return fetchJson(`${BASES.orchestrator}/option-contracts/option-bars?${qs.toString()}`)
      },
    },
  },
}
