import React, { useMemo, useState } from 'react'
import { api } from '../apiClient'

export default function Calendar() {
  const [daysGet, setDaysGet] = useState(7)
  const [symbolsGet, setSymbolsGet] = useState('')
  const [daysFetch, setDaysFetch] = useState(7)
  const [source, setSource] = useState('nasdaq')
  const [symbolsFetch, setSymbolsFetch] = useState('')
  const [meta, setMeta] = useState({ src: '-', col: '-' })
  const [viewTable, setViewTable] = useState(false)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)

  const cols = useMemo(() => ['symbol','company_name','earnings_date','time','eps_estimate','revenue_estimate','source'], [])

  const setMetaFromRes = (res) => {
    const src = res.headers.get('X-Calendar-Source') || '-'
    const col = res.headers.get('X-Calendar-Collection-Date') || '-'
    setMeta({ src, col })
  }

  async function onGet(e) {
    e?.preventDefault?.()
    setLoading(true)
    try {
      const { data, res } = await api.calendar.getUpcoming(daysGet, symbolsGet)
      setMetaFromRes(res)
      setData(data)
    } finally { setLoading(false) }
  }

  async function onFetch(e) {
    e?.preventDefault?.()
    if (source === 'yahoo' && !symbolsFetch) { alert('Yahoo requires symbols'); return }
    setLoading(true)
    try {
      const { data, res } = await api.calendar.fetchUpcoming(daysFetch, source, symbolsFetch)
      setMetaFromRes(res)
      setData(data)
    } finally { setLoading(false) }
  }

  function Table({ rows }) {
    if (!Array.isArray(rows) || rows.length === 0) return <div>No events</div>
    return (
      <table className="tbl">
        <thead><tr>{cols.map(c => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{cols.map(c => <td key={c}>{r?.[c] ?? ''}</td>)}</tr>
          ))}
        </tbody>
      </table>
    )
  }

  return (
    <section>
      <h2>Calendar Service (React)</h2>
      <div className="muted">Last collection: {meta.col} | Source: {meta.src}</div>
      <form onSubmit={onGet} style={{marginTop:'.25rem'}}>
        <label>Days Ahead <input type="number" min={0} max={180} value={daysGet} onChange={e=>setDaysGet(Number(e.target.value))}/></label>
        <label>Symbols <input value={symbolsGet} onChange={e=>setSymbolsGet(e.target.value)} placeholder="(optional) AAPL,MSFT"/></label>
        <button type="submit">Get (repo only)</button>
      </form>
      <form onSubmit={onFetch} style={{marginTop:'.6rem'}}>
        <label>Days Ahead <input type="number" min={0} max={180} value={daysFetch} onChange={e=>setDaysFetch(Number(e.target.value))}/></label>
        <label>Source <select value={source} onChange={e=>setSource(e.target.value)}>
          <option value="nasdaq">nasdaq</option>
          <option value="yahoo">yahoo</option>
          <option value="all">all</option>
        </select></label>
        <label>Symbols <input value={symbolsFetch} onChange={e=>setSymbolsFetch(e.target.value)} placeholder="Required for yahoo (e.g. AAPL,MSFT)"/></label>
        <button type="submit">Fetch & persist (external)</button>
      </form>
      <div className="toggle">
        <span>JSON</span>
        <label className="switch">
          <input type="checkbox" checked={viewTable} onChange={e=>setViewTable(e.target.checked)}/>
          <span className="slider"></span>
        </label>
        <span>Table</span>
      </div>
      {viewTable
        ? <div className="out">{loading ? 'Loading...' : <Table rows={data||[]}/>}</div>
        : <pre className="out">{loading ? 'Loading...' : JSON.stringify(data, null, 2)}</pre>
      }
    </section>
  )
}

