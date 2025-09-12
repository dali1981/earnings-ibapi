import React, { useState } from 'react'
import { api } from '../apiClient'

export default function Chains() {
  const [sym, setSym] = useState('AAPL')
  const [out, setOut] = useState('')
  const [status, setStatus] = useState('')
  const [snapDate, setSnapDate] = useState('')
  const [asOf, setAsOf] = useState('')
  const [asOfList, setAsOfList] = useState([])

  async function onSubmit(e) {
    e?.preventDefault?.()
    setStatus('Loading...')
    setOut('')
    try {
      const { data } = await api.chains.latest(sym)
      setOut(JSON.stringify(data, null, 2))
      setStatus('OK')
    } catch (err) {
      const msg = err?.message || String(err)
      setOut(`Error: ${msg}`)
      setStatus('Failed')
      // Also log for debugging
      try { console.error('[chains.latest] error', err) } catch {}
    }
  }

  async function loadAsOfs() {
    setStatus('Loading as-of list...')
    setOut('')
    try {
      const { data } = await api.chains.asOfs(sym)
      setAsOfList(data?.as_of_list || [])
      if (data?.latest) setAsOf(data.latest)
      setOut(JSON.stringify(data, null, 2))
      setStatus('OK')
    } catch (err) {
      setOut(`Error: ${err?.message || String(err)}`)
      setStatus('Failed')
    }
  }

  return (
    <section>
      <h2>Option Chain (React)</h2>
      <form onSubmit={onSubmit}>
        <label>Symbol <input value={sym} onChange={e=>setSym(e.target.value)} /></label>
        <button type="submit">Read Latest (repo)</button>
      </form>
      <div className="muted" style={{margin:'.25rem 0'}}>Reads from option_chains repository only.</div>
      <div style={{display:'flex', gap:'.5rem', alignItems:'center', margin:'.5rem 0'}}>
        <button onClick={loadAsOfs} type="button">List As-Ofs</button>
        <label>As-Of
          <select value={asOf} onChange={e=>setAsOf(e.target.value)} style={{marginLeft:'.4rem'}}>
            <option value="">--</option>
            {asOfList.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </label>
        <button type="button" onClick={async()=>{
          if (!asOf) { alert('Pick as_of'); return }
          setStatus('Loading summary...'); setOut('')
          try { const { data } = await api.chains.summary(sym, asOf); setOut(JSON.stringify(data, null, 2)); setStatus('OK') } catch(err){ setOut(String(err?.message||err)); setStatus('Failed') }
        }}>Summary</button>
        <button type="button" onClick={async()=>{
          if (!asOf) { alert('Pick as_of'); return }
          setStatus('Loading expiries...'); setOut('')
          try { const { data } = await api.chains.expiries(sym, asOf); setOut(JSON.stringify(data, null, 2)); setStatus('OK') } catch(err){ setOut(String(err?.message||err)); setStatus('Failed') }
        }}>Expiries</button>
        <button type="button" onClick={async()=>{
          if (!asOf) { alert('Pick as_of'); return }
          setStatus('Loading strikes...'); setOut('')
          try { const { data } = await api.chains.strikes(sym, asOf); setOut(JSON.stringify(data, null, 2)); setStatus('OK') } catch(err){ setOut(String(err?.message||err)); setStatus('Failed') }
        }}>Strikes</button>
        <button type="button" onClick={async()=>{
          if (!asOf) { alert('Pick as_of'); return }
          setStatus('Loading params...'); setOut('')
          try { const { data } = await api.chains.params(sym, asOf); setOut(JSON.stringify(data, null, 2)); setStatus('OK') } catch(err){ setOut(String(err?.message||err)); setStatus('Failed') }
        }}>Params</button>
      </div>
      <hr style={{borderColor:'#22263b', margin:'.75rem 0'}} />
      <form onSubmit={async (e) => {
        e?.preventDefault?.()
        const s = (sym || '').trim()
        if (!s) { alert('Please enter a symbol'); return }
        setStatus('Dispatching snapshot fetch to orchestrator...')
        setOut('')
        try {
          const payload = { symbols: [s.toUpperCase()] }
          if (snapDate) payload.snapshot_date = snapDate
          const { data } = await api.orchestrator.fetchOptionChainSnapshots(payload)
          setOut(JSON.stringify(data, null, 2))
          setStatus(`Dispatched ${data?.details?.dispatched ?? 1} task(s). Job ID: ${data?.details?.group_id || data?.job_id || 'N/A'}`)
        } catch (err) {
          const msg = err?.message || String(err)
          setOut(`Error: ${msg}`)
          setStatus('Dispatch failed')
          try { console.error('[option-chains.snapshots] error', err) } catch {}
        }
      }}>
        <label>Symbol <input value={sym} onChange={e=>setSym(e.target.value)} /></label>
        <label>Snapshot Date <input type="date" value={snapDate} onChange={e=>setSnapDate(e.target.value)} /></label>
        <button type="submit">Fetch via Orchestrator</button>
      </form>
      <div style={{margin:'.5rem 0', color:'#8a90ad', fontSize:'.9rem'}}>{status}</div>
      <pre className="out">{out}</pre>
    </section>
  )
}
