import React, { useState } from 'react'
import { api } from '../apiClient'

function parseList(s) { return (s||'').split(',').map(x=>x.trim()).filter(Boolean) }

export default function Orchestrator() {
  const [symbols, setSymbols] = useState('AAPL,MSFT')
  const [secTypes, setSecTypes] = useState('STK')
  const [currency, setCurrency] = useState('USD')
  const [exchanges, setExchanges] = useState('NYSE,NASDAQ,AMEX,ARCA,BATS,IEX')
  const [out, setOut] = useState('')

  async function run(kind) {
    const payload = {
      symbols: parseList(symbols).map(s=>s.toUpperCase()),
      sec_types: parseList(secTypes).map(s=>s.toUpperCase()),
      currency: (currency||'USD').toUpperCase(),
      exchanges: parseList(exchanges).map(s=>s.toUpperCase()),
    }
    let res
    if (kind === 'search') res = await api.orchestrator.search(payload)
    else if (kind === 'details') res = await api.orchestrator.details({ symbols: payload.symbols })
    else res = await api.orchestrator.setup(payload)
    setOut(JSON.stringify(res.data, null, 2))
  }

  return (
    <section>
      <h2>Contracts Orchestrator (React)</h2>
      <form onSubmit={(e)=>e.preventDefault()}>
        <label>Symbols <input value={symbols} onChange={e=>setSymbols(e.target.value)} placeholder="AAPL, MSFT, TSLA"/></label>
        <label>Sec Types <input value={secTypes} onChange={e=>setSecTypes(e.target.value)} /></label>
        <label>Currency <input value={currency} onChange={e=>setCurrency(e.target.value)} /></label>
        <label>Exchanges <input value={exchanges} onChange={e=>setExchanges(e.target.value)} /></label>
        <div style={{marginTop:'.3rem'}}>
          <button type="button" onClick={()=>run('search')}>Search Descriptions</button>
          <button type="button" onClick={()=>run('details')}>Fetch Details</button>
          <button type="button" onClick={()=>run('setup')}>Setup (Search + Details)</button>
        </div>
      </form>
      <pre className="out">{out}</pre>
    </section>
  )
}

