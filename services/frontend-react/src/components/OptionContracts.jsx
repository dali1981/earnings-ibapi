import React, { useState, useRef, useEffect } from 'react'
import { api } from '../apiClient'

export default function OptionContracts() {
  const [sym, setSym] = useState('AAPL')
  const [asOf, setAsOf] = useState('')
  const [expiry, setExpiry] = useState('')
  const [rights, setRights] = useState('C,P')
  const [mode, setMode] = useState('count')
  const [expiryCount, setExpiryCount] = useState('2')
  const [spotAuto, setSpotAuto] = useState(true)
  const [spotVal, setSpotVal] = useState('')
  const [strikesPerSide, setStrikesPerSide] = useState('5')
  const [moneynessTargets, setMoneynessTargets] = useState('0.9,1.0,1.1')
  const [deltaTargets, setDeltaTargets] = useState('0.25,0.5')
  const [ivAssumption, setIvAssumption] = useState('0.25')
  const [useUnderlyingIv, setUseUnderlyingIv] = useState(false)
  const [maxContracts, setMaxContracts] = useState('20')
  const [out, setOut] = useState('')
  const [status, setStatus] = useState('')
  const [contractsResult, setContractsResult] = useState(null)
  const [showTable, setShowTable] = useState(false)
  const hasEnriched = !!(contractsResult && Array.isArray(contractsResult.contracts) && contractsResult.contracts.some(c => c && c.conId))
  const [historicalRes, setHistoricalRes] = useState(null)
  const [active, setActive] = useState('select')
  const [previewParams, setPreviewParams] = useState({ expiry: '', right: 'C', strike: '', bar_size: '1 day', start_date:'', end_date:'' })
  const [previewRows, setPreviewRows] = useState([])
  const [previewTotal, setPreviewTotal] = useState(0)
  const selectTaskIdRef = useRef(null)
  const pollTimerRef = useRef(null)

  useEffect(()=>{
    return ()=> { if (pollTimerRef.current) clearInterval(pollTimerRef.current) }
  }, [])

  const R = () => rights.split(',').map(x=>x.trim().toUpperCase()).filter(x=>x==='C' || x==='P')

  const MTargets = () => moneynessTargets.split(',').map(x=>parseFloat(x)).filter(x=>!isNaN(x))
  const DTargets = () => deltaTargets.split(',').map(x=>parseFloat(x)).filter(x=>!isNaN(x))

  async function select() {
    setStatus('Selecting...'); setOut('')
    try {
      const payload = {
        symbol: (sym||'').trim().toUpperCase(),
        snapshot_date: asOf || undefined,
        expiry_filter: expiry || undefined,
        rights: R(),
        strikes_per_side: Number(strikesPerSide)||undefined,
        max_contracts: Number(maxContracts)||20,
        selection_mode: mode,
        expiry_count: Number(expiryCount)||2,
        spot_price: spotAuto ? undefined : (Number(spotVal)||undefined),
        moneyness_targets: mode==='moneyness' ? MTargets() : undefined,
        delta_targets: mode==='delta' ? DTargets() : undefined,
        iv_assumption: mode==='delta' ? (Number(ivAssumption)||undefined) : undefined,
        use_underlying_iv: mode==='delta' ? !!useUnderlyingIv : undefined,
      }
      const { data } = await api.orchestrator.optionContracts.select(payload)
      setOut(JSON.stringify(data, null, 2))
      const taskId = data?.details?.task_id
      if (!taskId) { setStatus('Dispatched select task (no task id returned)'); return }
      setStatus(`Select dispatched (task ${taskId}). Waiting for result...`)
      selectTaskIdRef.current = taskId
      if (pollTimerRef.current) clearInterval(pollTimerRef.current)
      pollTimerRef.current = setInterval(async ()=>{
        try {
          const { data: res } = await api.orchestrator.optionContracts.selectStatus(taskId)
          if (res && res.contracts) {
            clearInterval(pollTimerRef.current); pollTimerRef.current = null
            setContractsResult(res)
            setOut(JSON.stringify(res, null, 2))
            setStatus(`Selected ${res.contract_count||res.contracts.length} contracts for ${res.symbol}`)
          } else if (res && res.status) {
            setStatus(`Select in progress... (${res.status})`)
          } else {
            setStatus('Select in progress...')
          }
        } catch (err) {
          // 202 -> handled above; failures return 500
          setStatus(`Select failed or not ready: ${err?.message||String(err)}`)
        }
      }, 1000)
    } catch (err) {
      setOut(`Error: ${err?.message||String(err)}`); setStatus('Failed')
    }
  }

  async function enrich() {
    setStatus('Enriching...'); setOut('')
    try {
      if (!contractsResult) { alert('Paste contracts_result JSON below or run select and paste its output contracts_result'); return }
      const payload = { contracts_result: contractsResult, timeout_seconds: 15 }
      const { data } = await api.orchestrator.optionContracts.enrich(payload)
      setOut(JSON.stringify(data, null, 2))
      const taskId = data?.details?.task_id
      if (!taskId) { setStatus('Dispatched enrich task (no task id)'); return }
      setStatus(`Enrich dispatched (task ${taskId}). Waiting for result...`)
      if (pollTimerRef.current) clearInterval(pollTimerRef.current)
      pollTimerRef.current = setInterval(async ()=>{
        try {
          const { data: res } = await api.orchestrator.optionContracts.selectStatus(taskId)
          if (res && res.contracts) {
            clearInterval(pollTimerRef.current); pollTimerRef.current = null
            setContractsResult(res)
            setOut(JSON.stringify(res, null, 2))
            setStatus(`Enriched ${res.contract_count||res.contracts.length} contracts for ${res.symbol}`)
          } else if (res && res.status) {
            setStatus(`Enrich in progress... (${res.status})`)
          } else {
            setStatus('Enrich in progress...')
          }
        } catch (err) {
          setStatus(`Enrich failed or not ready: ${err?.message||String(err)}`)
        }
      }, 1000)
    } catch (err) {
      setOut(`Error: ${err?.message||String(err)}`); setStatus('Failed')
    }
  }

  async function save() {
    setStatus('Saving to contracts repo...'); setOut('')
    try {
      if (!contractsResult) { alert('Provide contracts_result JSON'); return }
      const payload = { contracts_result: contractsResult }
      const { data } = await api.orchestrator.optionContracts.save(payload)
      setOut(JSON.stringify(data, null, 2)); setStatus('Dispatched save task')
    } catch (err) {
      setOut(`Error: ${err?.message||String(err)}`); setStatus('Failed')
    }
  }

  async function historical() {
    setStatus('Dispatching option historical...'); setOut(''); setHistoricalRes(null)
    try {
      const payload = {
        symbol: (sym||'').trim().toUpperCase(),
        end_date_time: new Date().toISOString().slice(0,10).replace(/-/g,'') + ' 23:59:59',
        duration_str: '1 M',
        bar_size_setting: '1 day',
        what_to_show: 'TRADES',
        use_rth: 1,
        snapshot_date: asOf || undefined,
        expiry_filter: expiry || undefined,
        max_contracts: Number(maxContracts)||20,
        rights: R(),
        strikes_per_side: Number(strikesPerSide)||undefined,
        selection_mode: mode,
        expiry_count: Number(expiryCount)||2,
        spot_price: spotAuto ? undefined : (Number(spotVal)||undefined),
        moneyness_targets: mode==='moneyness' ? MTargets() : undefined,
        delta_targets: mode==='delta' ? DTargets() : undefined,
        iv_assumption: mode==='delta' ? (Number(ivAssumption)||undefined) : undefined,
        use_underlying_iv: mode==='delta' ? !!useUnderlyingIv : undefined,
        timeout_seconds: 20.0,
      }
      const { data } = await api.orchestrator.optionContracts.historical(payload)
      setOut(JSON.stringify(data, null, 2))
      const taskId = data?.details?.task_id
      if (!taskId) { setStatus('Dispatched historical task (no task id)'); return }
      setStatus(`Historical dispatched (task ${taskId}). Waiting for result...`)
      if (pollTimerRef.current) clearInterval(pollTimerRef.current)
      pollTimerRef.current = setInterval(async ()=>{
        try {
          const { data: res } = await api.orchestrator.optionContracts.selectStatus(taskId)
          // HistoricalResult shape: {processed, saved, failed, ...}
          if (res && (typeof res.processed !== 'undefined' || (res.results && Array.isArray(res.results)))) {
            clearInterval(pollTimerRef.current); pollTimerRef.current = null
            setHistoricalRes(res)
            setOut(JSON.stringify(res, null, 2))
            const p = res.processed||0, s = res.saved||0, f = res.failed||0
            setStatus(`Historical complete: processed ${p}, saved ${s}, failed ${f}`)
          } else if (res && res.status) {
            setStatus(`Historical in progress... (${res.status})`)
          } else {
            setStatus('Historical in progress...')
          }
        } catch (err) {
          setStatus(`Historical failed or not ready: ${err?.message||String(err)}`)
        }
      }, 1000)
    } catch (err) {
      setOut(`Error: ${err?.message||String(err)}`); setStatus('Failed')
    }
  }

  return (
    <section>
      <h2>Option Contracts (React)</h2>
      <nav style={{display:'flex', gap:'.5rem', borderBottom:'1px solid #e5e7eb', marginBottom:'.75rem'}}>
        {['select','enrich','historical','preview'].map(k=> (
          <button key={k} type="button" onClick={()=>setActive(k)} style={{borderBottom: active===k? '2px solid #6366f1':'2px solid transparent'}}>{k.toUpperCase()}</button>
        ))}
      </nav>

      {active==='select' && (
        <>
          <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(240px,1fr))', gap:'.5rem'}}>
            <label>Symbol
              <input value={sym} onChange={e=>setSym(e.target.value)} placeholder="AAPL" />
            </label>
            <label>As-Of (optional)
              <input type="date" value={asOf} onChange={e=>setAsOf(e.target.value)} />
            </label>
            <label>Expiry (ISO)
              <input value={expiry} onChange={e=>setExpiry(e.target.value)} placeholder="YYYY-MM-DD" />
            </label>
            <label>Rights
              <input value={rights} onChange={e=>setRights(e.target.value)} placeholder="C,P" />
            </label>
            <label>Selection Mode
              <select value={mode} onChange={e=>setMode(e.target.value)}>
                <option value="count">Count (ATM ± N)</option>
                <option value="moneyness">Moneyness</option>
                <option value="delta">Delta</option>
              </select>
            </label>
            <label>Expiry Count
              <input value={expiryCount} onChange={e=>setExpiryCount(e.target.value)} />
            </label>
            <label>Spot Source
              <select value={spotAuto? 'auto':'manual'} onChange={e=>setSpotAuto(e.target.value==='auto')}>
                <option value="auto">Auto (latest close)</option>
                <option value="manual">Manual</option>
              </select>
            </label>
            {!spotAuto && (
              <label>Spot Price
                <input value={spotVal} onChange={e=>setSpotVal(e.target.value)} placeholder="e.g., 192.5" />
              </label>
            )}
            {mode==='count' && (
              <label>Strikes/Side
                <input value={strikesPerSide} onChange={e=>setStrikesPerSide(e.target.value)} />
              </label>
            )}
            {mode==='moneyness' && (
              <label>Moneyness Targets
                <input value={moneynessTargets} onChange={e=>setMoneynessTargets(e.target.value)} placeholder="0.9,1.0,1.1" />
              </label>
            )}
            {mode==='delta' && (
              <>
                <label>Delta Targets
                  <input value={deltaTargets} onChange={e=>setDeltaTargets(e.target.value)} placeholder="0.25,0.5" />
                </label>
                <label>IV Assumption
                  <input value={ivAssumption} onChange={e=>setIvAssumption(e.target.value)} placeholder="0.25" />
                </label>
                <label>
                  <input type="checkbox" checked={useUnderlyingIv} onChange={e=>setUseUnderlyingIv(e.target.checked)} /> Use underlying IV index if available
                </label>
              </>
            )}
            <label>Max Contracts
              <input value={maxContracts} onChange={e=>setMaxContracts(e.target.value)} />
            </label>
          </div>
          <div style={{marginTop:'.5rem', display:'flex', gap:'.5rem', flexWrap:'wrap', alignItems:'center'}}>
            <button type="button" onClick={select}>Select</button>
          </div>
          <div style={{display:'flex', alignItems:'center', gap:'.5rem', marginTop:'.25rem'}}>
            <label style={{display:'inline-flex', alignItems:'center', gap:'.25rem'}}>
              <input type="checkbox" checked={showTable} onChange={e=>setShowTable(e.target.checked)} />
              Table view
            </label>
          </div>
          {showTable && contractsResult && Array.isArray(contractsResult.contracts) ? (
            <div style={{overflowX:'auto', marginTop:'.5rem'}}>
              <table style={{width:'100%', borderCollapse:'collapse'}}>
                <thead>
                  <tr>
                    <th style={th}>Expiry</th>
                    <th style={th}>Right</th>
                    <th style={th}>Strike</th>
                    <th style={th}>Exchange</th>
                    <th style={th}>Class</th>
                    <th style={th}>Multiplier</th>
                  </tr>
                </thead>
                <tbody>
                  {contractsResult?.contracts?.map((c, i) => (
                    <tr key={i}>
                      <td style={td}>{c.lastTradeDateOrContractMonth}</td>
                      <td style={td}>{c.right}</td>
                      <td style={td}>{Number(c.strike).toFixed(2)}</td>
                      <td style={td}>{c.exchange}</td>
                      <td style={td}>{c.tradingClass || c.trading_class || ''}</td>
                      <td style={td}>{c.multiplier}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </>
      )}

      {active==='enrich' && (
        <section>
          <h3>Enrich & Save</h3>
          <div className="muted" style={{margin:'.25rem 0'}}>Paste a contracts_result JSON or use output from Select (updated contracts will be saved only if conId present).</div>
          <textarea rows={6} value={contractsResult?JSON.stringify(contractsResult,null,2):''}
            onChange={e=>{ try{ setContractsResult(JSON.parse(e.target.value)) } catch{ setContractsResult(null) } }}
            placeholder={`{\n  "symbol": "AAPL", "contracts": []\n}`}
            style={{width:'100%', fontFamily:'monospace'}} />
          <div style={{marginTop:'.5rem', display:'flex', gap:'.5rem', alignItems:'center'}}>
            <button type="button" onClick={enrich}>Enrich</button>
            <button type="button" onClick={save} disabled={!hasEnriched} title={!hasEnriched ? 'Select + Enrich contracts first (need conId)' : ''}>Save to Contracts Repo</button>
            {!hasEnriched && <span style={{color:'#8a90ad', fontSize:'.9rem'}}>(needs conId — run Enrich)</span>}
          </div>
        </section>
      )}

      {active==='historical' && (
        <section>
          <h3>Historical & Full Flow</h3>
          <div style={{marginTop:'.5rem', display:'flex', gap:'.5rem', flexWrap:'wrap', alignItems:'center'}}>
            <button type="button" onClick={historical}>Run Historical</button>
            <button type="button" onClick={async ()=>{
              setStatus('Dispatching full flow...')
              try{
                const base = {
                  symbol: (sym||'').trim().toUpperCase(),
                  end_date_time: new Date().toISOString().slice(0,10).replace(/-/g,'') + ' 23:59:59',
                  duration_str: '1 M',
                  bar_size_setting: '1 day',
                  what_to_show: 'TRADES',
                  use_rth: 1,
                  snapshot_date: asOf || undefined,
                  expiry_filter: expiry || undefined,
                  max_contracts: Number(maxContracts)||20,
                  rights: R(),
                  strikes_per_side: Number(strikesPerSide)||undefined,
                  selection_mode: mode,
                  expiry_count: Number(expiryCount)||2,
                  spot_price: spotAuto ? undefined : (Number(spotVal)||undefined),
                  moneyness_targets: mode==='moneyness' ? MTargets() : undefined,
                  delta_targets: mode==='delta' ? DTargets() : undefined,
                  iv_assumption: mode==='delta' ? (Number(ivAssumption)||undefined) : undefined,
                  use_underlying_iv: mode==='delta' ? !!useUnderlyingIv : undefined,
                  timeout_seconds: 20.0,
                }
                const { data } = await api.orchestrator.optionContracts.fullflow(base)
                setOut(JSON.stringify(data, null, 2))
                const taskId = data?.details?.task_id
                if (!taskId) { setStatus('Dispatched full flow (no task id)'); return }
                setStatus(`Full flow dispatched (task ${taskId}). Waiting for result...`)
                if (pollTimerRef.current) clearInterval(pollTimerRef.current)
                pollTimerRef.current = setInterval(async ()=>{
                  try {
                    const { data: res } = await api.orchestrator.optionContracts.selectStatus(taskId)
                    if (res && (typeof res.processed !== 'undefined' || (res.results && Array.isArray(res.results)))) {
                      clearInterval(pollTimerRef.current); pollTimerRef.current = null
                      setHistoricalRes(res)
                      setOut(JSON.stringify(res, null, 2))
                      const p = res.processed||0, s = res.saved||0, f = res.failed||0
                      setStatus(`Full flow complete: processed ${p}, saved ${s}, failed ${f}`)
                    } else if (res && res.status) {
                      setStatus(`Full flow in progress... (${res.status})`)
                    } else {
                      setStatus('Full flow in progress...')
                    }
                  } catch (err) {
                    setStatus(`Full flow failed or not ready: ${err?.message||String(err)}`)
                  }
                }, 1000)
              } catch(err){ setStatus(`Full flow failed: ${err?.message||String(err)}`) }
            }}>Run Full Flow</button>
            {historicalRes && (
              <span style={{color:'#10b981', fontSize:'.9rem'}}>
                Saved {historicalRes.saved||0}, Failed {historicalRes.failed||0}
              </span>
            )}
          </div>
        </section>
      )}

      {active==='preview' && (
        <section>
          <h3>Preview Saved Option Bars</h3>
          <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(200px,1fr))', gap:'.5rem'}}>
            <label>Expiry
              <input value={previewParams.expiry} onChange={e=>setPreviewParams({...previewParams, expiry:e.target.value})} placeholder="YYYY-MM-DD" />
            </label>
            <label>Right
              <select value={previewParams.right} onChange={e=>setPreviewParams({...previewParams, right:e.target.value})}><option value="C">C</option><option value="P">P</option></select>
            </label>
            <label>Strike
              <input value={previewParams.strike} onChange={e=>setPreviewParams({...previewParams, strike:e.target.value})} placeholder="e.g., 135" />
            </label>
            <label>Bar Size
              <input value={previewParams.bar_size} onChange={e=>setPreviewParams({...previewParams, bar_size:e.target.value})} placeholder="1 day" />
            </label>
            <label>Start Date
              <input type="date" value={previewParams.start_date} onChange={e=>setPreviewParams({...previewParams, start_date:e.target.value})} />
            </label>
            <label>End Date
              <input type="date" value={previewParams.end_date} onChange={e=>setPreviewParams({...previewParams, end_date:e.target.value})} />
            </label>
          </div>
          <div style={{marginTop:'.5rem'}}>
            <button type="button" onClick={async ()=>{
              setStatus('Loading preview...')
              try{
                const params = { symbol:(sym||'').trim().toUpperCase(), expiry: previewParams.expiry, right: previewParams.right, strike: previewParams.strike, bar_size: previewParams.bar_size, start_date: previewParams.start_date, end_date: previewParams.end_date, limit: 200 }
                const { data } = await api.orchestrator.optionContracts.browseBars(params)
                setPreviewRows(data?.rows||[]); setPreviewTotal(data?.total||0)
                setStatus(`Preview loaded (${data?.rows?.length||0}/${data?.total||0})`)
              } catch(err){ setStatus(`Preview failed: ${err?.message||String(err)}`) }
            }}>Load Preview</button>
            <span className="muted" style={{marginLeft:'.5rem'}}>{previewTotal ? `${previewTotal} total (showing up to 200)` : ''}</span>
          </div>
          {previewRows.length>0 && (
            <div style={{overflowX:'auto', marginTop:'.5rem'}}>
              <table style={{borderCollapse:'collapse', minWidth:'720px'}}>
                <thead>
                  <tr>
                    {['time','expiry','right','strike','open','high','low','close','volume','bar_size'].map(h=> (<th key={h} style={th}>{h}</th>))}
                  </tr>
                </thead>
                <tbody>
                  {previewRows.map((r,i)=> (
                    <tr key={i}>
                      <td style={td}>{r.time}</td>
                      <td style={td}>{r.expiry}</td>
                      <td style={td}>{r.right}</td>
                      <td style={td}>{Number(r.strike).toFixed(2)}</td>
                      <td style={td}>{Number(r.open).toFixed(2)}</td>
                      <td style={td}>{Number(r.high).toFixed(2)}</td>
                      <td style={td}>{Number(r.low).toFixed(2)}</td>
                      <td style={td}>{Number(r.close).toFixed(2)}</td>
                      <td style={td}>{Number(r.volume||0).toFixed(0)}</td>
                      <td style={td}>{r.bar_size}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      <div style={{margin:'.5rem 0', color:'#8a90ad', fontSize:'.9rem'}}>{status}</div>
      <pre className="out">{out}</pre>
    </section>
  )
}

const th = { textAlign:'left', padding:'.4rem .5rem', borderBottom:'1px solid #e5e7eb', fontWeight:600, fontSize:'.9rem' }
const td = { textAlign:'left', padding:'.35rem .5rem', borderBottom:'1px solid #f1f5f9', fontSize:'.9rem', whiteSpace:'nowrap' }
