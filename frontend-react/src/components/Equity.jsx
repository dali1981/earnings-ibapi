import React, { useState } from 'react'
import { api } from '../apiClient'

export default function Equity() {
  const [sym, setSym] = useState('AAPL')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [bar, setBar] = useState('1 day')
  const [symbols, setSymbols] = useState('AAPL')
  const [duration, setDuration] = useState('5 D')
  const [barSize, setBarSize] = useState('1 day')
  const [endDate, setEndDate] = useState('')
  const [status, setStatus] = useState('')
  const [out, setOut] = useState('')
  const [datesInfo, setDatesInfo] = useState('')

  async function readRepo(e) {
    e?.preventDefault?.()
    setStatus('Reading from repository...')
    setOut('Loading...')
    try {
      const { data } = await api.equity.bars(sym, start, end, bar)
      setOut(JSON.stringify(data, null, 2))
      setStatus(`Read ${data?.rows || 0} bars from repository`)
    } catch (err) {
      setOut(`Error: ${err.message}`)
      setStatus('Read failed')
    }
  }

  async function loadDates(e) {
    e?.preventDefault?.()
    setDatesInfo('Loading available dates...')
    try {
      const { data } = await api.equity.dates(sym, bar)
      const arr = data?.dates || []
      if (arr.length) {
        const first = arr[0]
        const last = arr[arr.length - 1]
        setDatesInfo(`Available: ${arr.length} dates. Range: ${first} to ${last}`)
      } else {
        setDatesInfo('No dates available')
      }
    } catch (err) {
      setDatesInfo(`Error: ${err.message}`)
    }
  }

  async function fetchTWS(e) {
    e?.preventDefault?.()
    if (!symbols) { alert('Please enter symbol(s)'); return }
    const edt = endDate ? `${endDate.replace(/-/g, '')} 23:59:59` : new Date().toISOString().slice(0,10).replace(/-/g,'') + ' 23:59:59'
    setStatus('Fetching from TWS...')
    setOut('Dispatching request to orchestrator...')
    try {
      const { data } = await api.orchestrator.fetchHistoricalBars({
        symbols: symbols.split(',').map(s => s.trim()),
        end_date_time: edt,
        duration_str: duration,
        bar_size_setting: barSize,
        repository_type: 'equity'
      })
      setOut(JSON.stringify(data, null, 2))
      setStatus(`Fetching ${data?.processed || 0} symbol(s) from TWS. Job: ${data?.details?.group_id || data?.details?.chain_id || 'N/A'}`)
    } catch (err) {
      setOut(`Error: ${err.message}`)
      setStatus('Fetch failed')
    }
  }

  return (
    <section>
      <h2>Equity History (React)</h2>
      <form onSubmit={readRepo}>
        <label>Symbol <input value={sym} onChange={e=>setSym(e.target.value)} /></label>
        <label>Start <input type="date" value={start} onChange={e=>setStart(e.target.value)} /></label>
        <label>End <input type="date" value={end} onChange={e=>setEnd(e.target.value)} /></label>
        <label>Bar Size <input value={bar} onChange={e=>setBar(e.target.value)} /></label>
        <button type="submit">Read from Repository</button>
        <button type="button" onClick={loadDates} style={{marginLeft:'.5rem'}}>Available Dates</button>
      </form>
      {datesInfo && <div className="muted" style={{marginTop:'.3rem'}}>{datesInfo}</div>}
      <form onSubmit={fetchTWS} style={{marginTop:'.6rem'}}>
        <label>Symbol(s) <input value={symbols} onChange={e=>setSymbols(e.target.value)} placeholder="AAPL or AAPL,MSFT"/></label>
        <label>Duration <select value={duration} onChange={e=>setDuration(e.target.value)}>
          <option value="1 D">1 Day</option>
          <option value="5 D">5 Days</option>
          <option value="1 W">1 Week</option>
          <option value="2 W">2 Weeks</option>
          <option value="1 M">1 Month</option>
          <option value="3 M">3 Months</option>
          <option value="6 M">6 Months</option>
          <option value="1 Y">1 Year</option>
        </select></label>
        <label>Bar Size <select value={barSize} onChange={e=>setBarSize(e.target.value)}>
          <option value="1 min">1 minute</option>
          <option value="5 mins">5 minutes</option>
          <option value="15 mins">15 minutes</option>
          <option value="30 mins">30 minutes</option>
          <option value="1 hour">1 hour</option>
          <option value="1 day">1 day</option>
        </select></label>
        <label>End Date <input type="date" value={endDate} onChange={e=>setEndDate(e.target.value)} /></label>
        <button type="submit">Fetch from TWS (via Orchestrator)</button>
      </form>
      <div className="muted" style={{margin:'.5rem 0'}}>{status}</div>
      <pre className="out">{out}</pre>
    </section>
  )
}
