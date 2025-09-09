import React, { useState } from 'react'
import Calendar from './components/Calendar'
import Equity from './components/Equity'
import Chains from './components/Chains'
import OptionContracts from './components/OptionContracts'
import OptionHistory from './components/OptionHistory'
import Orchestrator from './components/Orchestrator'

const PAGES = [
  { key: 'calendar', label: 'Calendar', component: Calendar },
  { key: 'contracts', label: 'Contracts', component: Orchestrator },
  { key: 'equity', label: 'Equity History', component: Equity },
  { key: 'chains', label: 'Option Chains', component: Chains },
  { key: 'opt_sel', label: 'Option Contracts', component: OptionContracts },
  { key: 'opt_hist', label: 'Option History', component: OptionHistory },
]

export default function App() {
  const [active, setActive] = useState('calendar')
  const ActiveComp = (PAGES.find(p=>p.key===active)?.component) || Calendar
  return (
    <div className="layout">
      <aside className="sidebar">
        <div style={{fontWeight:700, marginBottom:'.5rem'}}>Trading UI</div>
        <nav className="nav-menu">
          {PAGES.map(p=> (
            <button key={p.key} type="button" onClick={()=>setActive(p.key)}
              className={`nav-item ${active===p.key? 'active':''}`}>
              {p.label}
            </button>
          ))}
        </nav>
      </aside>
      <main style={{padding:'1rem'}}>
        <header>
          <h1 style={{marginTop:0}}>{PAGES.find(p=>p.key===active)?.label || 'Trading Microservices — React UI'}</h1>
        </header>
        <ActiveComp />
      </main>
    </div>
  )
}
