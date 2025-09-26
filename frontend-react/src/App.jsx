import React, { useState } from 'react'
import { DataProvider } from './contexts/DataContext'
import Breadcrumb from './components/Breadcrumb'
import Home from './components/Home'
import Calendar from './components/Calendar'
import Equity from './components/Equity'
import Chains from './components/Chains'
import OptionContracts from './components/OptionContracts'
import OptionHistory from './components/OptionHistory'
import Orchestrator from './components/Orchestrator'

const PAGES = [
  { key: 'home', label: 'Home', component: Home, isHome: true },
  { key: 'calendar', label: 'Calendar', component: Calendar },
  { key: 'contracts', label: 'Contracts', component: Orchestrator },
  { key: 'equity', label: 'Equity History', component: Equity },
  { key: 'chains', label: 'Option Chains', component: Chains },
  { key: 'opt_sel', label: 'Option Contracts', component: OptionContracts },
  { key: 'opt_hist', label: 'Option History', component: OptionHistory },
]

export default function App() {
  const [active, setActive] = useState('home')
  const ActiveComp = (PAGES.find(p=>p.key===active)?.component) || Home
  const currentPage = PAGES.find(p=>p.key===active)
  
  return (
    <DataProvider>
      <div className="layout">
        <aside className="sidebar">
          <div 
            style={{fontWeight:700, marginBottom:'.5rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem'}}
            onClick={() => setActive('home')}
            className="sidebar-title"
          >
            🏠 Trading UI
          </div>
          <nav className="nav-menu">
            {PAGES.map(p=> (
              <button key={p.key} type="button" onClick={()=>setActive(p.key)}
                className={`nav-item ${active===p.key? 'active':''} ${p.isHome ? 'home-item' : ''}`}>
                {p.isHome ? '🏠' : ''} {p.label}
              </button>
            ))}
          </nav>
        </aside>
        <main style={{padding:'1rem'}}>
          <Breadcrumb currentPage={active} onNavigate={setActive} />
          <header style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
            <h1 style={{marginTop:0}}>{currentPage?.label || 'Trading Microservices — React UI'}</h1>
          </header>
          <ActiveComp />
        </main>
      </div>
    </DataProvider>
  )
}
