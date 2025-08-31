import React from 'react'
import Calendar from './components/Calendar'
import Equity from './components/Equity'
import Chains from './components/Chains'
import OptionContracts from './components/OptionContracts'
import Orchestrator from './components/Orchestrator'

export default function App() {
  return (
    <>
      <header>
        <h1>Trading Microservices — React UI</h1>
      </header>
      <main>
        <Calendar />
        <Orchestrator />
        <Equity />
        <Chains />
        <OptionContracts />
      </main>
    </>
  )
}
