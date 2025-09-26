import React from 'react'

export default function Home() {
  return (
    <section style={{ textAlign: 'center', padding: '2rem' }}>
      <h2>Trading Services Dashboard</h2>
      <p className="muted" style={{ marginBottom: '2rem' }}>
        Welcome to the Trading Microservices platform. Select a service from the sidebar to get started.
      </p>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', marginTop: '2rem' }}>
        <div className="service-card">
          <h3>📅 Calendar Service</h3>
          <p>View and fetch upcoming earnings announcements from multiple sources</p>
        </div>
        
        <div className="service-card">
          <h3>📈 Equity History</h3>
          <p>Access historical equity bar data and fetch new data from TWS</p>
        </div>
        
        <div className="service-card">
          <h3>🔗 Option Chains</h3>
          <p>Browse option chain snapshots and real-time data</p>
        </div>
        
        <div className="service-card">
          <h3>📋 Option Contracts</h3>
          <p>Manage option contract selection and enrichment</p>
        </div>
        
        <div className="service-card">
          <h3>📊 Option History</h3>
          <p>Query historical option bar data and analytics</p>
        </div>
        
        <div className="service-card">
          <h3>⚙️ Orchestrator</h3>
          <p>Contract search and setup operations</p>
        </div>
      </div>
    </section>
  )
}