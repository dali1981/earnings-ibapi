import React from 'react'

const pathMapping = {
  'home': 'Home',
  'calendar': 'Earnings Calendar',
  'equity': 'Equity History',
  'chains': 'Option Chains',
  'opt_sel': 'Option Contracts',
  'opt_hist': 'Option History',
  'contracts': 'Contracts'
}

export default function Breadcrumb({ currentPage, onNavigate }) {
  const breadcrumbs = []
  
  // Always start with Home
  breadcrumbs.push({ key: 'home', label: 'Home', isActive: false })
  
  // Add current page if not home
  if (currentPage && currentPage !== 'home') {
    breadcrumbs.push({ 
      key: currentPage, 
      label: pathMapping[currentPage] || currentPage,
      isActive: true 
    })
  }

  return (
    <nav 
      aria-label="Breadcrumb" 
      style={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: '0.5rem', 
        fontSize: '0.9rem', 
        color: 'var(--muted)',
        marginBottom: '1rem'
      }}
    >
      {breadcrumbs.map((item, index) => (
        <div key={item.key} style={{ display: 'flex', alignItems: 'center' }}>
          {index > 0 && (
            <span style={{ margin: '0 0.25rem', color: 'var(--line)' }}>›</span>
          )}
          
          {item.isActive ? (
            <span style={{ 
              color: 'var(--fg)', 
              fontWeight: '500',
              display: 'flex',
              alignItems: 'center',
              gap: '0.25rem'
            }}>
              {item.key === 'home' && '🏠'} {item.label}
            </span>
          ) : (
            <button
              onClick={() => onNavigate(item.key)}
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--muted)',
                cursor: 'pointer',
                padding: '0.25rem 0.5rem',
                borderRadius: '4px',
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem',
                fontSize: 'inherit'
              }}
              className="breadcrumb-btn"
            >
              {item.key === 'home' && '🏠'} {item.label}
            </button>
          )}
        </div>
      ))}
    </nav>
  )
}