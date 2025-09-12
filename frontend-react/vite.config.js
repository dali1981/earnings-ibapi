import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const targets = {
  calendar: process.env.VITE_CALENDAR_BASE || 'http://localhost:9001',
  equity: process.env.VITE_EQUITY_BASE || 'http://localhost:9002',
  chains: process.env.VITE_CHAINS_BASE || 'http://localhost:9004',
  orchestrator: process.env.VITE_ORCHESTRATOR_BASE || 'http://localhost:9005',
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: false,
    proxy: {
      '/svc/calendar': { target: targets.calendar, changeOrigin: true, rewrite: p => p.replace(/^\/svc\/calendar/, '') },
      '/svc/equity': { target: targets.equity, changeOrigin: true, rewrite: p => p.replace(/^\/svc\/equity/, '') },
      '/svc/chains': { target: targets.chains, changeOrigin: true, rewrite: p => p.replace(/^\/svc\/chains/, '') },
      '/svc/orchestrator': { target: targets.orchestrator, changeOrigin: true, rewrite: p => p.replace(/^\/svc\/orchestrator/, '') },
    }
  }
})
