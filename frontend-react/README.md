React Frontend (Experimental)

Overview
- A lightweight React UI that mirrors the current FastAPI/Jinja frontend.
- Uses Vite for local dev with a proxy to backend services.
- Non-destructive: the original frontend at services/frontend remains unchanged.

Prerequisites
- Node.js 18+ and npm (or pnpm/yarn) installed locally.

Install & Run (dev)
1) cd services/frontend-react
2) Copy env template and adjust targets if needed:
   cp .env.example .env
   # Edit .env to point to your running backend services
3) Install deps:
   npm install
4) Start dev server (with proxy):
   npm run dev
   # Open the URL printed by Vite (e.g., http://localhost:5173)

Build & Preview
- Production build: npm run build
- Local static preview: npm run preview

Service URLs
- The Vite dev server proxies requests to backend services to avoid CORS.
- Configure targets in .env (or use defaults):
  - VITE_CALENDAR_BASE=http://localhost:9001
  - VITE_EQUITY_BASE=http://localhost:9002
  - VITE_CHAINS_BASE=http://localhost:9004
  - VITE_ORCHESTRATOR_BASE=http://localhost:9005

Endpoints used by this UI
- Calendar: GET /calendar/upcoming, POST /calendar/fetch-upcoming
- Equity: GET /equity/:symbol/bars
- Chains: GET /chains/:symbol/latest
- Orchestrator (historical): POST /orchestrator/historical/bars-by-symbols
- Orchestrator (contracts): POST /orchestrator/contracts/search|details|setup

Notes
- The React UI focuses on read-only exploration and dispatch of jobs. The actual
  processing and persistence happen in the backend services.
- Keep both frontends side-by-side; use whichever suits your workflow.

