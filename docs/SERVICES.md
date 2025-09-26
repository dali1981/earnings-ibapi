# Services Architecture (FastAPI)

This project exposes data capabilities via multiple FastAPI microservices. Each service follows a consistent layout with clear separation of API boundary (routers/schemas), orchestration (service), and domain libraries (repositories, earnings, etc.).

## Layout
```
services/
  calendar_api/
    app.py                 # FastAPI app factory, includes routers
    routers/
      calendar.py          # /health, /calendar/upcoming
    schemas/
      calendar.py          # Pydantic models for API responses
    service/
      earnings_service.py  # Orchestrates fetcher and rate limits

  equity_api/
    app.py
    routers/equity.py      # /health, /equity/{symbol}/dates, /equity/{symbol}/bars
    schemas/equity.py

  option_history_api/
    app.py
    routers/options.py     # /health, /options/{underlying}/contracts, /options/{underlying}/bars
    schemas/options.py

  option_chain_api/
    app.py
    routers/chains.py      # /health, /chains/{symbol}/latest

  frontend/
    main.py                # FastAPI UI server
    templates/index.html   # Simple UI for manual calls
    static/styles.css
```

## Principles
- API boundary (routers, schemas) must not contain domain logic; they call the service layer.
- Service layer orchestrates repositories and fetchers from shared domain libs.
- Domain libraries (repositories/, earnings/, utils/, reliability/, ibx_*) remain shared and framework-free.
- Tests exist both in `tests/` (cross-service) and can be added per service.

## Running (Dev)
- `docker compose -f docker-compose.dev.yml up --build`
- Frontend: http://localhost:9000
- Services:
  - Calendar: http://localhost:9001
  - Equity: http://localhost:9002
  - Options: http://localhost:9003
  - Chains: http://localhost:9004

## Testing
- Health checks: `pytest tests/unit/test_services_health.py -q`
- Extend with service-specific tests under each service or in `tests/integration/`.

## Next Steps
- Add API gateway for single-origin access and aggregated OpenAPI docs.
- Add auth (API keys/JWT) via FastAPI dependencies.
- Add pagination/streaming for large datasets.
- Add service-level configuration via environment and shared settings helpers.

