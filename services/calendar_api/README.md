# Calendar API Service

FastAPI service exposing the earnings calendar repository and on-demand fetchers.

## Endpoints

- `GET /health`: Basic health check.
- `GET /calendar/upcoming`:
  - Read upcoming earnings from the repository only.
  - Query: `days` (default 14, 0–180), `symbols` (comma-separated optional).
  - Headers: `X-Calendar-Source=repo`, `X-Calendar-Collection-Date` (if available).
- `GET /calendar/get-upcoming`:
  - Same semantics as `/calendar/upcoming`; reads from repo only.
- `POST /calendar/fetch-upcoming`:
  - Fetch upcoming earnings from external sources and persist new events.
  - Query: `days` (default 14), `source` (one of `nasdaq`, `yahoo`, `all`; case-insensitive; default `nasdaq`), `symbols` (comma-separated; required for `yahoo`).
  - Convenience alias: `symbol` (single ticker) is accepted and merged into `symbols`.
  - Headers: `X-Calendar-Source=fetched`, `X-Calendar-Collection-Date=YYYY-MM-DD`.
  - Behavior: Fetch via async fetcher, then write only events not already present in the repo (dedup key: `(symbol, earnings_date, source)`).
- `POST /calendar/fetch-upcoming/yahoo`:
  - Convenience Yahoo-only fetch; requires `symbols`.
  - Same persistence semantics as above.
- `POST /calendar/events`:
  - Bulk add events. Deduplicates by `(symbol, earnings_date, source)`; logs duplicates.
- `GET /calendar/present`:
  - Query: `symbol`, `earnings_date`, optional `source`.
  - Returns whether an event is present in the repo for that exact key.
- `GET /calendar/symbol/{symbol}`:
  - Query: `days_back` (default 365). Returns recent historical earnings for a symbol.
- `GET /calendar/stats`:
  - Basic storage stats from the repository.

## Behavior & Design

- Read-only vs fetch:
  - Use `GET /calendar/upcoming` (or `get-upcoming`) to read what’s already stored.
  - Use `POST /calendar/fetch-upcoming` to pull from external sources and persist.
- No freshness cache:
  - The prior idea of a "freshness" cache is removed from the API layer. Persistence uses a presence check against the repo before writing, keeping the repo authoritative.
- Persistence:
  - Uses `EarningsCalendarRepository` (PyArrow/Parquet) with partitioning by `collection_date` and `source`.
  - Pre-filters fetched events by querying the repo for the min–max date window and matching `symbols`/`sources`; writes only truly new events.
  - Emits a file-based data-available event via `FileEventBus` on successful writes.
  - Safety: Generic `save()` is disabled for this dataset to prevent accidental overwrites. The service uses `store_earnings_batch()` which reads existing partition files, merges, deduplicates by `(symbol, earnings_date, source)`, and rewrites only the target `collection_date`/`source` partition.

## Sources & Fetching

- Abstraction: `EarningsCalendarSource` defines the async `fetch_earnings_range(start, end, symbols)` contract.
- Implementations: `NasdaqEarningsSource`, `YahooEarningsSource` (Yahoo requires `symbols`).
- Registry: `SourceRegistry` builds concrete source instances so fetchers don’t hard-code per-source wiring.
- Orchestrator: `AsyncEarningsCalendarFetcher` uses the registry to fetch from one or multiple sources concurrently and deduplicate results.
  - Yahoo specifics: API calls require realistic headers. Per-symbol lookup is supported (requires `symbols` in the route). The internal source also supports day-level bulk calendar queries.

## Schemas

- Response model: `EarningsEventModel` (Pydantic) mirrors the core event fields. Enum values (e.g., `time`, `source`) are normalized to strings in responses.
- Input model: `NewEarningsEvent` for bulk add.

## Usage Examples

- Read upcoming from repo:
  - `curl 'http://127.0.0.1:8000/calendar/upcoming?days=7'`
- Fetch and save NASDAQ upcoming:
  - `curl -X POST 'http://127.0.0.1:8000/calendar/fetch-upcoming?days=7'`
- Fetch and save Yahoo upcoming for specific symbols:
  - `curl -X POST 'http://127.0.0.1:8000/calendar/fetch-upcoming?days=7&source=yahoo&symbols=AAPL,MSFT'`
- Fetch both (Yahoo requires symbols):
  - `curl -X POST 'http://127.0.0.1:8000/calendar/fetch-upcoming?days=7&source=all&symbols=AAPL,MSFT'`

## Development Notes

- Code layout:
  - Routers: `routers/calendar.py`
  - Schemas: `schemas/calendar.py`
  - Services: `service/earnings_fetcher_service.py`, `service/calendar_service.py`
- Important changes:
  - Removed refresh/freshness decisions from API; repo is source of truth.
  - Added source selection (`nasdaq|yahoo|all`) to fetch endpoint and a Yahoo convenience route.
  - Normalized enum values for storage/response and added persistence presence checks.
  - Introduced `SourceRegistry` and refactored async fetcher to use source abstractions.

## Testing

- Create venv and install dev deps per repo guidelines.
- Suggested tests:
  - `POST /calendar/fetch-upcoming?source=yahoo` without `symbols` returns 400.
  - `GET /calendar/upcoming` returns repo data only.
  - Repeated `fetch-upcoming` calls do not increase stored rows for identical `(symbol, earnings_date, source)` keys.
