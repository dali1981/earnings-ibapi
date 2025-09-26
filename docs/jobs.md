# Jobs Overview

This page summarizes available jobs, their purpose, and current status.

## Taxonomy: Examples vs. Scripts vs. Jobs
- examples/ (read-only demos)
  - Purpose: Teach usage; quick inspections; no side effects.
  - External calls: None — only reads local repos.
  - Scheduling: Never; run ad-hoc.
  - I/O: Read from repo, print or sample small outputs (no writes).
  - Typical: get_with_options_example.py, small analytics snippets.

- scripts/ (local utilities)
  - Purpose: Small, focused repo helpers (operational but safe).
  - External calls: None by default — operate purely on local data.
  - Scheduling: Ad-hoc; handy dev tools.
  - I/O: Read/write local repos; may tidy data (e.g., dedup).
  - Typical: show_upcoming_earnings.py, dedup_earnings.py.

- jobs/ (operational workflows)
  - Purpose: Production/backfill flows; orchestrated tasks.
  - External calls: Yes (IB/API); persist then emit events.
  - Scheduling: Cron/systemd/airflow; checkpointed/idempotent.
  - I/O: Write to repos, publish data.available events; may read upstream repos first.
  - Tracing: Include a `job_id` in published payloads or run within a logging `log_context` so `correlation_id` is attached to all events.
  - Typical: backfill_option_chains.py, backfill_option_bars.py, option_chain_backfill_consumer.py, run_earnings_pipeline.py.

Rule of thumb: Teaching or showcasing? examples/. Need a safe local tool? scripts/. Anything that talks to IB/APIs or runs regularly? jobs/.

## Active Jobs
- `jobs/snapshot_option_chains.py`:
  - Snapshot current option chain (no historical chains via IB). Use `--date` as a label; offline safe mode writes to `data/exports/demo`.
- `jobs/contracts_snapshot.py`:
  - Snapshots contract descriptions via IB. Use `--patterns` to target specific symbols/patterns. Writes under `data/contract_descriptions`.
- `jobs/backfill_contracts_us_rt.py`:
  - Backfills US equity contract descriptions across exchanges using Celery.
  - Modes:
    - Direct IB (default): workers connect to IB (use `--use-service` to reuse long‑lived runtime per worker).
    - Via API: tasks call the FastAPI service (`services/ib_api`) with `--via-api --api-base http://host:9010`.
  - Usage:
    - Start a worker: `celery -A ibx_celery.app worker --loglevel=INFO`
    - Enqueue all defaults (direct IB): `python ops/jobs/backfill_contracts_us_rt.py --wait`
    - Enqueue via API: `python ops/jobs/backfill_contracts_us_rt.py --via-api --api-base http://localhost:9010 --wait`
    - Enqueue custom patterns (chunked): `python ops/jobs/backfill_contracts_us_rt.py --patterns AAPL MSFT NVDA --chunk-size 50 --wait`
  
- `jobs/backfill_option_chains.py` (deprecated):
  - Deprecated in favor of `jobs/snapshot_option_chains.py`. IB does not provide historical chains; this job now treats `--end` as a single snapshot date and logs a deprecation warning.
- `jobs/backfill_equity_bars.py` / `jobs/backfill_option_bars.py`:
  - Backfill historical equity/option bars via tasks. Support demo base with `--demo`.
- `jobs/option_chain_backfill_consumer.py`:
  - Event-driven consumer: reads `option_chains` events and backfills `option_bars`. Use with cron/systemd.
  - Correlation: consumer propagates `correlation_id` from inbound events into downstream emissions for end-to-end traceability.
- `jobs/run_earnings_pipeline.py`:
  - Discover/scoring/export/data collection pipeline for earnings-driven trading.
- `jobs/run_from_earnings.py`:
  - Bootstrap backfills based on recent earnings discoveries (chains, equity bars, option bars), optionally contracts.
- `jobs/run_setup_symbol.py` / `jobs/run_daily_update.py` / `jobs/run_daily_comprehensive.py`:
  - Orchestrated workflows for setup and daily updates (ensure dependencies, backfill as needed).

## Deprecated / Legacy
- `jobs/contracts_snapshot_one.py`:
  - Deprecated. Use `jobs/contracts_snapshot.py --patterns` to target specific symbols.
- `jobs/get_with_options.py`:
  - Deprecated ad-hoc inspection script. Prefer repository queries or add a usage example under `examples/`.
- `jobs/jobs_contract_details_backfill.py`:
  - Deprecated. Prefer `ibx_flows/contracts_backfill` or use `jobs/contracts_snapshot.py` for snapshots.

## Notes
- All jobs write under repo-root `data/…` (via `config.PROJECT_ROOT`). Running from subfolders will still write to `<repo>/data`.
- Consumers and backfill jobs emit `data.available` events via the filesystem bus. See `docs/data_ingestion.md`.
- Prefer repositories in `repositories/` over legacy `ibx_repos` when possible. Migrate legacy jobs as needed.
