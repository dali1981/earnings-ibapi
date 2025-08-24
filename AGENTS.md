# Repository Guidelines

This guide helps contributors work consistently across the earnings/trading codebase and its IB integrations.

## Project Structure & Module Organization
- Core modules at repo root: `main.py`, `runner.py`, `config.py`.
- Domain packages: `earnings/`, `trading/`, `repositories/`, `api/`, `ib/`, `utils/`.
- Tests: `tests/` with `unit/` and `integration/`; files named `test_*.py`.
- Scripts & examples: `scripts/`, `examples/` (runnable workflows).
- Data & outputs: `data/` (auto-created by `config.py`), plus `cache/`, `exports/`.
- Config & docs: `config/portfolio.yaml`, `docs/`.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`.
- Install (tests/dev): `pip install -r requirements-test.txt`.
- Run tests: `pytest -q` (or `pytest -m "unit"`, `pytest -m "integration"`).
- Make targets: `make test` (pytest + coverage), `make coverage` (HTML at `htmlcov/index.html`), `make clean` (purge caches/artifacts).
- Run examples: `python scripts/export_earnings_calendar.py` or `python main.py`.

## Coding Style & Naming Conventions
- Python 3.10+, PEP 8, 4-space indentation.
- Naming: modules `snake_case.py`; classes `CamelCase`; functions/vars `snake_case`; constants `UPPER_SNAKE`.
- Typing: add type hints and docstrings for public APIs.
- Formatting: `black .` and `isort .`; lint with `flake8`/`pylint` if configured.

## Testing Guidelines
- Framework: `pytest`; markers: `unit`, `integration`, `api`, `data`, `streaming`, `slow`.
- Coverage: enforce `--cov-fail-under=80`; view HTML via `make coverage` → `htmlcov/index.html`.
- Conventions: files `tests/**/test_*.py`; functions `test_*`; classes `Test*`.
- Mark external/IB-dependent tests as `integration` or `slow`.

## Commit & Pull Request Guidelines
- Commits: imperative, concise (e.g., "Add historical earnings discovery"); reference issues (e.g., `#123`) and rationale when relevant.
- PRs: clear description, linked issues, test plan/outputs, impact notes (data/IB usage), and small focused diffs.
- Ensure tests pass locally and coverage ≥80% before requesting review.

## Security & Configuration Tips
- Never commit secrets or IB credentials. Configure TWS/IB Gateway locally.
- Edit settings in `config.py` and `config/portfolio.yaml`.
- `config.py` creates `data/`; avoid committing large generated data/logs. Respect `.gitignore`.

## Architecture & Ingestion
- Local-first policy: producers always write to repositories, then notify via events; consumers read locally only.
- Event bus: file-based JSONL under `data/cache/events/`; ready flags in `data/cache/ready/`.
- Catch-up & recovery: see `docs/data_ingestion.md` for JSONL consumer, checkpoints, rotation, and reconciliation patterns.
- Consumers:
  - Option chain → option bars backfill: `jobs/option_chain_backfill_consumer.py`.
  - Make targets: `make consume-option-chains`, `make consume-option-chains-demo`.

## Jobs
- See `docs/jobs.md` for an overview of active vs. deprecated jobs and usage.

## Scripts vs. Jobs vs. Examples
- `scripts/`: Small, focused utilities that operate on the local repository only (no external calls by default). These are developer-facing helpers (e.g., `show_upcoming_earnings.py`, `dedup_earnings.py`).
- `jobs/`: Operational workflows that may call external services (IB, APIs), run on schedules, or perform backfills. These emit events after writing to repos and are designed for cron/systemd orchestration.
- `examples/`: Read-only demonstrations that show how to use repositories and utilities in a simple, reproducible way (no side effects, no network). Good for learning and quick inspection.
