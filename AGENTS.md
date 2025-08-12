# Repository Guidelines

## Project Structure & Module Organization
- Core modules: top-level Python files (e.g., `main.py`, `runner.py`, `config.py`).
- Domain packages: `earnings/`, `trading/`, `repositories/`, `api/`, `ib/`, `utils/`.
- Tests: `tests/` with `unit/`, `integration/`, and `test_*.py` modules.
- Scripts & examples: `scripts/` and `examples/` for runnable workflows.
- Data & cache: `data/` (auto-created via `config.py`), plus `cache/`, `exports/`.
- Docs & config: `docs/`, `config/portfolio.yaml` for portfolio inputs.

## Build, Test, and Development Commands
- Environment: `python -m venv .venv && source .venv/bin/activate`
- Install (tests/dev): `pip install -r requirements-test.txt`
- Run tests: `pytest -q` (or `pytest -m "unit"`, `-m "integration"`)
- Make targets: `make test`, `make coverage`, `make clean` (some targets assume optional configs).
- Coverage report: `make coverage` then open `htmlcov/index.html`.
- Run examples: `python scripts/export_earnings_calendar.py` or `python main.py`.

## Coding Style & Naming Conventions
- Python 3.10+, PEP 8, 4-space indentation.
- Naming: modules `snake_case.py`; classes `CamelCase`; functions/vars `snake_case`; constants `UPPER_SNAKE`.
- Typing: prefer type hints and docstrings for public APIs.
- Formatting: use `black .` and `isort .` if available; lint with `flake8`/`pylint` when configured.

## Testing Guidelines
- Framework: `pytest` configured via `pytest.ini`.
- Coverage: enforced threshold `--cov-fail-under=80` (HTML in `htmlcov/`).
- Markers: `unit`, `integration`, `api`, `data`, `streaming`, `slow`.
- Conventions: files `tests/**/test_*.py`, functions `test_*`, classes `Test*`.
- Examples: `pytest -m "unit"`, `pytest -m "integration and not slow"`.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise summary (e.g., "Add historical earnings discovery"), optional scope.
- Reference issues (e.g., `#123`) and explain rationale in the body when relevant.
- PRs: clear description, linked issues, test plan/outputs, impact notes (data/IB usage), and small focused diffs.
- Ensure tests pass and coverage remains ≥80% before requesting review.

## Security & Configuration Tips
- Do not commit secrets or IB credentials; configure TWS/IB Gateway locally.
- Edit settings in `config.py` and `config/portfolio.yaml`; data paths under `data/` are created automatically.
- Avoid committing large generated data/logs; respect existing `.gitignore`.
