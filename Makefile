# Makefile for Trading Project

.PHONY: help install install-dev test test-unit test-integration test-data test-api test-performance clean lint format coverage docs dev consume-option-chains consume-option-chains-demo

# Default target
help:
	@echo "Available targets:"
	@echo "  dev            Bootstrap local dev venv + deps"
	@echo "  dev-rebuild    Recreate ./venv and install deps"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-data      Run data validation tests"
	@echo "  test-api       Run API tests"
	@echo "  test-performance Run performance tests"
	@echo "  test-smoke     Run smoke tests (fast subset)"
	@echo "  coverage       Generate coverage report"
	@echo "  lint           Run code linting"
	@echo "  format         Format code"
	@echo "  clean          Clean up artifacts"
	@echo "  docs           Generate documentation"
	@echo "  consume-option-chains   Run option chain backfill consumer once"
	@echo "  consume-option-chains-demo  Run consumer on demo topic once"
	@echo "  show-earnings   Show upcoming earnings from local DB"
	@echo "  dedup-earnings  Deduplicate earnings repository partitions"
	@echo "  from-earnings  Bootstrap data for symbols with earning dates"
	@echo "  from-earnings-dry  Show planned actions without execution"
	@echo "  services-up    Run all FastAPI services (dev compose)"
	@echo "  services-down  Stop FastAPI services (dev compose)"
	@echo "  run-calendar   Run calendar API locally (uvicorn)"
	@echo "  run-equity     Run equity API locally (uvicorn)"
	@echo "  run-options    Run option history API locally (uvicorn)"
	@echo "  run-chains     Run option chain API locally (uvicorn)"
	@echo "  run-frontend   Run frontend locally (uvicorn)"
	@echo "  run-ib         Run IBX service API locally (uvicorn)"
	@echo "  test-frontend  Run frontend unit/integration tests (pytest)"
	@echo "  test-frontend-e2e  Run browser E2E tests (Playwright)"
	@echo "  test-calendar  Run calendar service tests"
	@echo "  test-events    Run event bus unit tests"
	@echo "  celery-worker  Start a Celery worker (ibx tasks)"
	@echo "  backfill-contracts-us  Enqueue US contracts backfill (Celery)"

# Installation targets
install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-test.txt
	pip install -r requirements-dev.txt

# Test targets
test-setup:
	python tests/test_runner.py --setup
	python tests/test_runner.py --validate

test: test-setup
	python tests/test_runner.py --report

test-unit:
	python tests/test_runner.py --unit

test-integration:
	python tests/test_runner.py --integration

test-data:
	python tests/test_runner.py --data

test-api:
	python tests/test_runner.py --api

test-performance:
	python tests/test_runner.py --performance

test-smoke:
	python tests/test_runner.py --smoke

# Coverage
coverage:
	pytest --cov=. --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

# Code quality
lint:
	flake8 --config .flake8 .
	pylint --rcfile .pylintrc *.py

format:
	black --config pyproject.toml .
	isort --settings-path pyproject.toml .

# Local development (creates ./venv if missing, installs core libs)
dev:
	@if [ ! -d venv ]; then \
		python3 -m venv venv; \
		echo "Created virtualenv at ./venv"; \
	fi
	. venv/bin/activate && \
	python -m pip install --upgrade pip && \
	python -m pip install -r requirements-test.txt && \
	python -m pip install -r requirements-local.txt && \
	python -m pip install black isort && \
	echo "Dev environment ready. Activate with: source venv/bin/activate"

dev-rebuild:
	rm -rf venv
	$(MAKE) dev

# Cleanup
clean:
	python tests/test_runner.py --clean
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/

# Documentation
docs:
	sphinx-build -b html docs/ docs/_build/html

# Bootstrap from earnings CSV
from-earnings:
	python ops/jobs/run_from_earnings.py --max-symbols 20 --min-days-until 0 --lookback-days 365

from-earnings-dry:
	python ops/jobs/run_from_earnings.py --max-symbols 20 --min-days-until 0 --lookback-days 365 --dry-run

# Development workflow
dev-setup: install-dev test-setup
	@echo "Development environment setup complete"

ci-test: test-setup
	python tests/test_runner.py --unit --parallel
	python tests/test_runner.py --integration
	python tests/test_runner.py --data

# Docker targets (if using containers)
docker-build:
	docker build -t trading-project .

docker-test:
	docker run --rm trading-project make test

# Database/Data management
reset-test-data:
	rm -rf tests/temp_data/
	python -c "from tests.conftest import setup_test_data; setup_test_data()"

# Performance profiling
profile-tests:
	python -m cProfile -o profile.stats tests/test_runner.py --performance
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# Event consumers
consume-option-chains:
	python ops/jobs/option_chain_backfill_consumer.py --service backfill-broker --topic option_chains --bar-size "1 day" --once

consume-option-chains-demo:
	python ops/jobs/option_chain_backfill_consumer.py --service backfill-broker --topic option_chains.demo --bar-size "1 day" --once

show-earnings:
	python ops/scripts/show_upcoming_earnings.py --days 30

dedup-earnings:
	python ops/scripts/dedup_earnings.py

# FastAPI services (dev compose)
services-up:
	docker compose -f docker-compose.dev.yml up --build

services-down:
	docker compose -f docker-compose.dev.yml down --remove-orphans

# Run individual services locally (requires venv with deps installed)
run-calendar:
	uvicorn services.calendar_api.app:app --host 0.0.0.0 --port 9001 --reload

run-equity:
	uvicorn services.equity_api.app:app --host 0.0.0.0 --port 9002 --reload

run-options:
	uvicorn services.option_history_api.app:app --host 0.0.0.0 --port 9003 --reload

run-chains:
	uvicorn services.option_chain_api.app:app --host 0.0.0.0 --port 9004 --reload

run-frontend:
	uvicorn services.frontend.main:app --host 0.0.0.0 --port 9000 --reload

run-ib:
	uvicorn services.ib_api.main:app --host 0.0.0.0 --port 9010 --reload

test-frontend:
	pytest -q -m "frontend and (unit or integration)" -o addopts=

test-frontend-e2e:
	python -m pip install -r requirements-test.txt >/dev/null 2>&1 || true
	npx --yes playwright install chromium >/dev/null 2>&1 || true
	pytest -q -m frontend_e2e -o addopts=

test-calendar:
	pytest -q -m "calendar and (unit or integration)" -o addopts=

test-events:
	pytest -q -m events -o addopts=

# Celery
celery-worker:
	celery -A ibx_celery.app worker --loglevel=INFO

backfill-contracts-us:
	python ops/jobs/backfill_contracts_us_rt.py --wait
