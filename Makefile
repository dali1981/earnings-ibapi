# Makefile for Trading Project

.PHONY: help install install-dev test test-unit test-integration test-data test-api test-performance clean lint format coverage docs

# Default target
help:
	@echo "Available targets:"
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