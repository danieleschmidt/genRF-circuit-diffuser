.PHONY: help install install-dev test test-cov lint format clean docs

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=genrf --cov-report=html --cov-report=term-missing

lint: ## Run linting
	flake8 genrf tests
	mypy genrf

format: ## Format code
	black genrf tests
	isort genrf tests

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs: ## Build documentation
	cd docs && make html

check: lint test ## Run all checks

build: clean ## Build package
	python -m build

upload-test: build ## Upload to TestPyPI
	python -m twine upload --repository testpypi dist/*

upload: build ## Upload to PyPI
	python -m twine upload dist/*

# Enhanced development targets
test-unit: ## Run unit tests only
	pytest tests/unit/

test-integration: ## Run integration tests only
	pytest tests/integration/

test-performance: ## Run performance tests only
	pytest tests/performance/ --benchmark-only

benchmark: ## Run performance benchmarks
	python benchmarks/run_benchmarks.py

security-scan: ## Run security scanning
	bandit -r genrf -f json -o bandit-report.json || true
	safety check --json --output safety-report.json || true

pre-commit-run: ## Run pre-commit hooks on all files
	pre-commit run --all-files

docker-build: ## Build Docker image
	docker build -t genrf:latest .

docker-dev: ## Run development container
	docker-compose up genrf-dev

docker-jupyter: ## Run Jupyter container
	docker-compose up genrf-jupyter