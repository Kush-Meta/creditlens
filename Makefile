# CreditLens - development entry points.
PY ?= python3
PORT ?= 8000
TICKER ?= MSFT

.DEFAULT_GOAL := help
.PHONY: help install dev-install seed ingest universe gen-eval demo demo-quick serve test test-cov \
        lint fmt typecheck eval eval-offline eval-json sweep ablation clean docker-build \
        docker-run reset check

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	$(PY) -m pip install -e .

dev-install: ## Install with development extras
	$(PY) -m pip install -e ".[dev]"

seed: ## Load the offline demo corpus (fictional issuers, synthetic data)
	$(PY) -m creditlens.cli seed

universe: ## Ingest the full 42-issuer credit-spectrum universe from SEC EDGAR
	$(PY) -m creditlens.cli universe --max-filings 12 --min-year 2021 --workers 4

gen-eval: ## Regenerate both generated evaluation suites
	$(PY) -m creditlens.cli gen-eval --source universe
	$(PY) -m creditlens.cli gen-eval --source fixtures

ingest: ## Ingest a real issuer from SEC EDGAR: make ingest TICKER=ORCL
	$(PY) -m creditlens.cli ingest --ticker $(TICKER) --min-year 2021

demo: seed universe gen-eval ## Full demo: synthetic corpus + 42-issuer universe + eval suite
	@$(PY) -m creditlens.cli companies

demo-quick: seed ## Faster demo: synthetic corpus plus four real issuers
	@for t in F MSFT ORCL AAPL; do \
	  echo "--- ingesting $$t"; \
	  $(PY) -m creditlens.cli ingest --ticker $$t --min-year 2021 >/dev/null || exit 1; \
	done
	@$(PY) -m creditlens.cli companies

serve: ## Run the API and web UI
	$(PY) -m creditlens.cli serve --port $(PORT)

test: ## Run the test suite
	$(PY) -m pytest

test-cov: ## Run tests with a coverage report
	$(PY) -m pytest --cov=creditlens --cov-report=term-missing --cov-report=html

lint: ## Lint
	$(PY) -m ruff check creditlens tests scripts

fmt: ## Auto-fix lint findings
	$(PY) -m ruff check --fix creditlens tests scripts

typecheck: ## Static type check
	$(PY) -m mypy creditlens

check: lint test ## Lint and test - what CI runs

eval: ## Run the full evaluation suite (golden + universe) with the ablation
	$(PY) -m creditlens.cli eval --suite golden,universe --ablation

eval-offline: ## Run the network-free suites - what CI gates on
	$(PY) -m creditlens.cli eval --suite golden,fixtures --offline

eval-json: ## Emit raw eval metrics as JSON
	$(PY) -m creditlens.cli eval --json

ablation: ## Retrieval ablation only (no LLM calls)
	$(PY) -m creditlens.cli eval --ablation --limit 1

sweep: ## Grid-search retrieval parameters against the labelled cases
	PYTHONPATH=. $(PY) scripts/sweep_retrieval.py

reset: ## Delete the local database
	rm -f data/creditlens.db data/creditlens.db-wal data/creditlens.db-shm

clean: reset ## Remove build artefacts and caches
	rm -rf .pytest_cache htmlcov .coverage build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

docker-build: ## Build the container image
	docker build -t creditlens:latest .

docker-run: ## Run the container
	docker run --rm -p 8000:8000 \
	  -e ANTHROPIC_API_KEY=$$ANTHROPIC_API_KEY \
	  -v $$(pwd)/data:/app/data creditlens:latest
