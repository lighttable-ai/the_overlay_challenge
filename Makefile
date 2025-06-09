.PHONY: help
.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: run-backend
run-backend: ## Run the backend on port 8001
	cd api && uv run --python 3.12 --dev uvicorn overlay.index:app --reload --port 8001

.PHONY: install-backend
install-backend: ## Install backend dependencies
	cd api && uv sync --dev

