# ============================================================================
# Code Horde — Makefile for macOS Local Development
# ============================================================================
# Usage:
#   make setup     — first-time setup (brew, venv, Docker, DB, keys)
#   make up        — start infrastructure (Docker) + app (uvicorn)
#   make down      — stop everything
#   make dev       — start app with hot-reload
#   make tunnel    — open ngrok tunnel for WhatsApp webhooks
#   make status    — check health of all components
#   make logs      — tail application logs
#   make test      — run test suite
#   make clean     — remove all generated data
# ============================================================================

.PHONY: setup up down dev tunnel status logs test clean \
        infra-up infra-down infra-status \
        venv deps ollama-setup keys db-reset \
        lint format check help cli cli-exec dashboard \
        app app-install app-open

SHELL       := /bin/bash
PROJECT_DIR := $(shell pwd)
VENV        := $(PROJECT_DIR)/.venv
PYTHON      := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
UVICORN     := $(VENV)/bin/uvicorn
COMPOSE     := docker compose -f docker/docker-compose.local.yaml
ENV_FILE    := .env

# ============================================================================
# HIGH-LEVEL COMMANDS
# ============================================================================

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: check-deps venv deps infra-up db-wait keys env-file ## First-time setup
	@echo ""
	@echo "╔══════════════════════════════════════════════════╗"
	@echo "║        Code Horde — Setup Complete!               ║"
	@echo "╚══════════════════════════════════════════════════╝"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit .env — add your API keys"
	@echo "  2. make ollama-setup   (install local LLM)"
	@echo "  3. make dev            (start with hot-reload)"
	@echo "  4. make tunnel         (expose webhooks via ngrok)"
	@echo ""

up: infra-up dev ## Start infrastructure + app

down: ## Stop everything
	@echo "Stopping app..."
	@-pkill -f "uvicorn src.main:app" 2>/dev/null || true
	@echo "Stopping infrastructure..."
	@$(COMPOSE) down
	@echo "All services stopped."

# ============================================================================
# INFRASTRUCTURE (Docker)
# ============================================================================

infra-up: ## Start PostgreSQL + Redis + Neo4j (Homebrew-first, Docker fallback)
	@echo "Starting infrastructure..."
	@# Try Homebrew first (native macOS, no Docker needed) for Redis + Postgres
	@if command -v brew >/dev/null 2>&1; then \
		echo "Using Homebrew for Redis + Postgres..."; \
		brew list redis >/dev/null 2>&1 || brew install redis; \
		brew services start redis 2>/dev/null; \
		for pg in postgresql@16 postgresql@15 postgresql; do \
			if brew list $$pg >/dev/null 2>&1; then \
				brew services start $$pg 2>/dev/null; \
				break; \
			fi; \
		done; \
		if ! brew list postgresql@16 >/dev/null 2>&1 && \
		   ! brew list postgresql@15 >/dev/null 2>&1 && \
		   ! brew list postgresql >/dev/null 2>&1; then \
			brew install postgresql@16 && brew services start postgresql@16; \
		fi; \
		echo "  Redis:    localhost:6379"; \
		echo "  Postgres: localhost:5432"; \
		if command -v docker >/dev/null 2>&1; then \
			echo "Starting Neo4j via Docker..."; \
			$(COMPOSE) up -d neo4j 2>/dev/null || true; \
			echo "  Neo4j:    localhost:7687  (UI: http://localhost:7474)"; \
		else \
			echo "  Neo4j:    skipped (Docker not available — using in-memory graph)"; \
		fi; \
		echo "Infrastructure ready."; \
	elif command -v docker >/dev/null 2>&1; then \
		echo "Using Docker..."; \
		$(COMPOSE) up -d; \
		echo "Infrastructure ready (Docker)."; \
		echo "  Redis:    localhost:6379"; \
		echo "  Postgres: localhost:5432"; \
		echo "  Neo4j:    localhost:7687  (UI: http://localhost:7474)"; \
		echo "  RabbitMQ: localhost:5672  (UI: http://localhost:15672)"; \
	else \
		echo "ERROR: Neither Homebrew nor Docker found."; \
		echo "  Install Homebrew: /bin/bash -c \"\$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""; \
		exit 1; \
	fi
	@# Start Ollama (native macOS — not Docker)
	@if command -v ollama >/dev/null 2>&1; then \
		if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then \
			echo "Starting Ollama..."; \
			ollama serve >/dev/null 2>&1 & \
			sleep 2; \
		fi; \
		echo "  Ollama:   localhost:11434"; \
	else \
		echo "  Ollama:   not installed (run: make ollama-setup)"; \
	fi

infra-down: ## Stop infrastructure services
	@echo "Stopping infrastructure..."
	@-brew services stop redis 2>/dev/null || true
	@-brew services stop postgresql@16 2>/dev/null || true
	@-brew services stop postgresql@15 2>/dev/null || true
	@-brew services stop postgresql 2>/dev/null || true
	@-$(COMPOSE) down 2>/dev/null || true
	@echo "Infrastructure stopped."

infra-status: ## Show infrastructure status
	@echo "=== Ports ==="
	@echo -n "  PostgreSQL (5432): " && (nc -z localhost 5432 2>/dev/null && echo "UP" || echo "DOWN")
	@echo -n "  Redis      (6379): " && (nc -z localhost 6379 2>/dev/null && echo "UP" || echo "DOWN")
	@echo -n "  RabbitMQ   (5672): " && (nc -z localhost 5672 2>/dev/null && echo "UP" || echo "DOWN (optional)")
	@echo ""
	@echo "=== Homebrew Services ==="
	@-brew services list 2>/dev/null | grep -E "redis|postgresql" || echo "  (no Homebrew services)"
	@echo ""
	@echo "=== Docker ==="
	@-$(COMPOSE) ps 2>/dev/null || echo "  (Docker not running)"

db-wait: ## Wait for PostgreSQL to be ready
	@echo "Waiting for PostgreSQL..."
	@for i in $$(seq 1 30); do \
		pg_isready -h localhost -p 5432 -U agentadmin -d code_horde >/dev/null 2>&1 && break; \
		sleep 1; \
	done
	@echo "PostgreSQL is ready."

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "Resetting database..."
	@$(COMPOSE) down -v postgres
	@$(COMPOSE) up -d postgres
	@$(MAKE) db-wait
	@echo "Database reset complete."

# ============================================================================
# PYTHON APPLICATION
# ============================================================================

venv: ## Create Python virtual environment
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		CHOSEN_PYTHON=""; \
		if [ "$$(uname -m)" = "arm64" ]; then \
			for bp in /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3; do \
				if [ -x "$$bp" ]; then CHOSEN_PYTHON="$$bp"; break; fi; \
			done; \
		fi; \
		if [ -z "$$CHOSEN_PYTHON" ]; then CHOSEN_PYTHON=python3; fi; \
		echo "Using $$CHOSEN_PYTHON ($$($$CHOSEN_PYTHON --version))"; \
		$$CHOSEN_PYTHON -m venv $(VENV); \
		echo "Virtual environment created at $(VENV)"; \
	else \
		echo "Virtual environment already exists."; \
	fi

deps: venv ## Install Python dependencies
	@echo "Installing dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -e ".[dev]"
	@echo "Dependencies installed."

dev: ## Start app with hot-reload
	@echo "Starting Code Horde (development mode)..."
	@$(UVICORN) src.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--reload-dir src \
		--log-level debug

cli: ## Interactive terminal (connect to running API)
	@$(PYTHON) scripts/cli.py

cli-exec: ## One-shot CLI command (usage: make cli-exec CMD="/status")
	@$(PYTHON) scripts/cli.py --exec "$(CMD)"

dashboard: ## Open Command Center in browser
	@echo "Dashboard: http://localhost:8000"
	@open http://localhost:8000 2>/dev/null || xdg-open http://localhost:8000 2>/dev/null || echo "Open http://localhost:8000 in your browser"

app: ## Build macOS .app bundle
	@echo "Building Code Horde.app..."
	@$(PYTHON) scripts/build_app.py
	@echo ""
	@echo "Run with: open dist/Code Horde.app"

app-install: ## Build + install to /Applications
	@$(PYTHON) scripts/build_app.py --install

app-open: ## Launch the desktop app (standalone mode)
	@$(PYTHON) -m src.desktop.app --standalone

run: ## Start app (production mode, no reload)
	@$(UVICORN) src.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--workers 2 \
		--log-level info

# ============================================================================
# OLLAMA (native macOS)
# ============================================================================

ollama-setup: ## Install and configure Ollama with recommended models
	@echo "Setting up Ollama..."
	@if ! command -v ollama >/dev/null 2>&1; then \
		echo "Installing Ollama via Homebrew..."; \
		brew install ollama; \
	fi
	@echo "Pulling recommended models..."
	@ollama pull qwen2.5-coder:7b
	@echo ""
	@echo "Ollama ready. Start it with: ollama serve"
	@echo "Available at: http://localhost:11434"

ollama-models: ## List installed Ollama models
	@ollama list

# ============================================================================
# WHATSAPP / NGROK
# ============================================================================

tunnel: ## Open ngrok tunnel for WhatsApp webhooks
	@if ! command -v ngrok >/dev/null 2>&1; then \
		echo "Installing ngrok..."; \
		brew install ngrok; \
	fi
	@echo "Opening tunnel to localhost:8000..."
	@echo "Configure this URL in Meta Developer Console as webhook:"
	@echo "  https://<your-domain>.ngrok-free.app/webhook/whatsapp"
	@echo ""
	@ngrok http 8000

# ============================================================================
# SECURITY
# ============================================================================

keys: ## Generate JWT secret and agent Ed25519 keys
	@echo "Generating secrets..."
	@mkdir -p keys
	@chmod 700 keys
	@# JWT secret
	@JWT=$$(openssl rand -hex 32) && \
		if [ -f "$(ENV_FILE)" ]; then \
			sed -i '' "s|AGENTARMY_SECURITY_JWT_SECRET=.*|AGENTARMY_SECURITY_JWT_SECRET=$$JWT|" $(ENV_FILE); \
		fi && \
		echo "JWT secret generated."
	@# Agent keys
	@for agent in commander sentinel builder inspector watcher scout scribe devops; do \
		openssl genpkey -algorithm ed25519 -out "keys/$${agent}_private.pem" 2>/dev/null; \
		chmod 600 "keys/$${agent}_private.pem"; \
		openssl pkey -in "keys/$${agent}_private.pem" -pubout -out "keys/$${agent}_public.pem" 2>/dev/null; \
		echo "  Key pair generated: $$agent"; \
	done
	@echo "All keys saved in ./keys/"

env-file: ## Create .env from template if it doesn't exist
	@if [ ! -f "$(ENV_FILE)" ]; then \
		cp .env.local $(ENV_FILE); \
		chmod 600 $(ENV_FILE); \
		echo ".env created from .env.local — edit it with your API keys."; \
	else \
		echo ".env already exists. Skipping."; \
	fi

# ============================================================================
# QUALITY
# ============================================================================

test: ## Run test suite
	@$(PYTHON) -m pytest tests/ -v --tb=short

lint: ## Run linters
	@$(VENV)/bin/ruff check src/ tests/

format: ## Format code
	@$(VENV)/bin/black src/ tests/
	@$(VENV)/bin/isort src/ tests/

check: lint ## Run all checks (lint + type check)
	@$(VENV)/bin/mypy src/ --ignore-missing-imports

# ============================================================================
# MONITORING
# ============================================================================

status: ## Check health of all components
	@echo "=== Infrastructure ==="
	@$(COMPOSE) ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "Docker not running"
	@echo ""
	@echo "=== Ollama ==="
	@curl -s http://localhost:11434/api/tags 2>/dev/null | python3 -c \
		"import sys,json; d=json.load(sys.stdin); [print(f'  {m[\"name\"]}') for m in d.get('models',[])]" \
		2>/dev/null || echo "  Not running (start with: ollama serve)"
	@echo ""
	@echo "=== Code Horde API ==="
	@curl -s http://localhost:8000/health 2>/dev/null | python3 -m json.tool 2>/dev/null \
		|| echo "  Not running (start with: make dev)"

logs: ## Tail Docker infrastructure logs
	@$(COMPOSE) logs -f --tail=50

# ============================================================================
# UTILITIES
# ============================================================================

check-deps: ## Verify system dependencies
	@echo "Checking dependencies..."
	@command -v python3 >/dev/null 2>&1 || (echo "ERROR: python3 not found" && exit 1)
	@command -v docker  >/dev/null 2>&1 || (echo "ERROR: Docker not found. Install Docker Desktop." && exit 1)
	@command -v openssl >/dev/null 2>&1 || (echo "ERROR: openssl not found" && exit 1)
	@command -v git     >/dev/null 2>&1 || (echo "ERROR: git not found" && exit 1)
	@echo "All dependencies OK."

clean: ## Remove generated data (volumes, venv, keys)
	@echo "Cleaning up..."
	@$(COMPOSE) down -v 2>/dev/null || true
	@rm -rf $(VENV) keys/ logs/ data/ artifacts/ __pycache__ .pytest_cache
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."
