# NoMorals Core — developer entry points.
# No mandatory dependencies: every target runs on a bare Python 3.11 install.

PYTHON ?= python3
PKG    := nomorals
NM     := $(PYTHON) -m $(PKG)

.DEFAULT_GOAL := help
.PHONY: help test test-verbose test-one lint gates fmt typecheck doctor tools config \
        ask run serve backup models memory clean lines

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── verification ──────────────────────────────────────────────────────────────

test: ## Run the full suite under pytest (offline; the canonical runner — executes plain functions AND unittest classes)
	@$(PYTHON) -u -m pytest tests

test-verbose: ## Run the full suite, naming each test
	@$(PYTHON) -u -m pytest tests -v 2>&1 | tail -40

test-one: ## Run one module: make test-one M=tests/test_agents.py
	@$(PYTHON) -u -m pytest $(M) -v

lint: ## Ruff check (absent tool = honest skip; installed tool's findings FAIL)
	@if ! $(PYTHON) -c "import ruff" 2>/dev/null; then \
		echo "ruff not installed — pip install ruff (optional); lint skipped"; \
	else \
		$(PYTHON) -m ruff check $(PKG) tests; \
	fi

gates: ## The gates CI enforces: ruff ratchet + committed-secret scan + personal-surface laws (docs/CI_GATES.md)
	@$(PYTHON) -m nomorals.tools.ruff_ratchet
	@$(PYTHON) -m nomorals.tools.secret_scan
	@$(PYTHON) -m unittest tests.test_surface_laws

typecheck: ## mypy check (absent tool = honest skip; installed tool's findings FAIL)
	@if ! command -v mypy >/dev/null 2>&1; then \
		echo "mypy not installed — pip install mypy (optional); typecheck skipped"; \
	else \
		mypy $(PKG); \
	fi

fmt: ## Format with ruff (absent tool = honest skip; a format error FAILS)
	@if ! $(PYTHON) -c "import ruff" 2>/dev/null; then \
		echo "ruff not installed — pip install ruff (optional); fmt skipped"; \
	else \
		$(PYTHON) -m ruff format $(PKG) tests; \
	fi

# ── inspection ────────────────────────────────────────────────────────────────

doctor: ## Environment capabilities, database health, optional packages
	@$(NM) doctor

tools: ## List every registered tool and the capability it requires
	@$(NM) tools

config: ## Print the effective configuration
	@$(NM) config

models: ## Model registry state and the curated catalog
	@$(NM) models --catalog

lines: ## Honest code count: files and non-blank non-comment lines
	@$(PYTHON) scripts/count_lines.py

# ── operation ─────────────────────────────────────────────────────────────────

ask: ## Single-turn chat: make ask P="explain CRDTs"
	@$(NM) ask "$(P)"

run: ## Run a goal through the orchestrator: make run G="research X"
	@$(NM) run "$(G)"

serve: ## Start the HTTP API (set NM_API_TOKEN)
	@$(NM) serve

backup: ## Create and rotate a versioned backup
	@$(NM) backup --create

memory: ## Memory stats: make memory Q="search terms" to recall instead
	@if [ -n "$(Q)" ]; then $(NM) memory --query "$(Q)"; else $(NM) memory --stats; fi

clean: ## Remove caches and build artifacts (never touches data/ or backups/)
	@find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	@echo "cleaned (data/, backups/, models/ untouched)"
