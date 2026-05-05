# Trophic — build / install / run helpers.
#
# Quick start on a new machine:
#
#   git clone https://github.com/dgonier/consciousness-ecology-codd.git
#   cd consciousness-ecology-codd
#   make install     # sets up .venv with uv + installs all deps + voter SDKs
#   make health      # pings every available apex voter
#   make eval-smoke  # runs a 5-scenario apex panel eval (cheap)
#
# Detect uv if present, fall back to python venv + pip.
PYTHON ?= python3.11
VENV ?= .venv
PY = $(VENV)/bin/python

.DEFAULT_GOAL := help

.PHONY: help
help:                    ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Setup

.PHONY: venv
venv:                    ## Create the virtualenv at .venv (uses uv if available)
	@if [ -d $(VENV) ]; then \
		echo "[make] $(VENV)/ already exists; rerun 'make clean-venv' to recreate"; \
	elif command -v uv >/dev/null 2>&1; then \
		echo "[make] uv detected — using it"; \
		uv venv --python $(PYTHON) $(VENV); \
	else \
		echo "[make] no uv; falling back to $(PYTHON) -m venv"; \
		$(PYTHON) -m venv $(VENV); \
	fi

.PHONY: install
install: venv            ## Install package + voter SDKs (idempotent)
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install --python $(PY) -e ".[voters,dev]"; \
	else \
		$(PY) -m pip install --upgrade pip; \
		$(PY) -m pip install -e ".[voters,dev]"; \
	fi
	@echo "[make] install done"

.PHONY: install-core
install-core: venv       ## Install core dependencies only (no API voter SDKs)
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install --python $(PY) -e ".[dev]"; \
	else \
		$(PY) -m pip install --upgrade pip; \
		$(PY) -m pip install -e ".[dev]"; \
	fi

.PHONY: clean-venv
clean-venv:              ## Wipe and recreate the virtualenv
	rm -rf $(VENV)
	$(MAKE) venv

##@ Voter health / quick eval

.PHONY: health
health:                  ## Ping each available apex voter (5 voters max)
	$(PY) -u scripts/diagnostics/voter_health_check.py

.PHONY: eval-smoke
eval-smoke:              ## 5-scenario apex panel eval (decomposer ON)
	$(PY) -u scripts/diagnostics/apex_vote_eval.py \
		--n-per-ticker 1 --tickers AAPL,GOOG,MSFT,AMZN,JPM \
		--max-scenarios 5 --decomposer

.PHONY: eval-20
eval-20:                 ## 20-scenario apex panel eval
	$(PY) -u scripts/diagnostics/apex_vote_eval.py \
		--n-per-ticker 4 --tickers AAPL,GOOG,MSFT,AMZN,JPM \
		--max-scenarios 20 --decomposer

.PHONY: eval-100
eval-100:                ## Full 100-scenario apex panel eval (~50 min wall)
	$(PY) -u scripts/diagnostics/apex_vote_eval.py \
		--n-per-ticker 20 --tickers AAPL,GOOG,MSFT,AMZN,JPM \
		--decomposer

.PHONY: bare-baseline
bare-baseline:           ## Bare Qwen + benchmark XML prompt (50 scenarios)
	$(PY) -u scripts/diagnostics/baseline_promptonly_stocknet.py

##@ Trophic-stack training (GPU)

.PHONY: train-seed
train-seed:              ## Train a stable-infra ckpt; override SEED=N LR=2e-5
	TROPHIC_SEED=$(or $(SEED),36) \
	TROPHIC_BINARY_HEAD=1 \
	TROPHIC_LR=$(or $(LR),2e-5) \
	TROPHIC_GRAD_CLIP=$(or $(GRAD_CLIP),0.25) \
		$(PY) -u scripts/train_sft_stocknet_stable.py

##@ Diagnostics

.PHONY: inspect
inspect:                 ## Capture per-tier signals JSONL for one scenario; CKPT=path SCEN=name
	$(PY) -u scripts/diagnostics/inspect_signals_jsonl.py \
		--ckpt $(or $(CKPT),checkpoints/sft_seed36_stable_best.pt) \
		--scenario $(or $(SCEN),stocknet_test_AAPL_2015-10-01)

.PHONY: viz
viz:                     ## Start the react-flow signal-flow viz on :5173
	cd viz && npm install && npm run dev

##@ Tests + housekeeping

.PHONY: test
test:                    ## Run pytest
	$(PY) -m pytest tests/ -x

.PHONY: lint-bigfiles
lint-bigfiles:           ## Print any tracked file >1MB (should be empty)
	@git ls-files | xargs -I{} sh -c 'sz=$$(stat -c "%s" "{}" 2>/dev/null); if [ "$$sz" -gt 1048576 ]; then echo "$${sz} {}"; fi' | sort -rn || true

.PHONY: clean
clean:                   ## Remove build artifacts + __pycache__ (keeps .venv)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info trophic/*.egg-info

##@ Environment file

# Source the env file at /home/dgonier/debaterhub/debaterhub-monorepo/apps/front/.env
# Override via ENVFILE=path/to/.env make eval-20
ENVFILE ?= /home/dgonier/debaterhub/debaterhub-monorepo/apps/front/.env

.PHONY: print-env
print-env:               ## Print which API-key env vars are loaded (values redacted)
	@if [ -f "$(ENVFILE)" ]; then \
		set -a; . "$(ENVFILE)" 2>/dev/null; set +a; \
		env | grep -E "^(OPENAI_API_KEY|AWS_|GEMINI_API_KEY|GOOGLE_API_KEY|OPENROUTER_API_KEY|BEDROCK_MODEL_ID|GEMINI_MODEL|OPENAI_MODEL)" | sed 's/=.*/=***/' | sort; \
	else \
		echo "no env file at $(ENVFILE) — set ENVFILE=... or skip"; \
	fi
