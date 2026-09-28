# AppleSupport AI agent — run `make help` for the list of targets.
#
# Knobs (pass on the command line, e.g. `make agent LIMIT=25 WORKERS=4`):
#   LIMIT    only run the first N golden cases
#   MODEL    override LLM_MODEL from .env for the agent
#   WORKERS  parallel LLM calls for the agent (default 2)
#   TWEET    the customer tweet for `make ask`

VENV    ?= .venv
PYTHON  ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
WORKERS ?= 2
LIMIT   ?=
MODEL   ?=
TWEET   ?=
export TWEET

PROC        := data/processed
PREDS       := $(PROC)/agent_preds.jsonl
JUDGED      := $(PROC)/judged.jsonl
ABL_PREDS   := $(PROC)/agent_no_retrieval_preds.jsonl
ABL_JUDGED  := $(PROC)/agent_no_retrieval_judged.jsonl

AGENT_ARGS = --workers $(WORKERS) $(if $(LIMIT),--limit $(LIMIT)) $(if $(MODEL),--model $(MODEL))

.DEFAULT_GOAL := help
.PHONY: help setup env data prepare splits golden baselines agent judge evaluate \
        run quick smoke ask web ablation offline clean

help:  ## Show this help
	@echo "AppleSupport AI agent"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36mmake %-10s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Typical first run:  make setup && make data && make quick"

# --- setup -------------------------------------------------------------------

setup:  ## Create .venv, install requirements, create .env from the example
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q --upgrade pip
	$(VENV)/bin/pip install -q -r requirements.txt
	@$(MAKE) --no-print-directory env

env:  ## Copy .env.example to .env if .env does not exist yet
	@if [ -f .env ]; then echo ".env already exists"; \
	else cp .env.example .env && echo "created .env -- fill in LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, JUDGE_MODEL"; fi

# --- data (no LLM calls) -----------------------------------------------------

data: prepare splits golden  ## Download TWCS, build splits, validate golden set (~6 min)

prepare:  ## Download TWCS and rebuild the AppleSupport cases (~4 min)
	$(PYTHON) src/prepare_data.py

splits:  ## Build corpus / dev / golden splits, assert no leakage (~2 min)
	$(PYTHON) src/make_splits.py

golden:  ## Validate the 220 hand-labelled golden cases
	$(PYTHON) src/build_golden.py

baselines:  ## B0 trivial + B1 TF-IDF/LogReg baselines (~2 min)
	$(PYTHON) src/baselines.py

# --- agent + evaluation ------------------------------------------------------

agent:  ## Run the agent on the golden set (resumable; LIMIT=, MODEL=, WORKERS=)
	$(PYTHON) src/agent.py $(AGENT_ARGS)

judge:  ## Score every agent reply with the LLM judge
	$(PYTHON) src/judge.py

evaluate:  ## Print the results table -> reports/results.json
	$(PYTHON) src/evaluate.py $(if $(wildcard $(ABL_JUDGED)),--ablation $(ABL_JUDGED))

run: baselines agent judge evaluate  ## Full headline run: baselines -> agent -> judge -> evaluate

quick:  ## End-to-end on 25 cases in ~2 minutes
	@$(MAKE) --no-print-directory run LIMIT=25

ablation:  ## Agent without retrieval, judged, for the ablation row in evaluate
	$(PYTHON) src/agent.py $(AGENT_ARGS) --no-retrieval --out $(ABL_PREDS)
	$(PYTHON) src/judge.py --preds $(ABL_PREDS) --out $(ABL_JUDGED)
	@$(MAKE) --no-print-directory evaluate

offline:  ## Re-evaluate from the LLM cache only, no API calls
	LLM_CACHE_ONLY=1 $(PYTHON) src/agent.py $(AGENT_ARGS)
	LLM_CACHE_ONLY=1 $(PYTHON) src/judge.py
	$(PYTHON) src/evaluate.py $(if $(wildcard $(ABL_JUDGED)),--ablation $(ABL_JUDGED))

# --- try it ------------------------------------------------------------------

ask:  ## Run the agent on one tweet: make ask TWEET="my iphone won't charge"
	@if [ -z "$$TWEET" ]; then echo 'usage: make ask TWEET="my iphone won'"'"'t charge"'; exit 1; fi
	$(PYTHON) src/agent.py --tweet "$$TWEET" $(if $(MODEL),--model $(MODEL))

web:  ## Start the demo dashboard at http://localhost:5001
	$(PYTHON) web/server.py

smoke:  ## Wiring test with a stubbed LLM -- no API key, no cost
	$(PYTHON) src/smoke_test.py

clean:  ## Remove predictions, judgments, and caches of compiled Python (keeps data + LLM cache)
	rm -f $(PREDS) $(JUDGED) $(ABL_PREDS) $(ABL_JUDGED) $(PROC)/_smoke_*.jsonl
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
