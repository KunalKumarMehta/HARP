# HARP — one-command entry points.
# CI (.github/workflows/ci.yml) runs the full 16-gate matrix; `make check` is the
# fast local contract-gate subset (gates 1-9). ponytail: gate list mirrors ci.yml;
# CI stays the source of truth.
PYTHON ?= python
export PYTHONPATH := $(CURDIR)

.PHONY: demo demo-offline demo-distributed serve check help

help:              ## show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## / — /'

demo:              ## run the whole spine on mocks (plan -> wire -> executor -> routed)
	$(PYTHON) -m demo.run_demo

demo-offline:      ## network drop -> everything fails closed to edge
	$(PYTHON) -m demo.run_demo --offline

demo-distributed:  ## run the edge tier on a separate fabric node (loopback)
	$(PYTHON) -m demo.run_demo --distributed

serve:             ## serve HARP as an OpenAI-compatible local model on :8765
	$(PYTHON) -m serve.openai_endpoint

check:             ## run the core contract gates locally (CI runs all 16)
	$(PYTHON) -c "import asyncio; from shared.harp_contract import _smoke; asyncio.run(_smoke())"
	$(PYTHON) -m shared.conformance
	$(PYTHON) -m fabric.sync_queue
	$(PYTHON) tests/e2e_smoke.py
	$(PYTHON) tests/ws_roundtrip.py
	$(PYTHON) -m shared.plan_codec
	$(PYTHON) tests/executor_smoke.py
	$(PYTHON) -m edge.genie_backend
	$(PYTHON) -m fabric.remote_backend
	@echo "OK - core contract gates passed (CI runs the full 16 + demo-integration)"
