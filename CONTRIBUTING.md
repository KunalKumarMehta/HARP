# Contributing to HARP

Thanks for your interest in HARP — a hardware-aware agentic assistant whose engine
is a calibrated edge↔cloud escalation gate that works offline. Contributions of all
kinds are welcome: bug reports, docs, tests, new backends, and routing improvements.

## Ground rule: the contract is frozen

`shared/harp_contract.py` is the v0 integration contract — the `Backend` ABC,
`InferRequest`/`Metrics`, the `PlanGraph` DAG, and the capability-negotiated
`Router`. Everything else depends on it; it depends on nothing. Changes to this
file are deliberately rare and ripple everywhere, so open an issue to discuss
before proposing one. New functionality should almost always be a new
implementation *behind* the contract, not a change *to* it.

## Development setup

```bash
git clone https://github.com/KunalKumarMehta/HARP
cd HARP
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # core runs on stdlib alone; extras add the strict CI paths
```

The runtime needs no third-party packages to import or to run the demo — the plan
codec and the fabric both fall back to the standard library. The `dev`/`full`
extras add `jsonschema`, `websockets`, and `httpx` for the stricter checks CI runs.

## Run it

```bash
python -m demo.run_demo               # cloud plan -> wire -> executor -> routed edge/cloud
python -m demo.run_demo --offline     # network drop -> everything fails closed to edge
python -m demo.run_demo --distributed # run the edge tier on a separate fabric node
```

## Run the gates before you push

These are the core contract gates (also `make check`); CI runs the full 15
(`.github/workflows/ci.yml`). Run them locally first — all must pass:

```bash
export PYTHONPATH="$PWD"
for g in \
  "shared.conformance" "fabric.sync_queue" "tests.e2e_smoke" \
  "tests.ws_roundtrip" "shared.plan_codec" "tests.executor_smoke" \
  "edge.genie_backend" "fabric.remote_backend"; do
  python -m "$g" || exit 1
done
python -c "import asyncio; from shared.harp_contract import _smoke; asyncio.run(_smoke())"
```

Any new `Backend` implementation must pass `shared/conformance.py`
(`assert_conforms`); wire it into the conformance set so the gate covers it.

## Style

- Target Python 3.11+. Keep the core import-light: no new hard dependency without
  a stdlib fallback or an entry in `[project.optional-dependencies]`.
- Match the surrounding code — naming, docstring density, and idiom.
- `ruff` is configured in `pyproject.toml`; run `ruff check .` before pushing.
- Tests and contract gates live in `tests/` and the `_smoke`/`__main__` blocks of
  the modules they cover. Add coverage for anything you change.

## Pull requests

1. Branch from `main`.
2. Keep PRs focused; describe what changed and why.
3. Make sure every gate above is green.
4. Reference any related issue.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
