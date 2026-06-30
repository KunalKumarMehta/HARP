# HARP

**A hardware-aware agentic assistant that runs on your device and escalates to the
cloud only when a task genuinely needs it.** Its engine is a calibrated edge↔cloud
gate that keeps working offline — running what it can on the Snapdragon NPU and
escalating only what it must to a cloud planner.

MIT-licensed and public from commit 1.

## How it works

The product is a device-first assistant. The **engine** — the part that makes it a
product and not a wrapper — is a **calibrated edge↔cloud escalation gate**
([`router/router_policy.py`](router/router_policy.py)): it decides, per task, *what
runs where*, with a conformal bound on dangerous mis-routes and a hardware/offline
guard that always has the final say. Everything else exists to serve that decision.

| Layer | Stack | What it proves |
|---|---|---|
| **Engine (the IP)** | calibrated complexity gate + capability negotiation | the decision *what runs where* — the defensible core |
| **Edge (execution)** | Qualcomm AI Hub · Genie / QNN · Hexagon NPU | on-device, low-latency, energy-efficient inference |
| **Cloud (planning)** | NeMo Agent Toolkit · NIM · Nemotron | multi-agent planning, heavy reasoning, measured speedups |

## Architecture

```
  cloud planner (NeMo ReWOO)                         edge device (Snapdragon)
  ──────────────────────────                         ────────────────────────
  plan_emitter ─► PlanGraph ─► plan_codec.to_json ──► from_json ─► PlanExecutor
                                  (validated JSON wire)               │
                                                                      ▼
                                                            PolicyRouter._select
                                                       (calibrated AUTO · pins ·
                                                        offline/capability guard)
                                                          │              │
                                                      edge backend   cloud backend
                                                   GenieBackend /     NIMBackend
                                                   QNNBackend         (Nemotron)
```

- **`shared/harp_contract.py`** — the frozen v0 contract: `Backend` ABC
  (`capabilities/infer/profile`), `PlanGraph` DAG, `RouteDecision{LOCAL,ESCALATE,AUTO}`.
  Everything depends on it; it depends on nothing.
- **`router/router_policy.py`** — the engine: an encoder hardness score → isotonic
  calibration → **conformal gate** that bounds silent under-routing at α. Resolves
  `AUTO` steps, honors planner pins, never fights the hardware guard.
- **`shared/plan_codec.py`** — the single cloud↔edge serialization boundary
  (JSON-schema shape + in-code DAG semantics), with a dependency-free fallback so the
  edge ships clean.
- **`fabric/executor.py`** — walks the DAG, dispatches each step through the router,
  threads `<step>_output` dataflow between dependent steps.
- **`fabric/sync_queue.py` + `ws_node.py`** — four-state offline mutation queue +
  WebSocket transport with drop→reconnect→idempotent (client-UUID-keyed) redelivery.

## Quickstart (zero setup — runs on mocks)

```bash
make demo               # cloud plan → wire → executor → routed across edge/cloud
make demo-offline       # network drop → everything fails closed to edge
make demo-distributed   # run the edge tier on a separate fabric node (phone)
```

You'll see a 4-step plan (transcribe · screen-scan · summarize · decide) routed
live: audio/text resolve on edge, vision and the deep-reason step escalate to cloud,
and each step's output threads into the next. No third-party deps required (the codec
and fabric fall back to stdlib).

Run the gates the way CI does: `make check` runs the core contract gates. CI runs the
full **15 gates + demo-integration** — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Backends — one contract, swap freely

| Backend | Loads | When to use |
|---|---|---|
| **`edge/genie_backend.py`** | a precompiled Genie context-binary bundle via `genie-t2t-run` | the **fast path** — any model already in the AI-Hub catalog |
| **`edge/qnn_backend.py`** | a self-compiled onnxruntime-genai model dir (QNN EP) | a model you compiled yourself (not in the catalog) |
| **`cloud/nim_backend.py`** | an OpenAI-compatible NIM endpoint (Nemotron) | the cloud planner / escalate tier |

The router dispatches by negotiating `capabilities()` — it never imports a concrete
backend, so edge↔cloud is a one-call swap and a new accelerator stubs in behind the
same interface.

### Multi-device

`fabric/remote_backend.py` adds a **`RemoteBackend`** — a `Backend` whose `infer()`
proxies over the WebSocket fabric to a peer node. Because it satisfies the same
contract, the laptop's router dispatches a step to the **phone** exactly as it would
a local backend: a plan step assigned to the edge tier executes on the phone's NPU
and streams its tokens back — and the offline guard fails it closed when the link
drops. `make demo-distributed` demonstrates this over a loopback node today.

### Serve it as a model

`python -m serve.openai_endpoint` exposes HARP as an OpenAI-compatible local model
with an advisory `/v1/route`. The [Hermes integration](integrations/hermes/README.md)
points an agent at it so every turn is routed automatically — NPU lane for what fits
on-device, cloud lane for what doesn't.

## On-device (Snapdragon X Elite / Qualcomm Device Cloud)

The precompiled path skips compilation entirely. One command provisions a barebone
QDC X Elite session (Windows ARM64) and runs the on-device gate:

```bat
edge\bootstrap_qdc.cmd     :: installs Python + deps, wires genie-t2t-run, runs Risk-A
python run_test.py         :: power + latency over build\qwen3-4b-w4a16 → evidence pack
```

It samples the NPU power rail *during* decode and emits TTFT / decode tok-s /
energy-per-token. Off-device it runs a conformant stub and **fails the gate by
design** (no silicon, no pass). Full walkthrough: [docs/USER_MANUAL.md](docs/USER_MANUAL.md).

## Repo layout

```
shared/        frozen contract, plan codec, conformance, JSON schema
router/        the engine — calibrated edge↔cloud escalation gate (the routing IP)
edge/          Genie + QNN backends, bench harness, power telemetry, AI-Hub compile
cloud/         NIM backend, ReWOO plan emitter, NAT middleware, mutation dedup
fabric/        offline queue, WebSocket transport, executor, remote backend (multi-device)
serve/         OpenAI-compatible endpoint + advisory /v1/route
integrations/  Hermes provider plugin + pre_llm_call hook, agentskills.io skill
demo/          run_demo.py — the whole spine in one command
tests/         the 15 CI gates
build/         precompiled Genie bundles (git-ignored; large binaries)
```

## Documentation

| Doc | Covers |
|---|---|
| [docs/PRD.md](docs/PRD.md) | problem, product, goals, requirements, success metrics |
| [docs/SDD.md](docs/SDD.md) | architecture, components, concurrency, requirement→CI-gate map |
| [docs/ADR.md](docs/ADR.md) | the 20 locked architectural decisions + rationale |
| [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) | SQLite outbox schema + plan-graph / fabric wire contracts |
| [docs/USER_MANUAL.md](docs/USER_MANUAL.md) | run locally, on QDC, multi-device, cloud; troubleshooting |

## License

MIT — see [LICENSE](LICENSE).
