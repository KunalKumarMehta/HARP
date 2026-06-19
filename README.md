# HARP — Hardware-Aware Routing Platform

**A hardware-agnostic agentic runtime that routes each task to the right model on
the right tier** — a lightweight edge gatekeeper that runs what it can on the
Snapdragon NPU and escalates only what it must to a cloud planner, degrading
gracefully to fully-offline when disconnected.

One repo, one architecture, three expressions:

| Plane | Stack | What it proves |
|---|---|---|
| **Edge (execution)** | Qualcomm AI Hub · Genie / QNN · Hexagon NPU | on-device, low-latency, energy-efficient inference |
| **Cloud (planning)** | NeMo Agent Toolkit · NIM · Nemotron | multi-agent planning, heavy reasoning, measured speedups |
| **Routing (the IP)** | calibrated complexity gate + capability negotiation | the decision *what runs where* — the defensible core |

MIT-licensed and public from commit 1.

---

## Quickstart (zero setup — runs on mocks)

```bash
python -m demo.run_demo              # cloud plan → wire → executor → routed across edge/cloud
python -m demo.run_demo --offline    # network drop → everything fails closed to edge
python -m demo.run_demo --distributed # run the edge tier on a separate fabric node (phone)
```

You'll see a 4-step plan (transcribe · screen-scan · summarize · decide) routed
live: audio/text resolve on edge, vision and the deep-reason step escalate to
cloud, and the dataflow from each step threads into the next.

### Run the full contract gate (9 checks, what CI runs)

```bash
for g in \
  "shared.conformance" "fabric.sync_queue" "tests.e2e_smoke" \
  "tests.ws_roundtrip" "shared.plan_codec" "tests.executor_smoke" \
  "edge.genie_backend" "fabric.remote_backend"; do
  python -m $g || exit 1
done
python -c "import asyncio; from shared.harp_contract import _smoke; asyncio.run(_smoke())"
```

No third-party deps required (the codec and fabric fall back to stdlib). CI adds
`websockets` + `jsonschema` for extra rigor.

---

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
  Everything else depends on this; it depends on nothing.
- **`shared/plan_codec.py`** — the single cloud↔edge serialization boundary.
  Two-layer validation: JSON-schema shape + in-code DAG semantics (unique ids,
  referential integrity, acyclicity). Dependency-free fallback so the edge ships clean.
- **`router/router_policy.py`** — `PolicyRouter`: an encoder hardness score →
  isotonic calibration → **conformal gate** that bounds silent under-routing at α.
  Resolves `AUTO` steps; honors planner pins; never fights the hardware guard.
- **`fabric/executor.py`** — walks `topo_order()`, dispatches each step through the
  router, threads `<step>_output` dataflow between dependent steps. The end-to-end loop.
- **`fabric/sync_queue.py` + `ws_node.py`** — four-state offline mutation queue
  (pending/in_flight/success/conflict) + WebSocket transport with drop→reconnect→
  idempotent (client-UUID-keyed) redelivery.

## Backends — one contract, swap freely

| Backend | Loads | When to use |
|---|---|---|
| **`edge/genie_backend.py`** | a **precompiled Genie context-binary bundle** (e.g. `build/qwen3-4b-w4a16/` from [`qualcomm/ai-hub-models`](https://github.com/qualcomm/ai-hub-models/tree/v0.56.0)) via `genie-t2t-run` | the **fast path** — no self-compilation, for any model already in the AI-Hub catalog |
| **`edge/qnn_backend.py`** | a self-compiled onnxruntime-genai model dir (QNN EP) | a model you compiled yourself (not in the catalog) |
| **`cloud/nim_backend.py`** | an OpenAI-compatible NIM endpoint (Nemotron) | the cloud planner / escalate tier |

The router dispatches by negotiating `capabilities()` — it never imports a
concrete backend, so edge↔cloud is a one-call swap and Apple/AMD stub in behind
the same interface.

### Multi-device

`fabric/remote_backend.py` adds a **`RemoteBackend`** — a `Backend` whose `infer()`
proxies over the WebSocket fabric to a peer node. Because it satisfies the same
contract, the laptop's router dispatches a step to the **phone** exactly as it
would to a local backend: `serve_backend(GenieBackend(...))` runs on the phone,
`PolicyRouter(edge=RemoteBackend("ws://phone:8770"), cloud=NIM)` runs on the
laptop. A plan step assigned to the edge tier executes on the phone's NPU and
streams its tokens back — orchestration that's *impossible on a single device* —
and the offline guard fails it closed when the link drops. `--distributed`
demonstrates this over a loopback node today.

---

## On-device (Snapdragon X Elite / Qualcomm Device Cloud)

The precompiled path skips compilation entirely. A fresh QDC X Elite session is
barebone (no Python, no deps, no `genie-t2t-run`) — one command provisions it all:

```bat
REM on the QDC Snapdragon X Elite (Windows ARM64), from the repo root:
edge\bootstrap_qdc.cmd             :: installs Python + deps, finds/wires genie-t2t-run,
                                   :: then runs the Risk-A gate over build\qwen3-4b-w4a16
```

`bootstrap_qdc.cmd` is idempotent (safe to re-run) and detect-first. Once provisioned:

```bat
python run_test.py                 :: integrated Risk-A gate: power+latency over the bundle
python run_test.py --target csv    :: CSV power fallback (Free HWiNFO)
```

A precompiled Genie bundle already lives in `build/qwen3-4b-w4a16/` (Qwen3-4B
w4a16, QAIRT 2.45, Hexagon v73); drop more bundles from `ai-hub-models` into
`build/` and `genie_swarm()` discovers them automatically — no code change.

`run_test.py` → `edge/bench_genie.py` loads the bundle through `GenieBackend`,
samples the NPU power rail *during* decode, and emits `evidence_pack_genie.md`
(TTFT / decode tok/s / energy-per-token) — the on-device efficiency evidence.
Off-device it runs a conformant stub and **fails the gate by design** (no silicon,
no pass). Self-compiling a non-catalog model instead? Use `edge/compile_qwen3.py`
(LLM) or `edge/compile_spike.py` (encoder/ASR), then `edge/bench.py`.

---

## Repo layout

```
shared/   frozen contract, plan codec, conformance, mutation key
router/   calibrated PolicyRouter (the routing IP)
edge/     Genie + QNN backends, bench harness, power telemetry, AI-Hub compile scripts
cloud/    NIM backend, ReWOO plan emitter, NAT middleware, mutation dedup
fabric/   offline queue, WebSocket transport, end-to-end executor, remote backend (multi-device)
demo/     run_demo.py — the whole spine in one command
tests/    e2e / ws-roundtrip / executor smoke (CI gates)
build/    precompiled Genie bundles (git-ignored; large binaries)
```

## Documentation

| Doc | Covers |
|---|---|
| [documentation/PRD.md](documentation/PRD.md) | problem, goals, requirements, success metrics, open decisions |
| [documentation/SDD.md](documentation/SDD.md) | architecture, components, concurrency, requirement→CI-gate test map |
| [documentation/ADR.md](documentation/ADR.md) | the 12 locked architectural decisions + rationale |
| [documentation/DATA_SCHEMA.md](documentation/DATA_SCHEMA.md) | SQLite outbox schema + plan-graph / fabric wire contracts |
| [documentation/USER_MANUAL.md](documentation/USER_MANUAL.md) | run locally, on QDC, multi-device, cloud; troubleshooting |

## License

MIT — see [LICENSE](LICENSE).
