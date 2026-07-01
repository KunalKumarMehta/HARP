# HARP — Software Design Document (SDD / TDD)

**Status:** living · **Owner:** Maintainers · **Last updated:** 2026-07-01
Pairs with [PRD.md](PRD.md), [ADR.md](ADR.md), [DATA_SCHEMA.md](DATA_SCHEMA.md).

## 1. Architecture overview

Three planes over one frozen contract:

```
  CONTROL (cloud)            WIRE              EXECUTION (edge device)
  ─────────────────   ───────────────────   ───────────────────────────────
  NeMo ReWOO planner   PlanGraph JSON         from_json → PlanExecutor
  → plan_emitter   →   (schema + DAG  )   →   → PolicyRouter._select
  → NIMBackend         validated, ~0.7 KB     → backend.infer (stream)
       │                                          │            │
   Nemotron                                   edge backend   cloud backend
   (NIM)                                      Genie/QNN/     NIMBackend
                                              Remote          (escalate)
```

The **contract** (`shared/harp_contract.py`) is frozen: it depends on nothing and
everything depends on it. CI gate 1 + conformance (gate 2) protect it.

## 2. Components

| Module | Responsibility | Key types |
|---|---|---|
| `shared/harp_contract.py` | the freeze | `Backend`, `Router`, `PlanGraph/PlanStep`, `Capability`, `InferRequest`, `Metrics`, enums `Tier/Modality/RouteDecision/SyncState` |
| `shared/plan_codec.py` | cloud↔edge serialization | `to_json/from_json`, two-layer validation, `PlanWireError` |
| `shared/conformance.py` | runtime ABC enforcement | `assert_conforms()` |
| `router/router_policy.py` | the routing IP | `PolicyRouter`, `RoutingPolicy`, `IsotonicCalibrator`, `ConformalGate` |
| `fabric/executor.py` | end-to-end plan execution | `PlanExecutor`, `StepResult`, `ExecutionResult` |
| `fabric/sync_queue.py` | offline four-state outbox | `OutboxQueue`, `Mutation` |
| `fabric/ws_node.py` | mutation-sync transport (laptop dual-role) | `FabricNode` |
| `fabric/remote_backend.py` | **multi-device**: a Backend over WS | `RemoteBackend`, `serve_backend()` |
| `edge/genie_backend.py` | precompiled Genie bundle backend (fast path) | `GenieBackend`, `genie_qwen3_4b()`, `genie_swarm()` |
| `edge/qnn_backend.py` | self-compiled onnxruntime-genai backend | `QNNBackend` |
| `edge/bench*.py`, `power.py` | Risk-A gate + energy/latency evidence | `run_gate`, `ProfilableBackend` |
| `cloud/*` | NIM backend, ReWOO emitter, NAT middleware, dedup | `NIMBackend`, `emit_plan_graph`, `emit()` |
| `serve/openai_endpoint.py` | OpenAI-compatible endpoint + advisory `/v1/route` | `make_app()`, `_resolve_route()`, NPU single-flight |
| `integrations/hermes/*` | Hermes model-provider plugin + `pre_llm_call` routing hook | `build_harp_provider_profile()`, `decide_override()` |
| `demo/run_demo.py` | the whole spine in one command | — |
| `demo/track_a_routing_demo.py` | per-turn `/v1/route` demo + trace artifact | `run()` |

## 3. The contract (interfaces)

```python
class Backend(ABC):
    async def capabilities(self) -> Capability        # negotiated, never imported
    def infer(self, req: InferRequest) -> AsyncIterator[str]   # streaming mandatory
    async def profile(self, req: InferRequest) -> Metrics
```
`RouteDecision = {LOCAL, ESCALATE, AUTO="undecided"}`. The planner pins LOCAL/
ESCALATE only when certain; AUTO means "router, you decide".

## 4. Routing design

`PolicyRouter._select(step)`:
1. If `decision == AUTO`: build `RoutingFeatures` from the query + `capabilities()`,
   run `RoutingPolicy.decide` → resolve to LOCAL/ESCALATE.
2. Defer to base `Router._select` for the **hardware/offline guard** (offline →
   edge; no NPU or unsupported modality → cloud; ESCALATE → cloud). The guard has
   final say; the policy never overrides it.

`RoutingPolicy` (the IP): encoder hardness score `u(x)=P(escalate)` → isotonic
calibration → **conformal gate** (`δ` = (1−α) quantile of `u` over calibration
queries the edge got right). Escalate iff `u>δ`. Guarantee: `Pr[edge wrong ∧
kept-local] ≤ α`. A regex floor short-circuits trivial turns; thermal/battery bias
work off a hot/dying NPU.

## 5. Wire protocol

`PlanGraph → to_json → {plan_id, steps:[{step_id, modality, decision, model_id,
prompt, depends_on}]}`. Validation is two-layer because JSON-Schema can't express a
DAG: **shape** via `plan_schema.json` (jsonschema in CI, stdlib fallback at the
edge) + **semantics** in code (unique ids, referential integrity, acyclicity via
`topo_order`). See [DATA_SCHEMA.md](DATA_SCHEMA.md).

## 6. Execution & dataflow

`PlanExecutor.execute(plan)`:
- Walk `plan.topo_order()` (raises on cycle → never half-runs a bad DAG).
- `_resolve_prompt`: single-pass, word-boundary regex substitution of
  `<dep>_output` refs with upstream outputs (no cascade corruption, no prefix
  clobber); literal-instruction steps get upstream context appended.
- Resolve the backend once via `router._select` (records tier), then stream.
- **Failure isolation:** a failed step taints its downstream cone — dependents are
  *skipped*, never run on empty/garbage upstream context.

## 7. Backend matrix

| Backend | Loads | Runtime | Notes |
|---|---|---|---|
| `GenieBackend` | precompiled Genie bundle (ctx-bins + `genie_config.json`) | `genie-t2t-run` subprocess | **fast path**; `genie_swarm()` auto-discovers `build/*`; `HARP_GENIE_BIN` override |
| `QNNBackend` | self-compiled ONNX model dir | onnxruntime-genai QNN EP | for non-catalog models |
| `NIMBackend` | OpenAI-compatible NIM endpoint | httpx SSE | cloud planner/escalate |
| `RemoteBackend` | a peer node's Backend | WebSocket RPC | **multi-device**; offline guard fails it closed |

## 8. Concurrency model

- **Async everywhere** at the contract boundary (`infer` is an async generator).
- `GenieBackend.infer` bridges a blocking `genie-t2t-run` subprocess on a worker
  thread to an `asyncio.Queue`; stderr is drained concurrently (no PIPE deadlock);
  the child is terminated if the consumer stops early (no zombie); a stream that
  ends without a `done`/`error` frame raises (no silent partial success).
- `OutboxQueue` is **single-writer** — confined to one worker thread (SQLite
  affinity), `BEGIN IMMEDIATE` atomic dual-write, WAL + tuned PRAGMAs.

## 9. Test strategy (TDD) — requirement → gate map

Every contract invariant is a runnable CI gate (`.github/workflows/ci.yml`).

| Gate | Command | Covers |
|---|---|---|
| 1 | `shared.harp_contract._smoke` | FR1/FR2 swap, offline fail-closed, metrics |
| 2 | `shared.conformance` | FR1 ABC conformance (mocks + any real backend) |
| 3 | `fabric.sync_queue` | FR8 four-state FSM, crash recovery, conflict |
| 4 | `tests.e2e_smoke` | FR2 AUTO calibration + pins + fail-closed |
| 5 | `tests.ws_roundtrip` | FR8 drop→reconnect→idempotent redelivery |
| 6 | `shared.plan_codec` | FR3 schema + DAG validation, round-trip, 6 rejections |
| 7 | `tests.executor_smoke` | FR4 dataflow threading, failure-skip, boundary-safety, cycle reject |
| 8 | `edge.genie_backend` | FR5 Genie conformance + swarm discovery (off-device stub) |
| 9 | `fabric.remote_backend` | FR7 multi-device over real socket + truncation-raises |
| 10 | `tests.test_endpoint_contract` + `tests.test_npu_single_flight` + `tests.test_tools_thinking_off` | serve schema/stream/tool_calls, NPU single-flight + overflow-shed, tools→no-CoT |
| 11 | `tests.test_router_contention` | contention shed when NPU busy; never-shed offline |
| 12 | `tests.test_hermes_provider` | Hermes ProviderProfile factory keys/values, aux-lane pin |
| 13 | `tests.test_route_endpoint` + `tests.test_pre_llm_call_hook` | advisory `/v1/route`; hook fail-safe + `**kwargs` |
| 14 | `tests.test_skill_packaging` | agentskills.io SKILL.md frontmatter, `hardware_probe` graceful |
| 15 | `tests.test_track_a_demo` | 6-turn routing table + `trace.jsonl` local/escalate split |

Backends that can't run off-device (Genie/QNN on NPU, NIM live) ship a conformant
stub/mock so CI is green everywhere; the **gate honestly FAILS** off-device when it
would assert silicon (no false PASS). Hardware-only variables (real tok-s/energy)
are measured on the QDC X Elite via `run_test.py`.

## 10. Deployment

- **Edge (QDC X Elite, Windows ARM64):** `edge/bootstrap_qdc.cmd` provisions
  Python + deps + `genie-t2t-run`, then runs Risk-A. ARM64-clean deps only.
- **Cloud:** `HARP_NIM_API_KEY` → live Nemotron NIM; ReWOO planner emits PlanGraph.
- **Multi-device:** phone runs `serve_backend(GenieBackend(...))`; laptop router
  holds `RemoteBackend("ws://phone:8770")`.

## 11. Known limitations / roadmap

- `genie-t2t-run` stdout framing is the one build-specific knob (`_OUTPUT_BEGIN/END`).
- RemoteBackend fabric is LAN/no-auth v0 (WSS+token is the production one-liner).
- Vision/ASR specialists are not yet precompiled locally → those steps escalate to
  cloud until their bundles land in `build/` (then `genie_swarm()` lights them up).
