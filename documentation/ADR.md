# HARP — Architecture Decision Records (ADR)

Format: each record is Context → Decision → Status → Consequences. Newest decisions
extend, never silently overturn, earlier ones.

---

## ADR-0001 — Python ABC over a C ABI for the backend contract
**Context:** A shipping runtime would use a C ABI (OrtEp/TRITONSERVER_*). The
orchestration layer is Python (QNN EP / Genie / NIM REST) and the C ABI is a
longer-term integration target.
**Decision:** The contract is a Python `Backend` ABC (`capabilities/infer/profile`).
**Status:** Accepted. **Consequences:** Fast parallel development; the C ABI is a
scale-roadmap line, not built.

## ADR-0002 — OpenAI-shaped payload as the boundary lingua franca
**Context:** Edge and cloud must share one request shape or the "hardware-agnostic"
story is aspirational. **Decision:** `InferRequest` is OpenAI-shaped; the same
struct hits QNN or NIM, only the backend swaps. **Status:** Accepted.
**Consequences:** One call site; backends are interchangeable behind the router.

## ADR-0003 — JSON wire for PlanGraph; Protobuf deferred
**Context:** Research pushed Protobuf for edge throughput. **Decision:** JSON
`PlanGraph` only; ~0.7 KB plans. **Status:** Accepted. **Consequences:** No schema
tooling cost; binary serialization is a scale-roadmap line. Validation is split:
JSON-Schema for shape, in-code for DAG semantics (Schema can't express a DAG).

## ADR-0004 — Cut CRDTs/vector clocks; four-state SQLite outbox + LWW
**Context:** Production-correct multi-device sync wants CRDTs; that is a multi-week
build that is out of scope for the current milestone. **Decision:** Four states (pending/in_flight/success/
conflict), client-UUID idempotency keys, monotonic integer revisions, at-least-once
redelivery, conflict quarantine. **Status:** Accepted.
**Consequences:** Offline resilience is demonstrable and CI-gateable without a CRDT
stack; cross-device merge is LWW, not convergent.

## ADR-0005 — Capability negotiation is the fallback logic
**Context:** Need a deterministic floor for edge↔cloud fallback. **Decision:** The
base `Router._select` negotiates `capabilities()` + network state (offline→edge;
no-NPU/unsupported-modality→cloud; ESCALATE→cloud) and always has final say.
**Status:** Accepted. **Consequences:** The learned policy slots *above* this guard;
it can never route into an impossible backend.

## ADR-0006 — `RouteDecision.AUTO` ("undecided") as the planner default
**Context:** A planner that pre-assigns the tier defeats a learned router.
**Decision:** Add `AUTO`; the ReWOO planner emits AUTO for deferred steps, pins
LOCAL/ESCALATE only when certain; `PolicyRouter` resolves AUTO. **Status:** Accepted
(merged into the freeze; wire value `"undecided"`). **Consequences:** Route is
endogenous; base router treats an unresolved AUTO as LOCAL.

## ADR-0007 — Encoder router (mmBERT-small), not a decoder
**Context:** Decoder routers (Qwen3-0.6B, Arch-Router-1.5B) are 50–150 ms and need
a static KV cache on Hexagon — too slow for an always-resident gatekeeper.
**Decision:** Stateless single-pass encoder head emitting `u(x)=P(escalate)`;
calibrate (isotonic) then gate (conformal), never raw argmax. **Status:** Accepted.
**Consequences:** <10 ms hot path; the calibration/conformal machinery is
backbone-agnostic and already coded.

## ADR-0008 — Cloud planner is ReWOO; PlanGraph is NVIDIA-agnostic
**Context:** ReAct agents resolve step N+1 only after observing N → no upfront DAG
to ship the edge. **Decision:** ReWOO planner; a `PreInvoke`/middleware hook
intercepts the plan; an adapter maps it to `PlanGraph` (we do NOT adopt
NVIDIA's ATIF as the runtime wire format). **Status:** Accepted. **Consequences:**
The edge-executes / cloud-plans split holds; the wire stays vendor-agnostic.

## ADR-0009 — `GenieBackend` for precompiled bundles, distinct from `QNNBackend`
**Context:** The precompiled AI-Hub asset (`build/qwen3-4b-w4a16`) is a **Genie**
context-binary bundle (`genie_config.json` + ctx-bins, run by `genie-t2t-run`),
which the onnxruntime-genai `QNNBackend` cannot load. The precompiled fast path had
no route into the contract. **Decision:** Add `GenieBackend` (drives `genie-t2t-run`)
as a first-class `Backend`; keep `QNNBackend` for self-compiled ONNX models;
`genie_swarm()` auto-discovers every bundle in `build/`. **Status:** Accepted.
**Consequences:** The precompiled fast path is now exercisable and measurable; self-compilation is only for
non-catalog models. The runtime is a Qualcomm SDK (QAIRT 2.45), provisioned by
`edge/bootstrap_qdc.cmd`, not pip.

## ADR-0010 — Multi-device is a Backend (`RemoteBackend`), not a new subsystem
**Context:** Multi-device orchestration requires coordinating inference across peer
nodes, but the executor/router should not need to learn about networking. **Decision:**
`RemoteBackend` satisfies the `Backend` ABC and proxies `infer()` over WebSocket to
a peer node running `serve_backend(...)`. **Status:** Accepted. **Consequences:** A
step routes to the phone exactly as to a local backend; the offline guard fails it
closed; zero change to the freeze or executor. Client raises on truncated streams
(no silent partial-success); the server drains/cleans up on client disconnect.

## ADR-0011 — Cut mid-stack neural-layer splitting (docs-only roadmap)
**Context:** The original thesis wanted to split a model's layers across edge/cloud.
Per-token activation round-trips saturate the WAN (~1.0–1.1× speedup) and invert
Qualcomm's energy score. **Decision:** Model-level routing (whole tasks to whole
models), not layer splitting. **Status:** Accepted. **Consequences:** Achieves the
same functional goal with lower complexity and better energy characteristics;
layer-splitting remains a roadmap item.

## ADR-0012 — One-command QDC provisioning; detect-first for `genie-t2t-run`
**Context:** A fresh QDC X Elite is barebone (no Python/deps/runtime) and ephemeral;
QAIRT is a login-gated Qualcomm SDK with no anonymous URL. **Decision:**
`edge/bootstrap_qdc.cmd` installs Python+deps, then **detects** a staged QAIRT
(common on QDC images) and wires `genie-t2t-run` onto PATH + `HARP_GENIE_BIN`,
accepting a `-QairtZip` fallback; it never fabricates a download URL. **Status:**
Accepted. **Consequences:** Repeatable one-command setup; honest failure with exact
next steps when the runtime genuinely isn't present.

## ADR-0013 — HARP exposed via an OpenAI-compatible endpoint + Hermes provider plugin
**Context:** Third-party agent frameworks (Hermes, OpenClaw) need to use HARP's
hardware-aware routing without adopting HARP's internals. The two losing options are
a fork (un-maintainable) or a generic MITM proxy (HARP would sit between the agent
and some *other* model, owning neither). **Decision:** Ship `serve/openai_endpoint.py`
— HARP *is* the origin model server, OpenAI-compatible (`/v1/chat/completions`,
`/v1/models`, `/health`) — plus a drop-in Hermes `model-provider` plugin
(`integrations/hermes/...`) that points the agent at it. The endpoint imports only
the frozen `Backend` interface (concrete backends are constructor args).
**Status:** Accepted. **Consequences:** Any OpenAI-speaking agent gets the NPU lane
for free; no fork, no proxy. Standard `chat.completion(.chunk)` + `tool_calls` on the
wire; no framework-proprietary events preserved. `harp_contract.py` untouched.

## ADR-0014 — NPU single-flight lock + O(N) overflow-shed to escalate
**Context:** The NPU single-context binary is **single-lane**. Two concurrent infers
against one context binary exhaust the FastRPC memory map (`fastrpc memory map for
fd ... failed with error: 0x1`), fail the SMMU domain, hit "Could not allocate
persistent weights buffer!", and crash — or silently collide in VTCM (identical
output for distinct prompts). Under queue, TTFT degrades O(N):
`TTFT_k ≈ TTFT_base + Σ T_exec(i<k)`. **Decision:** The endpoint guards all local
infers with one `asyncio.Lock` (exactly one in-flight). When the lane is busy and
projected wait exceeds `HARP_TTFT_BUDGET_S` (2.0s) **and** escalate is available, it
**sheds** the request to the cloud lane instead of queuing. Offline (no escalate) it
queues on the NPU — correctness > latency, never drop. **Status:** Accepted.
**Consequences:** No FastRPC 0x1 / VTCM-collision crash is reachable from the endpoint;
foreground TTFT stays bounded; offline never loses a request.

## ADR-0015 — Thinking disabled on the local lane when tools are present
**Context:** GenieAPIService emits real OpenAI `tool_calls`, but Qwen3 chain-of-thought
corrupts the Genie tool-interception path when a request carries `tools`. **Decision:**
When a request carries `tools` and the chosen lane is local, the endpoint forces
`thinking=False` on the local infer (threaded into `GenieBackend.infer`, defaulting on).
**Status:** Accepted. **Consequences:** Tool-calling on-device returns well-formed
`tool_calls`; CoT remains available for non-tool local turns. `InferRequest` is frozen,
so thinking/tools ride as optional kwargs on the concrete local backend, not as new
contract fields (see tension note in the PR).

## ADR-0016 — Contention axis added to the router; complexity + hardware gates unchanged
**Context:** The complexity gate (isotonic + conformal) and the base-class hardware
guard decide *can the edge run this well*; neither sees lane **contention**.
**Decision:** Extend `RoutingFeatures` with `npu_inflight`, `npu_queue_depth`,
`tools_present`, `offline`. A CONTENTION gate runs **after** the complexity gate: if
complexity says LOCAL but projected NPU wait exceeds the budget and escalate is
available, flip to ESCALATE with `reason="contention_shed"`. It never fires offline and
never overrides the existing gates — it only flips an already-LOCAL soft verdict. The
isotonic+conformal gate and the hardware guard stay authoritative. **Status:** Accepted.
**Consequences:** The router models lane pressure as a first-class axis; the synthetic
corpus emits the four new keys so training data matches the live feature contract.

## ADR-0017 — `default_aux_model` pins the Hermes aux lane to the NPU
**Context:** Hermes runs an aux lane for background work (summarization, title-gen,
memory compaction) — latency-tolerant and constant. **Decision:** The provider's
`default_aux_model="harp-edge"` pins that lane to the NPU. **Status:** Accepted.
**Consequences:** Background work stays on-device (private, free, no round-trip) and is
the single-stream NPU sweet spot; single-flight + overflow-shed keep it from contending
with the foreground lane, which stays `harp-auto` and free to escalate.

## ADR-0018 — Routing brain centralized; `/v1/route` is the single source of truth
**Context:** Track-A needs routing to be a visible, per-turn agentic decision, wired
into Hermes via a `pre_llm_call` hook. A hook that embeds its own classifier would
**drift** from the model the endpoint actually serves. **Decision:** Add an advisory,
side-effect-free `POST /v1/route` (no inference, no `commit_local` — it reuses the
existing `_resolve_route`). The Hermes hook is a thin adapter: it POSTs the turn to
`/v1/route` and forwards the returned `runtime_override`; it embeds **no** classifier.
**Status:** Accepted. **Consequences:** One routing brain (`router/router_policy.py`),
queried over HTTP; the hook can never disagree with the served model. The `/route` call
is on the hot path, so the hook uses a 0.4 s timeout and fails safe to `None`.

## ADR-0019 — ESCALATE returns `None`; the Hermes primary handles cloud turns natively
**Context:** On a local decision the hook overrides the turn to the `harp` provider; on
escalate it must hand off to the cloud. HARP's own endpoint tool path is Tier-0 and out
of scope this pass. **Decision:** On ESCALATE the hook returns `None` (passive) so
Hermes' configured **primary** model — Nemotron via NIM — handles the turn natively,
tools and all. The demo deliberately avoids local tool turns. **Status:** Accepted.
**Consequences:** Tool-calling correctness rides on the mature NIM path, not the
endpoint's local tool path; the Track-A demo stays clean. Routing local tool turns
through the endpoint is tracked separately (Tier-0).

## ADR-0020 — `/route` complexity classifier is a documented placeholder for the mmBERT head
**Context:** The trained mmBERT-small encoder head is not wired yet; `RoutingPolicy`
ships `mock_score_fn` (token length + complexity-keyword count). **Decision:** `/v1/route`
reuses the SAME calibrated `RoutingPolicy` (no second classifier), and `GET /health`
reports `route_classifier` with an explicit "placeholder for mmBERT-small head" label.
**Status:** Accepted. **Consequences:** Demo decisions are believable and honest about
their provenance; swapping in the trained head changes the complexity axis only and
touches neither the endpoint surface nor the hook/skill.
