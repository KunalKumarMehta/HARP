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
