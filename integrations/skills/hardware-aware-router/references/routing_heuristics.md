# Routing heuristics — LOCAL (NPU) vs ESCALATE (cloud planner)

The decision is HARP's, served by `POST /v1/route`. This file describes the policy
in prose for human and agent readers. It is documentation, not a second
implementation — the authoritative logic is `router/router_policy.py` (the gate)
and `serve/openai_endpoint.py::_resolve_route` (pins + contention shed).

## The axes, in order of authority

1. **Hardware / capability guard (authoritative).** No NPU, an unsupported
   modality, or an over-context prompt → ESCALATE (or, offline, degrade to LOCAL).
   This guard can never be overridden — it prevents routing into a backend that
   would OOM or crash.

2. **Connectivity.** Offline (escalate physically unavailable) → always LOCAL.
   Correctness/availability beats latency: a queued NPU turn still answers; a cloud
   turn with no network does not. The contention shed (below) is disabled offline.

3. **Complexity (the calibrated gate).** A single-pass encoder emits an
   uncertainty score `u(x)`; isotonic calibration maps it to a real edge-error
   probability; a conformal threshold gates escalation so under-routing (a hard
   query kept local) is bounded at α. Never raw argmax.
   - Trivial turns (greetings, acks, one-hop lookups) → LOCAL via a cheap floor,
     skipping the head entirely.
   - Long, multi-step, "prove / derive / design / optimize / step by step / diagnose
     / trade-off / architect" turns → high `u(x)` → ESCALATE.

4. **Contention (NPU single-lane pressure).** The NPU single-context binary is
   single-lane: one in-flight infer at a time, and TTFT degrades O(N) under queue.
   If complexity says LOCAL but the lane is in-flight with a deep enough queue that
   projected wait exceeds the TTFT budget (default 2.0 s), and escalate is
   available, flip to ESCALATE with reason `contention_shed`. Never offline.

5. **Tools.** A turn carrying `tools` is biased to the cloud planner, which handles
   function-calling natively. (On the local lane HARP forces chain-of-thought off
   for tool turns; tool turns simply escalate and the primary model answers — the
   hook returns `None`.)

## Reasons you will see on /route

| reason | axis | decision |
|---|---|---|
| `trivial_floor` | complexity (cheap floor) | local |
| `complexity_gate` | calibrated gate | local or escalate |
| `contention_shed` | NPU contention | escalate |
| `overflow_shed` | operational single-flight overflow | escalate (shed=true) |
| `capability_modality` / `capability_context` | hardware guard | escalate |
| `offline_forced_local` / `offline_degraded_*` | connectivity | local |
| `model_pin` | explicit `harp-edge` / `harp-cloud` | local / escalate |

## Tuning knobs (endpoint env)

- `HARP_TTFT_BUDGET_S` (2.0) — projected NPU wait above which a busy lane sheds.
- `HARP_NPU_EXEC_EST_S` (3.0) — per-infer NPU time estimate used to project wait.
- `HARP_ESCALATE_DISABLED` — force offline: queue on the NPU, never shed.

The conformal α and thermal/battery ceilings live in `RoutingPolicy`. Swapping the
placeholder `mock_score_fn` for the trained mmBERT-small head changes step 3 only;
every other axis is unchanged.
