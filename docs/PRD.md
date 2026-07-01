# HARP — Product Requirements Document (PRD)

**Status:** living · **Owner:** Maintainers · **Last updated:** 2026-07-01

## 1. Problem

The capable-device era has arrived: the machine in your hand or on your desk —
Apple Silicon, Snapdragon X, an AI PC, a flagship phone — is now strong enough to
run real models locally. Yet assistants still route every request to a cloud LLM:
expensive, slow, privacy-leaking, and dead without a network. Pure on-device
inference is the opposite failure — private and instant, but it can't reason hard
enough on its own. Neither extreme uses the hardware people already own well.

HARP is for people **with** a capable device — not for users with no local
hardware. The premise is that their device should be the default, with the cloud as
backup for the genuinely hard steps.

## 2. Product

**HARP** is a hardware-aware agentic assistant: it runs on-device and escalates to
the cloud only when a task genuinely needs it. A lightweight on-device gatekeeper
de-noises input and triages device state, runs what fits on local silicon (Apple
Silicon, Snapdragon X, an AI PC, a flagship phone), and escalates only the genuinely
hard steps to a cloud multi-agent planner — degrading gracefully to fully-offline
when disconnected.

The **engine** is what makes this a product and not a wrapper: a **calibrated
on-device↔cloud escalation gate** (`router/router_policy.py`) that decides, per
task, what runs where — with a conformal bound on dangerous mis-routes and a
hardware/offline guard that always has the final say. Everything else — the
on-device backends, the cloud planner, the offline fabric — exists to serve that
one decision.

| Layer | What it is | Why it matters |
|---|---|---|
| **Product** | a device-first agentic assistant, offline-resilient | the thing a user actually runs |
| **Engine (the IP)** | the calibrated on-device↔cloud escalation gate | the defensible core: *what runs where* |
| **Proof** | NPU efficiency (on-device) + multi-agent planning (cloud) | measured evidence both lanes of the gate are real |

## 3. Users

- **Primary:** owners of a capable device — an Apple Silicon Mac, a Snapdragon X
  laptop, an AI PC, or a flagship phone — who want a private, instant, offline-capable
  assistant that uses the hardware they already paid for, not a metered cloud round-trip.
- **A demonstrative scenario:** a field / low-connectivity operator whose device must
  keep working when the network doesn't — the offline lane made concrete.
- **Operators / evaluators:** teams assessing on-device efficiency, agentic
  architecture quality, and product legibility.
- **Developers:** contributors building specialists/backends against a frozen contract.

## 4. Goals & non-goals

**Goals**
- G1 Route `{local | escalate}` per task with a calibrated gate that **bounds
  silent under-routing** (dangerous mis-routes) at α.
- G2 Run a whole quantized SLM on the Hexagon NPU and report TTFT / tok-s /
  energy-per-token.
- G3 Execute a cloud-emitted plan DAG end-to-end on the edge with dataflow between
  steps; fail closed to edge when offline.
- G4 Multi-device: a plan step can execute on a *separate* device (phone) over the
  fabric — orchestration impossible on one device.
- G5 One public MIT repo, ARM64-clean, demoable in one command.

**Non-goals (explicitly cut; docs-only roadmap)**
- Mid-stack neural-layer splitting / per-token activation offload (bandwidth-fatal).
- RLHF (only the router is tuned, via SFT/LoRA).
- CRDT/vector-clock multi-device sync (replaced by a four-state queue + LWW).
- Protobuf wire (JSON is sufficient at demo scale).

## 5. Functional requirements

- FR1 **Backend contract**: `capabilities()/infer()/profile()`; streaming mandatory.
- FR2 **Router**: honor planner pins (LOCAL/ESCALATE); resolve AUTO via calibrated
  conformal gate; hardware/offline guard has final say.
- FR3 **Plan codec**: validated JSON wire (schema shape + DAG semantics); reject
  malformed/ cyclic plans loudly.
- FR4 **Executor**: topological execution, `<step>_output` dataflow threading,
  per-step failure isolation (a failed step skips its downstream cone).
- FR5 **Edge backends**: load a precompiled Genie bundle (`GenieBackend`, fast path)
  or a self-compiled ONNX model (`QNNBackend`); loud NPU-engagement assertion.
- FR6 **Cloud backend**: OpenAI-compatible NIM (`NIMBackend`); ReWOO planner emits
  PlanGraph; dedup escalated mutations on a stable id.
- FR7 **Multi-device**: `RemoteBackend` proxies inference to a peer node over WS.
- FR8 **Offline**: four-state outbox queue; reconnect → recover → at-least-once,
  idempotent redelivery; conflict quarantine.
- FR9 **Evidence**: a Risk-A gate that emits an on-device performance evidence pack (NPU engagement, tok/s, energy-per-token).
- FR10 **One-command provisioning** of a barebone QDC X Elite (Windows ARM64).

## 6. Non-functional requirements

- NFR1 Router hot-path overhead < 50 ms (target < 10 ms with the encoder head).
- NFR2 NPU decode ≥ 15 tok/s floor (separates NPU from silent CPU fallback).
- NFR3 ARM64-clean: no dependency requiring an unavailable `win_arm64` wheel on the
  edge runtime path (codec + fabric fall back to stdlib).
- NFR4 MIT-licensed, public from commit 1; no employer/3rd-party proprietary code.
- NFR5 Deterministic, gated correctness: every contract invariant is a CI gate.

## 7. Success metrics

- Risk-A gate **PASS** on real silicon (NPU engaged, ≥15 tok/s, energy reported).
- Risk-B: routing probe 0 silent mis-routes, p95 gate overhead < 50 ms.
- Demo runs end-to-end in one command, online and offline, and across two nodes.
- All CI gates green on every push (currently 16 + demo-integration).

## 8. Kill-risk gates (feasibility before commitment)

- **Risk A — edge executor is real:** a quantized SLM runs on the NPU with measured
  tok-s/energy. *De-risked:* a precompiled Qwen3-4B Genie bundle is provisioned into
  `build/` (git-ignored; staged by `edge\bootstrap_qdc.cmd`), so Risk A is now
  run-and-measure, not compile.
- **Risk B — router is real:** calibrated `{local|escalate}` under the latency
  budget with no silent mis-routes.

## 9. Open decisions

- **Vertical scope:** HARP leads as a **product** (a device-first agentic assistant)
  whose defensible core is the escalation engine. A specific default persona (e.g. a
  voice-first operator) is a packaging choice on top of that engine, not yet locked.
- **Router tuning:** router-only SFT/LoRA fine-tune is confirmed; Nemotron→edge
  distillation is a flex roadmap item.
- **Device topology (confirmed):** X Elite PC + Snapdragon 8 Elite phone + Arduino
  UNO Q + Cloud AI 100.

## 10. Constraints / risks

- QDC X Elite sessions are barebone and ephemeral → `edge/bootstrap_qdc.cmd`.
- `genie-t2t-run` (QAIRT 2.45) is a Qualcomm SDK, not pip-installable → bootstrap
  detects/wires it; bundle is QAIRT-2.45 / Hexagon-v73 specific.
