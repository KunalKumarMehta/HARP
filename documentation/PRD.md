# HARP — Product Requirements Document (PRD)

**Status:** living · **Owner:** Maintainers · **Last updated:** 2026-06-17

## 1. Problem

Humans are messy communicators and devices are heterogeneous. Dumping raw prompts
plus bloated device telemetry into one cloud LLM is expensive, slow, leaks
privacy, and dies without a network. Pure on-device inference can't reason hard
enough. Neither extreme wins.

## 2. Product

**HARP** is a hardware-aware agentic routing runtime: a lightweight edge
gatekeeper de-noises input and triages device state, then routes each task to the
right model on the right tier — a swarm of specialist small models on the
Snapdragon NPU, escalating only the genuinely hard steps to a cloud multi-agent
planner — degrading gracefully to fully-offline when disconnected.

It is one MIT codebase that addresses three complementary capability dimensions:

| Dimension | What HARP demonstrates |
|---|---|
| **On-device inference** | NPU inference, energy/latency efficiency, multi-device orchestration on Snapdragon hardware |
| **Cloud multi-agent planning** | cloud multi-agent planning on NeMo/Nemotron with measured optimization |
| **Hybrid edge-cloud vision** | the full hybrid architecture at Bharat scale |

## 3. Users

- **Primary (demo persona):** a Bharat micro-enterprise/kirana operator using a
  voice-first agent on a low-connectivity phone.
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
- All CI gates green on every push (currently 9).

## 8. Kill-risk gates (feasibility before commitment)

- **Risk A — edge executor is real:** a quantized SLM runs on the NPU with measured
  tok-s/energy. *De-risked:* a precompiled Qwen3-4B Genie bundle ships in `build/`;
  Risk A is now run-and-measure, not compile.
- **Risk B — router is real:** calibrated `{local|escalate}` under the latency
  budget with no silent mis-routes.

## 9. Open decisions

- **Vertical scope:** the current design is a horizontal platform; a Bharat-voice
  default persona remains under active consideration and is not yet locked.
- **Router tuning:** router-only SFT/LoRA fine-tune is confirmed; Nemotron→edge
  distillation is a flex roadmap item.
- **Device topology (confirmed):** X Elite PC + Snapdragon 8 Elite phone + Arduino
  UNO Q + Cloud AI 100.

## 10. Constraints / risks

- QDC X Elite sessions are barebone and ephemeral → `edge/bootstrap_qdc.cmd`.
- `genie-t2t-run` (QAIRT 2.45) is a Qualcomm SDK, not pip-installable → bootstrap
  detects/wires it; bundle is QAIRT-2.45 / Hexagon-v73 specific.
