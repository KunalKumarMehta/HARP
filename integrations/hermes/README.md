# HARP × Hermes — drop-in model provider

Point Hermes at HARP and every turn flows through HARP's hardware-aware router:
the NPU lane for what fits on-device, the cloud lane for what doesn't, graceful
offline fallback either way. HARP looks like one OpenAI-compatible model.

## Install (one screen)

1. **Drop the plugin in.**
   ```sh
   cp -r integrations/hermes/plugins/model-providers/harp \
         "$HERMES_HOME/plugins/model-providers/harp"
   ```

2. **Run the HARP endpoint** (off-device works — it falls back to the genie stub):
   ```sh
   python -m serve.openai_endpoint          # serves on :8765 by default
   ```

3. **Tell the plugin where HARP is.**
   ```sh
   export HARP_BASE_URL="http://127.0.0.1:8765/v1"
   export HARP_API_KEY="local"              # any non-empty value off-device
   ```

4. **Select it in Hermes.**
   ```sh
   hermes model        # choose: custom/harp
   ```
   Models exposed: `harp-auto` (router decides), `harp-edge` (pin NPU),
   `harp-cloud` (pin cloud).

## Routing hook (`pre_llm_call`) — routing as a per-turn decision

The provider above lets you *select* HARP. The **hook** makes HARP decide *every
turn automatically*: before each LLM call it asks HARP `POST /v1/route` and, on a
LOCAL verdict, pins that one turn to the NPU lane (`harp-edge`) via a request-scoped
`runtime_override`; on ESCALATE it returns `None` and your configured primary cloud
model (Nemotron/NIM) handles the turn.

1. **Drop the hook in** (alongside the provider):
   ```sh
   cp -r integrations/hermes/plugins/hooks/hardware-aware-router \
         "$HERMES_HOME/plugins/hooks/hardware-aware-router"
   ```
2. **Point it at HARP** (same var as the provider):
   ```sh
   export HARP_BASE_URL="http://127.0.0.1:8765/v1"
   ```
3. **Set your primary model to the cloud planner.** In Hermes, configure the
   primary as `nvidia-nim` / Nemotron. LOCAL turns get overridden to `harp-edge`;
   ESCALATE turns fall through to this primary.

The hook is fail-safe: the `/route` call has a 0.4 s timeout and returns `None` on
any error, so a slow or down router never stalls a turn. It never mutates the
conversation history or system prompt — only appends one ephemeral telemetry line
on local turns. See `integrations/skills/hardware-aware-router/SKILL.md` for the
agentskills.io packaging and the OpenClaw stdout path.

## Why `default_aux_model = harp-edge`

Hermes runs an *aux lane* for background work — conversation summarization,
title generation, memory compaction. That work is latency-tolerant and runs
constantly, which makes it the **NPU sweet spot**: pinning it to `harp-edge`
keeps it on-device (private, free, no cloud round-trip) and, because the NPU is
single-lane, HARP's single-flight + overflow-shed keeps it from ever contending
with the foreground turn. The foreground lane stays `harp-auto`, free to escalate.

## Endpoint env knobs

| Var | Default | Meaning |
|---|---|---|
| `HARP_ENDPOINT_PORT` | `8765` | listen port |
| `HARP_TTFT_BUDGET_S` | `2.0` | shed to cloud if projected NPU wait exceeds this |
| `HARP_NPU_EXEC_EST_S` | `3.0` | per-infer NPU time estimate (wait projection) |
| `HARP_ESCALATE_DISABLED` | `false` | force offline: queue on NPU, never shed |
| `HARP_LOCAL_MODEL` | `qwen3-4b` | local model id (must match a genie bundle) |

Observability: every response carries an `X-HARP-Route` header and a top-level
`harp_route` field — `{tier, reason, npu_inflight, shed}` — so you can watch the
router decide in real time.
