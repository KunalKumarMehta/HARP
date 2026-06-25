---
name: hardware-aware-router
description: Route each agent turn between an on-device NPU (Snapdragon/QNN) and a cloud planner (NVIDIA Nemotron/NIM) based on query complexity, NPU contention, connectivity, and tool use. Use when an agent runs on Snapdragon/Copilot+ hardware, when turns should stay on-device for privacy/offline/cost, when you need to route between on-device NPU and cloud planner, or when wiring HARP into Hermes (pre_llm_call hook) or OpenClaw (stdout routing).
license: MIT
metadata:
  provider: harp
  endpoint: /v1/route
  homepage: https://github.com/KunalKumarMehta/HARP
---

# Hardware-Aware Router (HARP)

Make routing a **visible, per-turn agentic decision**: short/simple/private turns
run on the on-device NPU; long/multi-step planning escalates to the cloud planner.
HARP is the routing brain; this skill is the thin adapter that wires it into an
agent framework.

## What it does

On every turn the agent asks HARP `POST /v1/route` (advisory — no inference, no
state change) and gets back:

```json
{ "tier": "edge", "reason": "complexity_gate", "shed": false,
  "decision": "local",
  "runtime_override": { "provider": "harp", "model": "harp-edge",
                        "base_url": "http://127.0.0.1:8765/v1",
                        "api_mode": "chat_completions" } }
```

- **`decision: "local"`** → pin this turn to the NPU lane (`harp` provider /
  `harp-edge`). Private, offline-capable, free.
- **`decision: "escalate"`** → `runtime_override` is `null`; the agent's primary
  cloud model (Nemotron via NIM) handles the turn natively, tools and all.

The decision comes from HARP's calibrated router (isotonic + conformal complexity
gate, NPU contention axis, offline/hardware guards). The classifier behind the gate
is reported on `GET /health` as `route_classifier`; the default `mock_score_fn`
(token length + complexity-keyword count) is a **placeholder for the trained
mmBERT-small encoder head** and is swappable without touching this skill.

## Hermes integration (pre_llm_call)

Hermes fires a `pre_llm_call` hook before each LLM call. HARP ships one at
`integrations/hermes/plugins/hooks/hardware-aware-router/`. The callback is
keyword-only and tolerates unknown kwargs:

```python
def _hook(*, session_id="", user_message="", conversation_history=None,
          is_first_turn=False, model="", platform="", sender_id="",
          chat_id="", **kwargs): ...
```

It returns one of:

- `{"runtime_override": {...}, "context": "<one telemetry line>"}` on a LOCAL turn
  (the override is request-scoped and auto-reverts after the turn);
- `None` on ESCALATE or on **any** error — a slow/down router (timeout 0.4 s) must
  never stall a turn. It never mutates `conversation_history` or `system_prompt`.

### Install

1. Run the HARP endpoint: `python -m serve.openai_endpoint` (off-device falls back
   to the genie stub; `:8765` by default).
2. Drop the hook in: copy
   `integrations/hermes/plugins/hooks/hardware-aware-router/` into
   `$HERMES_HOME/plugins/hooks/`.
3. Also install the HARP model-provider plugin (so the `harp` provider exists):
   copy `integrations/hermes/plugins/model-providers/harp/` into
   `$HERMES_HOME/plugins/model-providers/`.
4. `export HARP_BASE_URL=http://127.0.0.1:8765/v1`.
5. Configure your Hermes **primary** model as the cloud planner
   (`nvidia-nim` / Nemotron). LOCAL turns get overridden to `harp-edge`; ESCALATE
   turns fall through to this primary.

## OpenClaw integration (documented, not shipped this pass)

OpenClaw has no `pre_llm_call`; it routes by shelling out and reading **stdout**.
Use `scripts/hardware_probe.py` (NPU presence) plus a `curl` to `/v1/route`:

```sh
python scripts/hardware_probe.py            # {"npu_present": true|false, ...}
curl -s localhost:8765/v1/route -d '{"messages":[{"role":"user","content":"..."}]}'
```

A LOCAL decision → point OpenClaw's model at `runtime_override.base_url` with
`model=harp-edge`; ESCALATE → leave it on the cloud model. (The recursive-stdout
variant is future work.)

## Files

- `scripts/hardware_probe.py` — standalone NPU probe; JSON to stdout, exit 0 even
  with no Qualcomm SDK / onnxruntime. Reused by both framework paths.
- `references/routing_heuristics.md` — the LOCAL-vs-ESCALATE policy in prose.

## Why this matters

Routing is the product, made legible: each turn shows *where it ran and why*
(`X-HARP-Route` header + `harp_route` field + the hook's telemetry line). On-device
turns are private, offline-capable, and free; only genuinely hard planning pays for
the cloud.
