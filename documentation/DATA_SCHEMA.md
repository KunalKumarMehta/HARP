# HARP — Data Schemas

HARP has one embedded database (the offline outbox) and two on-wire data contracts
(the plan graph and the fabric RPC frames). All are stdlib/JSON — no server DB.

---

## 1. Offline outbox — SQLite (`fabric/sync_queue.py`)

Local, single-writer SQLite (`:memory:` in tests; a file on device). WAL +
`busy_timeout=5000`, `synchronous=NORMAL`, `cache_size=-64000` (~64 MB),
`temp_store=MEMORY`, `mmap_size=256 MB`, `foreign_keys=ON`. The connection is
confined to one worker thread (SQLite thread affinity); writes use
`BEGIN IMMEDIATE` for an atomic dual-write.

### Table `materialized` — the local read model
| Column | Type | Notes |
|---|---|---|
| `entity_id` | TEXT PRIMARY KEY | client-generated id (no server translation) |
| `payload` | TEXT NOT NULL | JSON blob of the entity's current state |
| `revision` | INTEGER NOT NULL | monotonic per entity (NOT a timestamp — clock-drift-immune) |

### Table `outbox` — the sync intent log
| Column | Type | Notes |
|---|---|---|
| `mutation_id` | TEXT PRIMARY KEY | client `uuid4` = **server idempotency key** |
| `entity_id` | TEXT NOT NULL | which entity this mutates |
| `op` | TEXT NOT NULL | `create` \| `update` \| `delete` |
| `payload` | TEXT NOT NULL | JSON blob of the mutation |
| `revision` | INTEGER NOT NULL | monotonic per entity |
| `status` | TEXT NOT NULL | `SyncState`: `pending`/`in_flight`/`success`/`conflict` |
| `retry_count` | INTEGER NOT NULL DEFAULT 0 | bumped on each claim |
| `created_at` | REAL NOT NULL | epoch seconds; FIFO ordering key |

Index: `idx_outbox_status (status, created_at)` — drives FIFO claim of `pending`.

### Four-state FSM
```
enqueue ─▶ pending ──next_in_flight──▶ in_flight ──ACK success──▶ (row deleted)
              ▲                            │
              └──────── recover() ─────────┘ (reconnect/crash: unknown fate → retransmit)
                                           │
                                           └──ACK 404/409/403──▶ conflict (quarantined)
```
- `enqueue`: atomic dual-write (materialized + outbox) under one `BEGIN IMMEDIATE`.
- `recover`: on every reconnect/startup, `in_flight → pending` (the mutation's fate
  is unknown; idempotency keys make blind retransmit safe → at-least-once).
- `next_in_flight`: FIFO claim oldest `pending` → `in_flight`, bump `retry_count`.
- `mark_success`: delete the row (minimal footprint).
- `mark_conflict`: quarantine; never silently overwrite, never infinite-retry.

> Note: the cloud-side `CloudMutationHandler` dedup path uses a **ULID** key
> (`shared/mutation.py`) for time-sortable idempotency; the outbox here uses
> `uuid4`. Both are client-minted-once and stable across retransmits.

---

## 2. Plan graph wire (`shared/plan_schema.json`, `shared/plan_codec.py`)

JSON, compact (~0.7 KB typical). The cloud↔edge boundary.

```json
{
  "plan_id": "plan-demo",
  "steps": [
    {
      "step_id": "s_asr",
      "modality": "text | audio | vision",
      "decision": "local | escalate | undecided",
      "model_id": "qwen3-4b",
      "prompt": "transcribe the call recording",
      "depends_on": ["..."]
    }
  ]
}
```
Constraints (`additionalProperties:false`; all step keys required):
- **Shape** (JSON-Schema): types, enum membership, no extra keys. Validated by
  `jsonschema` in CI, by a dependency-free structural checker at the edge.
- **Semantics** (in code, since Schema can't express a DAG): unique `step_id`s,
  every `depends_on` references an existing step, and the graph is acyclic
  (`PlanGraph.topo_order` raises otherwise). Any violation → `PlanWireError`
  (fail loud, fail closed; an executor never sees a half-valid plan).
- Dataflow convention: a step prompt may reference `"<dep_id>_output"`, substituted
  with the dependency's output at execution time.

---

## 3. Fabric RPC frames (`fabric/remote_backend.py`)

JSON request/response over WebSocket for multi-device inference. One short-lived
connection per call.

| Direction | Frame |
|---|---|
| client → server | `{"op": "capabilities"}` |
| client → server | `{"op": "infer", "req": {messages, model_id, modality, max_tokens, stream}}` |
| client → server | `{"op": "profile", "req": {...}}` |
| server → client | `{"capability": {...}}` / `{"metrics": {...}}` |
| server → client (infer stream) | `{"t": "<token>"}` … then `{"done": true}` |
| server → client (error) | `{"error": "<Type: message>"}` |

Rules: a stream that ends **without** a `done` or `error` frame is treated as a
failed inference and **raises** on the client (no silent partial success); the
server guards its error-reply against an already-closed socket and closes the
inference generator on client disconnect. Trust model: LAN, no auth (v0) — WSS +
bearer token is the documented production hardening.

---

## 4. Mutation sync frames (`fabric/ws_node.py`)

Cloud uplink frames for the outbox drain:
`{mutation_id, entity_id, op, payload, revision}` → ACK `{mutation_id, status}`
where `status ∈ SyncState`. The cloud dedups on `mutation_id` before any
side-effect (at-least-once → effectively-once).

---

## 5. Routing feature contract (`router/router_policy.py` · `RoutingFeatures`)

The gatekeeper's input contract. The synthetic-data generator
(`data/synth_routing_data.py`) emits **exactly** these keys per row so training
data matches the live features. Query + device fields are pre-existing; the four
**contention** keys were added with the NPU single-flight endpoint and default so
every existing caller is unchanged.

| Key | Type | Source | Notes |
|---|---|---|---|
| `query` | str | hot path | the turn text scored by the encoder head |
| `modality` | `Modality` | plan step | text \| audio \| vision |
| `online` | bool | network | false ⇒ escalate physically unavailable |
| `npu_present` | bool | `Capability` | no NPU ⇒ capability fallback to cloud |
| `edge_modalities` | tuple | `Capability` | modality coverage guard |
| `edge_max_context` | int | `Capability` | over-context ⇒ escalate |
| `approx_tokens` | int | cheap estimate | pre-tokenizer length proxy |
| `thermal_c` | float? | sensor | ≥ ceiling ⇒ bias escalate |
| `battery_pct` | float? | sensor | ≤ floor (not charging) ⇒ bias escalate |
| **`npu_inflight`** | bool | endpoint | an NPU infer is currently running |
| **`npu_queue_depth`** | int | endpoint | infers already committed to the NPU lane |
| **`tools_present`** | bool | request | request carries tools (local lane forces CoT off) |
| **`offline`** | bool | endpoint | escalate unavailable ⇒ contention gate **never** sheds |

**Contention gate:** projected NPU wait `= (npu_queue_depth + npu_inflight) ·
npu_exec_est_s`. If a soft-LOCAL verdict's projected wait exceeds
`contention_budget_s` (2.0s) and not offline, flip to ESCALATE
(`reason="contention_shed"`). It runs **after** the isotonic+conformal complexity
gate and never overrides it or the hardware guard. The four contention keys are
sampled independently of difficulty in the corpus — they are an inference-time
gate over a LOCAL verdict, **not** a supervised label — so existing labels/weights
are unchanged.
