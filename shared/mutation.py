"""
HARP · shared/mutation.py · edge↔cloud interface · MIT
=====================================================================
mutation_id keying — edge/cloud wiring notes (unblocks CloudMutationHandler)
=====================================================================
Notes on how `mutation_id` is keyed when wiring CloudMutationHandler in front
of the escalate path, grounded in the ARM64 fabric four-state-queue design:

 1. mutation_id is a CLIENT-generated ULID — generated at the edge, in the
    `pending` state, BEFORE any network exists. Not server-assigned. ULID over
    UUIDv4 deliberately: it is lexicographically sortable (48-bit time prefix),
    so it doubles as the FIFO ordering key the handler can sort on.
 2. It is assigned ONCE at enqueue and is STABLE across every retransmit. On
    reconnect the edge downgrades in_flight→pending and retransmits the SAME id
    (blind at-least-once delivery). Id stability is the entire basis of dedup.
 3. mutation_id is the SOLE idempotency key. CloudMutationHandler dedups on it:
    seen-id → suppress all side-effects, re-emit the ORIGINAL success ACK.
    (Maintain a high-speed recent-id cache; absorbing duplicates is mandatory,
    not best-effort.)
 4. mutation_id is NOT the conflict key. Collision detection uses
    (entity_id, entity_revision) with monotonic INTEGER revisions, server-wins
    last-write-wins — NEVER timestamps (clock drift across the triad). The id
    says "which attempt"; the revision says "is it stale".
 5. entity_id is ALSO a client-generated ULID. The cloud accepts it blindly as
    the global PK from inception — no temp-UUID→server-int translation, no
    cascading FK rewrite.
 6. ESCALATE PATH: an escalation is just a mutation with op=ESCALATE. Same
    envelope, same keying. CloudMutationHandler therefore dedups escalations
    with the identical recent-id-cache check — wire it in front of escalate
    exactly as in front of CREATE/UPDATE/DELETE. No special-casing.

Wiring summary: key on `mutation_id` for idempotency, `(entity_id,
entity_revision)` for conflict, sort by `mutation_id` for FIFO.
=====================================================================
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum

# ---- minimal pure-Python ULID (no dep — compatible with ARM64/win_arm64 wheels) ------
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"   # 32 symbols, excl I L O U


def _b32(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def new_ulid(ts_ms: int | None = None) -> str:
    """26-char Crockford-base32 ULID: 48-bit ms timestamp + 80-bit randomness.
    Monotonic-sortable by creation time → FIFO key for the outbox + cloud handler."""
    ts = int(time.time() * 1000) if ts_ms is None else ts_ms
    rand = int.from_bytes(os.urandom(10), "big")        # 80 bits
    return _b32(ts, 10) + _b32(rand, 16)


def new_mutation_id() -> str:
    return new_ulid()


class MutationStatus(str, Enum):
    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    SUCCESS = "success"
    CONFLICT = "conflict"


class MutationOp(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ESCALATE = "escalate"        # edge gatekeeper hands query to cloud (NIM)


@dataclass
class MutationEnvelope:
    """The serialized outbox row. Edge emits it; CloudMutationHandler consumes it.
    Field names are the frozen wire contract between the edge and cloud backends."""
    mutation_id: str                       # client ULID — idempotency + FIFO key
    entity_id: str                         # client ULID — global PK, accepted blindly
    entity_revision: int                   # monotonic int — conflict key (server-wins LWW)
    op: MutationOp
    endpoint: str                          # RPC command, e.g. "router.escalate"
    payload: dict
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    retry_count: int = 0
    status: MutationStatus = MutationStatus.PENDING

    @classmethod
    def new(cls, *, entity_id: str, entity_revision: int, op: MutationOp,
            endpoint: str, payload: dict) -> "MutationEnvelope":
        return cls(mutation_id=new_mutation_id(), entity_id=entity_id,
                   entity_revision=entity_revision, op=op, endpoint=endpoint,
                   payload=payload)

    def to_wire(self) -> dict:
        d = self.__dict__.copy()
        d["op"] = self.op.value
        d["status"] = self.status.value
        return d


# ---- reference idempotency check (CloudMutationHandler implements its own) ---
def is_duplicate(recent_ids, mutation_id: str) -> bool:
    """Reference semantics for the cloud handler: True → suppress side-effects,
    re-emit the original success ACK. `recent_ids` is any membership cache
    (set / LRU / Redis). Keyed strictly on mutation_id, never on payload hash."""
    return mutation_id in recent_ids


def is_stale(current_revision: int, incoming_revision: int) -> bool:
    """Conflict detection: True → 409 Conflict, client pulls authoritative state
    and discards its mutation. Integer compare only; no timestamps."""
    return incoming_revision <= current_revision
