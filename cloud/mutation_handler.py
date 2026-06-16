"""
HARP — Hardware-Aware Routing Platform
cloud/mutation_handler.py  ·  CCE owns this  ·  MIT

CTO mandate: "Cloud mutation handler MUST dedup on mutation_id before any side
effect (at-least-once redelivery is now proven)."

The edge offline queue is at-least-once: a single logical escalation can be
REDELIVERED (lost ack, reconnect, retry). The cloud side effect here is a real
NIM inference call — it costs tokens and may trigger downstream tool effects.
So this handler guarantees EXACTLY-ONCE EXECUTION per mutation_id:

  - First arrival: execute the side effect, cache the materialized result.
  - Redelivery (sequential): return the cached result, side effect NOT re-run.
  - Redelivery (concurrent, racing the first): coalesce — the duplicate awaits
    the in-flight execution and returns its result. Still one side effect.
  - Transient failure: entry is evicted so a later redelivery may retry. We do
    NOT cache failures as terminal (at-least-once expects eventual success).

mutation_id CONTRACT (edge must honor — flag to CEE/CTO):
  mutation_id is minted ONCE when the mutation is ENQUEUED and reused on EVERY
  redelivery. Stable per logical mutation, e.g. f"{plan_id}:{step_id}". It must
  NOT be regenerated per send attempt, or dedup cannot work.

Store is pluggable. Default is in-memory (correct within one process lifetime).
Cross-restart correctness needs a persistent store (Redis/SQLite) — scale
roadmap; the DedupStore interface is the swap point.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from shared.harp_contract import Backend, InferRequest


class MutState(str, Enum):
    IN_FLIGHT = "in_flight"
    DONE = "done"


@dataclass
class MutationEnvelope:
    """What the edge ships on escalation. Wraps the contract InferRequest with
    the dedup key + trace ids. Not a contract change — a cloud-receive envelope."""
    mutation_id: str
    request: InferRequest
    plan_id: str = ""
    step_id: str = ""


@dataclass
class _Entry:
    state: MutState
    event: asyncio.Event = field(default_factory=asyncio.Event)
    result: str | None = None
    created_at: float = field(default_factory=time.monotonic)


class DedupStore(Protocol):
    """Swap point for persistence. Default impl is in-memory below."""
    def get(self, mid: str) -> _Entry | None: ...
    def put(self, mid: str, entry: _Entry) -> None: ...
    def delete(self, mid: str) -> None: ...


class InMemoryDedupStore:
    def __init__(self, ttl_s: float | None = None):
        self._d: dict[str, _Entry] = {}
        self._ttl = ttl_s

    def get(self, mid: str) -> _Entry | None:
        e = self._d.get(mid)
        if e and self._ttl and e.state == MutState.DONE and (time.monotonic() - e.created_at) > self._ttl:
            del self._d[mid]
            return None
        return e

    def put(self, mid: str, entry: _Entry) -> None:
        self._d[mid] = entry

    def delete(self, mid: str) -> None:
        self._d.pop(mid, None)


@dataclass
class HandlerStats:
    executed: int = 0          # real side effects fired
    dedup_hits: int = 0        # sequential redeliveries served from cache
    coalesced: int = 0         # concurrent redeliveries that awaited in-flight
    failures: int = 0


class CloudMutationHandler:
    """Idempotent front door to the cloud Manager backend. Wrap any contract
    Backend; call handle() with a MutationEnvelope."""

    def __init__(self, backend: Backend, store: DedupStore | None = None):
        self._backend = backend
        self._store: DedupStore = store or InMemoryDedupStore()
        self._lock = asyncio.Lock()        # guards check-and-insert on the store
        self.stats = HandlerStats()

    async def handle(self, env: MutationEnvelope) -> str:
        mid = env.mutation_id
        if not mid:
            raise ValueError("mutation_id is required for dedup — edge must stamp it at enqueue")

        # --- DEDUP DECISION: must happen BEFORE any side effect ---
        async with self._lock:
            existing = self._store.get(mid)
            if existing is None:
                entry = _Entry(state=MutState.IN_FLIGHT)
                self._store.put(mid, entry)
                owner = True
            else:
                entry = existing
                owner = False
        # lock released — the side effect (slow NIM call) runs unlocked

        if not owner:
            if entry.state == MutState.DONE:
                self.stats.dedup_hits += 1          # sequential redelivery, cached
                return entry.result or ""
            await entry.event.wait()                # concurrent redelivery, coalesce
            self.stats.coalesced += 1
            if entry.result is None:                # original failed -> retry this one
                return await self.handle(env)
            return entry.result

        # --- OWNER executes the single side effect ---
        try:
            result = await self._collect(env.request)
            async with self._lock:
                entry.result = result
                entry.state = MutState.DONE
            self.stats.executed += 1
            entry.event.set()
            return result
        except Exception:
            self.stats.failures += 1
            async with self._lock:
                self._store.delete(mid)             # evict -> redelivery may retry
            entry.event.set()                       # wake waiters; they'll re-handle
            raise

    async def _collect(self, req: InferRequest) -> str:
        """Materialize the stream to a full string so the cached result can be
        replayed deterministically on redelivery."""
        return "".join([tok async for tok in self._backend.infer(req)])


# ---------------------------------------------------------------- self-test (fake backend, no network)
class _FakeBackend(Backend):
    """Counts real invocations so we can prove exactly-once execution."""
    def __init__(self):
        self.calls = 0

    async def capabilities(self): ...
    async def profile(self, req): ...

    async def infer(self, req: InferRequest):
        self.calls += 1
        await asyncio.sleep(0.05)                   # simulate NIM latency
        yield f"result#{self.calls}"


async def _demo() -> None:
    be = _FakeBackend()
    h = CloudMutationHandler(be)
    env = MutationEnvelope("plan-1:reason1",
                           InferRequest(messages=[{"role": "user", "content": "decide"}],
                                        model_id="nvidia/llama-3.3-nemotron-super-49b-v1.5"))

    # 1) first delivery
    r1 = await h.handle(env)
    # 2) sequential redelivery (lost ack) -> cached, no second call
    r2 = await h.handle(env)
    # 3) concurrent redelivery storm -> coalesced to the same single execution
    env2 = MutationEnvelope("plan-1:reason2", env.request)
    storm = await asyncio.gather(*[h.handle(env2) for _ in range(5)])

    print(f"first={r1!r} redelivery={r2!r} (same={r1==r2})")
    print(f"concurrent storm results all equal: {len(set(storm))==1} -> {storm[0]!r}")
    print(f"backend real calls: {be.calls}  (expect 2: one per distinct mutation_id)")
    print(f"stats: executed={h.stats.executed} dedup_hits={h.stats.dedup_hits} coalesced={h.stats.coalesced}")
    assert be.calls == 2, "EXACTLY-ONCE VIOLATED"
    assert r1 == r2, "redelivery returned a different result"
    assert len(set(storm)) == 1, "coalescing failed"
    print("\nPASS: dedup-before-side-effect holds under sequential + concurrent at-least-once redelivery")


if __name__ == "__main__":
    asyncio.run(_demo())
