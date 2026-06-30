"""
HARP — hardware-aware edge↔cloud routing
cloud/mutation_handler.py  ·  MIT

Design rule: the cloud mutation handler must dedup on mutation_id before any
side effect (at-least-once redelivery is the delivery guarantee).

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

mutation_id CONTRACT (edge must honor):
  mutation_id is minted ONCE when the mutation is ENQUEUED and reused on EVERY
  redelivery. Stable per logical mutation, e.g. f"{plan_id}:{step_id}". It must
  NOT be regenerated per send attempt, or dedup cannot work.

Store is pluggable. Default is in-memory (correct within one process lifetime).
Cross-restart correctness needs a persistent store (Redis/SQLite) — scale
roadmap; the DedupStore interface is the swap point.

SQLitePersistentDedupStore: survives process restarts, enables exactly-once
across deployments. Uses the same schema approach as fabric.sync_queue.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import queue
import sqlite3
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
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
    async def get(self, mid: str) -> _Entry | None: ...
    async def put(self, mid: str, entry: _Entry) -> None: ...
    async def delete(self, mid: str) -> None: ...
    async def update_result(self, mid: str, result: str) -> None: ...


class InMemoryDedupStore:
    def __init__(self, ttl_s: float | None = None):
        self._d: dict[str, _Entry] = {}
        self._ttl = ttl_s

    async def get(self, mid: str) -> _Entry | None:
        e = self._d.get(mid)
        if e and self._ttl and e.state == MutState.DONE and (time.monotonic() - e.created_at) > self._ttl:
            del self._d[mid]
            return None
        return e

    async def put(self, mid: str, entry: _Entry) -> None:
        self._d[mid] = entry

    async def delete(self, mid: str) -> None:
        self._d.pop(mid, None)
    
    async def update_result(self, mid: str, result: str) -> None:
        e = self._d.get(mid)
        if e:
            e.result = result
            e.state = MutState.DONE
            e.event.set()


# ---------------------------------------------------------------- persistent store
# SQLite-backed dedup store that survives process restarts.
# Thread-safe: uses a single writer thread + check_same_thread=False.

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS mutation_dedup (
    mutation_id TEXT PRIMARY KEY,
    request_json TEXT NOT NULL,
    response TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL,           -- 'in_flight' | 'done'
    created_at REAL NOT NULL,
    completed_at REAL DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_mutation_state ON mutation_dedup(state);
"""


class SQLitePersistentDedupStore:
    """Persistent exactly-once dedup store. Write-through design:

      - In-process `_mem` is authoritative for LIVE entries — it holds the shared
        `_Entry` (with its `asyncio.Event`) the handler uses to COALESCE concurrent
        redeliveries. (A reconstructed-per-get entry could not share that event, so
        a coalescing waiter would hang — the reason a naive SQLite store is wrong.)
      - SQLite persists COMPLETED results so a redelivery AFTER A RESTART finds the
        cached result and returns it without re-executing the side effect. Concurrent
        coalescing across a restart is physically impossible (separate processes), so
        nothing is lost.

    SQL runs on a dedicated writer thread fed by a thread-safe `queue.Queue`;
    futures resolve via `loop.call_soon_threadsafe`. The writer thread NEVER touches
    an asyncio.Event (that only happens on the handler's loop). This fixes the prior
    implementation, which mis-used `asyncio.Queue.get(timeout=)` from a thread and
    hung on every call.
    """

    def __init__(self, db_path: str, *, ttl_s: float | None = None):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_s
        self._mem: dict[str, _Entry] = {}
        self._q: "queue.Queue" = queue.Queue()         # thread-safe (NOT asyncio.Queue)
        self._shutdown = threading.Event()
        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer.start()
        atexit.register(self.close)

    def _writer_loop(self) -> None:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SQLITE_SCHEMA)
        try:
            while not self._shutdown.is_set():
                try:
                    item = self._q.get(timeout=0.1)    # queue.Queue.get HAS a timeout
                except queue.Empty:
                    continue
                if item is None:
                    break
                func, fut, loop = item
                try:
                    res = func(conn)
                    loop.call_soon_threadsafe(fut.set_result, res)
                except Exception as e:                 # surface to the awaiting coroutine
                    loop.call_soon_threadsafe(fut.set_exception, e)
        finally:
            conn.close()

    async def _exec(self, func):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._q.put((func, fut, loop))
        return await fut

    async def get(self, mid: str) -> _Entry | None:
        live = self._mem.get(mid)
        if live is not None:                           # in-process: shared event, coalescing
            return live
        row = await self._exec(lambda c: c.execute(
            "SELECT response, state, completed_at FROM mutation_dedup WHERE mutation_id=?",
            (mid,)).fetchone())
        if not row:
            return None
        resp, state, completed = row
        if self._ttl and state == "done" and completed and (time.monotonic() - completed) > self._ttl:
            await self.delete(mid)
            return None
        e = _Entry(state=MutState(state))
        if state == MutState.DONE.value and resp:      # cross-restart cache hit
            e.result = resp
            e.event.set()
            self._mem[mid] = e
        return e

    async def put(self, mid: str, entry: _Entry) -> None:
        self._mem[mid] = entry                         # shared in-process entry
        req_json = json.dumps(getattr(entry, "request_json", "") or "")
        await self._exec(lambda c: c.execute(
            "INSERT OR REPLACE INTO mutation_dedup (mutation_id, request_json, state, created_at) "
            "VALUES (?, ?, ?, ?)", (mid, req_json, entry.state.value, time.monotonic())))

    async def update_result(self, mid: str, result: str) -> None:
        e = self._mem.get(mid)
        if e is not None:                              # set the shared event on the loop thread
            e.result = result
            e.state = MutState.DONE
            e.event.set()
        await self._exec(lambda c: c.execute(
            "UPDATE mutation_dedup SET response=?, state=?, completed_at=? WHERE mutation_id=?",
            (result, MutState.DONE.value, time.monotonic(), mid)))

    async def delete(self, mid: str) -> None:
        self._mem.pop(mid, None)
        await self._exec(lambda c: c.execute(
            "DELETE FROM mutation_dedup WHERE mutation_id=?", (mid,)))

    def close(self) -> None:
        if self._writer.is_alive():
            self._shutdown.set()
            try:
                self._q.put_nowait(None)
            except Exception:
                pass
            self._writer.join(timeout=2.0)


# ---------------------------------------------------------------- stats & handler
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
            existing = await self._store.get(mid)
            if existing is None:
                entry = _Entry(state=MutState.IN_FLIGHT)
                await self._store.put(mid, entry)
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
            await self._store.update_result(mid, result)
            self.stats.executed += 1
            entry.event.set()
            return result
        except Exception:
            self.stats.failures += 1
            await self._store.delete(mid)             # evict -> redelivery may retry
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


async def _demo_persistent() -> None:
    """Prove the SQLite store: exactly-once in-process AND a cross-restart cache hit
    (a redelivery after the process restarts must NOT re-run the side effect)."""
    import os
    import tempfile
    db = os.path.join(tempfile.gettempdir(), "harp_dedup_demo.db")
    for ext in ("", "-wal", "-shm"):
        try:
            os.remove(db + ext)
        except OSError:
            pass

    env = MutationEnvelope(
        "plan-X:reason1",
        InferRequest(messages=[{"role": "user", "content": "decide"}], model_id="m"))

    be1 = _FakeBackend()
    store1 = SQLitePersistentDedupStore(db)
    h1 = CloudMutationHandler(be1, store=store1)
    r1 = await h1.handle(env)                         # first delivery: executes
    r1b = await h1.handle(env)                        # sequential redelivery: cached
    storm = await asyncio.gather(                     # concurrent: coalesced (shared event)
        *[h1.handle(MutationEnvelope("plan-X:reason2", env.request)) for _ in range(5)])
    assert be1.calls == 2, f"in-process exactly-once violated: {be1.calls}"
    assert r1 == r1b and len(set(storm)) == 1
    store1.close()

    # SIMULATE RESTART: fresh in-process state + fresh backend, same db file.
    be2 = _FakeBackend()
    store2 = SQLitePersistentDedupStore(db)
    h2 = CloudMutationHandler(be2, store=store2)
    r2 = await h2.handle(env)                         # served from SQLite; backend NOT called
    assert be2.calls == 0, f"cross-restart dedup failed: backend re-ran {be2.calls}x"
    assert r2 == r1, "cross-restart returned a different result"
    store2.close()
    for ext in ("", "-wal", "-shm"):
        try:
            os.remove(db + ext)
        except OSError:
            pass
    print("PASS (persistent): exactly-once in-process + cross-restart cache hit (no re-execution)")


if __name__ == "__main__":
    asyncio.run(_demo())
    asyncio.run(_demo_persistent())
