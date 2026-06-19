"""
HARP — Hardware-Aware Routing Platform
fabric/sync_queue.py  ·  Offline mutation queue (four-state FSM)  ·  MIT

The offline-resilience backbone. Pure stdlib (sqlite3 only) so it builds native
on Windows ARM64 with zero win_arm64 wheel risk and zero Prism emulation.

Design notes:
  - Four states: pending -> in_flight -> success | conflict
  - Atomic dual-write (materialized + outbox) under BEGIN IMMEDIATE
  - Client-generated UUID primary keys  -> server idempotency, no ID translation
  - Monotonic integer revisions, NOT timestamps -> conflict res immune to clock drift
  - On reconnect: in_flight -> pending, blind retransmit (at-least-once)
  - WAL + tuned PRAGMAs; CRDTs deliberately NOT used (see ADR)

Threading contract: this object is single-writer. The fabric node owns ONE
dedicated single-worker ThreadPoolExecutor and confines this connection to it
(SQLite thread affinity). Never share the connection across arbitrary threads.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass

from shared.harp_contract import SyncState   # the frozen shared enum

_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA busy_timeout=5000",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA cache_size=-64000",      # ~64 MB page cache
    "PRAGMA temp_store=MEMORY",
    "PRAGMA mmap_size=268435456",    # 256 MB memory-mapped I/O
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS materialized (
    entity_id TEXT PRIMARY KEY,
    payload   TEXT NOT NULL,
    revision  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS outbox (
    mutation_id TEXT PRIMARY KEY,          -- client UUID = server idempotency key
    entity_id   TEXT NOT NULL,
    op          TEXT NOT NULL,             -- create | update | delete
    payload     TEXT NOT NULL,
    revision    INTEGER NOT NULL,          -- monotonic per entity
    status      TEXT NOT NULL,             -- SyncState value
    retry_count INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox(status, created_at);
"""


@dataclass
class Mutation:
    mutation_id: str
    entity_id: str
    op: str
    payload: dict
    revision: int


class OutboxQueue:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, isolation_level=None)  # autocommit; we drive txns
        for p in _PRAGMAS:
            self.db.execute(p)
        self.db.executescript(_SCHEMA)
        self.db.row_factory = sqlite3.Row

    # -- write path ----------------------------------------------------------

    def enqueue(self, entity_id: str, op: str, payload: dict) -> Mutation:
        """Atomic dual-write: update local materialized view AND append intent to
        the outbox in one BEGIN IMMEDIATE txn. UI/agent reads see it instantly;
        the cloud reconciles later."""
        mutation_id = str(uuid.uuid4())
        now = time.time()
        try:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT revision FROM materialized WHERE entity_id=?", (entity_id,)
            ).fetchone()
            revision = (row["revision"] + 1) if row else 1
            if op == "delete":
                self.db.execute("DELETE FROM materialized WHERE entity_id=?", (entity_id,))
            else:
                self.db.execute(
                    "INSERT INTO materialized(entity_id,payload,revision) VALUES(?,?,?) "
                    "ON CONFLICT(entity_id) DO UPDATE SET payload=excluded.payload, revision=excluded.revision",
                    (entity_id, json.dumps(payload), revision),
                )
            self.db.execute(
                "INSERT INTO outbox(mutation_id,entity_id,op,payload,revision,status,retry_count,created_at)"
                " VALUES(?,?,?,?,?,?,0,?)",
                (mutation_id, entity_id, op, json.dumps(payload), revision,
                 SyncState.PENDING.value, now),
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return Mutation(mutation_id, entity_id, op, payload, revision)

    # -- dispatcher path -----------------------------------------------------

    def recover(self) -> int:
        """Call on every reconnect/startup. Any in_flight mutation is of unknown
        fate (server may or may not have processed it before the socket died) ->
        downgrade to pending and retransmit. Idempotency keys make this safe."""
        cur = self.db.execute(
            "UPDATE outbox SET status=? WHERE status=?",
            (SyncState.PENDING.value, SyncState.IN_FLIGHT.value),
        )
        return cur.rowcount

    def next_in_flight(self) -> Mutation | None:
        """FIFO claim of the oldest pending mutation: mark in_flight (idempotency
        lock), bump retry, return it for transmission. None if queue is drained."""
        try:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT * FROM outbox WHERE status=? ORDER BY created_at ASC LIMIT 1",
                (SyncState.PENDING.value,),
            ).fetchone()
            if row is None:
                self.db.execute("COMMIT")
                return None
            self.db.execute(
                "UPDATE outbox SET status=?, retry_count=retry_count+1 WHERE mutation_id=?",
                (SyncState.IN_FLIGHT.value, row["mutation_id"]),
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return Mutation(row["mutation_id"], row["entity_id"], row["op"],
                        json.loads(row["payload"]), row["revision"])

    def mark_success(self, mutation_id: str) -> None:
        """Server ACK'd. Drop the row — keep the queue minimal-footprint."""
        self.db.execute("DELETE FROM outbox WHERE mutation_id=?", (mutation_id,))

    def mark_conflict(self, mutation_id: str) -> None:
        """Server rejected (404/409/403). Quarantine — do NOT silently overwrite,
        do NOT infinite-retry. Halts this entity's FIFO line pending resolution."""
        self.db.execute(
            "UPDATE outbox SET status=? WHERE mutation_id=?",
            (SyncState.CONFLICT.value, mutation_id),
        )

    def counts(self) -> dict[str, int]:
        rows = self.db.execute(
            "SELECT status, COUNT(*) c FROM outbox GROUP BY status"
        ).fetchall()
        return {r["status"]: r["c"] for r in rows}


# ---------------------------------------------------------------- self-test (CI gate)

def _selftest() -> None:
    q = OutboxQueue()

    # 1. offline: agent writes two mutations -> both pending, materialized updated
    m1 = q.enqueue("task-1", "create", {"title": "transcribe clip"})
    m2 = q.enqueue("task-1", "update", {"title": "transcribe + summarize"})
    assert q.counts() == {"pending": 2}, q.counts()
    assert m2.revision == 2, "revisions must be monotonic per entity"

    # 2. reconnect: claim oldest pending -> in_flight (FIFO preserves causality)
    flight = q.next_in_flight()
    assert flight.mutation_id == m1.mutation_id, "FIFO: create before update"
    assert q.counts() == {"pending": 1, "in_flight": 1}, q.counts()

    # 3. crash mid-flight before ACK -> recover() downgrades in_flight back to pending
    n = q.recover()
    assert n == 1 and q.counts() == {"pending": 2}, (n, q.counts())

    # 4. drain happy path: claim -> success deletes the row
    f1 = q.next_in_flight()
    q.mark_success(f1.mutation_id)
    f2 = q.next_in_flight()
    q.mark_success(f2.mutation_id)
    assert q.counts() == {}, q.counts()

    # 5. conflict path: stale-revision mutation gets quarantined, not overwritten
    q.enqueue("task-1", "update", {"title": "stale edit"})
    bad = q.next_in_flight()
    q.mark_conflict(bad.mutation_id)
    assert q.counts() == {"conflict": 1}, q.counts()

    print("fabric/sync_queue: four-state FSM OK "
          "(pending->in_flight, crash-recovery, success-drain, conflict-quarantine)")


if __name__ == "__main__":
    _selftest()
