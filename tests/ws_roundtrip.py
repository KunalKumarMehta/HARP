"""
HARP — tests/ws_roundtrip.py  ·  Fabric transport proof (CTO test mandate)  ·  MIT

Real websockets over loopback TCP. Proves:
  A. happy path drains FIFO; a server-rejected mutation is quarantined (conflict).
  B. a mid-flight drop (server closes before ACK) triggers reconnect, recover()
     downgrades the orphaned in_flight -> pending, and the node redelivers. The
     cloud sees the mutation_id twice (at-least-once) and dedups it (idempotency).
"""

from __future__ import annotations

import asyncio
import json
import sys

from websockets.asyncio.server import serve

from fabric.sync_queue import OutboxQueue
from fabric.ws_node import FabricNode


class MockCloud:
    def __init__(self, conflict_entities=(), drop_first=False):
        self.seen: dict[str, str] = {}      # mutation_id -> status (idempotency cache)
        self.received: list[str] = []       # every frame, incl. redeliveries
        self.conn_count = 0
        self.conflict_entities = set(conflict_entities)
        self.drop_first = drop_first

    async def handler(self, ws):
        self.conn_count += 1
        drop = self.drop_first and self.conn_count == 1
        async for raw in ws:
            msg = json.loads(raw)
            mid = msg["mutation_id"]
            self.received.append(mid)
            if drop:
                await ws.close()            # sever before ACK -> client recv() raises
                return
            if mid in self.seen:            # idempotent: re-ACK, no double side effect
                status = self.seen[mid]
            else:
                status = "conflict" if msg["entity_id"] in self.conflict_entities else "success"
                self.seen[mid] = status
            await ws.send(json.dumps({"mutation_id": mid, "status": status}))


async def _scenario(cloud: MockCloud, port: int, enqueue: list[tuple]) -> OutboxQueue:
    q = OutboxQueue()
    for entity, op, payload in enqueue:
        q.enqueue(entity, op, payload)

    server = await serve(cloud.handler, "127.0.0.1", port)
    node = FabricNode(q, f"ws://127.0.0.1:{port}")
    try:
        await asyncio.wait_for(
            node.run_uplink(drain_then_stop=True, max_attempts=8), timeout=10)
    finally:
        server.close()
        await server.wait_closed()
    return q


async def main() -> int:
    fails: list[str] = []

    # ---- Scenario A: drain + conflict quarantine
    cloudA = MockCloud(conflict_entities={"task-stale"})
    qA = await _scenario(cloudA, 8799, [
        ("task-1", "create", {"t": "transcribe"}),
        ("task-1", "update", {"t": "transcribe+summarize"}),
        ("task-stale", "update", {"t": "stale edit"}),     # server will reject
    ])
    cA = qA.counts()
    print(f"A received={cloudA.received}  final_queue={cA}")
    if cA.get("pending", 0) or cA.get("in_flight", 0):
        fails.append("A: pending/in_flight should be drained")
    if cA.get("conflict", 0) != 1:
        fails.append("A: stale mutation should be quarantined as conflict")
    if cloudA.received[0:2] != [cloudA.received[0], cloudA.received[1]]:  # FIFO sanity
        fails.append("A: FIFO order violated")

    # ---- Scenario B: mid-flight drop -> reconnect -> recover -> idempotent resend
    cloudB = MockCloud(drop_first=True)
    qB = await _scenario(cloudB, 8800, [
        ("task-A", "create", {"t": "first"}),
        ("task-B", "create", {"t": "second"}),
    ])
    cB = qB.counts()
    redelivered = len(cloudB.received) > len(set(cloudB.received))
    print(f"B conns={cloudB.conn_count} received={cloudB.received} "
          f"unique={sorted(set(cloudB.received))} final_queue={cB}")
    if cB:
        fails.append("B: queue must be fully drained after recovery")
    if cloudB.conn_count < 2:
        fails.append("B: a reconnect should have occurred")
    if not redelivered:
        fails.append("B: orphaned mutation should have been redelivered (at-least-once)")
    if len(cloudB.seen) != 2:
        fails.append("B: cloud should have committed exactly 2 unique mutations (idempotent)")

    if fails:
        print("\nFAIL:\n  " + "\n  ".join(fails))
        return 1
    print("\nws_roundtrip OK: FIFO drain, conflict quarantine, drop->reconnect->"
          "recover->idempotent redelivery")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
