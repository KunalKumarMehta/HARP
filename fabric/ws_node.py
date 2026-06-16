"""
HARP — Hardware-Aware Routing Platform
fabric/ws_node.py  ·  Multi-device transport node  ·  MIT

The laptop's dual-role fabric node (DP3 / ARM64 Fabric doc):
  - SERVER: binds 0.0.0.0 so the phone can reach it on the LAN (never localhost).
  - CLIENT: one persistent uplink to the cloud orchestrator.
Both built on `websockets` (aaugustin) — pure-Python-capable, the only WS lib
that survives the missing-win_arm64-wheel reality without a C toolchain.

Resilience contract:
  - Reconnect: exponential backoff + jitter (D0=1s, cap=30s, alpha=0.5).
  - Keep-alive: protocol ping/pong (ping_interval/ping_timeout=20s) kills
    half-open TCP instead of hanging forever.
  - On every (re)connect: queue.recover() downgrades orphaned in_flight -> pending,
    then blind-retransmit. At-least-once delivery; the cloud dedups on mutation_id.
  - Drain is strict FIFO; conflict (404/409/403) quarantines, never overwrites.

CRDTs are deliberately absent — four-state SQLite queue carries the whole story.
"""

from __future__ import annotations

import asyncio
import json
import random

import websockets
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from fabric.sync_queue import OutboxQueue
from shared.harp_contract import SyncState


# ---------------------------------------------------------------- reconnect backoff

def backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0,
                  alpha: float = 0.5) -> float:
    """T(n) = min(cap, base*2^n) + U(0, alpha*capped). Jitter disperses the
    thundering herd so N edge nodes don't synchronise their reconnect storm."""
    capped = min(cap, base * (2 ** attempt))
    return capped + random.uniform(0.0, alpha * capped)


# ---------------------------------------------------------------- the node

class FabricNode:
    def __init__(self, queue: OutboxQueue, cloud_uri: str, *,
                 local_host: str = "0.0.0.0", local_port: int = 8765,
                 ping_interval: float = 20.0, ping_timeout: float = 20.0):
        self.queue = queue
        self.cloud_uri = cloud_uri
        self.local_host = local_host
        self.local_port = local_port
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

    # ---- CLIENT: uplink to cloud, with reconnect + drain -------------------

    async def run_uplink(self, *, drain_then_stop: bool = False,
                         max_attempts: int | None = None) -> None:
        attempt = 0
        while True:
            try:
                async with connect(self.cloud_uri,
                                   ping_interval=self.ping_interval,
                                   ping_timeout=self.ping_timeout) as ws:
                    attempt = 0                       # reset only on a clean handshake
                    recovered = self.queue.recover()  # orphaned in_flight -> pending
                    if recovered:
                        print(f"[uplink] recovered {recovered} in_flight -> pending")
                    drained = await self._drain(ws)
                    if drain_then_stop and drained:
                        return
            except (websockets.ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                attempt += 1
                if max_attempts is not None and attempt > max_attempts:
                    raise
                delay = backoff_delay(attempt)
                print(f"[uplink] drop ({type(e).__name__}); retry #{attempt} in {delay:.2f}s")
                await asyncio.sleep(min(delay, 0.2) if max_attempts else delay)  # test: compress sleeps

    async def _drain(self, ws) -> bool:
        """Strict-FIFO send-and-await-ACK. Returns True once the queue holds no
        more pending work (conflicts are terminal and don't block 'drained')."""
        while True:
            m = self.queue.next_in_flight()           # pending -> in_flight (idempotency lock)
            if m is None:
                return self._counts_clear()
            await ws.send(json.dumps({
                "mutation_id": m.mutation_id, "entity_id": m.entity_id,
                "op": m.op, "payload": m.payload, "revision": m.revision,
            }))
            ack = json.loads(await ws.recv())          # raises ConnectionClosed on drop
            if ack.get("mutation_id") != m.mutation_id:
                continue                               # out-of-band frame; keep waiting
            if ack.get("status") == SyncState.SUCCESS.value:
                self.queue.mark_success(m.mutation_id)
            else:
                self.queue.mark_conflict(m.mutation_id)
                print(f"[uplink] {m.entity_id} -> CONFLICT, quarantined")

    def _counts_clear(self) -> bool:
        c = self.queue.counts()
        return c.get(SyncState.PENDING.value, 0) == 0 and \
               c.get(SyncState.IN_FLIGHT.value, 0) == 0

    # ---- SERVER: local listener for the phone -------------------------------

    async def serve_local(self, ready: asyncio.Event | None = None):
        async def handler(ws):
            async for raw in ws:                       # phone pushes mutation intents
                msg = json.loads(raw)
                m = self.queue.enqueue(msg["entity_id"], msg["op"], msg["payload"])
                await ws.send(json.dumps({"mutation_id": m.mutation_id,
                                          "status": "queued"}))
        async with serve(handler, self.local_host, self.local_port) as server:
            if ready:
                ready.set()
            await server.serve_forever()
