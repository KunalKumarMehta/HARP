"""
HARP — tests/test_npu_single_flight.py  ·  MIT

The NPU single-context binary is SINGLE-LANE. This test wraps the local backend so
it RAISES on concurrent entry (simulating the FastRPC `error: 0x1` memory-map
exhaustion). It then fires N=4 concurrent local requests at the endpoint and
asserts:

  - no concurrency error ever (the asyncio single-flight lock serialises);
  - with escalate available, >=1 request is shed to cloud (harp_route.shed==True);
  - with escalate disabled + offline, all 4 serialise cleanly on the NPU lane —
    queued, none dropped, never two in flight.

Off-device: the spy backend replaces genie entirely, so no NPU is needed.
"""
from __future__ import annotations

import asyncio
import sys

from httpx import ASGITransport, AsyncClient

from shared.harp_contract import (
    Backend, Capability, InferRequest, Metrics, Modality, Tier,
)
from serve.openai_endpoint import make_app


class SingleLaneSpy(Backend):
    """Local backend that blows up if two infers overlap — exactly the FastRPC 0x1
    failure the single-flight lock exists to prevent."""

    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.entered = 0

    async def capabilities(self) -> Capability:
        return Capability("spy-edge", Tier.EDGE, npu_present=True, ram_gb=16,
                          max_context=4096, modalities=(Modality.TEXT,),
                          offline_capable=True, supports_streaming=True)

    async def infer(self, req: InferRequest, *, thinking: bool = True,
                    tools: list | None = None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered += 1
        if self.active > 1:
            self.active -= 1
            raise RuntimeError("fastrpc memory map for fd 42 failed with error: 0x1")
        try:
            for w in ("edge", "answer"):
                await asyncio.sleep(self.delay)
                yield w + " "
        finally:
            self.active -= 1

    async def profile(self, req: InferRequest) -> Metrics:
        return Metrics("spy-edge", ttft_ms=1.0, tokens_per_s=30.0)


class FastCloud(Backend):
    async def capabilities(self) -> Capability:
        return Capability("spy-cloud", Tier.CLOUD, npu_present=False, ram_gb=80,
                          max_context=128_000, modalities=(Modality.TEXT,),
                          offline_capable=False, supports_streaming=True)

    async def infer(self, req: InferRequest):
        for w in ("cloud", "answer"):
            yield w + " "

    async def profile(self, req: InferRequest) -> Metrics:
        return Metrics("spy-cloud", ttft_ms=1.0, tokens_per_s=120.0)


def _payload() -> dict:
    return {"model": "harp-edge",
            "messages": [{"role": "user", "content": "summarize this"}]}


async def _fire(app, n: int = 4) -> list[dict]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://harp") as client:
        results = await asyncio.gather(
            *[client.post("/v1/chat/completions", json=_payload()) for _ in range(n)])
    for r in results:
        assert r.status_code == 200, f"concurrency error leaked: {r.status_code} {r.text}"
    return [r.json() for r in results]


async def _with_escalate() -> None:
    spy = SingleLaneSpy()
    app = make_app(local_backend=spy, escalate_backend=FastCloud(),
                   ttft_budget_s=2.0, exec_est_s=3.0)
    bodies = await _fire(app, 4)
    assert spy.max_active == 1, f"two infers overlapped: max_active={spy.max_active}"
    shed = [b["harp_route"]["shed"] for b in bodies]
    assert sum(1 for s in shed if s) >= 1, f"expected >=1 shed to cloud, got {shed}"
    # The shed ones must actually be on cloud.
    for b in bodies:
        if b["harp_route"]["shed"]:
            assert b["harp_route"]["tier"] == "cloud", b["harp_route"]


async def _offline_serializes() -> None:
    spy = SingleLaneSpy()
    app = make_app(local_backend=spy, escalate_disabled=True,
                   ttft_budget_s=2.0, exec_est_s=3.0)
    bodies = await _fire(app, 4)
    assert spy.max_active == 1, f"two infers overlapped offline: {spy.max_active}"
    assert spy.entered == 4, f"all 4 must run locally (queued), got {spy.entered}"
    assert all(b["harp_route"]["shed"] is False for b in bodies), "offline must not shed"
    assert all(b["choices"][0]["message"]["content"] for b in bodies), "none dropped"


def test_single_flight_sheds_when_escalate_available() -> None:
    asyncio.run(_with_escalate())


def test_single_flight_serializes_when_offline() -> None:
    asyncio.run(_offline_serializes())


def _main() -> int:
    test_single_flight_sheds_when_escalate_available()
    print("  OK sheds to cloud under contention, no concurrency error")
    test_single_flight_serializes_when_offline()
    print("  OK serializes all 4 offline, none dropped")
    print("test_npu_single_flight: passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
