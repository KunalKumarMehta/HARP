"""
HARP — Hardware-Aware Routing Platform
shared/harp_contract.py  ·  CONTRACT FREEZE v0  ·  MIT

This module is the integration spine. CEE implements `Backend` over QNN.
CCE implements `Backend` over NIM. CAIO's router consumes `PlanGraph` and
selects a backend through `capabilities()` — never by importing a concrete
backend. Freeze this file at spike start; changes require CTO sign-off.

Design grounding:
  - Unified inference contract + capability negotiation  -> AI Runtime Abstraction doc
  - OpenAI-compatible message payload as the boundary lingua franca -> Nexa shim pattern
  - Plan-graph DAG, JSON wire (Protobuf = scale roadmap, NOT built) -> Agentic Systems doc
  - Four-state offline machine (pending/in_flight/success/conflict) -> Agentic Systems doc
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


# ---------------------------------------------------------------- enums

class Tier(str, Enum):
    EDGE = "edge"      # QNN / Snapdragon NPU
    CLOUD = "cloud"    # NIM / Nemotron


class Modality(str, Enum):
    TEXT = "text"
    AUDIO = "audio"    # ASR specialist
    VISION = "vision"


class RouteDecision(str, Enum):
    LOCAL = "local"
    ESCALATE = "escalate"


class SyncState(str, Enum):
    PENDING = "pending"      # executed locally, queued
    IN_FLIGHT = "in_flight"  # reconnected, syncing to cloud
    SUCCESS = "success"      # accepted -> drop from queue
    CONFLICT = "conflict"    # 404/409/403 -> suspend, do NOT silently overwrite


# ---------------------------------------------------------------- contract types

@dataclass(frozen=True)
class Capability:
    """What a backend can promise. The router negotiates against this — it is
    the only thing that decides edge-vs-cloud fallback at the hardware layer."""
    backend_id: str
    tier: Tier
    npu_present: bool
    ram_gb: float
    max_context: int
    modalities: tuple[Modality, ...]
    offline_capable: bool
    supports_streaming: bool


@dataclass
class InferRequest:
    """OpenAI-shaped payload. Identical struct hits QNN or NIM — only the base
    backend changes, never the caller (Nexa shim invariant)."""
    messages: list[dict]            # [{"role": "user", "content": "..."}]
    model_id: str
    modality: Modality = Modality.TEXT
    max_tokens: int = 512
    stream: bool = True


@dataclass
class Metrics:
    """CEE's 40% Qualcomm score lives here. profile() must populate ttft/tok-s
    on every backend; energy/thermal are edge-only and may be None on cloud."""
    backend_id: str
    ttft_ms: float
    tokens_per_s: float
    energy_mj_per_tok: float | None = None
    thermal_c: float | None = None


class Backend(ABC):
    """THE contract. Two real implementations: QNNBackend (CEE), NIMBackend (CCE).
    Apple/AMD are stubbed behind this same interface — narrative only, not built."""

    @abstractmethod
    async def capabilities(self) -> Capability: ...

    @abstractmethod
    def infer(self, req: InferRequest) -> AsyncIterator[str]:
        """Token stream. Streaming is mandatory so TTFT is real, not simulated."""
        ...

    @abstractmethod
    async def profile(self, req: InferRequest) -> Metrics: ...


# ---------------------------------------------------------------- plan graph (cloud -> edge)

@dataclass
class PlanStep:
    step_id: str
    modality: Modality
    decision: RouteDecision          # router-assigned tier for this step
    model_id: str
    prompt: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class PlanGraph:
    """Compressed DAG the cloud planner emits to the edge executor. JSON on the
    wire. We ship plans, not activations — model-level routing, not layer split."""
    plan_id: str
    steps: list[PlanStep]

    def topo_order(self) -> list[PlanStep]:
        done: set[str] = set()
        ordered: list[PlanStep] = []
        pending = {s.step_id: s for s in self.steps}
        while pending:
            ready = [s for s in pending.values()
                     if all(d in done for d in s.depends_on)]
            if not ready:
                raise ValueError("cycle in plan graph")
            for s in ready:
                ordered.append(s)
                done.add(s.step_id)
                del pending[s.step_id]
        return ordered


# ---------------------------------------------------------------- router (capability-negotiated dispatch)

class Router:
    """Proves the one-call swap: caller hits dispatch(), router picks the backend
    by negotiating capabilities + applying the offline/hardware guard. CAIO's
    learned policy slots in at `_decide` later; this is the deterministic floor."""

    def __init__(self, edge: Backend, cloud: Backend, online: bool = True):
        self.edge = edge
        self.cloud = cloud
        self.online = online

    async def dispatch(self, step: PlanStep) -> AsyncIterator[str]:
        backend = await self._select(step)
        req = InferRequest(
            messages=[{"role": "user", "content": step.prompt}],
            model_id=step.model_id,
            modality=step.modality,
        )
        async for tok in backend.infer(req):
            yield tok

    async def _select(self, step: PlanStep) -> Backend:
        edge_cap = await self.edge.capabilities()
        # Hardware-capability + network-state guards (AI Runtime doc, routing vectors 1 & 2)
        if not self.online:
            return self.edge                      # offline: escalate is disabled, fail to local
        if step.decision == RouteDecision.ESCALATE:
            return self.cloud
        if not edge_cap.npu_present or step.modality not in edge_cap.modalities:
            return self.cloud                      # capability fallback prevents OOM/crash
        return self.edge


# ---------------------------------------------------------------- mock backends (unblock parallel work TODAY)

class _MockBackend(Backend):
    def __init__(self, cap: Capability, ttft_ms: float, tok_s: float):
        self._cap, self._ttft, self._tok_s = cap, ttft_ms, tok_s

    async def capabilities(self) -> Capability:
        return self._cap

    async def infer(self, req: InferRequest) -> AsyncIterator[str]:
        await asyncio.sleep(self._ttft / 1000)
        for w in f"[{self._cap.backend_id}] handled: {req.messages[-1]['content']}".split():
            await asyncio.sleep(1 / self._tok_s)
            yield w + " "

    async def profile(self, req: InferRequest) -> Metrics:
        t0 = time.perf_counter()
        toks = [t async for t in self.infer(req)]
        dt = time.perf_counter() - t0
        return Metrics(
            backend_id=self._cap.backend_id,
            ttft_ms=self._ttft,
            tokens_per_s=len(toks) / dt if dt else 0.0,
            energy_mj_per_tok=0.8 if self._cap.tier == Tier.EDGE else None,
            thermal_c=41.0 if self._cap.tier == Tier.EDGE else None,
        )


def mock_edge() -> Backend:
    return _MockBackend(
        Capability("qnn-mock", Tier.EDGE, npu_present=True, ram_gb=16,
                   max_context=4096, modalities=(Modality.TEXT, Modality.AUDIO),
                   offline_capable=True, supports_streaming=True),
        ttft_ms=180, tok_s=30.3)            # Qwen3-4B on X Elite, per capability map


def mock_cloud() -> Backend:
    return _MockBackend(
        Capability("nim-mock", Tier.CLOUD, npu_present=False, ram_gb=80,
                   max_context=128_000, modalities=tuple(Modality),
                   offline_capable=False, supports_streaming=True),
        ttft_ms=300, tok_s=120)


# ---------------------------------------------------------------- smoke test (CTO test mandate #1)

async def _smoke() -> None:
    plan = PlanGraph("p0", [
        PlanStep("s1", Modality.AUDIO, RouteDecision.LOCAL, "whisper-base", "transcribe clip"),
        PlanStep("s2", Modality.TEXT, RouteDecision.LOCAL, "qwen3-4b", "summarize that", depends_on=["s1"]),
        PlanStep("s3", Modality.TEXT, RouteDecision.ESCALATE, "nemotron", "deep-reason the plan", depends_on=["s2"]),
    ])

    print("== ONLINE: router swaps QNN<->NIM behind one call ==")
    r = Router(mock_edge(), mock_cloud(), online=True)
    for step in plan.topo_order():
        out = "".join([t async for t in r.dispatch(step)])
        print(f"  {step.step_id} [{step.decision.value:8}] -> {out.strip()}")

    print("\n== OFFLINE: escalate is disabled, fails closed to edge ==")
    r_off = Router(mock_edge(), mock_cloud(), online=False)
    out = "".join([t async for t in r_off.dispatch(plan.steps[2])])  # an ESCALATE step
    print(f"  s3 forced local -> {out.strip()}")

    print("\n== profile() carries the Qualcomm 40% metrics ==")
    m = await mock_edge().profile(InferRequest([{"role": "user", "content": "hi"}], "qwen3-4b"))
    print(f"  {m.backend_id}: ttft={m.ttft_ms}ms  tok/s={m.tokens_per_s:.1f}  "
          f"energy={m.energy_mj_per_tok}mJ/tok  thermal={m.thermal_c}C")


if __name__ == "__main__":
    asyncio.run(_smoke())