"""
HARP — Hardware-Aware Routing Platform
cloud/nim_backend.py  ·  CCE owns this  ·  MIT

The CLOUD half of the one-call swap. Implements shared/harp_contract.Backend
over any OpenAI-compatible NIM endpoint (build.nvidia.com hosted NIM, a local
NIM container, or trtllm-serve). CTO's Router treats this identically to CEE's
QNNBackend — it negotiates Capability, calls infer()/profile(), never imports
this class. That invariant is the whole point of the contract.

Grounding (see project research):
  - NIM exposes OpenAI /v1/chat/completions with SSE stream=true  -> NeMo Planner Architecture doc §"NIM Microservices"
  - TTFT is measured at FIRST CONTENT token, empty frames disregarded -> NVIDIA Optimization Strategy doc §"Core Mathematical Metrics" (GenAI-Perf rule)
  - tok/s + TTFT carried in Metrics so the NVIDIA before/after delta is real, not simulated

UNVERIFIED until deep-research pass (do not hardcode): exact Nemotron NIM
model_id strings. Pass model_id via InferRequest; default is a placeholder.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

from shared.harp_contract import (
    Backend,
    Capability,
    InferRequest,
    Metrics,
    Modality,
    Tier,
)

# ---------------------------------------------------------------- config

@dataclass
class NIMConfig:
    base_url: str = os.getenv("HARP_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    api_key: str | None = os.getenv("HARP_NIM_API_KEY")  # build.nvidia.com key; None for local NIM
    # PLACEHOLDER — replace post-verification with the real catalog id.
    default_model: str = os.getenv("HARP_NIM_MODEL", "nvidia/nemotron-planner-PLACEHOLDER")
    request_timeout_s: float = 120.0
    connect_timeout_s: float = 10.0
    # Advertised ceilings — used only for capability negotiation, not enforcement.
    max_context: int = 128_000
    ram_gb: float = 80.0


# ---------------------------------------------------------------- backend

class NIMBackend(Backend):
    """Cloud backend: OpenAI-compatible NIM. Streaming is mandatory (contract)
    so TTFT measured here is a true wall-clock first-token, identical metric
    shape to what CEE reports on QNN. That symmetry is what lets the pitch say
    'same harness, two tiers, here is the delta.'"""

    def __init__(self, cfg: NIMConfig | None = None):
        self.cfg = cfg or NIMConfig()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.cfg.request_timeout_s, connect=self.cfg.connect_timeout_s),
            headers=self._auth_headers(),
        )

    def _auth_headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---- contract method 1: capability negotiation -----------------------
    async def capabilities(self) -> Capability:
        """Cloud promises all modalities + huge context, but NOT offline. The
        Router's offline guard reads exactly this to fail-closed to edge."""
        return Capability(
            backend_id="nim-cloud",
            tier=Tier.CLOUD,
            npu_present=False,
            ram_gb=self.cfg.ram_gb,
            max_context=self.cfg.max_context,
            modalities=(Modality.TEXT, Modality.AUDIO, Modality.VISION),
            offline_capable=False,
            supports_streaming=True,
        )

    # ---- contract method 2: token stream ---------------------------------
    async def infer(self, req: InferRequest) -> AsyncIterator[str]:
        """SSE token stream from /chat/completions. Yields content deltas only;
        the [DONE] sentinel and empty role-frames are dropped so the consumer
        (CTO Router) sees pure content — and so TTFT lands on real text."""
        payload = {
            "model": req.model_id or self.cfg.default_model,
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "stream": True,
            "temperature": 0.1,
        }
        url = f"{self.cfg.base_url}/chat/completions"
        async with self._client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                tok = delta.get("content")
                if tok:                      # drop empty/role-only frames (GenAI-Perf TTFT rule)
                    yield tok

    # ---- contract method 3: profiling (the NVIDIA-scoring metric) --------
    async def profile(self, req: InferRequest) -> Metrics:
        """One real generation, wall-clock instrumented. TTFT = time to first
        CONTENT token. tok/s = output tokens / (last - first). energy/thermal
        are None on cloud by contract (edge-only fields)."""
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        first_tok_t: float | None = None
        n = 0
        last_t = t0
        async for _tok in self.infer(req):
            now = time.perf_counter()
            if ttft_ms is None:
                ttft_ms = (now - t0) * 1000.0
                first_tok_t = now
            n += 1
            last_t = now
        gen_window = (last_t - first_tok_t) if (first_tok_t and n > 1) else None
        tok_s = (n / gen_window) if gen_window and gen_window > 0 else 0.0
        return Metrics(
            backend_id="nim-cloud",
            ttft_ms=ttft_ms if ttft_ms is not None else float("inf"),
            tokens_per_s=tok_s,
            energy_mj_per_tok=None,   # cloud: not measured (contract)
            thermal_c=None,
        )


# ---------------------------------------------------------------- smoke (live or skip)

async def _smoke() -> None:
    """Runs only if HARP_NIM_API_KEY (or a reachable local NIM) is set.
    Proves the contract methods over a real endpoint before CTO wires it in."""
    cfg = NIMConfig()
    if not cfg.api_key and "localhost" not in cfg.base_url and "127.0.0.1" not in cfg.base_url:
        print("SKIP live smoke: set HARP_NIM_API_KEY or point HARP_NIM_BASE_URL at a local NIM.")
        return
    be = NIMBackend(cfg)
    try:
        cap = await be.capabilities()
        print(f"capabilities: {cap.backend_id} tier={cap.tier.value} "
              f"modalities={[m.value for m in cap.modalities]} offline={cap.offline_capable}")
        req = InferRequest(
            messages=[{"role": "user", "content": "Reply with exactly: routing online."}],
            model_id=cfg.default_model,
            max_tokens=32,
        )
        print("infer stream:", end=" ")
        async for t in be.infer(req):
            print(t, end="", flush=True)
        print()
        m = await be.profile(req)
        print(f"profile: ttft={m.ttft_ms:.1f}ms tok/s={m.tokens_per_s:.1f}")
    finally:
        await be.aclose()


if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke())
