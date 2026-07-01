"""
HARP — hardware-aware edge↔cloud routing
cloud/nim_backend.py  ·  MIT

Cloud half of the one-call swap. Implements shared/harp_contract.Backend over
any OpenAI-compatible NIM endpoint. Router treats this identically to
QNNBackend — negotiates Capability, calls infer()/profile(), never
imports this class.

NIM API specification (NVIDIA Nemotron):
  - base: https://integrate.api.nvidia.com/v1 ; auth: Bearer $HARP_NIM_API_KEY
  - reasoning models stream CoT on delta.reasoning_content, answer on delta.content
    Both are handled; TTFT lands on the FIRST token of EITHER stream (true
    first-generation instant).
  - reasoning control via extra_body.chat_template_kwargs.enable_thinking +
    reasoning_budget / min_thinking_tokens
No hardcoded model strings: model resolved by Role through model_registry.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

from shared.harp_contract import (
    Backend, Capability, InferRequest, Metrics, Modality, Tier,
)
from cloud.model_registry import Role, ModelSpec, resolve


@dataclass
class NIMConfig:
    base_url: str = os.getenv("HARP_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    api_key: str | None = os.getenv("HARP_NIM_API_KEY")     # None for local NIM container
    # Default cloud role this backend serves when a request gives no explicit model.
    default_role: Role = Role.MANAGER_PRAGMATIC
    enable_thinking: bool = False        # planner steps flip this True
    reasoning_budget: int = 8192
    min_thinking_tokens: int = 0
    surface_reasoning: bool = False      # if True, infer() also yields CoT tokens
    request_timeout_s: float = 120.0
    connect_timeout_s: float = 10.0
    ram_gb: float = 80.0


class NIMBackend(Backend):
    def __init__(self, cfg: NIMConfig | None = None, role: Role | None = None):
        self.cfg = cfg or NIMConfig()
        self.role = role or self.cfg.default_role
        self.spec: ModelSpec = resolve(self.role)
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

    # ---- contract 1: capability negotiation ------------------------------
    async def capabilities(self) -> Capability:
        """Cloud: all modalities, large ctx, NOT offline. Router's offline guard
        reads offline_capable=False here and fails closed to edge."""
        mods = ((Modality.TEXT, Modality.AUDIO, Modality.VISION)
                if self.spec.multimodal else (Modality.TEXT,))
        return Capability(
            backend_id=f"nim-cloud:{self.role.value}",
            tier=Tier.CLOUD, npu_present=False, ram_gb=self.cfg.ram_gb,
            max_context=self.spec.context_window or 0,
            modalities=mods, offline_capable=False, supports_streaming=True,
        )

    def _payload(self, req: InferRequest) -> dict:
        model = req.model_id or self.spec.model_id
        body: dict = {
            "model": model, "messages": req.messages,
            "max_tokens": req.max_tokens, "stream": True, "temperature": 0.1,
        }
        if self.spec.reasoning:
            body["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self.cfg.enable_thinking},
                "reasoning_budget": self.cfg.reasoning_budget,
                "min_thinking_tokens": self.cfg.min_thinking_tokens,
            }
        return body

    @staticmethod
    def _deltas(chunk: dict) -> tuple[str | None, str | None]:
        d = (chunk.get("choices") or [{}])[0].get("delta", {})
        return d.get("content"), d.get("reasoning_content")

    # ---- contract 2: token stream ----------------------------------------
    async def infer(self, req: InferRequest) -> AsyncIterator[str]:
        """Yields answer content. If surface_reasoning, CoT tokens are yielded
        wrapped so the consumer can distinguish them. Empty/role frames dropped."""
        url = f"{self.cfg.base_url}/chat/completions"
        async with self._client.stream("POST", url, json=self._payload(req)) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                content, reasoning = self._deltas(chunk)
                if reasoning and self.cfg.surface_reasoning:
                    yield f"\u2039think\u203a{reasoning}"   # ‹think›… marker, opt-in
                if content:
                    yield content

    # ---- contract 3: profiling (TTFT / tok-s) ---------------
    async def profile(self, req: InferRequest) -> Metrics:
        """TTFT = wall-clock to first token of EITHER stream (generation truly
        began). tok/s counts ANSWER (content) tokens over the answer window —
        reasoning tokens excluded from tok/s so the throughput number is the
        user-visible rate, not inflated by CoT. energy/thermal None on cloud."""
        url = f"{self.cfg.base_url}/chat/completions"
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        first_answer_t: float | None = None
        last_answer_t = t0
        answer_toks = 0
        async with self._client.stream("POST", url, json=self._payload(req)) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                content, reasoning = self._deltas(chunk)
                now = time.perf_counter()
                if ttft_ms is None and (content or reasoning):
                    ttft_ms = (now - t0) * 1000.0          # first generation instant
                if content:
                    if first_answer_t is None:
                        first_answer_t = now
                    answer_toks += 1
                    last_answer_t = now
        window = (last_answer_t - first_answer_t) if (first_answer_t and answer_toks > 1) else None
        tok_s = (answer_toks / window) if window and window > 0 else 0.0
        return Metrics(
            backend_id=f"nim-cloud:{self.role.value}",
            ttft_ms=ttft_ms if ttft_ms is not None else float("inf"),
            tokens_per_s=tok_s, energy_mj_per_tok=None, thermal_c=None,
        )


async def _smoke() -> None:
    cfg = NIMConfig()
    if not cfg.api_key and "localhost" not in cfg.base_url and "127.0.0.1" not in cfg.base_url:
        print("SKIP live smoke: set HARP_NIM_API_KEY or point HARP_NIM_BASE_URL at a local NIM.")
        print(f"resolved role={NIMBackend(cfg).role.value} model={NIMBackend(cfg).spec.model_id}")
        return
    be = NIMBackend(cfg)
    try:
        cap = await be.capabilities()
        print(f"cap: {cap.backend_id} mods={[m.value for m in cap.modalities]} ctx={cap.max_context} offline={cap.offline_capable}")
        req = InferRequest(messages=[{"role": "user", "content": "Reply exactly: routing online."}],
                           model_id="", max_tokens=32)
        print("infer:", end=" ")
        async for t in be.infer(req):
            print(t, end="", flush=True)
        m = await be.profile(req)
        print(f"\nprofile: ttft={m.ttft_ms:.1f}ms tok/s={m.tokens_per_s:.1f}")
    finally:
        await be.aclose()


if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke())
