"""
HARP — serve/openai_endpoint.py  ·  MIT

The integration seam: HARP exposed to any agent framework (Hermes, OpenClaw, …)
as a hardware-aware, OpenAI-compatible local model. The agent points at this
endpoint and gets the NPU lane for free — no HARP knowledge required.

This server IS the origin model (not a proxy in front of another model). It emits
standard `chat.completion(.chunk)` + standard `tool_calls`; it preserves no
framework-proprietary events.

Three behaviours encode verified hardware facts:

  1. NPU single-flight. The single-context binary is SINGLE-LANE: two concurrent
     infers against one context binary exhaust the FastRPC memory map
     (`fastrpc memory map for fd ... failed with error: 0x1`), fail the SMMU
     domain, "Could not allocate persistent weights buffer!", and crash — or
     silently collide in VTCM. So exactly one local infer is in flight at a time,
     guarded by a per-app asyncio.Lock.

  2. Overflow shed. NPU TTFT degrades O(N) under queue (TTFT_k ≈ TTFT_base +
     Σ T_exec(i<k)). When the lane is busy and projected wait exceeds the TTFT
     budget AND escalate is available, shed the request to the cloud lane instead
     of queuing. Offline (no escalate) we queue on the NPU — correctness > latency.

  3. Tools force thinking off. GenieAPIService returns real OpenAI tool_calls, but
     CoT must be disabled when a request carries tools. local lane + tools →
     thinking=False on the local infer.

Routing: "harp-edge" pins LOCAL, "harp-cloud" pins ESCALATE, "harp-auto" (default)
defers to the calibrated PolicyRouter — including the contention axis.

No global mutable state: all runtime state lives on `app.state.harp`.
Config via env: HARP_ENDPOINT_PORT (8765), HARP_TTFT_BUDGET_S (2.0),
HARP_NPU_EXEC_EST_S (3.0), HARP_ESCALATE_DISABLED (bool), HARP_LOCAL_MODEL (qwen3-4b).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from shared.harp_contract import Backend, InferRequest, Modality, RouteDecision, Tier
from router.router_policy import RoutingFeatures, RoutingPolicy


# ---------------------------------------------------------------- config

def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------- model selector

_MODEL_PINS = {
    "harp-edge": RouteDecision.LOCAL,      # pin local (NPU, single-stream safe)
    "harp-cloud": RouteDecision.ESCALATE,  # pin cloud (escalate lane)
    "harp-auto": RouteDecision.AUTO,       # defer to the router
}
_MODELS = tuple(_MODEL_PINS)
DEFAULT_MODEL = "harp-auto"


# ---------------------------------------------------------------- tool-call parsing
# Qwen3 / Genie emit tool calls as <tool_call>{json}</tool_call> in the text
# stream. The frozen Backend.infer() yields plain str tokens, so the structured
# tool_calls object is reconstructed here — one parser for the real NPU output and
# the off-device stub alike.

_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _extract_tool_calls(text: str) -> list[dict]:
    calls: list[dict] = []
    for i, m in enumerate(_TOOL_RE.finditer(text)):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args)
        calls.append({
            "id": f"call_{i}_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": obj.get("name", ""), "arguments": args},
        })
    return calls


def _strip_tool_calls(text: str) -> str:
    return _TOOL_RE.sub("", text).strip()


# ---------------------------------------------------------------- default policy
# Same synthetic calibration the e2e smoke uses, so the endpoint's AUTO gate is
# the real conformal gate, not a stub.

def _default_policy() -> RoutingPolicy:
    cal_u = [i / 200.0 for i in range(200)]
    cal_err = [1 if (i % 100) / 100.0 < cal_u[i] else 0 for i in range(200)]
    return RoutingPolicy().calibrate(cal_u, cal_err)


# ---------------------------------------------------------------- runtime state

@dataclass
class EndpointState:
    """All mutable runtime state. Lives on app.state — never module-global."""
    local: Backend | None
    escalate: Backend | None
    policy: RoutingPolicy
    ttft_budget_s: float
    exec_est_s: float
    escalate_disabled: bool
    local_model_id: str
    npu_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    committed_local: int = 0      # infers committed to the NPU lane (incl. in-flight)
    inflight: bool = False        # an NPU infer is currently running

    # --- escalate availability == "online" (the only network signal we have) ---
    @property
    def escalate_available(self) -> bool:
        return self.escalate is not None and not self.escalate_disabled

    # --- contention reads (SYNCHRONOUS — no await between busy() and commit_local) ---
    def busy(self) -> bool:
        return self.committed_local > 0

    def projected_wait_s(self) -> float:
        # committed_local already counts the in-flight infer; O(N) TTFT growth.
        return self.committed_local * self.exec_est_s

    def commit_local(self) -> None:
        self.committed_local += 1

    async def local_stream(self, req: InferRequest, *, thinking: bool,
                           tools: list | None) -> AsyncIterator[str]:
        """Single-flight NPU lane. The lock guarantees exactly one in-flight local
        infer (FastRPC 0x1 / VTCM-collision guard). Caller MUST have called
        commit_local() synchronously before awaiting this (closes the shed race)."""
        async with self.npu_lock:
            self.inflight = True
            try:
                async for tok in _call_local(self.local, req, thinking, tools):
                    yield tok
            finally:
                self.inflight = False
                self.committed_local -= 1


async def _call_local(local: Backend, req: InferRequest, thinking: bool,
                      tools: list | None) -> AsyncIterator[str]:
    """Pass thinking/tools to a HARP local backend; fall back to the bare contract
    signature for any injected Backend that doesn't accept them. Never branches on
    concrete type — duck-typed, so the endpoint stays backend-agnostic."""
    try:
        gen = local.infer(req, thinking=thinking, tools=tools)
    except TypeError:
        gen = local.infer(req)
    async for tok in gen:
        yield tok


# ---------------------------------------------------------------- routing

@dataclass
class _Route:
    tier: Tier
    reason: str
    shed: bool
    npu_inflight: bool

    def to_dict(self) -> dict:
        return {"tier": self.tier.value, "reason": self.reason,
                "npu_inflight": self.npu_inflight, "shed": self.shed}

    def header(self) -> str:
        return f"{self.tier.value}:{self.reason}{':shed' if self.shed else ''}"


def _resolve_route(state: EndpointState, model: str, query: str, tools: list | None,
                   modality: Modality) -> _Route:
    """Pin or policy -> tier, then the operational single-flight shed. Returns the
    final lane. Synchronous: the busy()/commit_local() pair must not be split by an
    await, so the whole decision runs here and the caller commits immediately."""
    pin = _MODEL_PINS.get(model, RouteDecision.AUTO)

    if pin == RouteDecision.ESCALATE and state.escalate_available:
        return _Route(Tier.CLOUD, "model_pin", shed=False, npu_inflight=state.busy())
    if pin == RouteDecision.ESCALATE:        # pinned cloud but offline -> fail to local
        tier = Tier.EDGE
        reason = "pin_cloud_offline_fallback"
    elif pin == RouteDecision.LOCAL:
        tier, reason = Tier.EDGE, "model_pin"
    else:
        # AUTO: the calibrated PolicyRouter, contention axis included.
        f = RoutingFeatures(
            query=query, modality=modality, online=state.escalate_available,
            npu_present=True, edge_modalities=(Modality.TEXT, Modality.AUDIO),
            edge_max_context=4096, approx_tokens=max(1, len(query) // 4),
            npu_inflight=state.busy(), npu_queue_depth=state.committed_local,
            tools_present=bool(tools), offline=not state.escalate_available,
        )
        verdict = state.policy.decide(f)
        tier = Tier.CLOUD if verdict.decision == RouteDecision.ESCALATE else Tier.EDGE
        reason = verdict.reason

    # Operational overflow shed: even after routing says LOCAL, if the lane is busy
    # NOW and projected wait blows the TTFT budget and escalate is available, shed.
    if tier == Tier.EDGE and state.escalate_available and state.busy() \
            and state.projected_wait_s() > state.ttft_budget_s:
        return _Route(Tier.CLOUD, "overflow_shed", shed=True, npu_inflight=True)

    if tier == Tier.EDGE:
        return _Route(Tier.EDGE, reason, shed=False, npu_inflight=state.busy())
    return _Route(Tier.CLOUD, reason, shed=False, npu_inflight=state.busy())


# ---------------------------------------------------------------- response assembly

def _now() -> int:
    return int(time.time())


def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def _non_stream_body(cid: str, model: str, text: str, route: _Route) -> dict:
    tool_calls = _extract_tool_calls(text)
    if tool_calls:
        message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
        finish = "stop"
    approx_completion_toks = max(1, len(text.split()))
    return {
        "id": cid,
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": approx_completion_toks,
                  "total_tokens": approx_completion_toks},
        "harp_route": route.to_dict(),
    }


def _sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _chunk(cid: str, model: str, created: int, delta: dict, finish=None) -> dict:
    return {"id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model, "choices": [{"index": 0, "delta": delta,
                                          "finish_reason": finish}]}


async def _stream_response(token_stream: AsyncIterator[str], cid: str, model: str,
                           route: _Route, tools: list | None) -> AsyncIterator[str]:
    created = _now()
    yield _sse(_chunk(cid, model, created, {"role": "assistant"}))

    buffered: list[str] = []
    async for tok in token_stream:
        if tools:
            buffered.append(tok)            # buffer so tool_calls reassemble cleanly
        else:
            yield _sse(_chunk(cid, model, created, {"content": tok}))

    finish = "stop"
    if tools:
        full = "".join(buffered)
        calls = _extract_tool_calls(full)
        if calls:
            delta_calls = [{"index": i, "id": c["id"], "type": "function",
                            "function": c["function"]} for i, c in enumerate(calls)]
            yield _sse(_chunk(cid, model, created, {"tool_calls": delta_calls}))
            finish = "tool_calls"
        else:
            content = _strip_tool_calls(full) or full
            if content:
                yield _sse(_chunk(cid, model, created, {"content": content}))

    final = _chunk(cid, model, created, {}, finish=finish)
    final["harp_route"] = route.to_dict()   # non-standard, for demo proof
    yield _sse(final)
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------- dispatch

def _build_request(state: EndpointState, route: _Route, messages: list[dict],
                   max_tokens: int) -> InferRequest:
    # local lane needs a model_id in the genie manifest; cloud resolves by role.
    model_id = state.local_model_id if route.tier == Tier.EDGE else ""
    return InferRequest(messages=messages, model_id=model_id,
                        modality=Modality.TEXT, max_tokens=max_tokens, stream=True)


def _dispatch_stream(state: EndpointState, route: _Route, req: InferRequest,
                     tools: list | None) -> AsyncIterator[str]:
    if route.tier == Tier.EDGE:
        thinking = not bool(tools)          # tools present -> CoT off on the local lane
        return state.local_stream(req, thinking=thinking, tools=tools)
    # cloud lane via the frozen Backend interface (ponytail: NIM tool-passthrough is
    # a later pass; tool requests route local where the NPU sweet spot is anyway).
    return state.escalate.infer(req)


# ---------------------------------------------------------------- app factory

def make_app(
    *,
    local_backend: Backend | None = None,
    escalate_backend: Backend | None = None,
    policy: RoutingPolicy | None = None,
    ttft_budget_s: float | None = None,
    exec_est_s: float | None = None,
    escalate_disabled: bool | None = None,
    local_model_id: str | None = None,
) -> FastAPI:
    """Build the endpoint. Backends/policy are injectable for tests; defaults wire
    the real genie swarm + NIM cloud lane lazily (no network until called)."""
    disabled = escalate_disabled if escalate_disabled is not None \
        else _env_bool("HARP_ESCALATE_DISABLED")
    if local_backend is None:
        from edge.genie_backend import genie_swarm
        local_backend = genie_swarm()
    if escalate_backend is None and not disabled:
        from cloud.nim_backend import NIMBackend
        escalate_backend = NIMBackend()

    app = FastAPI(title="HARP", version="0.1.0")
    app.state.harp = EndpointState(
        local=local_backend,
        escalate=escalate_backend,
        policy=policy or _default_policy(),
        ttft_budget_s=ttft_budget_s if ttft_budget_s is not None
        else _env_float("HARP_TTFT_BUDGET_S", 2.0),
        exec_est_s=exec_est_s if exec_est_s is not None
        else _env_float("HARP_NPU_EXEC_EST_S", 3.0),
        escalate_disabled=disabled,
        local_model_id=local_model_id or os.getenv("HARP_LOCAL_MODEL", "qwen3-4b"),
    )

    @app.get("/v1/models")
    async def list_models() -> dict:
        return {"object": "list", "data": [
            {"id": m, "object": "model", "owned_by": "harp"} for m in _MODELS]}

    @app.get("/health")
    async def health() -> dict:
        state: EndpointState = app.state.harp
        npu_present = False
        if state.local is not None:
            try:
                npu_present = (await state.local.capabilities()).npu_present
            except Exception:
                npu_present = False
        return {
            "status": "ok",
            "npu_present": npu_present,
            "escalate_available": state.escalate_available,
            "queue_depth": state.committed_local,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        state: EndpointState = app.state.harp
        body = await request.json()
        messages = body.get("messages") or []
        if not messages:
            return JSONResponse({"error": "messages is required"}, status_code=400)
        model = body.get("model") or DEFAULT_MODEL
        stream = bool(body.get("stream", False))
        tools = body.get("tools") or None
        max_tokens = int(body.get("max_tokens") or 512)
        query = next((m.get("content", "") for m in reversed(messages)
                      if m.get("role") == "user"), "")

        # Route + commit must not be split by an await (shed-race guard).
        route = _resolve_route(state, model, query, tools, Modality.TEXT)
        if route.tier == Tier.EDGE:
            state.commit_local()

        req = _build_request(state, route, messages, max_tokens)
        token_stream = _dispatch_stream(state, route, req, tools)
        cid = _completion_id()
        headers = {"X-HARP-Route": route.header()}

        if stream:
            return StreamingResponse(
                _stream_response(token_stream, cid, model, route, tools),
                media_type="text/event-stream", headers=headers)

        text = "".join([t async for t in token_stream])
        return JSONResponse(_non_stream_body(cid, model, text, route), headers=headers)

    return app


# ---------------------------------------------------------------- __main__

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("HARP_ENDPOINT_PORT", "8765"))
    uvicorn.run(make_app(), host="0.0.0.0", port=port)
