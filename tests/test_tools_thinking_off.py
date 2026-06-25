"""
HARP — tests/test_tools_thinking_off.py  ·  MIT

GenieAPIService returns real tool_calls, but CoT must be DISABLED when a request
carries tools. This asserts the endpoint invokes the local backend with
thinking=False whenever tools are present (and thinking=True otherwise).
"""
from __future__ import annotations

import sys

from fastapi.testclient import TestClient

from shared.harp_contract import (
    Backend, Capability, InferRequest, Metrics, Modality, Tier,
)
from serve.openai_endpoint import make_app


class ThinkingSpy(Backend):
    def __init__(self):
        self.calls: list[dict] = []

    async def capabilities(self) -> Capability:
        return Capability("spy-edge", Tier.EDGE, npu_present=True, ram_gb=16,
                          max_context=4096, modalities=(Modality.TEXT,),
                          offline_capable=True, supports_streaming=True)

    async def infer(self, req: InferRequest, *, thinking: bool = True,
                    tools: list | None = None):
        self.calls.append({"thinking": thinking, "tools_present": bool(tools)})
        # echo a tool call so the tools path also returns valid OpenAI output
        if tools:
            yield '<tool_call>{"name": "noop", "arguments": {}}</tool_call> '
        else:
            yield "ok "

    async def profile(self, req: InferRequest) -> Metrics:
        return Metrics("spy-edge", ttft_ms=1.0, tokens_per_s=30.0)


def _client(spy: ThinkingSpy) -> TestClient:
    return TestClient(make_app(local_backend=spy, escalate_disabled=True))


def test_tools_force_thinking_off() -> None:
    spy = ThinkingSpy()
    r = _client(spy).post("/v1/chat/completions", json={
        "model": "harp-edge",
        "messages": [{"role": "user", "content": "use a tool"}],
        "tools": [{"type": "function", "function": {"name": "noop", "parameters": {}}}],
    })
    assert r.status_code == 200, r.text
    assert spy.calls, "local backend was never invoked"
    assert spy.calls[-1] == {"thinking": False, "tools_present": True}, spy.calls[-1]


def test_no_tools_keeps_thinking_on() -> None:
    spy = ThinkingSpy()
    r = _client(spy).post("/v1/chat/completions", json={
        "model": "harp-edge",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert r.status_code == 200, r.text
    assert spy.calls[-1] == {"thinking": True, "tools_present": False}, spy.calls[-1]


def _main() -> int:
    test_tools_force_thinking_off()
    print("  OK tools -> thinking disabled on local lane")
    test_no_tools_keeps_thinking_on()
    print("  OK no tools -> thinking stays on")
    print("test_tools_thinking_off: passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
