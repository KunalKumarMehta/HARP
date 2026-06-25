"""
HARP — tests/test_endpoint_contract.py  ·  MIT

OpenAI-contract conformance for serve/openai_endpoint over the off-device genie
stub. Asserts: /v1/models shape; non-stream completion is valid OpenAI schema;
stream yields chat.completion.chunk frames terminating in [DONE]; a request with
`tools` returns a well-formed tool_calls object.

Runs as a plain script (CI: `python tests/test_endpoint_contract.py`) and under
pytest (test_* functions). No NPU, no genie-t2t-run required.
"""
from __future__ import annotations

import json
import sys

from fastapi.testclient import TestClient

from edge.genie_backend import genie_swarm
from serve.openai_endpoint import make_app


def _client() -> TestClient:
    # local lane = genie stub (off-device); escalate disabled so the test is hermetic.
    app = make_app(local_backend=genie_swarm(), escalate_disabled=True)
    return TestClient(app)


def test_models_shape() -> None:
    r = _client().get("/v1/models")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list"
    ids = {m["id"] for m in body["data"]}
    assert ids == {"harp-auto", "harp-edge", "harp-cloud"}, ids


def test_health_shape() -> None:
    body = _client().get("/health").json()
    for k in ("npu_present", "escalate_available", "queue_depth"):
        assert k in body, f"/health missing {k}: {body}"
    assert body["escalate_available"] is False     # disabled in this fixture


def test_non_stream_completion() -> None:
    r = _client().post("/v1/chat/completions", json={
        "model": "harp-edge",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200, r.text
    assert r.headers.get("X-HARP-Route", "").startswith("edge:")
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "harp-edge"
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"], "stub must produce content"
    hr = body["harp_route"]
    assert hr["tier"] == "edge" and hr["shed"] is False


def test_stream_chunks_terminate_in_done() -> None:
    saw_chunk = saw_done = saw_role = False
    with _client().stream("POST", "/v1/chat/completions", json={
        "model": "harp-edge", "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                saw_done = True
                continue
            obj = json.loads(payload)
            assert obj["object"] == "chat.completion.chunk"
            saw_chunk = True
            if obj["choices"][0]["delta"].get("role") == "assistant":
                saw_role = True
    assert saw_role, "first chunk must carry the assistant role"
    assert saw_chunk, "expected at least one chunk frame"
    assert saw_done, "stream must terminate with data: [DONE]"


def test_tools_returns_tool_calls() -> None:
    tools = [{"type": "function", "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }}]
    r = _client().post("/v1/chat/completions", json={
        "model": "harp-edge",
        "messages": [{"role": "user", "content": "weather in Mumbai?"}],
        "tools": tools,
    })
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    calls = choice["message"]["tool_calls"]
    assert isinstance(calls, list) and len(calls) == 1, calls
    c = calls[0]
    assert c["type"] == "function" and c["id"]
    assert c["function"]["name"] == "get_weather"
    json.loads(c["function"]["arguments"])     # arguments must be a JSON string


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  OK {fn.__name__}")
    print(f"test_endpoint_contract: {len(fns)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
