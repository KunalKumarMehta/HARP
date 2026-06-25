"""
HARP — tests/test_route_endpoint.py  ·  MIT

POST /v1/route is advisory: it returns a tier + decision WITHOUT running inference
and WITHOUT mutating lane state (no commit_local). Asserts: a short query -> local
with a well-formed runtime_override (provider=harp, model=harp-edge, base_url
honored); a long/multi-step query -> escalate with runtime_override null; an offline
endpoint never escalates; a busy+online hint sheds.
"""
from __future__ import annotations

import sys

from fastapi.testclient import TestClient

from shared.harp_contract import (
    Backend, Capability, InferRequest, Metrics, Modality, Tier,
)
from serve.openai_endpoint import make_app
from edge.genie_backend import genie_swarm

_BASE = "http://harp.local:8765/v1"
_HARD = ("design a sharded irrigation planner and derive its schedule step by step "
         "across five plots accounting for rainfall and crop stage")


class _StubCloud(Backend):
    async def capabilities(self) -> Capability:
        return Capability("stub-cloud", Tier.CLOUD, npu_present=False, ram_gb=80,
                          max_context=128_000, modalities=(Modality.TEXT,),
                          offline_capable=False, supports_streaming=True)

    async def infer(self, req: InferRequest):
        if False:
            yield ""

    async def profile(self, req: InferRequest) -> Metrics:
        return Metrics("stub-cloud", ttft_ms=1.0, tokens_per_s=1.0)


def _online_client() -> TestClient:
    return TestClient(make_app(local_backend=genie_swarm(),
                               escalate_backend=_StubCloud(), base_url=_BASE))


def test_short_query_local_with_override() -> None:
    c = _online_client()
    r = c.post("/v1/route", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] == "local" and body["tier"] == "edge"
    ov = body["runtime_override"]
    assert ov is not None
    assert ov["provider"] == "harp"
    assert ov["model"] == "harp-edge"
    assert ov["base_url"] == _BASE            # base_url honored
    assert ov["api_mode"] == "chat_completions"


def test_hard_query_escalates_no_override() -> None:
    body = _online_client().post(
        "/v1/route", json={"messages": [{"role": "user", "content": _HARD}]}).json()
    assert body["decision"] == "escalate" and body["tier"] == "cloud", body
    assert body["runtime_override"] is None


def test_route_is_side_effect_free() -> None:
    c = _online_client()
    for _ in range(3):
        c.post("/v1/route", json={"messages": [{"role": "user", "content": _HARD}]})
    # no commit_local on the advisory path -> queue stays empty
    assert c.get("/health").json()["queue_depth"] == 0


def test_offline_never_escalates() -> None:
    c = TestClient(make_app(local_backend=genie_swarm(), escalate_disabled=True,
                            base_url=_BASE))
    body = c.post("/v1/route",
                  json={"messages": [{"role": "user", "content": _HARD}]}).json()
    assert body["decision"] == "local", body      # offline: correctness > latency
    assert body["runtime_override"] is not None


def test_busy_hint_sheds() -> None:
    body = _online_client().post("/v1/route", json={
        "messages": [{"role": "user", "content": "summarize this note"}],
        "npu_inflight": True, "npu_queue_depth": 4,
    }).json()
    assert body["decision"] == "escalate", body   # busy + online -> shed to cloud


def test_health_reports_classifier() -> None:
    body = _online_client().get("/health").json()
    assert "route_classifier" in body
    assert "placeholder" in body["route_classifier"]   # mmBERT placeholder labeled


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  OK {fn.__name__}")
    print(f"test_route_endpoint: {len(fns)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
