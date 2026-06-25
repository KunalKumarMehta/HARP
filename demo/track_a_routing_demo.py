"""
HARP — demo/track_a_routing_demo.py  ·  MIT

The Track-A live-demo artifact: a scripted multi-turn vernacular field-worker
workflow where HARP makes a VISIBLE, per-turn routing decision — on-device NPU vs
cloud planner. Screen-share this in the 12-minute slot.

Each turn calls HARP `POST /v1/route` (advisory; no inference) and prints a routing
table. It also writes demo/track_a_routing_trace.jsonl (one record per turn) so the
deck can cite real numbers.

Off-device safe: runs the endpoint in-process over stub backends (no NPU, no live
Hermes, no network). The decisions are the REAL router's — only the backends are
stubbed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from shared.harp_contract import (
    Backend, Capability, InferRequest, Metrics, Modality, Tier,
)
from serve.openai_endpoint import make_app
from edge.genie_backend import genie_swarm


# A no-op cloud backend so escalate is AVAILABLE (online) without a network call.
# /route never infers, so this is never actually driven — it just makes the cloud
# lane reachable for the routing decision.
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


# A field worker (vernacular code-switch) logging a wheat inspection on a handset.
# 5 turns belong on-device; ONE hard multi-step planning turn escalates to cloud.
TURNS: list[str] = [
    "Namaste — kaam shuru karein?",                                       # greet
    "I need to log today's wheat field inspection.",                      # intent parse
    "What's the recommended urea dose for wheat at tillering?",           # retrieval
    "Summarize my last three field notes in one line.",                   # on-device step
    ("Design a 3-day irrigation and fertilizer schedule across my five plots, "
     "step by step, accounting for the rainfall forecast and each crop's growth "
     "stage."),                                                            # HARD -> escalate
    "Haan, theek hai — save kar do.",                                     # confirm
]

_LABEL = {"local": "LOCAL / NPU", "escalate": "ESCALATE / Nemotron"}


def _client() -> TestClient:
    app = make_app(local_backend=genie_swarm(), escalate_backend=_StubCloud(),
                   base_url="http://127.0.0.1:8765/v1")
    return TestClient(app)


def run(trace_path: Path | None = None) -> list[dict]:
    """Drive the workflow through /route, return one record per turn, and write the
    trace JSONL. Pure of console output so tests can call it directly."""
    client = _client()
    records: list[dict] = []
    for i, query in enumerate(TURNS, start=1):
        r = client.post("/v1/route", json={
            "messages": [{"role": "user", "content": query}]}).json()
        records.append({
            "turn": i,
            "query": query,
            "decision": r["decision"],
            "tier": r["tier"],
            "reason": r["reason"],
            "shed": r["shed"],
            "runtime_override": r["runtime_override"],
        })
    if trace_path is None:
        trace_path = Path(__file__).resolve().parent / "track_a_routing_trace.jsonl"
    with open(trace_path, "w") as fh:
        for rec in records:
            # the override is implied by decision; keep the trace compact for the deck
            slim = {k: rec[k] for k in ("turn", "decision", "tier", "reason", "shed")}
            slim["query"] = rec["query"]
            fh.write(json.dumps(slim) + "\n")
    return records


def _print_table(records: list[dict]) -> None:
    print("\n  HARP · Track-A per-turn routing  (on-device NPU vs cloud planner)\n")
    print(f"  {'#':<2} {'query':<46} {'decision':<22} {'reason':<16} tier")
    print("  " + "-" * 92)
    for rec in records:
        q = rec["query"]
        q = (q[:43] + "...") if len(q) > 46 else q
        print(f"  {rec['turn']:<2} {q:<46} {_LABEL[rec['decision']]:<22} "
              f"{rec['reason']:<16} {rec['tier']}")
    local = sum(1 for r in records if r["decision"] == "local")
    esc = len(records) - local
    print("  " + "-" * 92)
    print(f"\n  {local}/{len(records)} turns resolved on-device; {esc} escalated to "
          f"cloud planner — privacy-preserving, offline-capable, cost-routed.\n")


def main() -> int:
    records = run()
    _print_table(records)
    trace = Path(__file__).resolve().parent / "track_a_routing_trace.jsonl"
    print(f"  trace written: {trace}")
    # sanity: the demo must actually demonstrate a split
    local = sum(1 for r in records if r["decision"] == "local")
    esc = len(records) - local
    if local < 1 or esc < 1:
        print(f"  WARN: degenerate split (local={local}, escalate={esc})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
