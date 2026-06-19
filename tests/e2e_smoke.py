"""
HARP — tests/e2e_smoke.py  ·  End-to-end routing smoke  ·  MIT

Spans two layers (shared freeze + router policy), so it lives ABOVE shared/ —
the freeze stays router-agnostic. Wires the PolicyRouter into a full
PlanGraph dispatch and asserts the integration invariants:

  1. planner pins (LOCAL / ESCALATE) are honored verbatim
  2. AUTO is resolved by the calibrated gate: trivial -> edge, hard -> cloud
  3. offline fails closed: every step lands on edge, escalate is unavailable
  4. dispatch streams real tokens end-to-end through the chosen backend

Exits non-zero on any mismatch.
"""

from __future__ import annotations

import asyncio
import sys

from shared.harp_contract import (
    Modality, PlanGraph, PlanStep, RouteDecision, Tier, mock_edge, mock_cloud,
)
from router.router_policy import PolicyRouter, RoutingPolicy


def _build_policy() -> RoutingPolicy:
    # synthetic calibration: edge error rises with uncertainty u
    cal_u = [i / 200.0 for i in range(200)]
    cal_err = [1 if (i % 100) / 100.0 < cal_u[i] else 0 for i in range(200)]
    return RoutingPolicy().calibrate(cal_u, cal_err)


def _plan() -> PlanGraph:
    return PlanGraph("e2e", [
        PlanStep("s1", Modality.AUDIO, RouteDecision.LOCAL, "whisper-base",
                 "transcribe the clip"),                                   # pin LOCAL
        PlanStep("s2", Modality.TEXT, RouteDecision.AUTO, "qwen3-4b",
                 "what time is it", depends_on=["s1"]),                    # AUTO -> trivial -> edge
        PlanStep("s3", Modality.TEXT, RouteDecision.AUTO, "qwen3-4b",
                 "design a multi-agent planner and derive its latency budget step by step",
                 depends_on=["s2"]),                                        # AUTO -> hard -> cloud
        PlanStep("s4", Modality.TEXT, RouteDecision.ESCALATE, "nemotron",
                 "deep-reason the final plan", depends_on=["s3"]),          # pin ESCALATE
    ])


async def _tier_of(router: PolicyRouter, step: PlanStep) -> Tier:
    backend = await router._select(step)
    return (await backend.capabilities()).tier


async def main() -> int:
    policy = _build_policy()
    plan = _plan()
    failures: list[str] = []

    # ---- ONLINE: pins honored, AUTO resolved by the gate, tokens actually stream
    online = PolicyRouter(mock_edge(), mock_cloud(), policy, online=True)
    expected_online = {
        "s1": Tier.EDGE,    # LOCAL pin
        "s2": Tier.EDGE,    # AUTO, trivial
        "s3": Tier.CLOUD,   # AUTO, hard
        "s4": Tier.CLOUD,   # ESCALATE pin
    }
    print("== ONLINE: PolicyRouter over full plan ==")
    for step in plan.topo_order():
        tier = await _tier_of(online, step)
        toks = "".join([t async for t in online.dispatch(step)])
        ok = tier == expected_online[step.step_id]
        failures += [] if ok else [f"online {step.step_id}: {tier} != {expected_online[step.step_id]}"]
        assert toks.strip(), f"{step.step_id} streamed no tokens"
        print(f"  [{'OK ' if ok else 'MIS'}] {step.step_id} in={step.decision.value:9} "
              f"-> {tier.value:5} :: {toks.strip()}")

    # ---- OFFLINE: every step fails closed to edge; escalate is physically gone
    offline = PolicyRouter(mock_edge(), mock_cloud(), policy, online=False)
    print("\n== OFFLINE: fail-closed to edge ==")
    for step in plan.steps:
        tier = await _tier_of(offline, step)
        ok = tier == Tier.EDGE
        failures += [] if ok else [f"offline {step.step_id}: {tier} != EDGE"]
        print(f"  [{'OK ' if ok else 'MIS'}] {step.step_id} in={step.decision.value:9} -> {tier.value}")

    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\ne2e_smoke OK: pins honored, AUTO calibrated, offline fail-closed, tokens streamed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
