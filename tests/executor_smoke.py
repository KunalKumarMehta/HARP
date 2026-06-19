"""
HARP · tests/executor_smoke.py · CI Gate 7 · MIT
Proves the end-to-end spine as one runnable path:

    PlanGraph --topo--> PlanExecutor --PolicyRouter--> backends --> result

Asserts the four things that actually break in the field:
  1. Dataflow threading — a dependent step's prompt receives its upstream output.
  2. AUTO resolution — undecided steps route by the calibrated gate.
  3. Planner pins honored — ESCALATE -> cloud when online.
  4. Offline fail-closed — every step (incl. ESCALATE) collapses to edge.
  5. Cyclic plan is rejected loudly (executor never half-runs a bad DAG).
"""
from __future__ import annotations

import asyncio

from shared.harp_contract import (
    Modality, PlanGraph, PlanStep, RouteDecision, mock_cloud, mock_edge,
)
from router.router_policy import RoutingPolicy, PolicyRouter
from fabric.executor import PlanExecutor, render_trace


def _calibrated_policy() -> RoutingPolicy:
    cal_u = [i / 200.0 for i in range(200)]
    cal_err = [1 if (i % 100) / 100.0 < cal_u[i] else 0 for i in range(200)]
    return RoutingPolicy().calibrate(cal_u, cal_err)


def _plan() -> PlanGraph:
    return PlanGraph("p-exec", [
        PlanStep("s1", Modality.AUDIO, RouteDecision.AUTO, "whisper-base",
                 "transcribe the call recording"),
        PlanStep("s2", Modality.TEXT, RouteDecision.AUTO, "qwen3-4b",
                 "summarize s1_output", depends_on=["s1"]),
        PlanStep("s3", Modality.TEXT, RouteDecision.ESCALATE, "nemotron",
                 "deep-reason over s2_output", depends_on=["s2"]),
    ])


async def _main() -> None:
    pol = _calibrated_policy()

    # --- ONLINE -------------------------------------------------------------
    ex = PlanExecutor(PolicyRouter(mock_edge(), mock_cloud(), pol, online=True))
    res = await ex.execute(_plan())
    print(render_trace(_plan(), res))

    tiers = {s.step_id: s.tier for s in res.steps}
    assert res.ok, "all steps must succeed online"
    assert tiers["s1"] == "edge", f"s1 (AUTO/audio) -> edge, got {tiers['s1']}"
    assert tiers["s2"] == "edge", f"s2 (AUTO/short) -> edge, got {tiers['s2']}"
    assert tiers["s3"] == "cloud", f"s3 (ESCALATE pin) -> cloud, got {tiers['s3']}"

    # dataflow: s2's output must reflect s1's substituted-in text
    by = res.by_id
    assert "transcribe the call recording" in by["s2"].output, \
        f"s1 output must thread into s2 prompt; got: {by['s2'].output!r}"
    assert "summarize" in by["s2"].output, "s2 own instruction must survive"

    # --- OFFLINE: escalate disabled, everything fails closed to edge --------
    ex_off = PlanExecutor(PolicyRouter(mock_edge(), mock_cloud(), pol, online=False))
    res_off = await ex_off.execute(_plan())
    assert res_off.ok, "offline run must still complete on edge"
    assert all(s.tier == "edge" for s in res_off.steps), \
        f"offline must fail closed to edge, got {[(s.step_id, s.tier) for s in res_off.steps]}"

    # --- failure propagation: a failed step skips its downstream cone -------
    class _BoomEdge:
        async def capabilities(self):
            return (await mock_edge().capabilities())
        async def infer(self, req):
            raise RuntimeError("edge boom")
            yield  # pragma: no cover (make it an async generator)
        async def profile(self, req):
            return await mock_edge().profile(req)

    boom_plan = PlanGraph("p-fail", [
        PlanStep("f1", Modality.TEXT, RouteDecision.LOCAL, "qwen3-4b", "do thing"),
        PlanStep("f2", Modality.TEXT, RouteDecision.LOCAL, "qwen3-4b",
                 "use f1_output", depends_on=["f1"]),
    ])
    rfail = await PlanExecutor(
        PolicyRouter(_BoomEdge(), mock_cloud(), pol, online=True)).execute(boom_plan)
    byf = rfail.by_id
    assert not byf["f1"].ok, "f1 must record the backend error"
    assert byf["f2"].error and "skipped" in byf["f2"].error, \
        "f2 must be SKIPPED (not run on empty upstream context), got: " + str(byf["f2"].error)
    assert byf["f2"].tier is None, "skipped step never reaches a backend"

    # --- dataflow boundary safety: prefix step-ids don't cross-clobber ------
    from fabric.executor import _resolve_prompt
    s = PlanStep("x2", Modality.TEXT, RouteDecision.AUTO, "m",
                 "combine x_output and x2_output", depends_on=["x", "x2"])
    got = _resolve_prompt(s, {"x": "AAA", "x2": "BBB"})
    assert got == "combine AAA and BBB", f"boundary-safe substitution failed: {got!r}"

    # --- cyclic plan rejected ----------------------------------------------
    cyclic = PlanGraph("bad", [
        PlanStep("a", Modality.TEXT, RouteDecision.AUTO, "m", "x", depends_on=["b"]),
        PlanStep("b", Modality.TEXT, RouteDecision.AUTO, "m", "y", depends_on=["a"]),
    ])
    try:
        await PlanExecutor(PolicyRouter(mock_edge(), mock_cloud(), pol)).execute(cyclic)
    except ValueError:
        pass
    else:
        raise AssertionError("cyclic plan must raise, not half-execute")

    print("\nexecutor_smoke OK: dataflow threaded, AUTO calibrated, pins honored, "
          "offline fail-closed, cycle rejected")


if __name__ == "__main__":
    asyncio.run(_main())
