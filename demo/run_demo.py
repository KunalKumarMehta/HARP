"""
HARP · demo/run_demo.py · MIT
The whole spine in one runnable command — end-to-end demo.

    cloud planner (ReWOO adapter)
        → PlanGraph
        → JSON wire  (to_json / from_json: the cloud↔edge boundary, validated)
        → PlanExecutor
        → PolicyRouter  (calibrated AUTO · honored pins · offline fail-closed)
        → backends      (edge: Genie/Qwen3-4B on NPU when present, else mock;
                         cloud: NIM/Nemotron when HARP_NIM_API_KEY set, else mock)
        → threaded dataflow result

Runs TODAY with zero setup (mocks). Lights up real silicon automatically:
  • edge  — set if `genie-t2t-run` is on PATH and build/qwen3-4b-w4a16/ exists.
  • cloud — set HARP_NIM_API_KEY to plan/escalate against a real Nemotron NIM.

Usage:
    python -m demo.run_demo                 # mock plan, auto-detect backends
    python -m demo.run_demo --offline       # show graceful fail-closed to edge
    python -m demo.run_demo --genie         # force the Genie edge path (stub off-device)
    python -m demo.run_demo --distributed   # run the edge tier on a remote fabric node (multi-device)
    python -m demo.run_demo --live "..."    # plan a custom task via the live NIM planner
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys

from shared.harp_contract import Backend, PlanGraph, Tier, mock_cloud, mock_edge
from shared.plan_codec import from_json, to_json
from cloud.plan_emitter import RawReWOOStep, emit_plan_graph
from router.router_policy import PolicyRouter, RoutingPolicy
from fabric.executor import PlanExecutor, render_trace
from fabric.remote_backend import RemoteBackend, serve_backend

# A multi-modal task: a call recording + a screen scan, summarized,
# then a decision. Exercises all three modalities and both tiers.
_MOCK_STEPS = [
    RawReWOOStep("s_asr", "asr_transcribe", "transcribe the call recording", [], "edge"),
    RawReWOOStep("s_vis", "vision_screen", "extract text from the screen scan", [], "edge"),
    RawReWOOStep("s_sum", "text_summarize",
                 "summarize s_asr_output and s_vis_output", ["s_asr", "s_vis"], "edge"),
    RawReWOOStep("s_dr", "deep_reason",
                 "decide the next action from s_sum_output", ["s_sum"], "cloud"),
]


def _calibrated_policy() -> RoutingPolicy:
    cal_u = [i / 200.0 for i in range(200)]
    cal_err = [1 if (i % 100) / 100.0 < cal_u[i] else 0 for i in range(200)]
    return RoutingPolicy().calibrate(cal_u, cal_err)


def _edge_backend(force_genie: bool) -> tuple[Backend, str]:
    """Real Genie/Qwen3-4B if the runtime is present (or forced), else the mock.
    GenieBackend itself stays honest off-device (stub + npu_present=False)."""
    genie_on_path = (shutil.which("genie-t2t-run") is not None
                     or bool(os.environ.get("HARP_GENIE_BIN")))
    if force_genie or genie_on_path:
        from edge.genie_backend import genie_swarm
        be = genie_swarm()                       # auto-discovers every bundle in build/
        ids = ",".join(sorted(be._specs))
        mode = (f"Genie swarm NPU [{ids}]" if genie_on_path
                else f"Genie swarm off-device stub [{ids}]")
        return be, mode
    return mock_edge(), "mock edge (Qwen3-4B @ 30.3 tok/s profile)"


def _cloud_backend() -> tuple[Backend, str]:
    if os.environ.get("HARP_NIM_API_KEY"):
        from cloud.nim_backend import NIMBackend, NIMConfig
        return NIMBackend(NIMConfig()), "NIM/Nemotron (live)"
    return mock_cloud(), "mock cloud (Nemotron NIM profile)"


async def _build_plan(live_task: str | None) -> PlanGraph:
    if live_task:
        # Lazy import: only touches the NIM planner path when actually planning live.
        from cloud.emit_first_plan import emit
        return await emit(live_task)
    return emit_plan_graph("plan-demo", _MOCK_STEPS)


async def _main(argv: list[str]) -> int:
    offline = "--offline" in argv
    force_genie = "--genie" in argv
    distributed = "--distributed" in argv
    live_task = None
    if "--live" in argv:
        i = argv.index("--live")
        live_task = " ".join(argv[i + 1:]) or "Analyze the call recording and screen scan, then decide the next action"

    print("=" * 72)
    print("HARP — hardware-aware routing, end to end")
    print("=" * 72)

    # 1) cloud planner emits a DAG
    plan = await _build_plan(live_task)

    # 2) cross the validated JSON wire (this is the real cloud→edge boundary)
    wire = to_json(plan)
    plan = from_json(wire)                  # edge decodes verbatim; rejects malformed
    print(f"\n[cloud→wire→edge]  plan={plan.plan_id}  steps={len(plan.steps)}  "
          f"wire={len(wire.encode())} bytes  (schema+DAG validated)")

    # 3) assemble the router over whichever backends are live
    edge, edge_mode = _edge_backend(force_genie)
    cloud, cloud_mode = _cloud_backend()

    # 3b) multi-device: run the edge tier on a separate fabric node. Here the node
    # is an in-process loopback server; on hardware it's the phone (bind 0.0.0.0).
    node_task = None
    if distributed:
        host, port = "127.0.0.1", 8770
        ready = asyncio.Event()
        node_task = asyncio.create_task(serve_backend(edge, host, port, ready))
        await asyncio.wait_for(ready.wait(), timeout=5.0)
        edge = RemoteBackend(f"ws://{host}:{port}", label=f"remote→{edge_mode}")
        edge_mode = f"RemoteBackend → fabric node ({edge_mode})"

    print(f"[backends]         edge = {edge_mode}\n"
          f"                   cloud= {cloud_mode}\n"
          f"[network]          {'OFFLINE (escalate disabled)' if offline else 'online'}")

    router = PolicyRouter(edge, cloud, _calibrated_policy(), online=not offline)
    executor = PlanExecutor(router)

    # 4) run it
    print("\n--- execution ---------------------------------------------------------")
    result = await executor.execute(plan)
    print(render_trace(plan, result))

    # 5) summary
    edge_n = sum(1 for s in result.steps if s.tier == Tier.EDGE.value)
    cloud_n = sum(1 for s in result.steps if s.tier == Tier.CLOUD.value)
    print("\n--- summary -----------------------------------------------------------")
    print(f"  routed: {edge_n} on edge · {cloud_n} on cloud · "
          f"{len(result.steps)} total   ({'all OK' if result.ok else 'PARTIAL — see errors'})")
    print(f"  wall:   {result.total_ms:.0f} ms")
    final = next((s for s in reversed(result.steps) if s.ok), None)
    if final:
        print(f"  result: {final.output[:160]}")

    # close any live cloud client + tear down the fabric node cleanly
    aclose = getattr(cloud, "aclose", None)
    if aclose:
        await aclose()
    if node_task is not None:
        node_task.cancel()
        try:
            await node_task
        except (asyncio.CancelledError, Exception):
            pass
    return 0 if result.ok else 1


def main() -> None:
    sys.exit(asyncio.run(_main(sys.argv[1:])))


if __name__ == "__main__":
    main()
