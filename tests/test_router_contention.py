"""
HARP — tests/test_router_contention.py  ·  MIT

The contention axis in router_policy. A LOCAL query stays LOCAL when the NPU lane
is idle, sheds to ESCALATE (reason=contention_shed) when the lane is in-flight with
a deep queue, and stays LOCAL when offline regardless of queue (escalate is gone;
correctness > latency). The complexity + hardware gates are unchanged.
"""
from __future__ import annotations

import sys

from shared.harp_contract import Modality, RouteDecision
from router.router_policy import RoutingFeatures, RoutingPolicy

_LOCAL_QUERY = "summarize this paragraph in one line"
_EDGE_MODS = (Modality.TEXT, Modality.AUDIO)


def _policy() -> RoutingPolicy:
    cal_u = [i / 200.0 for i in range(200)]
    cal_err = [1 if (i % 100) / 100.0 < cal_u[i] else 0 for i in range(200)]
    return RoutingPolicy().calibrate(cal_u, cal_err)


def _features(*, inflight: bool, depth: int, online: bool, offline: bool) -> RoutingFeatures:
    return RoutingFeatures(
        query=_LOCAL_QUERY, modality=Modality.TEXT, online=online,
        npu_present=True, edge_modalities=_EDGE_MODS, edge_max_context=4096,
        approx_tokens=max(1, len(_LOCAL_QUERY) // 4),
        npu_inflight=inflight, npu_queue_depth=depth,
        tools_present=False, offline=offline,
    )


def test_idle_stays_local() -> None:
    v = _policy().decide(_features(inflight=False, depth=0, online=True, offline=False))
    assert v.decision == RouteDecision.LOCAL, v
    assert v.reason != "contention_shed"


def test_contended_sheds_to_escalate() -> None:
    v = _policy().decide(_features(inflight=True, depth=4, online=True, offline=False))
    assert v.decision == RouteDecision.ESCALATE, v
    assert v.reason == "contention_shed", v.reason


def test_offline_never_sheds() -> None:
    # Same deep-queue contention, but offline: escalate is physically unavailable.
    v_flag = _policy().decide(_features(inflight=True, depth=8, online=True, offline=True))
    assert v_flag.decision == RouteDecision.LOCAL, v_flag
    v_online_false = _policy().decide(
        _features(inflight=True, depth=8, online=False, offline=False))
    assert v_online_false.decision == RouteDecision.LOCAL, v_online_false


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  OK {fn.__name__}")
    print(f"test_router_contention: {len(fns)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
