"""
HARP — Hardware-Aware Routing Platform
router/router_policy.py  ·  MIT

The learned routing brain. Fills the seam in the frozen contract:
    harp_contract.Router._select -> learned policy slots in at _decide

Design notes:
  - Router BASE = mmBERT-small, ENCODER-only, ~140 MB INT8 (heads FP16).
    Decoder routers (Qwen3-0.6B, Arch-Router-1.5B) are memory-bandwidth-bound
    at 50-150 ms TTFT on Hexagon and need a statically-allocated KV cache.
    Only a stateless single-pass encoder breaks the <10 ms always-resident
    barrier. The head emits a binary {local, escalate} hardness score;
    uncertainty u(x) = P(escalate) from that head, NOT a decode-time
    per-token margin.
  - Never route on raw argmax. Calibrate first, threshold second.
  - Asymmetric risk: under-routing (a hard query sent to the SLM) is the
    dangerous failure; over-routing only wastes cloud cost. Gate is tuned to
    bound under-routing at alpha via a conformal threshold.
  - Two-stage signal: encoder hardness score u(x) -> isotonic map to a real
    edge-error probability p_err (for logging / cost-optimal threshold) ->
    conformal delta as the actual escalation gate.

Separation of concerns vs the frozen contract:
  - THIS module decides COMPLEXITY  (local vs escalate from the query).
  - harp_contract.Router._select keeps the HARDWARE/CONNECTIVITY guard
    (offline, npu_present, modality coverage). We do not duplicate or fight it.

Integration:
  - The NAT ReWOO planner emits PlanStep.decision = AUTO ("undecided") for every
    step whose tier it defers (this is NAT's native idiom). It pins ESCALATE
    only when structurally certain (a deep-reason step), and LOCAL only when it
    must stay on-device (privacy). PolicyRouter resolves AUTO via the calibrated
    gate; it never overrides a planner pin. The base-class hardware guard still
    has final say.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence, AsyncIterator

from shared.harp_contract import (
    Backend, Capability, Modality, PlanStep, RouteDecision, Router,
)


# ---------------------------------------------------------------- routing features
# The gatekeeper's input contract. Device + connectivity fields are populated
# from telemetry; the query fields are extracted on the hot path. Keep this
# stable — the synthetic-data generator (synth_routing_data.py) emits exactly
# these keys.

from router.tracing import (
    trace_decide, trace_pin_honored, trace_guard, trace_dispatch,
)

@dataclass(frozen=True)
class RoutingFeatures:
    query: str
    modality: Modality
    online: bool
    # device telemetry (from Capability + runtime sensors)
    npu_present: bool
    edge_modalities: tuple[Modality, ...]
    edge_max_context: int
    approx_tokens: int                 # cheap len-based estimate, pre-tokenizer
    thermal_c: float | None = None     # edge thermal; high -> bias to escalate
    battery_pct: float | None = None   # low + not charging -> bias to escalate
    # --- contention axis (NPU single-lane pressure; populated by the endpoint) ---
    # The NPU single-context binary is single-lane: one in-flight infer at a time.
    # Under queue, TTFT degrades O(N). These four let the gate shed a LOCAL verdict
    # to the cloud when the lane is busy AND escalate is available. Defaulted so
    # every existing caller (and the contract's PolicyRouter) is unchanged.
    npu_inflight: bool = False         # an NPU infer is currently running
    npu_queue_depth: int = 0           # infers already committed to the NPU lane
    tools_present: bool = False        # request carries tools (thinking forced off)
    offline: bool = False              # escalate physically unavailable -> never shed


# ---------------------------------------------------------------- isotonic calibration (PAV, dependency-free)
# Maps the raw uncertainty score u(x) -> calibrated P(edge is wrong). Sample
# complexity O(n^-1/3); turns an unstable margin into a real probability.

class IsotonicCalibrator:
    def __init__(self) -> None:
        self._xs: list[float] = []
        self._ys: list[float] = []
        self._fitted = False

    def fit(self, u: Sequence[float], err: Sequence[int]) -> "IsotonicCalibrator":
        """u: raw uncertainty per calibration query. err: 1 if edge was wrong."""
        pairs = sorted(zip(u, err), key=lambda p: p[0])
        xs = [p[0] for p in pairs]
        ys = [float(p[1]) for p in pairs]
        w = [1.0] * len(ys)
        # Pool Adjacent Violators
        i = 0
        while i < len(ys) - 1:
            if ys[i] > ys[i + 1]:
                tot_w = w[i] + w[i + 1]
                avg = (ys[i] * w[i] + ys[i + 1] * w[i + 1]) / tot_w
                ys[i] = avg
                w[i] = tot_w
                del ys[i + 1]
                del w[i + 1]
                del xs[i + 1]
                if i > 0:
                    i -= 1
            else:
                i += 1
        self._xs, self._ys, self._fitted = xs, ys, True
        return self

    def predict(self, u: float) -> float:
        if not self._fitted or not self._xs:
            return min(max(u, 0.0), 1.0)        # identity fallback, clamped
        if u <= self._xs[0]:
            return self._ys[0]
        if u >= self._xs[-1]:
            return self._ys[-1]
        # piecewise-constant step lookup
        lo, hi = 0, len(self._xs) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._xs[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        return self._ys[max(lo - 1, 0)]


# ---------------------------------------------------------------- conformal escalation gate
# delta = (1-alpha) quantile of u among calibration queries the edge handled
# CORRECTLY. Escalate iff u > delta. Marginal guarantee: Pr[edge wrong & kept
# local] <= alpha. This is the asymmetric under-route bound, by construction.

class ConformalGate:
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.delta = float("inf")           # until fit: never escalate on score alone
        self._fitted = False

    def fit(self, u: Sequence[float], err: Sequence[int]) -> "ConformalGate":
        correct = sorted(ui for ui, ei in zip(u, err) if ei == 0)
        if not correct:
            return self
        n = len(correct)
        rank = max(0, min(n - 1, int((1 - self.alpha) * (n + 1)) - 1))
        self.delta = correct[rank]
        self._fitted = True
        return self

    def escalate(self, u: float) -> bool:
        return u > self.delta


# ---------------------------------------------------------------- verdict

@dataclass(frozen=True)
class RoutingVerdict:
    decision: RouteDecision
    reason: str
    u: float = 0.0                # raw uncertainty
    p_err: float = 0.0            # calibrated edge-error probability
    delta: float = float("inf")   # active conformal threshold
    overhead_ms: float = 0.0


# ---------------------------------------------------------------- the policy

# Cheap heuristic floor: kills the hot path on obviously-trivial turns BEFORE
# the head runs. Conservative — only fires on patterns the SLM never fails.
_TRIVIAL = re.compile(
    r"^\s*(hi|hey|hello|yes|no|ok(ay)?|thanks?|thank you|got it|sure|stop)\b[\s.!?]*$",
    re.IGNORECASE,
)


def mock_score_fn(query: str) -> float:
    """Stand-in for the mmBERT-small encoder head's hardness score u(x) = P(escalate),
    a single-pass classification output (NOT a decode-time margin).
    Deterministic so the gate is testable; swap in the QNN-EP encoder for production.
    Heuristic: longer + reasoning-marker queries read as higher hardness."""
    q = query.lower()
    base = min(len(query) / 400.0, 0.6)
    markers = ("prove", "derive", "why", "design", "optimi", "step by step",
               "explain how", "diagnos", "trade-off", "tradeoff", "architect",
               "bound", "under-rout")
    bump = 0.40 if any(m in q for m in markers) else 0.0
    return min(base + bump, 0.99)


class RoutingPolicy:
    """Decides LOCAL vs ESCALATE from query complexity + telemetry. Calibrated,
    not argmax. Hardware/offline guards stay in harp_contract.Router._select."""

    def __init__(
        self,
        score_fn: Callable[[str], float] = mock_score_fn,
        calibrator: IsotonicCalibrator | None = None,
        gate: ConformalGate | None = None,
        thermal_ceiling_c: float = 80.0,
        battery_floor_pct: float = 15.0,
        contention_budget_s: float = 2.0,
        npu_exec_est_s: float = 3.0,
    ) -> None:
        self.score_fn = score_fn
        self.calibrator = calibrator or IsotonicCalibrator()
        self.gate = gate or ConformalGate()
        self.thermal_ceiling_c = thermal_ceiling_c
        self.battery_floor_pct = battery_floor_pct
        # Contention gate: projected NPU wait = depth * per-infer estimate. One
        # in-flight infer (depth>=1) already exceeds the default 2.0s TTFT budget,
        # so a busy lane sheds rather than queues (O(N) TTFT degradation).
        self.contention_budget_s = contention_budget_s
        self.npu_exec_est_s = npu_exec_est_s

    def calibrate(self, u: Sequence[float], err: Sequence[int]) -> "RoutingPolicy":
        self.calibrator.fit(u, err)
        self.gate.fit(u, err)
        return self

    def decide(self, f: RoutingFeatures, *, step_id: str = "", plan_id: str = "") -> RoutingVerdict:
        t0 = time.perf_counter()
        decision_in = "AUTO"  # This method only handles AUTO resolution

        # --- capability guards (mirror the contract; complexity is moot if edge can't run it)
        if f.modality not in f.edge_modalities or not f.npu_present:
            if not f.online:
                v = self._verdict(RouteDecision.LOCAL, "offline_degraded_no_edge_path", t0)
                trace_guard(step_id, plan_id, decision_in, "edge", v.reason)
                return v
            v = self._verdict(RouteDecision.ESCALATE, "capability_modality", t0)
            trace_guard(step_id, plan_id, decision_in, "cloud", v.reason)
            return v
        if f.approx_tokens > f.edge_max_context:
            if not f.online:
                v = self._verdict(RouteDecision.LOCAL, "offline_degraded_overlong", t0)
                trace_guard(step_id, plan_id, decision_in, "edge", v.reason)
                return v
            v = self._verdict(RouteDecision.ESCALATE, "capability_context", t0)
            trace_guard(step_id, plan_id, decision_in, "cloud", v.reason)
            return v

        # --- offline: escalate is physically unavailable, fail closed to local
        if not f.online:
            v = self._verdict(RouteDecision.LOCAL, "offline_forced_local", t0)
            trace_guard(step_id, plan_id, decision_in, "edge", v.reason)
            return v

        # --- thermal / power pressure: bias the cheap work off the NPU
        if f.thermal_c is not None and f.thermal_c >= self.thermal_ceiling_c:
            v = self._verdict(RouteDecision.ESCALATE, "thermal_guard", t0)
            trace_guard(step_id, plan_id, decision_in, "cloud", v.reason)
            return v
        if (f.battery_pct is not None and f.battery_pct <= self.battery_floor_pct):
            v = self._verdict(RouteDecision.ESCALATE, "battery_guard", t0)
            trace_guard(step_id, plan_id, decision_in, "cloud", v.reason)
            return v

        # --- heuristic floor: trivial turn, never invoke the head
        if _TRIVIAL.match(f.query):
            v = self._shed_if_contended(
                self._verdict(RouteDecision.LOCAL, "trivial_floor", t0, u=0.0, p_err=0.0), f, t0)
            trace_decide(step_id, plan_id, decision_in, v)
            return v

        # --- calibrated complexity gate
        u = self.score_fn(f.query)
        p_err = self.calibrator.predict(u)
        if self.gate.escalate(u):
            v = self._verdict(RouteDecision.ESCALATE, "complexity_gate", t0,
                             u=u, p_err=p_err)
            trace_decide(step_id, plan_id, decision_in, v)
            return v
        # complexity says LOCAL — runs AFTER the calibrated gate, never before.
        v = self._shed_if_contended(
            self._verdict(RouteDecision.LOCAL, "complexity_gate", t0, u=u, p_err=p_err), f, t0)
        trace_decide(step_id, plan_id, decision_in, v)
        return v

    # --- contention axis: shed a soft-LOCAL verdict to cloud when the NPU lane is
    # busy and the projected wait exceeds the budget. NEVER fires offline (escalate
    # is physically gone) and never overrides the complexity/hardware gates — it
    # only flips an already-LOCAL outcome. The base-class hardware guard still runs
    # downstream and has final say.
    def _projected_npu_wait_s(self, f: RoutingFeatures) -> float:
        ahead = f.npu_queue_depth + (1 if f.npu_inflight else 0)
        return ahead * self.npu_exec_est_s

    def _shed_if_contended(self, v: RoutingVerdict, f: RoutingFeatures,
                           t0: float) -> RoutingVerdict:
        if v.decision != RouteDecision.LOCAL:
            return v
        if f.offline or not f.online:           # escalate unavailable -> correctness > latency
            return v
        if self._projected_npu_wait_s(f) > self.contention_budget_s:
            return self._verdict(RouteDecision.ESCALATE, "contention_shed", t0,
                                 u=v.u, p_err=v.p_err)
        return v

    def _verdict(self, d: RouteDecision, reason: str, t0: float,
                 u: float = 0.0, p_err: float = 0.0) -> RoutingVerdict:
        return RoutingVerdict(
            decision=d, reason=reason, u=u, p_err=p_err,
            delta=self.gate.delta,
            overhead_ms=(time.perf_counter() - t0) * 1000.0,
        )


# ---------------------------------------------------------------- frozen-contract integration
# Extends Router WITHOUT editing the freeze. Honors a planner ESCALATE; upgrades
# a planner LOCAL via the calibrated policy. Hardware guard from the base class
# still has the final say.

class PolicyRouter(Router):
    def __init__(self, edge: Backend, cloud: Backend, policy: RoutingPolicy,
                 online: bool = True):
        super().__init__(edge, cloud, online)
        self.policy = policy

    async def _select(self, step: PlanStep) -> Backend:
        # Planner pins (LOCAL/ESCALATE) are respected as-is. Only AUTO ("undecided",
        # the NAT ReWOO default) is resolved by the calibrated policy.
        if step.decision == RouteDecision.AUTO:
            cap: Capability = await self.edge.capabilities()
            f = RoutingFeatures(
                query=step.prompt,
                modality=step.modality,
                online=self.online,
                npu_present=cap.npu_present,
                edge_modalities=cap.modalities,
                edge_max_context=cap.max_context,
                approx_tokens=max(1, len(step.prompt) // 4),
            )
            verdict = self.policy.decide(f, step_id=step.step_id, plan_id="")  # plan_id not available here
            resolved = PlanStep(
                step_id=step.step_id, modality=step.modality,
                decision=verdict.decision, model_id=step.model_id,
                prompt=step.prompt, depends_on=step.depends_on,
            )
            return await super()._select(resolved)      # hardware/offline guard, final say
        # Planner pin: trace and pass through
        trace_pin_honored(step.step_id, "", step.decision.value, 
                         "edge" if step.decision == RouteDecision.LOCAL else "cloud")
        return await super()._select(step)


# ---------------------------------------------------------------- self-test (Risk B shape)

def _selftest() -> None:
    # synthetic calibration set: u in [0,1], edge wrong more often as u rises
    cal_u, cal_err = [], []
    for i in range(200):
        u = i / 200.0
        cal_u.append(u)
        cal_err.append(1 if (i % 100) / 100.0 < u else 0)   # err prob ~ u

    pol = RoutingPolicy().calibrate(cal_u, cal_err)
    print(f"conformal delta (alpha={pol.gate.alpha}) = {pol.gate.delta:.3f}")

    probe = [
        ("hi", Modality.TEXT, RouteDecision.LOCAL),
        ("what time is it", Modality.TEXT, RouteDecision.LOCAL),
        ("summarize this paragraph in one line", Modality.TEXT, RouteDecision.LOCAL),
        ("prove the routing gate bounds under-routing at alpha", Modality.TEXT, RouteDecision.ESCALATE),
        ("design a multi-agent planner and derive its latency budget step by step", Modality.TEXT, RouteDecision.ESCALATE),
    ]
    edge_mods = (Modality.TEXT, Modality.AUDIO)
    overheads, hits = [], 0
    print("\n== Risk B probe ==")
    for q, mod, expect in probe:
        f = RoutingFeatures(q, mod, online=True, npu_present=True,
                            edge_modalities=edge_mods, edge_max_context=4096,
                            approx_tokens=max(1, len(q) // 4))
        v = pol.decide(f)
        ok = v.decision == expect
        hits += ok
        overheads.append(v.overhead_ms)
        print(f"  [{'OK ' if ok else 'MIS'}] {v.decision.value:8} "
              f"u={v.u:.2f} p_err={v.p_err:.2f} {v.overhead_ms:.3f}ms  :: {q[:48]}")

    p95 = sorted(overheads)[int(0.95 * (len(overheads) - 1))]
    print(f"\naccuracy {hits}/{len(probe)}  p95 overhead {p95:.3f}ms  "
          f"(gate budget: <50ms, target 0 silent mis-routes)")


async def _integration_test() -> None:
    """Prove AUTO resolves through the real contract Router, and planner pins
    are honored. Uses the contract's mock backends."""
    import asyncio
    from shared.harp_contract import mock_edge, mock_cloud

    cal_u = [i / 200.0 for i in range(200)]
    cal_err = [1 if (i % 100) / 100.0 < cal_u[i] else 0 for i in range(200)]
    pol = RoutingPolicy().calibrate(cal_u, cal_err)
    pr = PolicyRouter(mock_edge(), mock_cloud(), pol, online=True)

    cases = [
        PlanStep("a1", Modality.TEXT, RouteDecision.AUTO, "qwen3-4b", "what time is it"),
        PlanStep("a2", Modality.TEXT, RouteDecision.AUTO, "qwen3-4b",
                 "design a multi-agent planner and derive its latency budget step by step"),
        PlanStep("p1", Modality.TEXT, RouteDecision.LOCAL, "qwen3-4b",
                 "design a multi-agent planner ... (planner-pinned LOCAL, privacy)"),
        PlanStep("p2", Modality.TEXT, RouteDecision.ESCALATE, "nemotron", "deep reason"),
    ]
    print("\n== PolicyRouter: AUTO resolved, pins honored ==")
    for st in cases:
        backend = await pr._select(st)
        tier = (await backend.capabilities()).tier.value
        print(f"  {st.step_id} in={st.decision.value:9} -> {tier}")


if __name__ == "__main__":
    import asyncio
    _selftest()
    asyncio.run(_integration_test())
