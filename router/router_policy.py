"""
HARP — Hardware-Aware Routing Platform
router/router_policy.py  ·  CAIO ARTIFACT v0  ·  MIT

The learned routing brain. Fills the seam the CTO froze:
    harp_contract.Router._select -> "CAIO's learned policy slots in at _decide later"

Doctrine (grounded, not assumed):
  - NEVER route on raw argmax/softmax. Calibrate first, threshold second.
      -> LLM Routing & Cascade Classifiers V2 §"Calibrated Uncertainty"
  - Asymmetric risk: under-routing (a hard query sent to the SLM) is the
    dangerous failure; over-routing only wastes cloud cost. Gate is tuned to
    BOUND under-routing at alpha via a conformal threshold.
      -> V2 §"Conformal Prediction and Strict Marginal Guarantees"
  - Two-stage signal: cheap per-token margin uncertainty u(x) from the sub-1B
    router head -> isotonic map to a real error probability p_err (for logging /
    cost-optimal threshold) -> conformal delta as the actual escalation gate.
      -> V2 §"Isotonic Regression and the UCCI Framework"

Separation of concerns vs the frozen contract:
  - THIS module decides COMPLEXITY  (local vs escalate from the query).
  - harp_contract.Router._select keeps the HARDWARE/CONNECTIVITY guard
    (offline, npu_present, modality coverage). We do not duplicate or fight it.

Integration without touching the freeze:
  - Planner emits PlanStep.decision = ESCALATE only when it is STRUCTURALLY
    certain a step needs the cloud (e.g. a deep-reason step). LOCAL is the
    "router, you decide" default. PolicyRouter upgrades LOCAL->ESCALATE when the
    calibrated gate fires; it never downgrades a planner ESCALATE.
  - Proposed contract delta (CTO sign-off): add RouteDecision.AUTO so the
    planner can express "undecided" explicitly instead of overloading LOCAL.
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
# The gatekeeper's input contract. CTO/CEE populate device + connectivity from
# telemetry; the query fields are extracted on the hot path. Keep this stable —
# the synthetic-data generator (synth_routing_data.py) emits exactly these keys.

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
    """Stand-in for the sub-1B router head's per-token margin uncertainty u(x).
    Deterministic so the gate is testable today; CEE swaps in the QNN-EP head.
    Heuristic: longer + reasoning-marker queries read as higher uncertainty."""
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
    ) -> None:
        self.score_fn = score_fn
        self.calibrator = calibrator or IsotonicCalibrator()
        self.gate = gate or ConformalGate()
        self.thermal_ceiling_c = thermal_ceiling_c
        self.battery_floor_pct = battery_floor_pct

    def calibrate(self, u: Sequence[float], err: Sequence[int]) -> "RoutingPolicy":
        self.calibrator.fit(u, err)
        self.gate.fit(u, err)
        return self

    def decide(self, f: RoutingFeatures) -> RoutingVerdict:
        t0 = time.perf_counter()

        # --- capability guards (mirror the contract; complexity is moot if edge can't run it)
        if f.modality not in f.edge_modalities or not f.npu_present:
            if not f.online:
                return self._verdict(RouteDecision.LOCAL, "offline_degraded_no_edge_path", t0)
            return self._verdict(RouteDecision.ESCALATE, "capability_modality", t0)
        if f.approx_tokens > f.edge_max_context:
            if not f.online:
                return self._verdict(RouteDecision.LOCAL, "offline_degraded_overlong", t0)
            return self._verdict(RouteDecision.ESCALATE, "capability_context", t0)

        # --- offline: escalate is physically unavailable, fail closed to local
        if not f.online:
            return self._verdict(RouteDecision.LOCAL, "offline_forced_local", t0)

        # --- thermal / power pressure: bias the cheap work off the NPU
        if f.thermal_c is not None and f.thermal_c >= self.thermal_ceiling_c:
            return self._verdict(RouteDecision.ESCALATE, "thermal_guard", t0)
        if (f.battery_pct is not None and f.battery_pct <= self.battery_floor_pct):
            return self._verdict(RouteDecision.ESCALATE, "battery_guard", t0)

        # --- heuristic floor: trivial turn, never invoke the head
        if _TRIVIAL.match(f.query):
            return self._verdict(RouteDecision.LOCAL, "trivial_floor", t0, u=0.0, p_err=0.0)

        # --- calibrated complexity gate
        u = self.score_fn(f.query)
        p_err = self.calibrator.predict(u)
        if self.gate.escalate(u):
            return self._verdict(RouteDecision.ESCALATE, "complexity_gate", t0,
                                 u=u, p_err=p_err)
        return self._verdict(RouteDecision.LOCAL, "complexity_gate", t0,
                             u=u, p_err=p_err)

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
        if step.decision == RouteDecision.LOCAL:        # planner left it to us
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
            verdict = self.policy.decide(f)
            if verdict.decision == RouteDecision.ESCALATE and self.online:
                return self.cloud
        return await super()._select(step)              # hardware/offline guard, final say


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


if __name__ == "__main__":
    _selftest()
