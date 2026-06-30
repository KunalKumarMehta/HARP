"""
HARP · fabric/executor.py · MIT
The end-to-end edge executor loop — the last unbuilt edge of the spine.

This is what turns the frozen parts into ONE runnable path:

    cloud planner → PlanGraph (JSON wire) → from_wire() → [THIS] → result

It consumes a validated PlanGraph, walks it in dependency order (topo_order),
resolves each step's tier through the Router/PolicyRouter (calibrated AUTO,
honored pins, offline fail-closed — all already proven), dispatches the step to
the chosen backend, and threads each step's output into the prompts of the steps
that depend on it (the dataflow the planner encodes as `<step_id>_output` refs).

Design constraints honored:
  - Does NOT edit the freeze. Imports only the public shared.harp_contract.
  - Tier-aware WITHOUT double-routing: it resolves the backend once via the
    Router's own selection seam, records the tier, then streams from that backend
    — reproducing Router.dispatch() with observability the demo/trace needs.
  - Works with the base Router (deterministic) or PolicyRouter (learned) — the
    executor is blind to which, exactly like every other caller.
  - Failure-isolating: a step that throws is recorded and quarantined; the run
    continues so the trace shows the whole DAG, and dependents inherit the error
    context rather than crashing the executor.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from shared.harp_contract import (
    InferRequest, PlanGraph, PlanStep, Router,
)

# Optional tracing (no-op if HARP_TRACE not set). `_trace()` is True only when the
# module imported AND HARP_TRACE is enabled — so no TraceEvent is built on the hot
# path when tracing is off.
try:
    from router.tracing import get_emitter, TraceEvent, enabled as _trace_enabled, _now_iso
    _HAS_TRACING = True
except ImportError:
    _HAS_TRACING = False
    get_emitter = None        # type: ignore
    TraceEvent = None         # type: ignore
    _trace_enabled = lambda: False   # type: ignore
    _now_iso = lambda: ""            # type: ignore


def _trace() -> bool:
    return _HAS_TRACING and _trace_enabled()


@dataclass
class StepResult:
    step_id: str
    model_id: str
    modality: str
    decision_in: str          # planner's pin/AUTO as it arrived on the wire
    tier: str | None          # "edge" / "cloud" the router actually chose
    output: str
    tokens: int
    ms: float
    depends_on: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ExecutionResult:
    plan_id: str
    steps: list[StepResult] = field(default_factory=list)
    total_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    @property
    def by_id(self) -> dict[str, StepResult]:
        return {s.step_id: s for s in self.steps}

    @property
    def final_output(self) -> str:
        """Output of the last successful TERMINAL step — one no other step depends
        on (out-degree 0 in the DAG), computed from recorded depends_on, not from
        execution order."""
        if not self.steps:
            return ""
        all_deps: set[str] = set()
        for s in self.steps:
            all_deps.update(s.depends_on)
        terminals = [s for s in self.steps if s.step_id not in all_deps]
        return next((s.output for s in reversed(terminals) if s.ok), "")


def _resolve_prompt(step: PlanStep, outputs: dict[str, str]) -> str:
    """Substitute `<dep>_output` dataflow refs with real upstream text in a SINGLE
    pass — so a value inserted for one dep can't be re-matched as another dep's ref
    — with word boundaries, so `s_output` never clobbers `s_detail_output`. If the
    planner gave a literal instruction (no refs) but the step has deps, append the
    upstream outputs as context so information still flows along the edge."""
    resolved = step.prompt
    referenced = False
    if step.depends_on:
        refs = {f"{dep}_output": outputs.get(dep, "") for dep in step.depends_on}
        # longest ref first so the alternation prefers the most specific token
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(r) for r in sorted(refs, key=len, reverse=True)) + r")\b")

        def _sub(m: "re.Match[str]") -> str:
            nonlocal referenced
            referenced = True
            return refs[m.group(0)]

        resolved = pattern.sub(_sub, resolved)
    if step.depends_on and not referenced:
        ctx = "\n".join(f"[{d}]: {outputs.get(d, '')}" for d in step.depends_on)
        resolved = f"{resolved}\n\nContext:\n{ctx}"
    return resolved


class PlanExecutor:
    """Runs a PlanGraph through a Router. Stateless across runs; cheap to reuse."""

    def __init__(self, router: Router):
        self.router = router

    async def execute(self, plan: PlanGraph, *, on_event=None) -> ExecutionResult:
        """on_event(StepResult) is an optional live hook for the demo trace."""
        result = ExecutionResult(plan_id=plan.plan_id)
        outputs: dict[str, str] = {}
        failed: set[str] = set()
        t_run = time.perf_counter()

        if _trace():
            get_emitter().emit(TraceEvent(
                timestamp=_now_iso(),
                event="exec.start",
                step_id="", plan_id=plan.plan_id,
                decision_in="", decision_out="",
                tier=None, reason="plan_start",
            ))

        for step in plan.topo_order():                 # raises on a cyclic plan
            dead = [d for d in step.depends_on if d in failed]
            if dead:
                # Failure isolation: never run a step on empty/garbage upstream
                # context. A failed dependency taints its whole downstream cone.
                sr = StepResult(
                    step_id=step.step_id, model_id=step.model_id,
                    modality=step.modality.value, decision_in=step.decision.value,
                    tier=None, output="", tokens=0, ms=0.0,
                    depends_on=list(step.depends_on),
                    error=f"skipped: upstream failed [{', '.join(dead)}]",
                )
                if _trace():
                    get_emitter().emit(TraceEvent(
                        timestamp=_now_iso(),
                        event="exec.skip",
                        step_id=step.step_id, plan_id=plan.plan_id,
                        decision_in=step.decision.value, decision_out="SKIPPED",
                        tier=None, reason=f"upstream_failed: {','.join(dead)}",
                    ))
            else:
                resolved = PlanStep(
                    step_id=step.step_id, modality=step.modality, decision=step.decision,
                    model_id=step.model_id, prompt=_resolve_prompt(step, outputs),
                    depends_on=step.depends_on,
                )
                if _trace():
                    get_emitter().emit(TraceEvent(
                        timestamp=_now_iso(),
                        event="exec.step_start",
                        step_id=step.step_id, plan_id=plan.plan_id,
                        decision_in=step.decision.value, decision_out="",
                        tier=None, reason="dataflow_resolved",
                        metadata={"prompt_chars": len(resolved.prompt), "deps": step.depends_on},
                    ))
                sr = await self._run_step(step, resolved)
            if not sr.ok:
                failed.add(step.step_id)
            outputs[step.step_id] = sr.output
            result.steps.append(sr)
            if on_event:
                on_event(sr)

        result.total_ms = (time.perf_counter() - t_run) * 1000.0

        if _trace():
            get_emitter().emit(TraceEvent(
                timestamp=_now_iso(),
                event="exec.complete",
                step_id="", plan_id=plan.plan_id,
                decision_in="", decision_out="",
                tier=None, reason="plan_complete",
                metadata={"total_steps": len(result.steps), "failed": len(failed), "total_ms": result.total_ms},
            ))
        return result

    async def _run_step(self, original: PlanStep, resolved: PlanStep) -> StepResult:
        t0 = time.perf_counter()
        tier: str | None = None
        try:
            # Resolve the backend through the router's own seam (PolicyRouter
            # resolves AUTO + honors pins; base Router applies the hardware/offline
            # guard). One selection, then stream from it — no double-routing.
            backend = await self.router._select(resolved)
            tier = (await backend.capabilities()).tier.value
            req = InferRequest(
                messages=[{"role": "user", "content": resolved.prompt}],
                model_id=resolved.model_id, modality=resolved.modality,
            )
            chunks: list[str] = []
            async for tok in backend.infer(req):
                chunks.append(tok)
            out = "".join(chunks).strip()
            return StepResult(
                step_id=original.step_id, model_id=original.model_id,
                modality=original.modality.value, decision_in=original.decision.value,
                tier=tier, output=out, tokens=len(chunks),
                ms=(time.perf_counter() - t0) * 1000.0,
                depends_on=list(original.depends_on),
            )
        except Exception as e:                          # quarantine; keep the run alive
            return StepResult(
                step_id=original.step_id, model_id=original.model_id,
                modality=original.modality.value, decision_in=original.decision.value,
                tier=tier, output="", tokens=0,
                ms=(time.perf_counter() - t0) * 1000.0,
                depends_on=list(original.depends_on), error=f"{type(e).__name__}: {e}",
            )


def render_trace(plan: PlanGraph, result: ExecutionResult) -> str:
    """Human-readable execution trace for the demo / evidence."""
    lines = [f"plan {result.plan_id} · {len(result.steps)} steps · "
             f"{result.total_ms:.0f} ms · {'OK' if result.ok else 'PARTIAL'}"]
    for s in result.steps:
        flag = "ok " if s.ok else "ERR"
        arrow = f"{s.decision_in:>9} -> {s.tier or '?':5}"
        lines.append(f"  [{flag}] {s.step_id:8} {arrow} {s.model_id:14} "
                     f"{s.tokens:3}tok {s.ms:6.0f}ms")
        if s.error:
            lines.append(f"        ! {s.error}")
        elif s.output:
            preview = s.output.replace("\n", " ")
            lines.append(f"        = {preview[:88]}")
    return "\n".join(lines)
