"""
HARP — Hardware-Aware Routing Platform
cloud/plan_emitter.py  ·  CCE owns this  ·  MIT

Cloud planner -> edge executor bridge. Converts a NAT ReWOO planner's upfront
Dependency Graph into the contract's PlanGraph JSON, then ships it. This is the
"we emit plans, not activations" line made literal: model-level routing across
the tier boundary, kilobyte-scale wire payload.

CRITICAL GROUNDING (NeMo Agent Toolkit Execution Plan Export doc):
  NAT has NO `export_plan_only=True` toggle. ReWOO computes the full DAG before
  any tool runs, but to emit it to an EXTERNAL edge executor you must INTERCEPT
  it. The supported, minimal interception vector is the IntermediateStep /
  Middleware hook: catch the planner's emitted graph BEFORE the Executor Node
  dispatches, serialize, and short-circuit local execution. `emit_plan_graph`
  below is that interception adapter. `RawReWOOStep` mirrors the planner's
  internal step shape; the real hook fills it from the ATIF/IntermediateStep
  payload. Until NAT is installed + verified, the synthetic builder lets the
  edge team integrate against real JSON today.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.harp_contract import (
    Modality,
    PlanGraph,
    PlanStep,
    RouteDecision,
)

# ---------------------------------------------------------------- raw planner shape

@dataclass
class RawReWOOStep:
    """What the NAT ReWOO planner produces internally per node, normalized.
    Field names map onto the ATIF / IntermediateStep trajectory payload:
      id        <- step identifier
      tool      <- tool/agent name selected by the planner
      args      <- planner-synthesized argument payload (the 'prompt')
      deps      <- antecedent step ids (DAG edges)
      hint      <- optional tier hint the planner attached ('edge'|'cloud')
    """
    id: str
    tool: str
    args: str
    deps: list[str]
    hint: str | None = None


# ---------------------------------------------------------------- tool -> modality map

# The planner names tools; the edge executor needs modality + a concrete model.
# This table is the single place that binding lives. CAIO owns the model_ids;
# CCE owns the tool->modality mapping. Keep them in sync at swarm-lock.
_TOOL_TABLE: dict[str, tuple[Modality, str]] = {
    # tool_name           : (modality,          edge/cloud model_id)
    "asr_transcribe":       (Modality.AUDIO,   "whisper-base"),
    "text_summarize":       (Modality.TEXT,    "qwen3-4b"),
    "text_parse":           (Modality.TEXT,    "qwen3-4b"),
    "vision_screen":        (Modality.VISION,  "vision-specialist"),
    "deep_reason":          (Modality.TEXT,    "nemotron-planner"),  # escalate target
}

# Tools that are inherently heavy => escalate unless the planner hint overrides.
_ESCALATE_BY_DEFAULT = {"deep_reason"}


def _decision_for(step: RawReWOOStep) -> RouteDecision:
    """Deterministic floor. CAIO's learned router refines this later, but the
    emitter must never produce an unroutable step. Planner hint wins; else the
    tool's default tier; else local."""
    if step.hint == "cloud":
        return RouteDecision.ESCALATE
    if step.hint == "edge":
        return RouteDecision.LOCAL
    if step.tool in _ESCALATE_BY_DEFAULT:
        return RouteDecision.ESCALATE
    return RouteDecision.LOCAL


def _modality_model(tool: str) -> tuple[Modality, str]:
    if tool not in _TOOL_TABLE:
        # Fail loud at emit time, not silently at the edge.
        raise KeyError(f"unmapped planner tool '{tool}' — add it to _TOOL_TABLE before lock")
    return _TOOL_TABLE[tool]


# ---------------------------------------------------------------- interception adapter

def emit_plan_graph(plan_id: str, raw: list[RawReWOOStep]) -> PlanGraph:
    """THE interception point. Call this from the NAT Middleware/IntermediateStep
    hook with the ReWOO DAG, BEFORE the Executor Node fires. Returns a contract
    PlanGraph; caller serializes + ships, and suppresses NAT's local execution."""
    steps: list[PlanStep] = []
    for r in raw:
        modality, model_id = _modality_model(r.tool)
        steps.append(
            PlanStep(
                step_id=r.id,
                modality=modality,
                decision=_decision_for(r),
                model_id=model_id,
                prompt=r.args,
                depends_on=list(r.deps),
            )
        )
    graph = PlanGraph(plan_id=plan_id, steps=steps)
    graph.topo_order()  # validate acyclicity at emit time — never ship a cyclic plan
    return graph


# ---------------------------------------------------------------- wire serialization

def to_wire(graph: PlanGraph) -> str:
    """Compact JSON the edge consumes. Strips the planning context entirely —
    only the DAG crosses the boundary (kilobyte-scale, per architecture doc).
    Protobuf is the scale roadmap; JSON is the build."""
    obj = {
        "plan_id": graph.plan_id,
        "steps": [
            {
                "step_id": s.step_id,
                "modality": s.modality.value,
                "decision": s.decision.value,
                "model_id": s.model_id,
                "prompt": s.prompt,
                "depends_on": s.depends_on,
            }
            for s in graph.steps
        ],
    }
    return json.dumps(obj, separators=(",", ":"))


def from_wire(blob: str) -> PlanGraph:
    """Edge-side inverse — included here so both sides share one codec and
    can't drift. CTO can move this into /shared at integration."""
    obj = json.loads(blob)
    steps = [
        PlanStep(
            step_id=s["step_id"],
            modality=Modality(s["modality"]),
            decision=RouteDecision(s["decision"]),
            model_id=s["model_id"],
            prompt=s["prompt"],
            depends_on=list(s.get("depends_on", [])),
        )
        for s in obj["steps"]
    ]
    return PlanGraph(plan_id=obj["plan_id"], steps=steps)


# ---------------------------------------------------------------- synthetic planner (NAT-free integration today)

def synthetic_rewoo_plan(user_request: str) -> list[RawReWOOStep]:
    """Stand-in for the NAT ReWOO planner so the edge team integrates against
    real PlanGraph JSON before the live Nemotron planner lands. Heuristic, not
    cognitive — deliberately. Swap for the real interception hook post-verify."""
    raw: list[RawReWOOStep] = []
    has_audio = bool(re.search(r"\b(audio|clip|recording|voice|call)\b", user_request, re.I))
    has_vision = bool(re.search(r"\b(screen|image|photo|scan|document)\b", user_request, re.I))
    needs_reason = bool(re.search(r"\b(analy|plan|cross-reference|decide|strategy|why)\b", user_request, re.I))

    last = None
    if has_audio:
        raw.append(RawReWOOStep("s_asr", "asr_transcribe", "transcribe the provided audio", []))
        last = "s_asr"
    if has_vision:
        raw.append(RawReWOOStep("s_vis", "vision_screen", "extract text/regions from the image", []))
        last = "s_vis" if last is None else last
    deps = [d for d in ["s_asr", "s_vis"] if any(x.id == d for x in raw)]
    raw.append(RawReWOOStep("s_sum", "text_summarize", f"summarize: {user_request}", deps))
    if needs_reason:
        raw.append(RawReWOOStep("s_reason", "deep_reason",
                                "deep-reason over the summary; emit decision", ["s_sum"]))
    return raw


# ---------------------------------------------------------------- self-test

def _demo() -> None:
    req = "Analyze this support call recording and cross-reference the screen scan to decide next action."
    raw = synthetic_rewoo_plan(req)
    graph = emit_plan_graph("plan-001", raw)
    wire = to_wire(graph)

    print("== emitted PlanGraph (cloud) ==")
    for s in graph.topo_order():
        print(f"  {s.step_id:9} [{s.decision.value:8}] {s.modality.value:6} -> {s.model_id:18} deps={s.depends_on}")
    print(f"\nwire bytes: {len(wire.encode())}  (target: kilobyte-scale)")
    print(f"wire: {wire}")

    # round-trip integrity (edge would do from_wire)
    back = from_wire(wire)
    assert to_wire(back) == wire, "codec drift — cloud/edge JSON mismatch"
    print("\nround-trip OK: cloud emit == edge decode")


if __name__ == "__main__":
    _demo()
