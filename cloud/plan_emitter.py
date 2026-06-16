"""
HARP — Hardware-Aware Routing Platform
cloud/plan_emitter.py  ·  CCE owns this  ·  MIT  ·  v2 (verified against DR-2)

Cloud planner -> edge executor bridge. Converts the NAT ReWOO Planner Node's
upfront DAG into the contract's PlanGraph JSON. "We emit plans, not activations."

VERIFIED interception mechanism (NeMo Agent Toolkit Workflow doc, NAT v1.7.0):
  rewoo_agent is native (nat.agent.rewoo_agent.register). The ONLY pre-execution
  interception vector is the PreInvoke MIDDLEWARE hook. IntermediateStepManager,
  callbacks, and ATIF are ALL retrospective/post-flight — unusable for pre-exec.
  Correct path: custom middleware (PEP-420 ns pkg nat.plugins.dag_extractor)
  subscribed to PreInvoke -> assert target == ReWOO Executor phase -> extract DAG
  from input_data -> model_dump_json() -> return HALT_EXECUTION carrying the JSON.
  That cleanly bypasses the Executor Node without raising / corrupting telemetry.

LOCAL steps carry edge model_ids (CAIO's domain). ESCALATE steps resolve their
cloud model from the Manager/Worker registry — no hardcoded strings here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.harp_contract import Modality, PlanGraph, PlanStep, RouteDecision
from cloud.model_registry import Role, TOOL_TO_ROLE, resolve


@dataclass
class RawReWOOStep:
    """NAT ReWOO node, normalized from the DAG extracted at the PreInvoke hook.
    Mirrors the dict the Planner Node parses (nat.agent.rewoo_agent.agent)."""
    id: str
    tool: str
    args: str
    deps: list[str]
    hint: str | None = None          # planner-attached tier hint: 'edge'|'cloud'


# tool -> (modality, edge model_id used when the step stays LOCAL). CAIO owns these.
_EDGE_BINDING: dict[str, tuple[Modality, str]] = {
    "asr_transcribe": (Modality.AUDIO,  "whisper-base"),
    "vision_screen":  (Modality.VISION, "vision-specialist"),
    "text_summarize": (Modality.TEXT,   "qwen3-4b"),
    "text_parse":     (Modality.TEXT,   "qwen3-4b"),
    "deep_reason":    (Modality.TEXT,   "qwen3-4b"),   # local fallback if forced offline
}
_ESCALATE_BY_DEFAULT = {"deep_reason"}


def _decision_for(step: RawReWOOStep) -> RouteDecision:
    if step.hint == "cloud":
        return RouteDecision.ESCALATE
    if step.hint == "edge":
        return RouteDecision.LOCAL
    if step.tool in _ESCALATE_BY_DEFAULT:
        return RouteDecision.ESCALATE
    return RouteDecision.LOCAL


def _bind(step: RawReWOOStep, decision: RouteDecision) -> tuple[Modality, str]:
    if step.tool not in _EDGE_BINDING:
        raise KeyError(f"unmapped planner tool '{step.tool}' — add to _EDGE_BINDING/TOOL_TO_ROLE before lock")
    modality, edge_model = _EDGE_BINDING[step.tool]
    if decision == RouteDecision.ESCALATE:
        role: Role = TOOL_TO_ROLE.get(step.tool, Role.MANAGER_PRAGMATIC)
        return modality, resolve(role).model_id      # registry-resolved, swappable
    return modality, edge_model


def emit_plan_graph(plan_id: str, raw: list[RawReWOOStep]) -> PlanGraph:
    """Called by the PreInvoke middleware with the extracted ReWOO DAG, BEFORE
    the Executor Node fires. Returns a contract PlanGraph; caller serializes,
    ships, and returns HALT_EXECUTION to suppress local NAT execution."""
    steps: list[PlanStep] = []
    for r in raw:
        decision = _decision_for(r)
        modality, model_id = _bind(r, decision)
        # Contract requires prompt: str. Never ship "" — synthesize the dataflow
        # binding so the edge executor always has a concrete input handle.
        prompt = (r.args or "").strip() or (
            " + ".join(f"{d}_output" for d in r.deps) if r.deps else r.id + "_input")
        steps.append(PlanStep(step_id=r.id, modality=modality, decision=decision,
                              model_id=model_id, prompt=prompt, depends_on=list(r.deps)))
    g = PlanGraph(plan_id=plan_id, steps=steps)
    g.topo_order()      # reject cyclic plans at emit time
    return g


def to_wire(g: PlanGraph) -> str:
    return json.dumps({
        "plan_id": g.plan_id,
        "steps": [{"step_id": s.step_id, "modality": s.modality.value,
                   "decision": s.decision.value, "model_id": s.model_id,
                   "prompt": s.prompt, "depends_on": s.depends_on} for s in g.steps],
    }, separators=(",", ":"))


def from_wire(blob: str) -> PlanGraph:
    o = json.loads(blob)
    return PlanGraph(o["plan_id"], [
        PlanStep(s["step_id"], Modality(s["modality"]), RouteDecision(s["decision"]),
                 s["model_id"], s["prompt"], list(s.get("depends_on", []))) for s in o["steps"]])


# ---------------------------------------------------------------- PreInvoke middleware skeleton (NAT v1.7)
# Real integration target. Lands in a PEP-420 ns pkg: nat.plugins.dag_extractor.
# Subscribes to PreInvoke; fires exactly at the Planner->Executor hand-off.
_PREINVOKE_MIDDLEWARE_SKELETON = '''
# nat/plugins/dag_extractor/middleware.py   (NAT v1.7.0)
from nat.builder.middleware import BaseMiddleware, PreInvoke, HALT_EXECUTION  # framework base
from harp.cloud.plan_emitter import RawReWOOStep, emit_plan_graph, to_wire

class DagExtractorMiddleware(BaseMiddleware):
    hook = PreInvoke
    def on_pre_invoke(self, ctx):
        # fire only at the ReWOO Executor boundary, not on every tool call
        if "rewoo_agent" not in ctx.target_function_or_group:
            return None
        dag = ctx.input_data          # parsed Dependency Graph dict from Planner Node
        raw = [RawReWOOStep(n["id"], n["tool"], n["args"], n.get("deps", []), n.get("hint"))
               for n in dag["nodes"]]
        wire = to_wire(emit_plan_graph(dag.get("plan_id", "plan"), raw))
        # suppress local execution, carry the serialized DAG back up the stack
        return HALT_EXECUTION(payload=wire)
'''


def preinvoke_skeleton() -> str:
    return _PREINVOKE_MIDDLEWARE_SKELETON


# ---------------------------------------------------------------- synthetic planner (NAT-free integration today)
def synthetic_rewoo_plan(user_request: str) -> list[RawReWOOStep]:
    raw: list[RawReWOOStep] = []
    audio = bool(re.search(r"\b(audio|clip|recording|voice|call)\b", user_request, re.I))
    vision = bool(re.search(r"\b(screen|image|photo|scan|document)\b", user_request, re.I))
    reason = bool(re.search(r"\b(analy|plan|cross-reference|decide|strategy|why)\b", user_request, re.I))
    if audio:
        raw.append(RawReWOOStep("s_asr", "asr_transcribe", "transcribe the provided audio", []))
    if vision:
        raw.append(RawReWOOStep("s_vis", "vision_screen", "extract text/regions from the image", []))
    deps = [d for d in ["s_asr", "s_vis"] if any(x.id == d for x in raw)]
    raw.append(RawReWOOStep("s_sum", "text_summarize", f"summarize: {user_request}", deps))
    if reason:
        raw.append(RawReWOOStep("s_reason", "deep_reason", "deep-reason the summary; emit decision", ["s_sum"]))
    return raw


def _demo() -> None:
    req = "Analyze this support call recording and cross-reference the screen scan to decide next action."
    g = emit_plan_graph("plan-001", synthetic_rewoo_plan(req))
    wire = to_wire(g)
    print("== emitted PlanGraph (cloud, registry-resolved escalate target) ==")
    for s in g.topo_order():
        print(f"  {s.step_id:9} [{s.decision.value:8}] {s.modality.value:6} -> {s.model_id}")
    print(f"\nwire bytes: {len(wire.encode())}")
    assert to_wire(from_wire(wire)) == wire, "codec drift"
    print("round-trip OK")


if __name__ == "__main__":
    _demo()
