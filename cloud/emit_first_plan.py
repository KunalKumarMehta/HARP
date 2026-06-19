"""
HARP — Hardware-Aware Routing Platform
cloud/emit_first_plan.py  ·  MIT

FIRST REAL PLAN-GRAPH EMISSION. Drives the MANAGER (planner) NIM directly
through our verified NIMBackend, parses its JSON DAG into the contract
PlanGraph, and ships the wire payload. This path has ZERO dependency on the
NAT framework-plugin executor internals — it is the fastest, most robust route
to a real graph the moment HARP_NIM_API_KEY is set. The NAT rewoo config +
dag_extractor middleware are the full-system productionization of the same
boundary; this is the direct-invocation equivalent that emits now.

Run live:   HARP_NIM_API_KEY=... python -m cloud.emit_first_plan "Analyze the call recording and screen scan, then decide next action"
Run offline (parse proof): python -m cloud.emit_first_plan --mock
"""

from __future__ import annotations

import asyncio
import json
import re
import sys

from shared.harp_contract import PlanGraph
from cloud.model_registry import Role, resolve
from cloud.nim_backend import NIMBackend, NIMConfig
from cloud.plan_emitter import RawReWOOStep, emit_plan_graph, to_wire, from_wire

# Tools the planner may schedule (must match the edge executor's real capabilities).
_TOOLS = {
    "asr_transcribe": "transcribe an audio clip to text",
    "vision_screen": "extract text/regions from an image or screen",
    "text_summarize": "summarize text",
    "deep_reason": "deep multi-step reasoning over a summary (escalate to cloud)",
}

_SCHEMA = ('{"plan_id": str, "nodes": [{"id": str, "tool": str, "args": str, '
           '"deps": [str], "hint": "edge"|"cloud"|null}]}')


def _planner_messages(user_request: str) -> list[dict]:
    tool_lines = "\n".join(f"  - {t}: {d}" for t, d in _TOOLS.items())
    system = (
        "You are HARP's cloud planner. Decompose the user task into a minimal "
        "execution DAG using ONLY these tools:\n" + tool_lines + "\n\n"
        "Rules: parallelize independent steps (empty deps); chain dependent ones "
        "via deps (antecedent step ids). Mark deep_reason nodes hint=\"cloud\"; "
        "perception/summary nodes hint=\"edge\". EVERY node MUST have non-empty args: "
        "either a literal instruction OR a dataflow reference to upstream outputs "
        "in the form '<step_id>_output'; root nodes (no deps) reference the task input. "
        "Output STRICT JSON only, no prose, "
        "no markdown fences, matching exactly:\n" + _SCHEMA
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user_request}]


def _extract_json(text: str) -> dict:
    """Robust: strip ```fences, then grab the first balanced {...} object."""
    t = re.sub(r"```(?:json)?", "", text).strip()
    start = t.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in planner output:\n{text[:300]}")
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError("unbalanced JSON in planner output")


def _norm_args(node: dict, task_input: str) -> str:
    """Never emit empty args. If the planner left args blank, synthesize the
    dataflow binding the edge executor needs: upstream outputs for dependent
    nodes, the task input for roots."""
    a = (node.get("args") or "").strip()
    if a:
        return a
    deps = node.get("deps") or []
    return " + ".join(f"{d}_output" for d in deps) if deps else task_input


def _to_raw(dag: dict, task_input: str) -> list[RawReWOOStep]:
    nodes = dag.get("nodes") or []
    if not nodes:
        raise ValueError("planner returned no nodes")
    bad = [n["tool"] for n in nodes if n["tool"] not in _TOOLS]
    if bad:
        raise ValueError(f"planner hallucinated unknown tools: {bad}")
    return [RawReWOOStep(n["id"], n["tool"], _norm_args(n, task_input),
                         n.get("deps", []), n.get("hint")) for n in nodes]


async def emit(user_request: str, mock_output: str | None = None) -> PlanGraph:
    if mock_output is not None:
        raw_text = mock_output
        plan_id = "plan-mock-1"
    else:
        role = Role.MANAGER_PRAGMATIC
        be = NIMBackend(NIMConfig(enable_thinking=True), role=role)   # thinking=True LOCKED: yields dataflow-wired DAG
        from shared.harp_contract import InferRequest
        try:
            raw_text = "".join([t async for t in be.infer(
                InferRequest(messages=_planner_messages(user_request),
                             model_id=resolve(role).model_id,           # explicit, never ""
                             max_tokens=2048))])
        finally:
            await be.aclose()
        plan_id = "plan-live-1"
    dag = _extract_json(raw_text)
    dag.setdefault("plan_id", plan_id)
    graph = emit_plan_graph(dag["plan_id"], _to_raw(dag, user_request))
    return graph


def _print(graph: PlanGraph) -> None:
    wire = to_wire(graph)
    print("== FIRST PLAN-GRAPH (cloud planner -> edge) ==")
    for s in graph.topo_order():
        print(f"  {s.step_id:9} [{s.decision.value:8}] {s.modality.value:6} -> {s.model_id}")
    print(f"\nwire bytes: {len(wire.encode())}")
    print(f"wire: {wire}")
    assert to_wire(from_wire(wire)) == wire, "codec drift"
    print("round-trip OK — edge can decode this verbatim")


_MOCK = (
    '{"plan_id":"plan-mock-1","nodes":['
    '{"id":"s_asr","tool":"asr_transcribe","args":"transcribe the call recording","deps":[],"hint":"edge"},'
    '{"id":"s_vis","tool":"vision_screen","args":"extract text from the screen scan","deps":[],"hint":"edge"},'
    '{"id":"s_sum","tool":"text_summarize","args":"summarize transcript + screen text","deps":["s_asr","s_vis"],"hint":"edge"},'
    '{"id":"s_dr","tool":"deep_reason","args":"decide the next action","deps":["s_sum"],"hint":"cloud"}]}'
)


def main() -> None:
    args = sys.argv[1:]
    if "--mock" in args or not args:
        graph = asyncio.run(emit("(mock)", mock_output=_MOCK))
    else:
        req = " ".join(a for a in args if not a.startswith("--"))
        graph = asyncio.run(emit(req))
    _print(graph)


if __name__ == "__main__":
    main()
