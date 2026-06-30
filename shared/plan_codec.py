"""
HARP — hardware-aware edge↔cloud routing
shared/plan_codec.py  ·  PlanGraph wire codec  ·  MIT

The single serialization boundary for the cloud->edge plan handoff. The cloud
ATIF->PlanGraph adapter emits through to_wire(); the edge executor parses
through from_wire(). One codec, both sides — neither hand-rolls JSON.

Validation is two-layered because JSON Schema alone is insufficient:
  - SHAPE  (plan_schema.json): types, enums, required keys, additionalProperties.
    Uses `jsonschema` when installed (CI rigor); falls back to a dependency-free
    structural check so the edge runtime ships clean on Windows ARM64 (no
    win_arm64 wheel risk — same minimal-dependency design rule as the fabric).
  - SEMANTICS (here, in code): JSON Schema cannot express a DAG. We enforce
    unique step_ids, referential integrity of depends_on, and acyclicity
    (via PlanGraph.topo_order) — the constraints that actually break an executor.

ATIF richness (Step.extra: timeouts, hardware hints) collapses to the six wire
fields at the cloud adapter; `decision` is the only thing derived from extra that
crosses the wire. The schema stays NVIDIA-agnostic by design. If hardware hints
must reach the edge later, that is an explicit schema rev (optional `hints`),
not a silent additionalProperties leak.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from shared.harp_contract import Modality, PlanGraph, PlanStep, RouteDecision

_SCHEMA_PATH = Path(__file__).with_name("plan_schema.json")

try:
    import jsonschema
    _HAVE_JSONSCHEMA = True
except ImportError:                      # edge runtime path: no dep, fallback validates
    _HAVE_JSONSCHEMA = False

_STEP_KEYS = {"step_id", "modality", "decision", "model_id", "prompt", "depends_on"}
_MODALITIES = {m.value for m in Modality}
_DECISIONS = {d.value for d in RouteDecision}


class PlanWireError(ValueError):
    """Malformed or semantically-invalid plan on the wire. Fail loud, fail closed."""


@lru_cache(maxsize=1)
def _schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text())


# ---------------------------------------------------------------- shape validation

def _validate_shape(doc: object) -> None:
    if _HAVE_JSONSCHEMA:
        try:
            jsonschema.validate(doc, _schema())
        except jsonschema.ValidationError as e:
            raise PlanWireError(f"schema: {e.message}") from e
        return

    # dependency-free fallback — enforces the schema's hard constraints verbatim
    if not isinstance(doc, dict):
        raise PlanWireError("plan must be a JSON object")
    if doc.keys() - {"plan_id", "steps"}:
        raise PlanWireError(f"unexpected top-level keys: {doc.keys() - {'plan_id', 'steps'}}")
    if not isinstance(doc.get("plan_id"), str) or not doc["plan_id"]:
        raise PlanWireError("plan_id must be a non-empty string")
    steps = doc.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PlanWireError("steps must be a non-empty array")
    for s in steps:
        if not isinstance(s, dict):
            raise PlanWireError("each step must be an object")
        if s.keys() - _STEP_KEYS:
            raise PlanWireError(f"step has unexpected keys: {s.keys() - _STEP_KEYS}")
        if _STEP_KEYS - s.keys():
            raise PlanWireError(f"step missing keys: {_STEP_KEYS - s.keys()}")
        if not isinstance(s["step_id"], str) or not s["step_id"]:
            raise PlanWireError("step_id must be a non-empty string")
        if not isinstance(s["model_id"], str) or not s["model_id"]:
            raise PlanWireError("model_id must be a non-empty string")
        if not isinstance(s["prompt"], str):
            raise PlanWireError("prompt must be a string")
        if s["modality"] not in _MODALITIES:
            raise PlanWireError(f"bad modality {s['modality']!r}; want {_MODALITIES}")
        if s["decision"] not in _DECISIONS:
            raise PlanWireError(f"bad decision {s['decision']!r}; want {_DECISIONS}")
        if not isinstance(s["depends_on"], list) or not all(isinstance(d, str) for d in s["depends_on"]):
            raise PlanWireError("depends_on must be an array of strings")


# ---------------------------------------------------------------- semantic validation

def _validate_dag(steps: list[PlanStep]) -> None:
    ids: set[str] = set()
    for s in steps:
        if s.step_id in ids:
            raise PlanWireError(f"duplicate step_id {s.step_id!r}")
        ids.add(s.step_id)
    for s in steps:
        for d in s.depends_on:
            if d not in ids:
                raise PlanWireError(f"step {s.step_id!r} depends_on unknown step {d!r}")
    try:
        PlanGraph("_", steps).topo_order()   # raises ValueError on cycle
    except ValueError as e:
        raise PlanWireError(f"not a DAG: {e}") from e


# ---------------------------------------------------------------- encode

def to_wire(plan: PlanGraph) -> dict:
    """PlanGraph -> schema-valid dict. Asserts the DAG before emitting so a
    malformed plan never leaves the cloud."""
    _validate_dag(plan.steps)
    doc = {
        "plan_id": plan.plan_id,
        "steps": [{
            "step_id": s.step_id,
            "modality": s.modality.value,
            "decision": s.decision.value,
            "model_id": s.model_id,
            "prompt": s.prompt,
            "depends_on": list(s.depends_on),
        } for s in plan.steps],
    }
    _validate_shape(doc)
    return doc


def to_json(plan: PlanGraph) -> str:
    return json.dumps(to_wire(plan), separators=(",", ":"))   # compact, kilobyte-scale


# ---------------------------------------------------------------- decode

def from_wire(doc: object) -> PlanGraph:
    """Wire dict -> PlanGraph. Validates shape, coerces enums, enforces DAG
    integrity. Anything off -> PlanWireError; the edge executor never sees a
    half-valid plan."""
    _validate_shape(doc)
    assert isinstance(doc, dict)
    steps: list[PlanStep] = []
    for raw in doc["steps"]:
        try:
            modality = Modality(raw["modality"])
            decision = RouteDecision(raw["decision"])
        except ValueError as e:
            raise PlanWireError(f"enum coercion failed: {e}") from e
        steps.append(PlanStep(
            step_id=raw["step_id"], modality=modality, decision=decision,
            model_id=raw["model_id"], prompt=raw["prompt"],
            depends_on=list(raw["depends_on"]),
        ))
    _validate_dag(steps)
    return PlanGraph(plan_id=doc["plan_id"], steps=steps)


def from_json(text: str) -> PlanGraph:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise PlanWireError(f"invalid JSON: {e}") from e
    return from_wire(doc)


# ---------------------------------------------------------------- self-test (CI gate)

def _selftest() -> None:
    plan = PlanGraph("p1", [
        PlanStep("s1", Modality.AUDIO, RouteDecision.LOCAL, "whisper-base", "transcribe"),
        PlanStep("s2", Modality.TEXT, RouteDecision.AUTO, "qwen3-4b", "summarize", depends_on=["s1"]),
        PlanStep("s3", Modality.TEXT, RouteDecision.ESCALATE, "nemotron", "deep reason", depends_on=["s2"]),
    ])

    # 1. round-trip identity (incl. AUTO <-> "undecided")
    back = from_json(to_json(plan))
    assert back == plan, "round-trip must be lossless"
    assert '"decision":"undecided"' in to_json(plan), "AUTO must serialize as 'undecided'"

    def expect_fail(label: str, doc: object) -> None:
        try:
            from_wire(doc)
        except PlanWireError:
            return
        raise AssertionError(f"{label}: expected PlanWireError, got none")

    good = to_wire(plan)

    # 2. cycle
    cyc = json.loads(json.dumps(good))
    cyc["steps"][0]["depends_on"] = ["s3"]
    expect_fail("cycle", cyc)

    # 3. dangling dependency
    dangle = json.loads(json.dumps(good))
    dangle["steps"][1]["depends_on"] = ["s9"]
    expect_fail("dangling depends_on", dangle)

    # 4. unknown enum value
    bad_dec = json.loads(json.dumps(good))
    bad_dec["steps"][0]["decision"] = "maybe"
    expect_fail("bad decision enum", bad_dec)

    # 5. additionalProperties (extra key)
    extra = json.loads(json.dumps(good))
    extra["steps"][0]["secret"] = "leak"
    expect_fail("extra step key", extra)

    # 6. duplicate step_id
    dup = json.loads(json.dumps(good))
    dup["steps"][1]["step_id"] = "s1"
    expect_fail("duplicate step_id", dup)

    # 7. empty steps
    expect_fail("empty steps", {"plan_id": "p", "steps": []})

    backend = "jsonschema" if _HAVE_JSONSCHEMA else "fallback"
    print(f"shared/plan_codec: round-trip + 6 rejections OK (shape backend: {backend})")


if __name__ == "__main__":
    _selftest()
