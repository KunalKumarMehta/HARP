"""
HARP — hardware-aware edge↔cloud routing
cloud/dag_extractor_middleware.py  ·  MIT  ·  NAT v1.7.0

Pre-execution DAG interception + suppression, built against the verified
installed API (introspected from nvidia-nat 1.7.0); note that the
conceptual `PreInvoke`/`HALT_EXECUTION` symbols do not exist in this version.

Real mechanism:
  - Subclass nat.middleware.FunctionMiddleware.
  - `enabled` property gates execution.
  - Override `function_middleware_invoke(*args, call_next, context, **kwargs)`
    for full control. To suppress the ReWOO Executor: detect the target by
    `context.name`, serialize the planned DAG, and return without calling
    `call_next`. Not calling call_next is the real "halt" — clean, no exception.
  - Register as a function-middleware plugin so the WorkflowBuilder attaches it
    at the Planner->Executor boundary.

What this intercepts: the Executor's tool-dispatch call. At that point the
ReWOO Planner has already produced the full DAG; we lift it out of the call
args, map it to the contract PlanGraph via plan_emitter, and short-circuit.
"""

from __future__ import annotations

import logging
from typing import Any

from nat.middleware import FunctionMiddleware, CallNext, FunctionMiddlewareContext
from nat.middleware.middleware import InvocationContext

from cloud.plan_emitter import RawReWOOStep, emit_plan_graph, to_wire

logger = logging.getLogger("harp.dag_extractor")

# Function names that signal the ReWOO Executor boundary. Plugin-dependent, so
# we match defensively on a set of substrings rather than one exact string.
_EXECUTOR_MARKERS = ("rewoo", "executor", "execute_plan", "plan_execute")


def _looks_like_executor(name: str) -> bool:
    n = (name or "").lower()
    return any(m in n for m in _EXECUTOR_MARKERS)


def _coerce_raw(dag: Any) -> list[RawReWOOStep]:
    """Normalize whatever the planner produced into RawReWOOStep list. Accepts
    our strict-JSON schema {'nodes':[...]} or a pydantic model exposing it."""
    if hasattr(dag, "model_dump"):
        dag = dag.model_dump()
    if isinstance(dag, str):
        import json
        dag = json.loads(dag)
    nodes = dag.get("nodes") if isinstance(dag, dict) else None
    if not nodes:
        raise ValueError("no 'nodes' in intercepted DAG payload")
    return [RawReWOOStep(n["id"], n["tool"], n.get("args", ""),
                         n.get("deps", []), n.get("hint")) for n in nodes]


class DagExtractorMiddleware(FunctionMiddleware):
    """Lifts the ReWOO DAG out at the Executor boundary, emits the contract
    PlanGraph wire JSON, and suppresses local execution. The emitted JSON is
    stashed on `self.last_emission` and returned as the function output so the
    caller (NAT runner / FastAPI front end) receives the plan instead of an
    executed result."""

    def __init__(self, *, on_emit=None) -> None:
        super().__init__(is_final=False)
        self._on_emit = on_emit          # optional sink: ship to edge transport
        self.last_emission: str | None = None

    @property
    def enabled(self) -> bool:
        return True

    # pre/post are abstract in the base; we do all work in the invoke override,
    # so these are pass-through no-ops.
    async def pre_invoke(self, context: InvocationContext) -> InvocationContext | None:
        return None

    async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
        return None

    async def function_middleware_invoke(
        self,
        *args: Any,
        call_next: CallNext,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> Any:
        # Only act at the ReWOO Executor boundary; everything else passes through.
        if not _looks_like_executor(context.name):
            return await call_next(*args, **kwargs)

        # The planned DAG is the Executor's input. It is the first positional arg
        # or a 'plan'/'dag' kwarg depending on the plugin's signature.
        dag = (args[0] if args else None) or kwargs.get("plan") or kwargs.get("dag")
        try:
            raw = _coerce_raw(dag)
        except Exception as e:                       # noqa: BLE001
            logger.warning("DAG extract failed at %s: %s — passing through", context.name, e)
            return await call_next(*args, **kwargs)  # fail open: let NAT run normally

        plan_id = getattr(dag, "plan_id", None) or (
            dag.get("plan_id") if isinstance(dag, dict) else None) or "plan"
        wire = to_wire(emit_plan_graph(plan_id, raw))
        self.last_emission = wire
        logger.info("intercepted ReWOO DAG at %s -> emitted %d-byte PlanGraph",
                    context.name, len(wire.encode()))
        if self._on_emit is not None:
            await _maybe_await(self._on_emit, wire)

        # Suppress: do not call call_next. Return the wire JSON as the output. The
        # local Executor never fires; the plan crosses to the edge instead.
        return wire


async def _maybe_await(fn, *a):
    import inspect
    r = fn(*a)
    if inspect.isawaitable(r):
        await r


# Plugin registration shape (NAT entry point). Lands in a PEP-420 ns package
# nat.plugins.harp_dag_extractor with this in its register module:
#
#   from nat.cli.register_workflow import register_function_middleware
#   @register_function_middleware(config_type=DagExtractorConfig)
#   async def build(cfg, builder):
#       yield DagExtractorMiddleware(on_emit=builder.get_transport("edge"))
#
# (register decorator name verified family: nat.cli.register_workflow.*)
