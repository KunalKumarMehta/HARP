"""
HARP — Observability/Tracing for Routing Decisions  ·  MIT

Structured logging + timing for PolicyRouter decisions. Designed to be
lightweight (no external dependencies) but extensible to OpenTelemetry, etc.

Key events traced:
  - route.decide: AUTO step resolved (LOCAL/ESCALATE) with u, p_err, delta, reason
  - route.pin_honored: planner LOCAL/ESCALATE pin respected
  - route.guard: hardware/offline guard forced LOCAL
  - route.dispatch: backend selected, duration, tokens, tier
  - route.error: step failed, error type
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


# --- Configuration -----------------------------------------------------------

_TRACING_ENABLED = os.getenv("HARP_TRACE", "0").lower() in ("1", "true", "yes")
_TRACE_LEVEL = os.getenv("HARP_TRACE_LEVEL", "INFO").upper()  # DEBUG, INFO, WARN
_TRACE_OUTPUT = os.getenv("HARP_TRACE_OUTPUT", "stdout")  # stdout, stderr, file:/path
_TRACE_JSON = os.getenv("HARP_TRACE_JSON", "0").lower() in ("1", "true", "yes")


def enabled() -> bool:
    """Cheap check call sites use to skip ALL trace work (event construction +
    timestamp) on the hot path when HARP_TRACE is unset. Zero-cost-when-off."""
    return _TRACING_ENABLED


def _now_iso() -> str:
    # timezone-aware UTC; datetime.utcnow() is deprecated in 3.12+.
    return datetime.now(timezone.utc).isoformat()


# --- Data Classes ------------------------------------------------------------

@dataclass(frozen=True)
class TraceEvent:
    """A single routing trace event."""
    timestamp: str
    event: str                    # route.decide, route.pin_honored, etc.
    step_id: str
    plan_id: str
    decision_in: str             # AUTO, LOCAL, ESCALATE
    decision_out: str            # LOCAL, ESCALATE
    tier: Optional[str]          # edge, cloud
    u: Optional[float] = None
    p_err: Optional[float] = None
    delta: Optional[float] = None
    reason: str = ""
    overhead_ms: float = 0.0
    tokens: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        if not self.metadata:
            d.pop("metadata", None)
        return d


# --- Event Emitter -----------------------------------------------------------

class TraceEmitter:
    """Emits trace events to configured output."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._file = None
        if _TRACE_OUTPUT.startswith("file:"):
            path = _TRACE_OUTPUT[5:]
            self._file = open(path, "a", buffering=1)
    
    def emit(self, event: TraceEvent) -> None:
        if not _TRACING_ENABLED:
            return
        if _TRACE_LEVEL == "DEBUG" or event.event in ("route.decide", "route.pin_honored", "route.guard", "route.error"):
            pass  # always emit key events
        elif _TRACE_LEVEL == "WARN" and not event.error:
            return  # only emit errors/warnings
        
        with self._lock:
            if _TRACE_JSON:
                line = json.dumps(event.to_dict())
            else:
                # Compact human-readable format
                parts = [
                    event.timestamp,
                    event.event,
                    f"step={event.step_id}",
                    f"plan={event.plan_id}",
                    f"in={event.decision_in}",
                    f"out={event.decision_out}",
                ]
                if event.tier:
                    parts.append(f"tier={event.tier}")
                if event.u is not None:
                    parts.append(f"u={event.u:.3f}")
                if event.p_err is not None:
                    parts.append(f"p_err={event.p_err:.3f}")
                if event.reason:
                    parts.append(f"reason={event.reason}")
                if event.overhead_ms:
                    parts.append(f"overhead={event.overhead_ms:.2f}ms")
                if event.duration_ms:
                    parts.append(f"duration={event.duration_ms:.1f}ms")
                if event.error:
                    parts.append(f"error={event.error}")
                line = " ".join(parts)
            
            if self._file:
                self._file.write(line + "\n")
            else:
                stream = sys.stderr if _TRACE_OUTPUT == "stderr" else sys.stdout
                print(line, file=stream)
    
    def close(self):
        if self._file:
            self._file.close()


# Global emitter (initialized on first use)
_default_emitter: Optional[TraceEmitter] = None

def get_emitter() -> TraceEmitter:
    global _default_emitter
    if _default_emitter is None:
        _default_emitter = TraceEmitter()
    return _default_emitter


# --- Convenience Functions ---------------------------------------------------

def trace_decide(step_id: str, plan_id: str, decision_in: str, verdict: Any) -> None:
    """Trace a routing decision from PolicyRouter.decide()."""
    if not _TRACING_ENABLED:
        return
    get_emitter().emit(TraceEvent(
        timestamp=_now_iso(),
        event="route.decide",
        step_id=step_id,
        plan_id=plan_id,
        decision_in=decision_in,
        decision_out=verdict.decision.value,
        tier=None,
        u=verdict.u,
        p_err=verdict.p_err,
        delta=verdict.delta,
        reason=verdict.reason,
        overhead_ms=verdict.overhead_ms,
    ))


def trace_pin_honored(step_id: str, plan_id: str, decision_in: str, tier: str) -> None:
    """Trace a planner pin being honored (LOCAL/ESCALATE)."""
    if not _TRACING_ENABLED:
        return
    get_emitter().emit(TraceEvent(
        timestamp=_now_iso(),
        event="route.pin_honored",
        step_id=step_id,
        plan_id=plan_id,
        decision_in=decision_in,
        decision_out=decision_in,
        tier=tier,
        reason="planner_pin",
    ))


def trace_guard(step_id: str, plan_id: str, decision_in: str, tier: str, reason: str) -> None:
    """Trace hardware/offline guard forcing a tier."""
    if not _TRACING_ENABLED:
        return
    get_emitter().emit(TraceEvent(
        timestamp=_now_iso(),
        event="route.guard",
        step_id=step_id,
        plan_id=plan_id,
        decision_in=decision_in,
        decision_out="LOCAL" if tier == "edge" else "ESCALATE",
        tier=tier,
        reason=reason,
    ))


def trace_dispatch(step_id: str, plan_id: str, decision_in: str, decision_out: str,
                   tier: str, tokens: int, duration_ms: float, error: str | None = None) -> None:
    """Trace a backend dispatch result."""
    if not _TRACING_ENABLED:
        return
    get_emitter().emit(TraceEvent(
        timestamp=_now_iso(),
        event="route.error" if error else "route.dispatch",
        step_id=step_id,
        plan_id=plan_id,
        decision_in=decision_in,
        decision_out=decision_out,
        tier=tier,
        tokens=tokens,
        duration_ms=duration_ms,
        error=error,
    ))


# --- OpenTelemetry Bridge (optional, for future) ----------------------------

def setup_otel(exporter_endpoint: str = "http://localhost:4317", service_name: str = "harp-router"):
    """Set up OpenTelemetry tracing (requires opentelemetry-sdk)."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        
        provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
        trace.set_tracer_provider(provider)
        exporter = OTLPSpanExporter(endpoint=exporter_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        return trace.get_tracer(__name__)
    except ImportError:
        return None


# --- Demo --------------------------------------------------------------------

if __name__ == "__main__":
    # _TRACING_ENABLED is read at import; force it on for this self-demo.
    _TRACING_ENABLED = True
    _TRACE_JSON = True

    # Simulate some events
    from types import SimpleNamespace
    
    v = SimpleNamespace(decision=SimpleNamespace(value="LOCAL"), u=0.1, p_err=0.05, delta=0.5, 
                        reason="complexity_gate", overhead_ms=0.5)
    trace_decide("s1", "plan-1", "AUTO", v)
    trace_pin_honored("s2", "plan-1", "LOCAL", "edge")
    trace_guard("s3", "plan-1", "ESCALATE", "edge", "offline_forced_local")
    trace_dispatch("s1", "plan-1", "AUTO", "LOCAL", "edge", 10, 150.0)
    trace_dispatch("s4", "plan-1", "ESCALATE", "ESCALATE", "cloud", 25, 500.0, "timeout")
    
    get_emitter().close()
    print("Trace demo complete")