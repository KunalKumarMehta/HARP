"""
HARP — Hermes pre_llm_call routing hook  ·  MIT

The agentic face of HARP: a hardware-aware routing decision on EVERY turn. Before
each LLM call, the hook asks the HARP endpoint (POST /v1/route — advisory, no
inference) whether the turn belongs on-device (NPU) or in the cloud planner, and:

  - LOCAL  -> returns a request-scoped runtime_override pinning this turn to the
              `harp` provider / `harp-edge` model (the NPU lane). Auto-reverts after
              the turn.
  - ESCALATE -> returns None: Hermes' configured primary model (Nemotron via NIM)
                handles the turn natively, tools and all.

The routing BRAIN lives in the HARP endpoint (router/router_policy.py). This hook
is a thin adapter — it embeds NO classifier, so it can never drift from the model
the endpoint actually serves.

Hot-path safety: the /route call has a small timeout and fails to None on ANY
error, so a slow or down router never stalls a turn. Hermes also catches hook
exceptions, but we never rely on that — we fail safe here.
"""
from __future__ import annotations

import os

DEFAULT_BASE_URL = "http://127.0.0.1:8765/v1"
_VALID_API_MODES = {"chat_completions", "anthropic_messages",
                    "codex_responses", "bedrock_converse"}


def _post_route(url: str, payload: dict, timeout_s: float) -> dict:
    """Single seam for the HTTP call — tests monkeypatch this. httpx is imported
    LAZILY: it is an optional HARP dependency, so a Hermes runtime without it must
    degrade to a router-unavailable error (caught by decide_override -> None), NOT
    crash at module import. Raises on any transport/HTTP/import error."""
    import httpx
    resp = httpx.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def decide_override(
    user_message: str,
    conversation_history: list[dict] | None = None,
    *,
    harp_base_url: str = DEFAULT_BASE_URL,
    tools: list | None = None,
    timeout_s: float = 0.4,
) -> dict | None:
    """Pure, unit-testable core. POST the turn to {harp_base_url}/route and return
    the runtime_override dict on a LOCAL decision, or None on ESCALATE / any error.

    Fail-safe: ANY exception (timeout, connection refused, bad JSON, HTTP 5xx) ->
    None, so the turn falls through to Hermes' primary model. A router problem must
    never break a conversation."""
    try:
        url = harp_base_url.rstrip("/") + "/route"
        payload: dict = {"messages": [{"role": "user", "content": user_message or ""}]}
        if tools:
            payload["tools"] = tools
        data = _post_route(url, payload, timeout_s)
        if data.get("decision") != "local":
            return None                       # escalate (or unknown) -> primary model
        override = data.get("runtime_override")
        if not isinstance(override, dict):
            return None
        # Defensive: only forward a well-formed override (provider + model required).
        if not override.get("provider") or not override.get("model"):
            return None
        if override.get("api_mode") and override["api_mode"] not in _VALID_API_MODES:
            return None
        return override
    except Exception:
        return None                           # fail safe to the primary model


def _telemetry_line(override: dict) -> str:
    """One ephemeral line so the routing decision is visible in the transcript.
    Appended as `context`; never mutates history or the system prompt. provider/
    model cross a network boundary, so newlines are stripped — the docstring
    promises ONE line and we keep it one line regardless of the response."""
    provider = str(override.get("provider")).replace("\n", " ")
    model = str(override.get("model")).replace("\n", " ")
    return (f"[HARP] this turn routed on-device → {provider}/{model} "
            f"(NPU lane, privacy-preserving, offline-capable)")


def _hook(*, session_id: str = "", user_message: str = "",
          conversation_history: list[dict] | None = None, is_first_turn: bool = False,
          model: str = "", platform: str = "", sender_id: str = "", chat_id: str = "",
          **kwargs) -> dict | None:
    """Hermes pre_llm_call callback. KEYWORD-ONLY + **kwargs for forward-compat.
    Returns a dict with runtime_override (+ one ephemeral telemetry line) on a LOCAL
    turn, or None to defer to the primary model. Never raises — fail safe to None."""
    try:
        base_url = os.environ.get("HARP_BASE_URL", DEFAULT_BASE_URL)
        tools = kwargs.get("tools")
        override = decide_override(
            user_message, conversation_history,
            harp_base_url=base_url, tools=tools)
        if not override:
            return None
        return {"runtime_override": override, "context": _telemetry_line(override)}
    except Exception:
        return None


def register(ctx) -> None:
    """Entry point Hermes calls to install the hook."""
    ctx.register_hook("pre_llm_call", _hook)


# Auto-register when loaded inside a Hermes runtime; a no-op standalone (tests).
try:
    from hermes.plugins import current_context  # type: ignore

    register(current_context())
except Exception:  # not inside a Hermes runtime
    pass
