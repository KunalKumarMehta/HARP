"""
HARP — tests/test_pre_llm_call_hook.py  ·  MIT

The Hermes pre_llm_call hook. Verifies the EXACT verified callback contract
(keyword-only + **kwargs tolerance), and that it fails safe to None on escalate or
any /route error. The HTTP call is mocked — no endpoint, no network.

Asserts:
  - local  -> {"runtime_override": {...valid keys...}, "context": <one line>}
  - escalate -> None
  - route server error / timeout -> None (no exception leaks)
  - the callback accepts the full verified kwarg set PLUS an unknown kwarg.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_HOOK = os.path.join(
    os.path.dirname(__file__), "..", "integrations", "hermes", "plugins", "hooks",
    "hardware-aware-router", "__init__.py")

_LOCAL_RESP = {
    "tier": "edge", "reason": "complexity_gate", "shed": False, "decision": "local",
    "runtime_override": {"provider": "harp", "model": "harp-edge",
                         "base_url": "http://127.0.0.1:8765/v1",
                         "api_mode": "chat_completions"},
}
_ESCALATE_RESP = {
    "tier": "cloud", "reason": "complexity_gate", "shed": False,
    "decision": "escalate", "runtime_override": None,
}

# The exact verified pre_llm_call kwargs, plus an unknown one (forward-compat).
_KW = dict(
    session_id="s1", user_message="hi", conversation_history=[],
    is_first_turn=True, model="nvidia-nim/nemotron", platform="cli",
    sender_id="u1", chat_id="c1", some_future_field="ignored-by-kwargs",
)


def _load():
    spec = importlib.util.spec_from_file_location("harp_hook", _HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patch(mod, resp=None, exc=None):
    def fake(url, payload, timeout_s):
        assert url.endswith("/route"), url
        assert "messages" in payload
        if exc is not None:
            raise exc
        return resp
    mod._post_route = fake


def test_decide_override_local() -> None:
    mod = _load()
    _patch(mod, resp=_LOCAL_RESP)
    ov = mod.decide_override("hi", [], harp_base_url="http://x/v1")
    assert ov == _LOCAL_RESP["runtime_override"]


def test_decide_override_escalate_is_none() -> None:
    mod = _load()
    _patch(mod, resp=_ESCALATE_RESP)
    assert mod.decide_override("hard multi-step plan", []) is None


def test_decide_override_fails_safe_on_error() -> None:
    mod = _load()
    _patch(mod, exc=TimeoutError("router slow"))
    assert mod.decide_override("anything", []) is None      # no exception leaks


def test_hook_local_returns_override_and_context() -> None:
    mod = _load()
    _patch(mod, resp=_LOCAL_RESP)
    out = mod._hook(**_KW)                                   # full verified kwargs + unknown
    assert isinstance(out, dict)
    ov = out["runtime_override"]
    assert ov["provider"] == "harp" and ov["model"] == "harp-edge"
    assert ov["api_mode"] == "chat_completions"
    assert isinstance(out.get("context"), str) and out["context"].startswith("[HARP]")
    assert "\n" not in out["context"]                       # one ephemeral line only


def test_hook_escalate_returns_none() -> None:
    mod = _load()
    _patch(mod, resp=_ESCALATE_RESP)
    assert mod._hook(**_KW) is None


def test_hook_error_returns_none() -> None:
    mod = _load()
    _patch(mod, exc=RuntimeError("route 500"))
    assert mod._hook(**_KW) is None                          # daemon-safe fail


def test_register_uses_pre_llm_call() -> None:
    mod = _load()
    captured = {}

    class Ctx:
        def register_hook(self, name, cb):
            captured["name"] = name
            captured["cb"] = cb

    mod.register(Ctx())
    assert captured["name"] == "pre_llm_call"
    assert captured["cb"] is mod._hook


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  OK {fn.__name__}")
    print(f"test_pre_llm_call_hook: {len(fns)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
