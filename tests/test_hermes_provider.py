"""
HARP — tests/test_hermes_provider.py  ·  MIT

The Hermes model-provider plugin must build a correct ProviderProfile kwargs dict
standalone (no Hermes runtime present). Asserts the required keys/values,
default_aux_model pins the aux lane to harp-edge (NPU), and base_url honors
HARP_BASE_URL at call time.

The plugin lives under a `model-providers` dir (hyphen — not import-safe), so it's
loaded by file path.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_PLUGIN = os.path.join(
    os.path.dirname(__file__), "..", "integrations", "hermes", "plugins",
    "model-providers", "harp", "__init__.py")


def _load():
    spec = importlib.util.spec_from_file_location("harp_provider", _PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # registration is guarded -> no-op standalone
    return mod


def test_profile_required_fields() -> None:
    p = _load().build_harp_provider_profile()
    assert p["name"] == "harp"
    assert p["aliases"] == ("harp-router",)
    assert p["display_name"] == "HARP Hardware-Aware Router"
    assert p["description"]
    assert p["signup_url"] == ""
    assert p["env_vars"] == ("HARP_API_KEY", "HARP_BASE_URL")
    assert p["auth_type"] == "api_key"
    assert p["api_mode"] == "chat_completions"
    assert p["fallback_models"] == ("harp-auto", "harp-edge", "harp-cloud")


def test_default_aux_model_is_edge() -> None:
    # aux lane -> NPU: background summarization is the single-stream sweet spot.
    assert _load().build_harp_provider_profile()["default_aux_model"] == "harp-edge"


def test_base_url_honors_env() -> None:
    build = _load().build_harp_provider_profile
    saved = os.environ.get("HARP_BASE_URL")
    try:
        os.environ.pop("HARP_BASE_URL", None)
        assert build()["base_url"] == "http://127.0.0.1:8765/v1"   # default
        os.environ["HARP_BASE_URL"] = "https://harp.example/v1"
        assert build()["base_url"] == "https://harp.example/v1"    # env wins
    finally:
        if saved is None:
            os.environ.pop("HARP_BASE_URL", None)
        else:
            os.environ["HARP_BASE_URL"] = saved


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  OK {fn.__name__}")
    print(f"test_hermes_provider: {len(fns)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
