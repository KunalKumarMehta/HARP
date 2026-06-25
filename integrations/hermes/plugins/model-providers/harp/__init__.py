"""
HARP — Hermes model-provider plugin  ·  MIT

Registers HARP as a custom model provider inside a Hermes install. The agent then
selects `custom/harp` and points at the HARP OpenAI-compatible endpoint; every
turn flows through HARP's hardware-aware NPU/cloud router.

The real `providers` package only exists inside a Hermes runtime, so registration
is guarded — this module imports and unit-tests standalone via the pure factory
`build_harp_provider_profile()`.
"""
from __future__ import annotations

import os


def build_harp_provider_profile() -> dict:
    """Pure factory for the ProviderProfile kwargs. Standalone-testable; honors
    HARP_BASE_URL at call time so the endpoint location is deployment-config, not
    code. default_aux_model pins the aux lane to harp-edge: the aux lane is
    background work (summarization, title-gen) — the NPU single-stream sweet spot —
    so it never contends with the foreground lane and never leaves the device."""
    return dict(
        name="harp",
        aliases=("harp-router",),
        display_name="HARP Hardware-Aware Router",
        description=(
            "Routes each turn to the right model on the right tier — on-device NPU "
            "vs. cloud planner — with graceful offline fallback. OpenAI-compatible."
        ),
        signup_url="",
        env_vars=("HARP_API_KEY", "HARP_BASE_URL"),
        base_url=os.environ.get("HARP_BASE_URL", "http://127.0.0.1:8765/v1"),
        auth_type="api_key",
        api_mode="chat_completions",
        default_aux_model="harp-edge",          # aux lane -> NPU (single-stream safe)
        fallback_models=("harp-auto", "harp-edge", "harp-cloud"),
    )


# Module-level registration — a no-op unless we're inside a Hermes runtime.
try:
    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(ProviderProfile(**build_harp_provider_profile()))
except Exception:  # not inside a Hermes runtime (the common, standalone case)
    pass
