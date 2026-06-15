"""
HARP — shared/conformance.py  ·  Backend contract enforcement  ·  MIT

Closes the CI hole: the smoke test proves the MOCKS conform, but a real
QNNBackend / NIMBackend that drifts from the `Backend` ABC must fail the gate
too. CEE and CCE import assert_conforms() in their own test and the gate runs
it on edge/** and cloud/** changes.
"""

from __future__ import annotations

import asyncio
import inspect

from shared.harp_contract import (
    Backend, Capability, Metrics, InferRequest, Modality,
)


async def assert_conforms(backend: Backend) -> None:
    """Verify a backend satisfies the frozen contract at runtime — not just by
    subclassing, but by returning correctly-typed payloads from each method."""
    assert isinstance(backend, Backend), "must subclass Backend ABC"

    cap = await backend.capabilities()
    assert isinstance(cap, Capability), "capabilities() must return Capability"
    assert cap.modalities, "must declare at least one modality"

    req = InferRequest(
        messages=[{"role": "user", "content": "ping"}],
        model_id="conformance-probe",
        modality=Modality.TEXT,
    )

    stream = backend.infer(req)
    assert inspect.isasyncgen(stream) or hasattr(stream, "__aiter__"), \
        "infer() must return an async iterator (streaming is mandatory)"
    got = [tok async for tok in stream]
    assert got, "infer() must yield at least one token"

    m = await backend.profile(req)
    assert isinstance(m, Metrics), "profile() must return Metrics"
    assert m.ttft_ms >= 0 and m.tokens_per_s >= 0, "profile must populate ttft/tok-s"

    print(f"conformance OK: {cap.backend_id} ({cap.tier.value})")


if __name__ == "__main__":
    # Self-check against the reference mocks so the gate has a baseline.
    from shared.harp_contract import mock_edge, mock_cloud

    async def _run() -> None:
        await assert_conforms(mock_edge())
        await assert_conforms(mock_cloud())

    asyncio.run(_run())
