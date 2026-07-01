"""
HARP — tests/test_mac_demo.py  ·  MIT

Gates the CANONICAL demo (mac_demo/harp_demo.py) in --mock mode: no models, no
keys, stdlib only. Asserts the routing gate produces a believable split (>=1
on-device + >=1 escalate) across two distinct triggers (complexity + contention),
and writes a trace. The live path (Ollama + Nemotron) is exercised by hand; this
locks the routing logic off-device.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

_DEMO = os.path.join(os.path.dirname(__file__), "..", "mac_demo", "harp_demo.py")


def _load():
    spec = importlib.util.spec_from_file_location("harp_mac_demo", _DEMO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mock_run_splits_two_axes() -> None:
    mod = _load()
    with tempfile.TemporaryDirectory() as d:
        trace = mod.run(mock=True, trace_path=os.path.join(d, "t.jsonl"))
        assert len(trace) == len(mod.TURNS) == 6, trace
        decisions = [r["decision"] for r in trace]
        assert decisions.count("local") >= 1, decisions
        assert decisions.count("escalate") >= 1, decisions
        reasons = {r["reason"] for r in trace}
        assert "contention_shed" in reasons, reasons          # busy-lane shed axis
        assert "complexity_gate" in reasons, reasons          # complexity axis
        # tiers are labeled "on-device" (not "edge") in the canonical demo
        assert any(r["tier"] == "on-device" for r in trace), trace
        assert os.path.getsize(os.path.join(d, "t.jsonl")) > 0    # trace written


def test_offline_fails_closed_to_on_device() -> None:
    mod = _load()
    with tempfile.TemporaryDirectory() as d:
        trace = mod.run(offline=True, mock=True, trace_path=os.path.join(d, "t.jsonl"))
        assert all(r["tier"] == "on-device" for r in trace), trace   # nothing escalates offline
        assert all(r["decision"] == "local" for r in trace), trace


def _main() -> int:
    test_mock_run_splits_two_axes()
    print("  OK mock run: 6 turns, on-device + escalate across complexity + contention")
    test_offline_fails_closed_to_on_device()
    print("  OK offline: every turn fails closed to on-device")
    print("test_mac_demo: passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
