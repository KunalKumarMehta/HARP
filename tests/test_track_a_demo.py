"""
HARP — tests/test_track_a_demo.py  ·  MIT

The Track-A demo must run end-to-end off-device, produce a 6-row routing table, and
write a trace.jsonl with >=1 escalate and >=1 local (a real, believable split).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

from demo.track_a_routing_demo import run, TURNS


def test_demo_runs_and_splits() -> None:
    with tempfile.TemporaryDirectory() as d:
        trace = os.path.join(d, "trace.jsonl")
        records = run(trace_path=trace)

        assert len(records) == len(TURNS) == 6, records
        decisions = [r["decision"] for r in records]
        assert decisions.count("local") >= 1, decisions
        assert decisions.count("escalate") >= 1, decisions
        # the reason column must demonstrate more than one routing axis (e.g.
        # complexity_gate AND contention_shed) — not the same string every row.
        reasons = {r["reason"] for r in records}
        assert len(reasons) >= 2, f"demo should surface multiple axes, got {reasons}"
        assert "contention_shed" in reasons, reasons

        lines = [json.loads(x) for x in open(trace) if x.strip()]
        assert len(lines) == 6
        assert {"turn", "decision", "tier", "reason", "query"} <= set(lines[0])
        assert any(l["decision"] == "escalate" for l in lines)
        assert any(l["decision"] == "local" for l in lines)


def test_local_turns_carry_override() -> None:
    with tempfile.TemporaryDirectory() as d:
        records = run(trace_path=os.path.join(d, "t.jsonl"))
    for r in records:
        if r["decision"] == "local":
            ov = r["runtime_override"]
            assert ov and ov["provider"] == "harp" and ov["model"] == "harp-edge"
        else:
            assert r["runtime_override"] is None


def _main() -> int:
    test_demo_runs_and_splits()
    print("  OK demo runs, 6 rows, >=1 escalate + >=1 local")
    test_local_turns_carry_override()
    print("  OK local turns carry harp-edge override, escalate turns null")
    print("test_track_a_demo: passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
