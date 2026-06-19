"""
HARP · edge/bench_genie.py · MIT
Risk-A gate for a PRECOMPILED Genie bundle (the fast path: no self-compile).

`bench.py` benchmarks a self-compiled onnxruntime-genai model (QNNBackend).
This runs the SAME gate + evidence pack against a precompiled Genie
context-binary bundle (GenieBackend) — e.g. build/qwen3-4b-w4a16/ from
qualcomm/ai-hub-models. It reuses bench.py's run_gate / render_evidence_pack /
write_artifacts verbatim, because those are backend-agnostic (they only call
profile_detailed + assert_npu_engaged, which both backends implement).

On the QDC X Elite (genie-t2t-run on PATH):
    python -m edge.bench_genie                 # default WoS HWiNFO power rail
    python -m edge.bench_genie --target csv     # CSV power fallback (Free HWiNFO)
    python -m edge.bench_genie --target android # ADB sysfs (phone)
Off-device (no QAIRT): runs the conformant stub so the harness is verifiable,
and assert_npu_engaged() correctly FAILS the gate (stub < 15 tok/s) — it never
reports a passing NPU number without real silicon.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from edge.bench import GATE_PROBE, GateResult, run_gate, render_evidence_pack, write_artifacts
from edge.genie_backend import genie_qwen3_4b
from edge.power import get_sampler


async def main(target: str = "wos") -> None:
    backend = genie_qwen3_4b()                  # autodiscovers build/qwen3-4b-w4a16
    cap = await backend.capabilities()
    if not cap.npu_present:
        print("[warn] genie-t2t-run not found — running the off-device stub. Numbers "
              "below are NOT silicon; the gate FAILS by design.\n"
              "       On the QDC X Elite, provision it once:  edge\\bootstrap_qdc.cmd\n")

    try:
        sampler = get_sampler(target)
        baseline = sampler.measure_baseline(3.0)
    except Exception as e:                       # no power rail off-device — gate still runs
        print(f"[warn] power sampler '{target}' unavailable ({e}); latency-only run.")
        sampler, baseline = None, 0.0

    gate = await run_gate(backend, power=sampler, baseline_w=baseline, probe=GATE_PROBE)
    # Throughput alone can't prove NPU engagement — the stub is "fast" too. If the
    # Genie runtime isn't present, the gate FAILS regardless of measured tok/s.
    if not cap.npu_present:
        gate = GateResult(
            False, gate.metrics, gate.avg_watts, gate.peak_watts,
            "NPU NOT ENGAGED: genie-t2t-run absent (off-device stub). Provision "
            "QAIRT 2.45 on the X Elite (genie-t2t-run on PATH) to run for real.")
    write_artifacts(gate, None, None, "evidence_pack_genie.md", "evidence_pack_genie.json")
    print(render_evidence_pack(gate, None, None))
    print(f"\nartifacts: evidence_pack_genie.md · evidence_pack_genie.json")


if __name__ == "__main__":
    tgt = "wos"
    if "--target" in sys.argv:
        tgt = sys.argv[sys.argv.index("--target") + 1]
    asyncio.run(main(tgt))
