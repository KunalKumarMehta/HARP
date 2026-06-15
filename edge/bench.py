"""
HARP · edge/bench.py · CEE-owned · MIT
The Risk-A gate instrument and the Qualcomm "energy efficiency" evidence engine.
One run produces: gate verdict, the NPU-vs-CPU A/B table winners show on screen,
and a markdown evidence pack mapped to the 40% Technical Implementation rubric.

Grounding (NPU Profiling Guide):
  - prefill TTFT (compute-bound) vs decode tok/s (bandwidth-bound) reported separately   §"Deconstructing TTFT"
  - energy-per-token via ∫P dt / N, idle baseline subtracted                              §"Energy-per-Token"
  - A/B split-screen NPU vs CPU (tok/s + watts) is the canonical winning artifact         §"Strategy 3 ... A/B Comparison"
  - sustained run reveals thermal throttle; report peak-retention %                        §"Thermal Throttling"
  - concurrent specialist residency under VTCM/RAM budget                                  CEE test mandate #3
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass

from shared.harp_contract import InferRequest, Metrics, Modality
from qnn_backend import QNNBackend, QnnModelSpec


# ---- probe set --------------------------------------------------------------
GATE_PROBE = InferRequest(
    messages=[{"role": "user",
               "content": "Summarize the tradeoffs of w4a16 quantization in two sentences."}],
    model_id="qwen3-4b",
    modality=Modality.TEXT,
    max_tokens=128,
)


@dataclass
class GateResult:
    passed: bool
    metrics: Metrics
    avg_watts: float | None
    peak_watts: float | None
    peak_thermal_c: float | None
    note: str


# ---- Risk A gate ------------------------------------------------------------
async def run_gate(backend: QNNBackend, *, power=None, baseline_w: float = 0.0,
                   thermal_fn=None, probe: InferRequest = GATE_PROBE) -> GateResult:
    """Compile-already-done assumption: backend points at the embed_in_onnx wrapper.
    This measures and renders the verdict. Raises via assert_npu_engaged on fail."""
    m = await backend.profile(probe, power=power, baseline_w=baseline_w, thermal_fn=thermal_fn)
    note = "PASS"
    passed = True
    try:
        backend.assert_npu_engaged(m)
    except RuntimeError as e:
        passed, note = False, str(e)
    return GateResult(
        passed=passed, metrics=m,
        avg_watts=(power.trace.avg_watts() if power else None),
        peak_watts=(power.trace.peak_watts() if power else None),
        peak_thermal_c=m.thermal_c, note=note,
    )


# ---- NPU vs CPU A/B (the on-screen winning artifact) ------------------------
async def run_ab(npu_backend: QNNBackend, cpu_backend: QNNBackend, *,
                 npu_power=None, cpu_power=None, baseline_w: float = 0.0,
                 probe: InferRequest = GATE_PROBE) -> dict:
    """Same probe, two EPs. cpu_backend is a QNNBackend built with htp_dll pointed
    at QnnCpu.dll (or an XNNPACK wrapper) — the deliberate slow baseline."""
    npu = await npu_backend.profile(probe, power=npu_power, baseline_w=baseline_w)
    cpu = await cpu_backend.profile(probe, power=cpu_power, baseline_w=baseline_w)
    speedup = (npu.tokens_per_s / cpu.tokens_per_s) if cpu.tokens_per_s else float("inf")
    return {"npu": npu, "cpu": cpu, "decode_speedup_x": speedup}


# ---- concurrent specialist residency (test mandate #3) ----------------------
async def run_concurrent_load(backend: QNNBackend, specs: list[QnnModelSpec]) -> dict:
    """Fire ASR + SLM + embedding inferences concurrently; report whether all
    completed without an OOM/op-create failure. Proves swarm co-residency under
    the 16 GB / VTCM budget (or forces the prompt-chaining workaround if not)."""
    reqs = [InferRequest([{"role": "user", "content": "probe"}], s.model_id, s.modality, max_tokens=16)
            for s in specs]

    async def drain(r):
        t0 = time.perf_counter()
        n = 0
        try:
            async for _ in backend.infer(r):
                n += 1
            return {"model_id": r.model_id, "ok": True, "tokens": n,
                    "wall_s": round(time.perf_counter() - t0, 3)}
        except Exception as e:
            return {"model_id": r.model_id, "ok": False, "error": str(e)}

    results = await asyncio.gather(*(drain(r) for r in reqs))
    return {"co_resident_ok": all(r["ok"] for r in results), "per_model": results}


# ---- reporting --------------------------------------------------------------
def render_evidence_pack(gate: GateResult, ab: dict | None = None,
                         concurrent: dict | None = None) -> str:
    m = gate.metrics
    L = []
    L.append("# HARP Edge — Qualcomm Technical Implementation Evidence Pack")
    L.append(f"_backend: `{m.backend_id}` · generated {time.strftime('%Y-%m-%d %H:%M %Z')}_\n")

    L.append("## Risk A Gate — On-Device Executor Is Real")
    L.append(f"**Verdict: {'✅ PASS' if gate.passed else '❌ FAIL'}**\n")
    L.append("| Metric | Value | Phase / Bound |")
    L.append("|---|---|---|")
    L.append(f"| TTFT | {m.ttft_ms:.0f} ms | prefill · compute-bound |")
    L.append(f"| Decode rate | {m.tokens_per_s:.1f} tok/s | decode · bandwidth-bound |")
    if m.energy_mj_per_tok is not None:
        L.append(f"| Energy / token | {m.energy_mj_per_tok:.1f} mJ | idle-baseline subtracted |")
    if gate.avg_watts is not None:
        L.append(f"| Avg system power | {gate.avg_watts:.1f} W | sustained |")
        L.append(f"| Peak system power | {gate.peak_watts:.1f} W | burst |")
    if m.thermal_c is not None:
        L.append(f"| Peak skin/zone temp | {m.thermal_c:.1f} °C | OEM cap ~45 °C |")
    if not gate.passed:
        L.append(f"\n> {gate.note}")

    if ab:
        n, c = ab["npu"], ab["cpu"]
        L.append("\n## A/B — NPU (QNN HTP) vs CPU fallback")
        L.append("| Backend | TTFT (ms) | Decode (tok/s) |")
        L.append("|---|---|---|")
        L.append(f"| Hexagon NPU | {n.ttft_ms:.0f} | {n.tokens_per_s:.1f} |")
        L.append(f"| Oryon CPU | {c.ttft_ms:.0f} | {c.tokens_per_s:.1f} |")
        L.append(f"\n**Decode speedup: {ab['decode_speedup_x']:.1f}× on NPU.**")

    if concurrent:
        L.append("\n## Concurrent Swarm Residency (16 GB / VTCM budget)")
        L.append(f"**Co-resident: {'✅ yes' if concurrent['co_resident_ok'] else '❌ no — prompt-chain fallback required'}**\n")
        L.append("| Model | OK | Tokens | Wall (s) |")
        L.append("|---|---|---|---|")
        for r in concurrent["per_model"]:
            L.append(f"| {r['model_id']} | {'✅' if r['ok'] else '❌'} | "
                     f"{r.get('tokens','—')} | {r.get('wall_s','—')} |")

    L.append("\n## Rubric mapping (40% Technical Implementation)")
    L.append("- **Latency** → TTFT + decode tok/s, prefill/decode split above.")
    L.append("- **Energy efficiency** → mJ/token + avg/peak watts, baseline-isolated.")
    L.append("- **Heterogeneous orchestration** → NPU-only LLM path proven via A/B vs CPU.")
    L.append("- **Robustness** → loud NPU-engagement assertion + SWA prefill guard (no `0xc0000409`).")
    return "\n".join(L)


def write_artifacts(gate: GateResult, ab: dict | None, concurrent: dict | None,
                    md_path: str, json_path: str) -> None:
    with open(md_path, "w") as f:
        f.write(render_evidence_pack(gate, ab, concurrent))
    blob = {
        "gate": {**asdict(gate), "metrics": asdict(gate.metrics)},
        "ab": ({"npu": asdict(ab["npu"]), "cpu": asdict(ab["cpu"]),
                "decode_speedup_x": ab["decode_speedup_x"]} if ab else None),
        "concurrent": concurrent,
    }
    with open(json_path, "w") as f:
        json.dump(blob, f, indent=2, default=str)


# ---- on-device entrypoint ---------------------------------------------------
async def main():
    """Runs ON the Snapdragon target after CEE has compiled Qwen3-4B to a Context
    Binary (embed_in_onnx) per the deployment walkthrough. power injected per D4 device."""
    from power import get_sampler  # local import: only available on the target

    backend = QNNBackend(models=[
        QnnModelSpec("qwen3-4b", "./models/qwen3-4b-w4a16", Modality.TEXT, sliding_window=None),
    ])
    sampler = get_sampler("android")           # or "wos" with reader=...
    baseline = sampler.measure_baseline(3.0)

    gate = await run_gate(backend, power=sampler, baseline_w=baseline,
                          thermal_fn=lambda: sampler.thermal_c())
    write_artifacts(gate, None, None,
                    "evidence_pack.md", "evidence_pack.json")
    print(render_evidence_pack(gate))


if __name__ == "__main__":
    asyncio.run(main())
