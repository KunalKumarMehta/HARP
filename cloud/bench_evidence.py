"""
HARP — Hardware-Aware Routing Platform
cloud/bench_evidence.py  ·  MIT

The NVIDIA scoring metric, operationalized. This harness drives the contract's
profile() against two configs (baseline vs optimized), computes the GenAI-Perf
metric family, and prints an evidence table with lines that map metrics to
business value.

Metric definitions:
  - TTFT = first CONTENT token; ITL = (e2e - TTFT)/(out_tokens - 1)
  - "goodput" = throughput under a fixed latency SLA (TTFT ceiling) — the metric
    that matters under a latency SLA
  - Key angles: roofline (memory- vs compute-bound), TCO (3x TPS at same SLA = 1/3 the GPUs)
  - Real sweep is `genai-perf analyze --sweep-type concurrency` on the deployed
    TRT-LLM engine in Triton; this harness is the in-process pre-flight + the
    table generator. Final numbers come from GenAI-Perf on the GPU box.

Runs against the mock cloud backend (no GPU needed) so the evidence pipeline is
built and tested before the real NIM/TRT-LLM engine is provisioned.
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass

from shared.harp_contract import Backend, InferRequest, Modality, mock_cloud

# Interactive-agent SLA: TTFT ceiling for fluid tool-use.
TTFT_SLA_MS = 500.0


@dataclass
class BenchResult:
    label: str
    ttft_ms_p50: float
    ttft_ms_p99: float
    tok_s_mean: float
    n_runs: int
    sla_pass_rate: float        # fraction of runs under TTFT_SLA_MS == goodput proxy

    def speedup_vs(self, base: "BenchResult") -> tuple[float, float]:
        ttft_x = base.ttft_ms_p50 / self.ttft_ms_p50 if self.ttft_ms_p50 else 0.0
        tps_x = self.tok_s_mean / base.tok_s_mean if base.tok_s_mean else 0.0
        return ttft_x, tps_x


async def run_config(be: Backend, label: str, req: InferRequest, runs: int = 12) -> BenchResult:
    """Drive profile() N times, aggregate GenAI-Perf-style percentiles. Warm-up
    run discarded (TRT-LLM/JIT warm path skew)."""
    await be.profile(req)  # warm-up, discarded
    ttfts: list[float] = []
    tok_s: list[float] = []
    passes = 0
    for _ in range(runs):
        m = await be.profile(req)
        ttfts.append(m.ttft_ms)
        tok_s.append(m.tokens_per_s)
        if m.ttft_ms <= TTFT_SLA_MS:
            passes += 1
    ttfts.sort()
    p99 = ttfts[min(len(ttfts) - 1, int(0.99 * len(ttfts)))]
    return BenchResult(
        label=label,
        ttft_ms_p50=statistics.median(ttfts),
        ttft_ms_p99=p99,
        tok_s_mean=statistics.fmean(tok_s),
        n_runs=runs,
        sla_pass_rate=passes / runs,
    )


# Defensible bands. A measured delta OUTSIDE these means the baseline is wrong.
#   HF/PyTorch static -> vLLM     : order-of-magnitude (up to ~75% fleet reduction)
#   vLLM -> TensorRT-LLM          : +15-30% throughput, -10-20% p50 TTFT  (NOT a big multiplier)
TRTLLM_TPS_BAND = (1.15, 1.30)
TRTLLM_TTFT_BAND = (1.10, 1.20)


def _flag(x: float, band: tuple[float, float]) -> str:
    lo, hi = band
    if x < lo:   return "below band"
    if x > hi:   return "ABOVE band — verify baseline is vLLM, not HF/PyTorch"
    return "defensible"


def render_pack(base: BenchResult, opt: BenchResult, baseline_engine: str = "vLLM") -> str:
    ttft_x, tps_x = opt.speedup_vs(base)
    gpu_fraction = (1.0 / tps_x) if tps_x else float("inf")
    tps_note = _flag(tps_x, TRTLLM_TPS_BAND) if baseline_engine == "vLLM" else "baseline=HF/PyTorch (large multiplier expected)"
    ttft_note = _flag(ttft_x, TRTLLM_TTFT_BAND) if baseline_engine == "vLLM" else ""
    lines = [
        "================ NVIDIA EVIDENCE PACK (cloud planner workload) ================",
        f"baseline engine: {baseline_engine}   optimized engine: TensorRT-LLM / Triton",
        f"{'metric':<22}{'baseline':>14}{'optimized':>14}{'delta':>12}",
        "-" * 62,
        f"{'TTFT p50 (ms)':<22}{base.ttft_ms_p50:>14.1f}{opt.ttft_ms_p50:>14.1f}{ttft_x:>11.2f}x  ({ttft_note})",
        f"{'TTFT p99 (ms)':<22}{base.ttft_ms_p99:>14.1f}{opt.ttft_ms_p99:>14.1f}",
        f"{'throughput (tok/s)':<22}{base.tok_s_mean:>14.1f}{opt.tok_s_mean:>14.1f}{tps_x:>11.2f}x  ({tps_note})",
        f"{'goodput @SLA<%dms' % TTFT_SLA_MS:<22}{base.sla_pass_rate:>13.0%}{opt.sla_pass_rate:>14.0%}",
        "-" * 62,
        "EVIDENCE FRAMING (cite measured values; ban aspirational TFLOPS):",
        f"  • TTFT: {ttft_x:.2f}x ({ttft_note}). State baseline explicitly: vs {baseline_engine}.",
        f"  • Throughput: {tps_x:.2f}x at fixed TTFT SLA = goodput, not raw TPS.",
        f"  • TCO: {tps_x:.2f}x goodput => ~{gpu_fraction:.2f} of the fleet for equal users.",
        "    (The order-of-magnitude win is HF->vLLM ~75% fleet cut; TRT-LLM is the",
        "     +15-30% efficiency-extraction layer on top — frame it as exactly that.)",
        "  • Roofline: show decode moving toward peak HBM BW via Paged KV + in-flight",
        "    batching (Nsight Compute). Ban peak-TFLOPS claims; decode is BW-bound.",
        "  • Safety, no latency tax: NeMo Guardrails stream_first=True, parallel rails.",
        "==============================================================================",
        "REAL NUMBERS (2026 — config.pbtxt MUST set exclude_input_in_output: true):",
        "  legacy:  genai-perf analyze -m <model> --backend tensorrtllm --endpoint-type chat \\",
        "           --streaming --sweep-type concurrency --sweep-range 1:256 --output-tokens-mean-deterministic",
        "  modern:  aiperf profile --model <model> --backend tensorrtllm \\",
        "           --search-space 'concurrency:1,1000:int' --search-metric output_token_throughput \\",
        "           --search-direction maximize    (Bayesian/Optuna; GenAI-Perf is deprecating)",
    ]
    return "\n".join(lines)


async def _demo() -> None:
    req = InferRequest(
        messages=[{"role": "user", "content": "deep-reason this multi-step plan and emit a decision graph"}],
        model_id="nemotron-planner",
        modality=Modality.TEXT,
    )
    # Baseline = mock cloud as-is. "Optimized" stub = same mock (delta ~1x here);
    # on the GPU box, baseline=HF/PyTorch NIM-off, optimized=TRT-LLM engine in Triton.
    base = await run_config(mock_cloud(), "baseline (pre-TRT-LLM)", req)
    opt = await run_config(mock_cloud(), "optimized (TRT-LLM/Triton)", req)
    print(render_pack(base, opt))


if __name__ == "__main__":
    asyncio.run(_demo())
