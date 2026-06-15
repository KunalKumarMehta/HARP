"""
HARP · edge/qnn_backend.py · CEE-owned · MIT
Real edge implementation of shared/harp_contract.Backend over the Hexagon NPU.

Grounding:
  - onnxruntime-genai + onnxruntime-qnn HTP EP                      Deployment Walkthrough
  - backend_path=QnnHtp.dll; default/CPU = silent slow fallback     §"EP Registration and DLL Linkage"
  - SWA prefill < window -> 0xc0000409                              §"Sliding Window Mismatches"
  - NO native timing hook; manual host-clock instrumentation        DR2 §"Timing Semantics"
  - VLMs need compute_logits() BEFORE generate_next_token()         DR2 §"Decode Phase"
  - max_length MUST be set or theoretical-max KV alloc crashes       DR2 §"append_tokens"
  - host clock is an UPPER bound on async NPU dispatch -> also emit
    qnn-profiling-data.csv via profiling_level for kernel truth      DR2 §"Extracting QNN-Specific Telemetry"
  - DRAM co-residency YES / VTCM exec co-residency NO; serialize via
    SEQUENTIAL_WITH_VA_OPTIMIZATION (~0.146 ms switch)               DR3 §"Architectural Recommendations"
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

from shared.harp_contract import (
    Backend, Capability, InferRequest, Metrics, Modality, Tier,
)

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

# X-Elite Qwen3-4B decode target 30.3 tok/s; Oryon CPU ~8 tok/s (Profiling Guide).
# 15 tok/s floor cleanly separates "NPU engaged" from "silent CPU fallback".
NPU_DECODE_FLOOR_TOK_S = 15.0


@dataclass
class QnnModelSpec:
    model_id: str
    model_dir: str
    modality: Modality
    sliding_window: int | None = None
    pad_token_id: int = 0
    requires_compute_logits: bool = False   # DR2: True for VLM/vision specialists
    weight_gb: float = 0.0                   # DR3: for DRAM-residency budgeting


@dataclass
class DetailedMetrics:
    """Richer than the frozen contract Metrics: carries the prefill/decode split
    the Qualcomm rubric wants. profile() still returns the contract Metrics."""
    backend_id: str
    prefill_ms: float        # append_tokens() wall time (compute-bound)
    ttft_ms: float           # t_start -> first generate_next_token (DR2 Table 1)
    decode_tok_s: float
    tokens: int
    energy_mj_per_tok: float | None = None
    thermal_c: float | None = None

    def to_contract(self) -> Metrics:
        return Metrics(self.backend_id, self.ttft_ms, self.decode_tok_s,
                       self.energy_mj_per_tok, self.thermal_c)


class QNNBackend(Backend):
    def __init__(
        self,
        models: list[QnnModelSpec],
        *,
        ram_gb: float = 16.0,
        max_context: int = 4096,
        htp_dll: str = "QnnHtp.dll",
        perf_mode: str = "burst",
        vtcm_mb: int = 0,
        # DR3: serialize execution; one model gets 100% VTCM/HVX/HMX at a time.
        residency: str = "sequential",
        graph_opt_mode: str = "3",          # htp_graph_finalization_optimization_mode
        backend_id: str = "qnn-x-elite",
        kernel_profiling: bool = False,     # DR2: emit qnn-profiling-data.csv
    ):
        self._specs = {m.model_id: m for m in models}
        self._loaded: dict[str, "og.Model"] = {}      # DRAM co-residency cache (DR3)
        self._ram_gb = ram_gb
        self._max_context = max_context
        self._htp_dll = htp_dll
        self._perf_mode = perf_mode
        self._vtcm_mb = vtcm_mb
        self._residency = residency
        self._graph_opt = graph_opt_mode
        self._backend_id = backend_id
        self._kernel_profiling = kernel_profiling
        self._mods = tuple(sorted({m.modality for m in models}, key=lambda x: x.value))

    async def capabilities(self) -> Capability:
        return Capability(
            backend_id=self._backend_id, tier=Tier.EDGE,
            npu_present=og is not None, ram_gb=self._ram_gb,
            max_context=self._max_context, modalities=self._mods,
            offline_capable=True, supports_streaming=True,
        )

    # ---- DR3: load every specialist into DRAM once at startup -------------------
    def warm_dram_residency(self) -> dict:
        """Preload all weights so per-query QnnContext_createFromBinary deserialize
        (multi-second from UFS/NVMe) never hits the hot path. ~2.2 GB for the
        Whisper+Gemma+Qwen3-4B swarm — trivial vs 16 GB budget (DR3)."""
        t0 = time.perf_counter()
        for mid in self._specs:
            self._model(mid)
        budget = sum(s.weight_gb for s in self._specs.values())
        return {"loaded": list(self._loaded), "weight_gb": round(budget, 2),
                "load_s": round(time.perf_counter() - t0, 3)}

    def _model(self, model_id: str) -> "og.Model":
        if og is None:
            raise RuntimeError("onnxruntime-genai absent: provision native ARM64 "
                               "runtime (oga_setup.ps1) before inference.")
        if model_id in self._loaded:
            return self._loaded[model_id]
        spec = self._specs.get(model_id)
        if spec is None:
            raise KeyError(f"{model_id} not in swarm manifest: {list(self._specs)}")

        cfg = og.Config(spec.model_dir)
        cfg.clear_providers()
        cfg.append_provider("qnn")
        cfg.set_provider_option("qnn", "backend_path", self._htp_dll)
        cfg.set_provider_option("qnn", "htp_performance_mode", self._perf_mode)
        cfg.set_provider_option("qnn", "htp_graph_finalization_optimization_mode", self._graph_opt)
        # DR3: SEQUENTIAL_WITH_VA_OPTIMIZATION — grant full VTCM to one graph; avoids
        # spill-fill from oversubscribing the 8 MB scratchpad across models.
        if self._residency == "sequential":
            cfg.set_provider_option("qnn", "htp_vtcm_optimization", "SEQUENTIAL_WITH_VA_OPTIMIZATION")
        if self._vtcm_mb:
            cfg.set_provider_option("qnn", "vtcm_mb", str(self._vtcm_mb))
        if self._kernel_profiling:                 # DR2: kernel-level latency CSV
            cfg.set_provider_option("qnn", "profiling_level", "detailed")
        m = og.Model(cfg)
        self._loaded[model_id] = m
        return m

    @staticmethod
    def _guard_prefill_length(tokens: list[int], spec: QnnModelSpec) -> list[int]:
        w = spec.sliding_window
        if w and len(tokens) < w:
            return tokens + [spec.pad_token_id] * (w - len(tokens))
        return tokens

    def _encode(self, model: "og.Model", req: InferRequest) -> list[int]:
        tok = og.Tokenizer(model)
        try:
            prompt = tok.apply_chat_template(messages=req.messages, add_generation_prompt=True)
        except Exception:
            prompt = "\n".join(m["content"] for m in req.messages)
        ids = list(tok.encode(prompt))
        if not ids:                                # DR2: empty KV prefill can fault
            ids = [self._specs[req.model_id].pad_token_id]
        return ids

    def _generate_blocking(self, req: InferRequest, on_token, on_first, timing=None):
        spec = self._specs[req.model_id]
        model = self._model(req.model_id)
        tok = og.Tokenizer(model)
        stream = tok.create_stream()
        ids = self._guard_prefill_length(self._encode(model, req), spec)

        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=min(len(ids) + req.max_tokens, self._max_context),  # DR2: never unset
            past_present_share_buffer=True,
        )
        gen = og.Generator(model, params)
        if self._kernel_profiling:
            gen.set_runtime_option("enable_profiling", f"{self._backend_id}_{req.model_id}")

        t_pre0 = time.perf_counter()
        gen.append_tokens(ids)                      # prefill (compute-bound)
        if timing is not None:
            timing["prefill_ms"] = (time.perf_counter() - t_pre0) * 1000.0

        first = True
        while not gen.is_done():
            if spec.requires_compute_logits:        # DR2: VLM state machine
                gen.compute_logits()
            gen.generate_next_token()
            if first:
                on_first()
                first = False
            piece = stream.decode(gen.get_next_tokens()[0])
            if piece:
                on_token(piece)
        del gen                                     # free KV cache immediately

    async def infer(self, req: InferRequest):
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[str | None] = asyncio.Queue()

        def emit(piece): loop.call_soon_threadsafe(q.put_nowait, piece)
        def noop(): pass

        def worker():
            try:
                self._generate_blocking(req, emit, noop)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            piece = await q.get()
            if piece is None:
                break
            yield piece

    async def profile_detailed(self, req: InferRequest, power=None,
                               baseline_w: float = 0.0, thermal_fn=None) -> DetailedMetrics:
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        ttft = {"v": None}
        n = {"v": 0}
        timing: dict = {}

        def on_first(): ttft["v"] = (time.perf_counter() - t0) * 1000.0
        def on_token(_): n["v"] += 1

        ctx = power if power is not None else _NullCtx()
        with ctx:
            await loop.run_in_executor(
                None, lambda: self._generate_blocking(req, on_token, on_first, timing))
        total_s = time.perf_counter() - t0

        decode_s = max(total_s - (ttft["v"] or 0) / 1000.0, 1e-6)
        # DR2 Table 1: TPS = (total - 1) / (t_end - t_first_token)
        tok_s = max(n["v"] - 1, 0) / decode_s if n["v"] > 1 else (n["v"] / decode_s)

        energy = None
        if power is not None and n["v"]:
            energy = (power.trace.energy_joules(baseline_w) / n["v"]) * 1000.0

        return DetailedMetrics(
            backend_id=self._backend_id,
            prefill_ms=timing.get("prefill_ms", 0.0),
            ttft_ms=ttft["v"] or 0.0,
            decode_tok_s=tok_s, tokens=n["v"],
            energy_mj_per_tok=energy,
            thermal_c=(thermal_fn() if thermal_fn else None),
        )

    async def profile(self, req: InferRequest, **kw) -> Metrics:
        return (await self.profile_detailed(req, **kw)).to_contract()

    def assert_npu_engaged(self, metrics: Metrics) -> None:
        if metrics.tokens_per_s < NPU_DECODE_FLOOR_TOK_S:
            raise RuntimeError(
                f"NPU NOT ENGAGED: {metrics.tokens_per_s:.1f} tok/s < floor "
                f"{NPU_DECODE_FLOOR_TOK_S}. Likely silent CPU EP fallback — verify "
                f"backend_path={self._htp_dll} on PATH and context-binary HTP arch (vXX).")


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
