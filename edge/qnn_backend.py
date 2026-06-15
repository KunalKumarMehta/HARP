"""
HARP · edge/qnn_backend.py · CEE-owned · MIT
The real edge implementation of shared/harp_contract.Backend over the Snapdragon
Hexagon NPU. CCE's NIMBackend is the cloud twin; the router never imports either.

Grounding:
  - onnxruntime-genai (autoregressive loop, KV cache) + onnxruntime-qnn plugin EP   Deployment Walkthrough §"Inference Environment"
  - backend_path -> QnnHtp.dll / libQnnHtp.so; CPU/GPU DLL = silent slow fallback    §"EP Registration and DLL Linkage"
  - htp_performance_mode = burst; past_present_share_buffer; vtcm_mb                  §"Memory-Budget Management", §"thermal throttling"
  - sliding_window prefill shorter than window -> 0xc0000409 in QnnHtp.dll           §"Access Violations and Sliding Window Mismatches"
  - profile() must carry ttft/tok-s always, energy/thermal edge-only                 harp_contract.Metrics

Two guards are mandatory and are the reason this file exists rather than a thin wrapper:
  G1  assert_npu_engaged  — Risk A is meaningless if we silently ran on CPU. We FAIL
      LOUD here. (Production capability-fallback is the router's job, not the gate's.)
  G2  guard_prefill_length — pad short prompts to the window boundary so the HTP
      driver never executes the out-of-bounds access that hard-kills the process.

streaming: og's generate loop is blocking; contract.infer is AsyncIterator[str].
We run the loop in a worker thread and bridge tokens to an asyncio.Queue so TTFT
is the real wall-clock to first emitted token, not a simulated value.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

# Import the frozen contract. In-repo this is `from shared.harp_contract import ...`
from harp_contract import (
    Backend, Capability, InferRequest, Metrics, Modality, Tier,
)

try:
    import onnxruntime_genai as og
except ImportError:                       # host/dev box without the ARM64 runtime
    og = None                             # methods raise clearly if used un-provisioned

# CPU-fallback discriminator. X-Elite Qwen3-4B decode target is 30.3 tok/s (cap map);
# Oryon CPU decode for a 4B model is ~8 tok/s (Profiling Guide table). A 15 tok/s
# floor cleanly separates "NPU engaged" from "silently fell back to CPU".
NPU_DECODE_FLOOR_TOK_S = 15.0


@dataclass
class QnnModelSpec:
    """One compiled swarm specialist: a Context-Binary-backed ONNX wrapper dir
    (genai_config.json + tokenizer + the embed_in_onnx .onnx)."""
    model_id: str
    model_dir: str                        # holds genai_config.json + wrapper + tokenizer
    modality: Modality
    sliding_window: int | None = None     # set iff compiled with SWA (e.g. Qwen3-4B)
    pad_token_id: int = 0


class QNNBackend(Backend):
    def __init__(
        self,
        models: list[QnnModelSpec],
        *,
        ram_gb: float = 16.0,             # X Elite LPDDR5X budget (cap map / D4)
        max_context: int = 4096,
        htp_dll: str = "QnnHtp.dll",      # libQnnHtp.so on Linux/Android
        perf_mode: str = "burst",         # burst | sustained_high_performance
        vtcm_mb: int = 0,                 # 0 = driver picks max contiguous VTCM
        backend_id: str = "qnn-x-elite",
    ):
        self._specs = {m.model_id: m for m in models}
        self._loaded: dict[str, "og.Model"] = {}
        self._ram_gb = ram_gb
        self._max_context = max_context
        self._htp_dll = htp_dll
        self._perf_mode = perf_mode
        self._vtcm_mb = vtcm_mb
        self._backend_id = backend_id
        self._mods = tuple(sorted({m.modality for m in models}, key=lambda x: x.value))

    # ---- contract: capabilities -------------------------------------------------
    async def capabilities(self) -> Capability:
        return Capability(
            backend_id=self._backend_id,
            tier=Tier.EDGE,
            npu_present=og is not None,
            ram_gb=self._ram_gb,
            max_context=self._max_context,
            modalities=self._mods,
            offline_capable=True,         # whole point of the edge tier
            supports_streaming=True,
        )

    # ---- model loading + QNN EP wiring -----------------------------------------
    def _model(self, model_id: str) -> "og.Model":
        if og is None:
            raise RuntimeError("onnxruntime-genai not present: provision the native "
                               "ARM64 runtime (oga_setup.ps1) before inference.")
        if model_id in self._loaded:
            return self._loaded[model_id]
        spec = self._specs.get(model_id)
        if spec is None:
            raise KeyError(f"{model_id} not in swarm manifest: {list(self._specs)}")

        # Force the QNN HTP EP explicitly. Defaulting = CPU EP = the catastrophic
        # "successful but fractions-of-a-token/s" path the walkthrough warns about.
        cfg = og.Config(spec.model_dir)
        cfg.clear_providers()
        cfg.append_provider("qnn")
        cfg.set_provider_option("qnn", "backend_path", self._htp_dll)
        cfg.set_provider_option("qnn", "htp_performance_mode", self._perf_mode)
        if self._vtcm_mb:
            cfg.set_provider_option("qnn", "vtcm_mb", str(self._vtcm_mb))
        m = og.Model(cfg)
        self._loaded[model_id] = m
        return m

    # ---- G2: sliding-window prefill guard --------------------------------------
    @staticmethod
    def _guard_prefill_length(tokens: list[int], spec: QnnModelSpec) -> list[int]:
        w = spec.sliding_window
        if w and len(tokens) < w:
            # Pad to the compiled window boundary; a short prefill into a fixed
            # SWA tensor is the 0xc0000409 native access violation.
            return tokens + [spec.pad_token_id] * (w - len(tokens))
        return tokens

    def _encode(self, model: "og.Model", req: InferRequest) -> list[int]:
        tok = og.Tokenizer(model)
        # genai_config chat template is applied by apply_chat_template when present;
        # fall back to a flat join for specialists without a template.
        try:
            prompt = tok.apply_chat_template(messages=req.messages, add_generation_prompt=True)
        except Exception:
            prompt = "\n".join(m["content"] for m in req.messages)
        return list(tok.encode(prompt))

    # ---- the blocking generate loop (runs in a worker thread) ------------------
    def _generate_blocking(self, req: InferRequest, on_token, on_first):
        spec = self._specs[req.model_id]
        model = self._model(req.model_id)
        tok = og.Tokenizer(model)
        stream = tok.create_stream()

        ids = self._guard_prefill_length(self._encode(model, req), spec)

        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=min(len(ids) + req.max_tokens, self._max_context),
            past_present_share_buffer=True,   # single static KV buffer; walkthrough §"past_present_share_buffer"
        )
        gen = og.Generator(model, params)
        gen.append_tokens(ids)                # prefill

        first = True
        while not gen.is_done():
            gen.generate_next_token()
            if first:
                on_first()                    # mark TTFT at first decoded token
                first = False
            piece = stream.decode(gen.get_next_tokens()[0])
            if piece:
                on_token(piece)
        del gen                               # free the KV cache immediately

    # ---- contract: infer (async token stream) ----------------------------------
    async def infer(self, req: InferRequest):
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[str | None] = asyncio.Queue()

        def emit(piece: str): loop.call_soon_threadsafe(q.put_nowait, piece)
        def noop(): pass

        def worker():
            try:
                self._generate_blocking(req, on_token=emit, on_first=noop)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel

        threading.Thread(target=worker, daemon=True).start()
        while True:
            piece = await q.get()
            if piece is None:
                break
            yield piece

    # ---- contract: profile (the Qualcomm 40% metrics) --------------------------
    async def profile(self, req: InferRequest, power=None, baseline_w: float = 0.0,
                      thermal_fn=None) -> Metrics:
        """Wall-clock TTFT + decode tok/s. If a PowerSampler is passed, also returns
        energy mJ/tok (idle-baseline subtracted). thermal_fn() -> °C if available."""
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        ttft = {"v": None}
        n = {"v": 0}

        def on_first(): ttft["v"] = (time.perf_counter() - t0) * 1000.0
        def on_token(_): n["v"] += 1

        ctx = power if power is not None else _NullCtx()
        with ctx:
            await loop.run_in_executor(
                None, lambda: self._generate_blocking(req, on_token, on_first))
        total_s = time.perf_counter() - t0

        decode_s = max(total_s - (ttft["v"] or 0) / 1000.0, 1e-6)
        tok_s = n["v"] / decode_s

        energy = None
        if power is not None and n["v"]:
            energy = (power.trace.energy_joules(baseline_w) / n["v"]) * 1000.0  # J->mJ/tok

        return Metrics(
            backend_id=self._backend_id,
            ttft_ms=ttft["v"] or 0.0,
            tokens_per_s=tok_s,
            energy_mj_per_tok=energy,
            thermal_c=(thermal_fn() if thermal_fn else None),
        )

    # ---- G1: Risk A gate assertion ---------------------------------------------
    def assert_npu_engaged(self, metrics: Metrics) -> None:
        """FAIL LOUD if decode collapsed to CPU-fallback territory. This is the
        binary Risk-A pass/fail discriminator — not a production fallback path."""
        if metrics.tokens_per_s < NPU_DECODE_FLOOR_TOK_S:
            raise RuntimeError(
                f"NPU NOT ENGAGED: {metrics.tokens_per_s:.1f} tok/s < floor "
                f"{NPU_DECODE_FLOOR_TOK_S}. Likely silent CPU EP fallback — verify "
                f"backend_path={self._htp_dll} resolves on PATH and the context "
                f"binary matches this device's HTP arch (vXX).")


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
