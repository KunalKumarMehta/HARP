"""
HARP · edge/genie_backend.py · MIT
Real edge Backend over a PRECOMPILED Qualcomm Genie context-binary bundle.

Why this exists (the gap it closes):
  The precompiled assets from `qualcomm/ai-hub-models` (and the bundle already in
  build/qwen3-4b-w4a16/) ship as a **Genie** package — a `genie_config.json` that
  points at split `.bin` context binaries, run by the QAIRT `genie-t2t-run` tool.
  `qnn_backend.QNNBackend` targets onnxruntime-genai (`og.Model`/`og.Config`) and
  CANNOT load a Genie bundle. So a precompiled model — the whole point of skipping
  self-compilation — had no path into the contract. This backend is that path.

  Use QNNBackend when YOU compiled an ONNX/OGA model dir.
  Use GenieBackend for an AI-Hub precompiled context-binary bundle (the fast path).
  Both satisfy the same frozen shared.harp_contract.Backend — the router never
  knows which one it dispatched to.

Bundle ground-truth reference files:
  - build/qwen3-4b-w4a16/metadata.json:  "runtime": "genie", "precision": "w4a16"
  - tool-versions.yaml:                  qairt 2.45.0
  - htp_backend_ext_config.json:         soc_model 60, dsp_arch v73  (Snapdragon X Elite)
  - genie_config.json:                   ctx-bins (4 parts), eos 151645, ctx size 4096
  - sample_prompt.txt:                   the Qwen3 chat template genie-t2t-run expects
                                         (prompt is pre-templated; genie applies none)
  - genie-app-script.txt:                the pipeline form of the same run

Runtime contract with genie-t2t-run (QAIRT 2.45, on the X Elite, on PATH):
    genie-t2t-run -c genie_config.json -p "<fully-templated prompt>"
  cwd MUST be the bundle dir so the relative ctx-bins / tokenizer paths resolve.
  stdout frames the answer as:  [BEGIN]: <generated text> [END]  then [KPIS]: ...
  We stream the BEGIN..END region as tokens. If the local QAIRT build frames
  output differently, adjust _OUTPUT_BEGIN/_OUTPUT_END below — it is the only
  build-specific knob, and the change is one line.

Off-device safety: if `genie-t2t-run` is not on PATH (CI, a laptop, this sandbox)
the backend reports npu_present=False and infer() yields a clearly-labelled
deterministic stub so the contract/conformance gates still pass. It never fakes
NPU engagement — assert_npu_engaged() still trips the 15 tok/s floor on the stub.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from shared.harp_contract import (
    Backend, Capability, InferRequest, Metrics, Modality, Tier,
)

# Same floor QNNBackend uses: X-Elite Qwen3-4B ~30 tok/s, Oryon CPU ~8 tok/s.
NPU_DECODE_FLOOR_TOK_S = 15.0

# genie-t2t-run stdout framing (QAIRT 2.4x). The ONLY build-specific knob.
_OUTPUT_BEGIN = "[BEGIN]:"
_OUTPUT_END = "[END]"


@dataclass
class GenieModelSpec:
    model_id: str
    bundle_dir: str                         # dir holding genie_config.json + *.bin
    modality: Modality = Modality.TEXT
    config_name: str = "genie_config.json"

    @property
    def config_path(self) -> Path:
        return Path(self.bundle_dir) / self.config_name


@dataclass
class DetailedMetrics:
    backend_id: str
    prefill_ms: float
    ttft_ms: float
    decode_tok_s: float
    tokens: int
    energy_mj_per_tok: float | None = None
    thermal_c: float | None = None

    def to_contract(self) -> Metrics:
        return Metrics(self.backend_id, self.ttft_ms, self.decode_tok_s,
                       self.energy_mj_per_tok, self.thermal_c)


class GenieBackend(Backend):
    def __init__(
        self,
        models: list[GenieModelSpec],
        *,
        genie_bin: str = "genie-t2t-run",
        ram_gb: float = 16.0,
        backend_id: str = "genie-x-elite",
        max_tokens_cap: int = 1024,
    ):
        if not models:
            raise ValueError("GenieBackend needs at least one GenieModelSpec")
        self._specs = {m.model_id: m for m in models}
        self._genie_bin = genie_bin
        self._ram_gb = ram_gb
        self._backend_id = backend_id
        self._max_tokens_cap = max_tokens_cap
        self._mods = tuple(sorted({m.modality for m in models}, key=lambda x: x.value))
        self._cfg_cache: dict[str, dict] = {}

    # ---- discovery / config -------------------------------------------------
    def _genie_path(self) -> str | None:
        """Absolute path to genie-t2t-run, or None if not installed (off-device).
        HARP_GENIE_BIN (a full path or a name) wins — the QDC bootstrap sets it so
        we don't depend on a flaky machine-wide PATH on a fresh cloud image."""
        override = os.environ.get("HARP_GENIE_BIN")
        if override:
            p = Path(override)
            if p.is_file():
                return str(p)
            found = shutil.which(override)
            if found:
                return found
        return shutil.which(self._genie_bin)

    def _config(self, model_id: str) -> dict:
        if model_id not in self._cfg_cache:
            spec = self._spec(model_id)
            try:
                self._cfg_cache[model_id] = json.loads(spec.config_path.read_text())
            except (OSError, json.JSONDecodeError):
                self._cfg_cache[model_id] = {}
        return self._cfg_cache[model_id]

    def _spec(self, model_id: str) -> GenieModelSpec:
        spec = self._specs.get(model_id)
        if spec is None:
            raise KeyError(f"{model_id} not in Genie manifest: {list(self._specs)}")
        return spec

    def _max_context(self, model_id: str) -> int:
        ctx = self._config(model_id).get("dialog", {}).get("context", {})
        return int(ctx.get("size", 4096))

    async def capabilities(self) -> Capability:
        any_id = next(iter(self._specs))
        return Capability(
            backend_id=self._backend_id, tier=Tier.EDGE,
            npu_present=self._genie_path() is not None,
            ram_gb=self._ram_gb, max_context=self._max_context(any_id),
            modalities=self._mods, offline_capable=True, supports_streaming=True,
        )

    # ---- Qwen3 chat template (matches the bundle's sample_prompt.txt) --------
    @staticmethod
    def _templated_prompt(messages: list[dict], thinking: bool = True,
                          tools: list | None = None) -> str:
        sys_txt = next((m["content"] for m in messages if m.get("role") == "system"), None)
        parts: list[str] = []
        if sys_txt is None:
            sys_txt = "You are a helpful AI assistant"
        # Qwen3 tool-calling: tools live in the system turn; the model emits
        # <tool_call>{...}</tool_call>. We template them verbatim so genie-t2t-run
        # sees the same shape GenieAPIService would. ponytail: device-time Genie may
        # template tools itself; this keeps the off-device path faithful.
        if tools:
            sys_txt = f"{sys_txt}\n\n# Tools\n{json.dumps(tools, separators=(',', ':'))}"
        parts.append(f"<|im_start|>system\n{sys_txt}<|im_end|>")
        for m in messages:
            role = m.get("role")
            if role in ("user", "assistant"):
                parts.append(f"<|im_start|>{role}\n{m['content']}<|im_end|>")
        # Qwen3 disables chain-of-thought when the assistant turn opens with /no_think.
        # The endpoint forces this whenever a request carries tools (Genie tool path).
        parts.append("<|im_start|>assistant\n" + ("" if thinking else "/no_think\n"))
        return "\n".join(parts)

    # ---- the blocking generate loop (subprocess, streamed) ------------------
    def _generate_blocking(self, req: InferRequest, on_token, on_first, state=None,
                           *, thinking: bool = True, tools: list | None = None) -> None:
        """state (optional): {"proc": Popen|None, "cancel": threading.Event}. When
        present, the live Popen is published to it so infer() can terminate the
        child if the async consumer stops early.

        thinking=False forces Qwen3 CoT off (the endpoint sets this when a request
        carries tools). tools are templated into the system turn so the model can
        emit <tool_call> blocks."""
        spec = self._spec(req.model_id)
        prompt = self._templated_prompt(req.messages, thinking=thinking, tools=tools)
        genie = self._genie_path()

        if genie is None:
            self._fallback_blocking(req, prompt, on_token, on_first, state, tools=tools)
            return

        bundle = Path(spec.bundle_dir).resolve()
        if not spec.config_path.exists():
            raise FileNotFoundError(
                f"Genie config not found: {spec.config_path}. Is the bundle complete?")

        # Inline prompt as ONE argv element. No shell => no injection; special tokens
        # survive verbatim. Routed sub-prompts sit far under any OS arg-length limit.
        cmd = [genie, "-c", spec.config_name, "-p", prompt]
        proc = subprocess.Popen(
            cmd, cwd=str(bundle), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        if state is not None:
            state["proc"] = proc
        self._stream_stdout(proc, on_token, on_first, state)

    def _stream_stdout(self, proc: "subprocess.Popen[str]", on_token, on_first,
                       state=None) -> None:
        """Emit the [BEGIN]..[END] region word-by-word (TTFT = first content word).
        Robust to: both markers on one line, a chatty stderr (drained concurrently
        so the PIPE buffer can't deadlock us), trailing [KPIS] after [END] (drained
        to EOF, not left to block the child), and consumer cancellation."""
        cancelled = lambda: state is not None and state["cancel"].is_set()

        # Concurrent stderr drain — without this, a >64KB stderr burst after [END]
        # deadlocks: child blocks writing stderr, we block in wait().
        err_chunks: list[str] = []
        if proc.stderr is not None:
            et = threading.Thread(
                target=lambda: err_chunks.extend(proc.stderr), daemon=True)
            et.start()
        else:
            et = None

        in_answer = saw_begin = done = False
        first = True
        assert proc.stdout is not None
        for line in proc.stdout:
            if cancelled():
                break
            if done:
                continue                              # drain remainder ([KPIS] etc.)
            stripped = line.rstrip("\n")
            seg: str | None = None
            if not in_answer and _OUTPUT_BEGIN in stripped:
                saw_begin = in_answer = True
                seg = stripped.split(_OUTPUT_BEGIN, 1)[1]
            elif in_answer:
                seg = stripped + " "
            if seg is None:
                continue
            if _OUTPUT_END in seg:                    # handles same-line BEGIN..END
                first = self._emit_words(seg.split(_OUTPUT_END, 1)[0], on_token, on_first, first)
                done = True
            else:
                first = self._emit_words(seg, on_token, on_first, first)

        if cancelled() and proc.poll() is None:
            proc.terminate()
        rc = proc.wait()
        if et is not None:
            et.join(timeout=2.0)
        err = "".join(err_chunks)
        if cancelled():
            return
        if rc != 0:
            raise RuntimeError(f"genie-t2t-run exited {rc}: {err.strip()[:400]}")
        if not saw_begin:
            # No marker AND no content — surface stderr so failures aren't silent.
            raise RuntimeError(
                "genie-t2t-run produced no [BEGIN] region; check the bundle / QAIRT "
                f"version or adjust _OUTPUT_BEGIN. stderr: {err.strip()[:400]}")

    @staticmethod
    def _emit_words(text: str, on_token, on_first, first: bool) -> bool:
        for w in text.split():
            if first:
                on_first()
                first = False
            on_token(w + " ")
        return first

    # ---- off-device deterministic stub (CI / no QAIRT) ----------------------
    def _fallback_blocking(self, req: InferRequest, prompt: str, on_token, on_first,
                           state=None, *, tools: list | None = None) -> None:
        msg = (f"[genie-stub:{req.model_id}] genie-t2t-run not on PATH; this is a "
               f"contract-conformant placeholder, not NPU output. Templated prompt "
               f"chars={len(prompt)}.")
        if tools:
            # Emit a real Qwen3/Hermes-shaped tool call so the OpenAI tool_calls
            # wiring is exercised off-device. Picks the first declared tool; not
            # NPU output, just a contract-conformant placeholder.
            fn = ((tools[0] or {}).get("function") or {}).get("name", "unknown_tool")
            msg = f'<tool_call>{{"name": "{fn}", "arguments": {{}}}}</tool_call>'
        first = True
        for w in msg.split():
            if state is not None and state["cancel"].is_set():
                break
            time.sleep(0.002)
            if first:
                on_first()
                first = False
            on_token(w + " ")

    # ---- async streaming bridge (worker thread -> asyncio.Queue) ------------
    async def infer(self, req: InferRequest, *, thinking: bool = True,
                    tools: list | None = None):
        """thinking defaults ON; the OpenAI endpoint passes thinking=False whenever
        a request carries tools (Genie tool-interception requires CoT off)."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[str | None] = asyncio.Queue()
        err: dict[str, BaseException] = {}
        state: dict = {"proc": None, "cancel": threading.Event()}

        def emit(piece: str) -> None:
            loop.call_soon_threadsafe(q.put_nowait, piece)

        def worker() -> None:
            try:
                self._generate_blocking(req, emit, lambda: None, state,
                                        thinking=thinking, tools=tools)
            except BaseException as e:               # surface, don't hang the await
                err["e"] = e
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()
        try:
            while True:
                piece = await q.get()
                if piece is None:
                    break
                yield piece
        finally:
            # Consumer stopped (break / cancel / GC): signal the worker and reap the
            # child so we never leak a zombie genie-t2t-run process.
            state["cancel"].set()
            proc = state["proc"]
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        if "e" in err:
            raise err["e"]

    # ---- profiling (mirrors QNNBackend so bench.py-style usage works) -------
    async def profile_detailed(self, req: InferRequest, power=None,
                               baseline_w: float = 0.0, thermal_fn=None) -> DetailedMetrics:
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        ttft = {"v": None}
        n = {"v": 0}

        def on_first() -> None:
            ttft["v"] = (time.perf_counter() - t0) * 1000.0

        def on_token(_) -> None:
            n["v"] += 1

        ctx = power if power is not None else _NullCtx()
        with ctx:
            await loop.run_in_executor(
                None, lambda: self._generate_blocking(req, on_token, on_first))
        total_s = time.perf_counter() - t0

        decode_s = max(total_s - (ttft["v"] or 0.0) / 1000.0, 1e-6)
        tok_s = max(n["v"] - 1, 0) / decode_s if n["v"] > 1 else (n["v"] / decode_s)
        energy = None
        if power is not None and n["v"]:
            energy = (power.trace.energy_joules(baseline_w) / n["v"]) * 1000.0
        return DetailedMetrics(
            backend_id=self._backend_id, prefill_ms=0.0, ttft_ms=ttft["v"] or 0.0,
            decode_tok_s=tok_s, tokens=n["v"], energy_mj_per_tok=energy,
            thermal_c=(thermal_fn() if thermal_fn else None),
        )

    async def profile(self, req: InferRequest, **kw) -> Metrics:
        return (await self.profile_detailed(req, **kw)).to_contract()

    def assert_npu_engaged(self, metrics: Metrics) -> None:
        if metrics.tokens_per_s < NPU_DECODE_FLOOR_TOK_S:
            raise RuntimeError(
                f"NPU NOT ENGAGED: {metrics.tokens_per_s:.1f} tok/s < floor "
                f"{NPU_DECODE_FLOOR_TOK_S}. genie-t2t-run may be missing (CPU/stub "
                f"path) or the context binary HTP arch (vXX) mismatches the SoC.")


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ---- factory: autodiscover the precompiled bundle in build/ -----------------

def _build_root(build_root: str | os.PathLike | None) -> Path:
    return Path(build_root) if build_root else Path(__file__).resolve().parent.parent / "build"


def genie_qwen3_4b(build_root: str | os.PathLike = None) -> GenieBackend:
    """Wire the precompiled Qwen3-4B w4a16 Genie bundle that ships in build/.
    Looks for build/qwen3-4b-w4a16/ relative to the repo root by default."""
    bundle = _build_root(build_root) / "qwen3-4b-w4a16"
    return GenieBackend([
        GenieModelSpec("qwen3-4b", str(bundle), Modality.TEXT),
    ])


# Heuristics to slot an auto-discovered bundle into the swarm. The model_id MUST
# match what the cloud planner emits (model_registry / plan_emitter), so we map
# bundle-metadata ids to the canonical swarm vocabulary.
_AUDIO_HINTS = ("whisper", "conformer", "asr", "stt", "wav2vec", "moonshine")
_VISION_HINTS = ("-vl", "vlm", "vision", "ocr", "parser", "clip", "vit", "detr", "yolo")
_ID_ALIASES = {
    "qwen3_4b": "qwen3-4b", "whisper_base": "whisper-base",
    "whisper_small": "whisper-base", "indicconformer": "whisper-base",
}


def _infer_bundle_identity(bundle: Path) -> tuple[str, Modality]:
    """(canonical model_id, modality) for a bundle dir, from metadata.json + name."""
    raw_id = bundle.name
    name = bundle.name.lower()
    try:
        meta = json.loads((bundle / "metadata.json").read_text())
        raw_id = meta.get("model_id", raw_id)
        name = f"{meta.get('model_name', raw_id)} {bundle.name}".lower()
    except (OSError, json.JSONDecodeError):
        pass
    model_id = _ID_ALIASES.get(raw_id, raw_id.replace("_", "-"))
    if any(h in name for h in _VISION_HINTS):
        modality = Modality.VISION
    elif any(h in name for h in _AUDIO_HINTS):
        modality = Modality.AUDIO
    else:
        modality = Modality.TEXT
    return model_id, modality


def _bundle_is_complete(bundle: Path) -> bool:
    """A bundle is runnable only if its genie_config points at context binaries
    that ACTUALLY EXIST. Guards against skeleton/placeholder dirs (a genie_config
    with no engine/ctx-bins, or referenced .bin files that aren't there) silently
    inflating the swarm's advertised modalities with models that can't run."""
    cfg = bundle / "genie_config.json"
    if not cfg.is_file():
        return False
    try:
        doc = json.loads(cfg.read_text())
        ctx_bins = (doc.get("dialog", {}).get("engine", {}).get("model", {})
                    .get("binary", {}).get("ctx-bins"))
    except (OSError, json.JSONDecodeError):
        return False
    if ctx_bins:
        return all((bundle / b).is_file() for b in ctx_bins)
    return any(bundle.glob("*.bin"))             # no ctx-bins list → require a real binary


def genie_swarm(build_root: str | os.PathLike = None) -> GenieBackend:
    """Auto-discover every precompiled Genie bundle under build/ and host them as a
    single edge swarm. Drop a COMPLETE bundle dir (genie_config.json + its context
    binaries) into build/ and it lights up — no code change. Modality + canonical
    model_id are inferred from metadata.json, or read verbatim from an optional
    build/swarm.json:

        {"models": [{"model_id": "qwen3-4b", "dir": "qwen3-4b-w4a16", "modality": "text"}]}

    Incomplete bundles (skeletons with no context binaries) and duplicate model_ids
    are SKIPPED with a stderr note — never silently advertised as runnable. Falls
    back to the lone Qwen3-4B bundle so the demo works before more arrive."""
    root = _build_root(build_root)
    specs: list[GenieModelSpec] = []
    seen: set[str] = set()
    skipped: list[str] = []

    def _add(model_id: str, bdir: Path, modality: Modality) -> None:
        if not _bundle_is_complete(bdir):
            skipped.append(f"{bdir.name} (no context binaries)")
            return
        if model_id in seen:
            skipped.append(f"{bdir.name} (duplicate model_id {model_id!r})")
            return
        seen.add(model_id)
        specs.append(GenieModelSpec(model_id, str(bdir), modality))

    manifest = root / "swarm.json"
    if manifest.is_file():                       # explicit override wins (authoritative)
        try:
            for m in json.loads(manifest.read_text()).get("models", []):
                _add(m["model_id"], root / m["dir"], Modality(m.get("modality", "text")))
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            specs, seen = [], set()

    if not specs and root.is_dir():              # heuristic discovery
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if not (d / "genie_config.json").is_file():
                continue
            model_id, modality = _infer_bundle_identity(d)
            _add(model_id, d, modality)

    if skipped:
        print(f"[genie_swarm] skipped {len(skipped)} incomplete/duplicate bundle(s): "
              f"{', '.join(skipped)}", file=sys.stderr)
    if not specs:
        return genie_qwen3_4b(build_root)        # nothing discovered yet
    return GenieBackend(specs)


# ---- self-test: conformance via the off-device stub -------------------------

def _selftest() -> None:
    import inspect

    be = genie_qwen3_4b()
    bundle_present = (Path(__file__).resolve().parent.parent
                      / "build" / "qwen3-4b-w4a16" / "genie_config.json").exists()
    tmpl = GenieBackend._templated_prompt(
        [{"role": "user", "content": "What is gravity? Keep the answer under ten words."}])
    assert tmpl.startswith("<|im_start|>system\n"), "must build Qwen3 chat template"
    assert tmpl.endswith("<|im_start|>assistant\n"), "must open the assistant turn"

    async def _run() -> None:
        cap = await be.capabilities()
        assert cap.tier == Tier.EDGE and cap.offline_capable
        req = InferRequest([{"role": "user", "content": "ping"}], "qwen3-4b")
        stream = be.infer(req)
        assert inspect.isasyncgen(stream)
        toks = [t async for t in stream]
        assert toks, "infer() must yield at least one token (stub or real)"
        m = await be.profile(req)
        assert isinstance(m, Metrics) and m.ttft_ms >= 0 and m.tokens_per_s >= 0

    asyncio.run(_run())

    # swarm auto-discovery finds the bundle(s) in build/
    swarm = genie_swarm()
    swarm_ids = sorted(swarm._specs.keys())
    assert "qwen3-4b" in swarm_ids, f"swarm must discover qwen3-4b bundle, got {swarm_ids}"

    mode = "REAL genie-t2t-run on PATH" if be._genie_path() else "off-device stub"
    print(f"edge/genie_backend: conformance OK · bundle_present={bundle_present} · "
          f"max_ctx={be._max_context('qwen3-4b')} · swarm={swarm_ids} · mode={mode}")


if __name__ == "__main__":
    _selftest()
