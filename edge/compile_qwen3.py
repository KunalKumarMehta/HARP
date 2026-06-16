"""
HARP · edge/compile_qwen3.py · CEE-owned · MIT
AI-Hub AOT pipeline: Qwen3-4B FP -> w4a16 Context Binary wrapped in ONNX, ready
for onnxruntime-genai + QNN HTP EP on the Snapdragon X Elite (Risk A target).

RUNS ON THE x86-64 COMPILE HOST — NOT on the device. qai_hub_models supports
only AMDx64 Python on Windows/Linux; the heavy AOT compile happens on AI Hub's
server-side farm, so this consumes ZERO Qualcomm Device Cloud interactive minutes.
Only bench.py (on the QDC X Elite over SSH) spends minutes.

Pipeline (grounding = Snapdragon LLM Deployment Walkthrough):
  0 preflight     token + ruamel pin + device resolve          §"Dependency Breakages"
  1 source graphs prompt_processor + token_generator ONNX      §"Multi-Graph Linking"
  2 quantize      w4a16, mse_minimizer, AIMET PCQ, calibration  §"Quantization and Calibration"
  3 compile+link  embed_in_onnx=True, weight-shared link        §"submit_compile_and_link_jobs"
  4 assemble      genai_config.json + tokenizer + wrapper        §"Runtime Invocation"
  5 profile (opt) AI Hub on-device latency cross-check (free)    Profiling Guide

Usage:
  python compile_qwen3.py --device "Snapdragon X Elite CRD" --out ./build/qwen3-4b-w4a16
  python compile_qwen3.py --dry-run          # validate plan without submitting jobs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# w4a16 standard; embeddings + LM head kept w8a16 (AIMET selective) per walkthrough.
GRAPH_NAMES = ["prompt_processor", "token_generator"]
WEIGHT_DTYPE = "int4"
ACT_DTYPE = "int16"
SENSITIVE_W8 = ["embed_tokens", "lm_head"]   # selective w8a16 islands
MAX_CONTEXT = 4096
SLIDING_WINDOW = None                        # set if compiling Qwen3 SWA; see G2 guard


@dataclass
class CompilePlan:
    device: str
    out_dir: Path
    src_graphs: dict           # graph_name -> path to FP ONNX
    calib_dir: Path
    sliding_window: int | None
    profile_after: bool


# ---- 0 preflight ------------------------------------------------------------
def preflight(plan: CompilePlan) -> "object":
    # ruamel.yaml minor-version breakage halts qai_hub_models export with an
    # obscure ModuleNotFoundError — pin it (walkthrough §"Dependency Breakages").
    try:
        import ruamel.yaml  # noqa
    except ImportError:
        sys.exit("FATAL: ruamel.yaml missing. Run: pip install ruamel-yaml==0.18.10")

    try:
        import qai_hub as hub
    except ImportError:
        sys.exit("FATAL: qai_hub missing. This must run on an x86-64 host. "
                 "pip install qai-hub qai-hub-models ; qai-hub configure --api_token <TOKEN>")

    if not (os.getenv("QAI_HUB_API_TOKEN") or Path.home().joinpath(".qai_hub", "client.ini").exists()):
        sys.exit("FATAL: no AI Hub token. Run: qai-hub configure --api_token <TOKEN>")

    devs = [d for d in hub.get_devices() if plan.device.lower() in d.name.lower()]
    if not devs:
        sys.exit(f"FATAL: device '{plan.device}' not in AI Hub catalog. "
                 f"List with: python -c 'import qai_hub as h;[print(d.name) for d in h.get_devices()]'")
    dev = devs[0]
    # Context Binary is SoC-specific: X Elite is HTP v73/v75; a v81 (8 Elite) binary
    # will not run here. The device object pins the arch into the compiled .bin.
    print(f"[0] device resolved: {dev.name}  (attrs: {getattr(dev,'attributes',[])})")
    return hub, dev


# ---- 1 source graphs --------------------------------------------------------
def source_graphs(plan: CompilePlan) -> dict:
    """prompt_processor (prefill, compute-bound) and token_generator (decode,
    bandwidth-bound) are exported separately so the linker can weight-share them.

    Produce these once via the qai_hub_models export front-end, e.g.:
        python -m qai_hub_models.models.qwen3_4b.export \
            --target-runtime onnx --skip-compiling --output-dir ./export
    which traces the HF checkpoint to the two ONNX variants. This script then
    quantizes + links them. We validate they exist rather than re-tracing (the
    3B trace needs >64 GB RAM; do it deliberately, not on every compile)."""
    missing = [g for g, p in plan.src_graphs.items() if not Path(p).exists()]
    if missing:
        sys.exit(f"FATAL: missing source ONNX graphs {missing}. Run the "
                 f"qai_hub_models export step first (see docstring).")
    print(f"[1] source graphs present: {list(plan.src_graphs)}")
    return plan.src_graphs


# ---- 2 quantize -------------------------------------------------------------
def quantize(hub, dev, plan: CompilePlan, graphs: dict) -> dict:
    """w4a16 via AI Hub quantize jobs. Calibration variance is the #1 silent
    killer: a thin calibration set yields distorted deep-layer scales -> the
    linker ICE-crashes on the full model while a 4-layer slice compiles fine
    (walkthrough §"Calibration Breakdown"). We assert a minimum sample count."""
    calib = sorted(plan.calib_dir.glob("*.npy")) if plan.calib_dir.exists() else []
    if len(calib) < 32:
        sys.exit(f"FATAL: calibration set too thin ({len(calib)} < 32). Provide a "
                 f"diverse multi-domain calibration set or expect a deep-layer ICE.")
    qmodels = {}
    for name, path in graphs.items():
        print(f"[2] quantize {name}: w4a16 (mse_minimizer, per-channel GEMM), "
              f"{len(calib)} calib samples, w8a16 islands={SENSITIVE_W8}")
        job = hub.submit_quantize_job(
            model=path,
            calibration_data={"input_ids": [str(p) for p in calib]},
            weights_dtype=WEIGHT_DTYPE,
            activations_dtype=ACT_DTYPE,
            # selective precision + per-channel GEMM are AIMET-ONNX 2.31 options;
            # exact kwarg names vary by client version — confirm on the pinned build.
            options=f"--mixed_precision --sensitive_layers {','.join(SENSITIVE_W8)} "
                    f"--algorithm mse_minimizer --per_channel",
        )
        qmodels[name] = job.get_target_model()
    return qmodels


# ---- 3 compile + link -------------------------------------------------------
def compile_and_link(hub, dev, plan: CompilePlan, qmodels: dict):
    """Single weight-shared Context Binary across prefill+decode, wrapped in ONNX.
    Independent compiles would duplicate the ~2.8 GB weights and OOM the device
    (walkthrough §"weight sharing"). embed_in_onnx replaces the deprecated
    --target_runtime precompiled_qnn_onnx string."""
    print(f"[3] compile_and_link graphs={GRAPH_NAMES} embed_in_onnx=True device={dev.name}")
    job = hub.submit_compile_and_link_jobs(
        models=[qmodels[g] for g in GRAPH_NAMES],
        graph_names=GRAPH_NAMES,
        device=dev,
        embed_in_onnx=True,                  # wrap .bin in ONNX protobuf shell
        options="--quantized_io",            # int64 input_ids; no FP tensor injection
    )
    target = job.get_target_model()
    plan.out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = plan.out_dir / "model.onnx"
    target.download(str(wrapper))
    print(f"[3] context-binary-in-ONNX downloaded -> {wrapper}")
    return wrapper


# ---- 4 assemble runtime manifest -------------------------------------------
def assemble(plan: CompilePlan, wrapper: Path) -> Path:
    """genai_config.json the ARM64 onnxruntime-genai engine consumes. Encodes the
    QNN HTP provider options + the memory levers bench.py/qnn_backend.py rely on."""
    cfg = {
        "model": {
            "type": "qwen3",
            "context_length": MAX_CONTEXT,
            "decoder": {
                "session_options": {
                    "provider_options": [{
                        "qnn": {
                            "backend_path": "QnnHtp.dll",
                            "htp_performance_mode": "burst",
                            "htp_graph_finalization_optimization_mode": "3",
                            # DR3: full VTCM to one graph at a time
                            "htp_vtcm_optimization": "SEQUENTIAL_WITH_VA_OPTIMIZATION",
                        }
                    }],
                    "past_present_share_buffer": True,   # single static KV block
                },
                "filename": wrapper.name,
            },
        },
        "search": {"max_length": MAX_CONTEXT, "past_present_share_buffer": True},
    }
    if plan.sliding_window:
        # G2: declaring a window hardcodes the prefill tensor shape; bench pads
        # short prompts to this boundary to avoid the 0xc0000409 access violation.
        cfg["model"]["decoder"]["sliding_window"] = {
            "window_size": plan.sliding_window, "pad_value": 0,
        }
    out = plan.out_dir / "genai_config.json"
    out.write_text(json.dumps(cfg, indent=2))
    # tokenizer assets must sit beside the wrapper; copy from the export dir.
    print(f"[4] wrote {out} ; ensure tokenizer.json + tokenizer_config.json are in {plan.out_dir}")
    return out


# ---- 5 optional AI Hub profile (free latency cross-check) -------------------
def profile(hub, dev, wrapper: Path):
    """AI Hub runs this on its OWN X Elite farm — costs no QDC minutes. Gives a
    latency baseline to cross-check bench.py's host-clock numbers (DR2: host clock
    is an upper bound on async NPU dispatch)."""
    print(f"[5] submit_profile_job on {dev.name} (server-side, free of QDC minutes)")
    job = hub.submit_profile_job(model=str(wrapper), device=dev)
    print(f"[5] profile job: {job.url}  -> compare its on-device latency to bench.py")


# ---- orchestration ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="Snapdragon X Elite CRD")
    ap.add_argument("--out", default="./build/qwen3-4b-w4a16")
    ap.add_argument("--export-dir", default="./export")
    ap.add_argument("--calib-dir", default="./calib")
    ap.add_argument("--sliding-window", type=int, default=SLIDING_WINDOW)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    plan = CompilePlan(
        device=a.device, out_dir=Path(a.out),
        src_graphs={g: str(Path(a.export_dir) / f"{g}.onnx") for g in GRAPH_NAMES},
        calib_dir=Path(a.calib_dir), sliding_window=a.sliding_window,
        profile_after=a.profile,
    )

    if a.dry_run:
        print("DRY-RUN plan:")
        print(json.dumps({"device": plan.device, "out": str(plan.out_dir),
                          "graphs": plan.src_graphs, "calib_dir": str(plan.calib_dir),
                          "sliding_window": plan.sliding_window,
                          "w4a16": {"w": WEIGHT_DTYPE, "a": ACT_DTYPE, "w8_islands": SENSITIVE_W8}},
                         indent=2))
        return

    hub, dev = preflight(plan)
    graphs = source_graphs(plan)
    qmodels = quantize(hub, dev, plan, graphs)
    wrapper = compile_and_link(hub, dev, plan, qmodels)
    assemble(plan, wrapper)
    if plan.profile_after:
        profile(hub, dev, wrapper)
    print(f"\nDONE. scp {plan.out_dir}/ to the QDC X Elite, then run bench.py over SSH.")


if __name__ == "__main__":
    main()
