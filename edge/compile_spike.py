"""
HARP · edge/compile_spike.py · MIT
AI Hub compile for the NON-autoregressive specialists: the mmBERT router
(encoder + classification head) and IndicConformer ASR (CTC/RNNT). Neither uses
a KV cache or genai_config — they run on a plain onnxruntime + QNN EP session,
single forward pass. That is the core difference from compile_qwen3.py.

Grounding:
  - encoders/ASR need STATIC input shapes on HTP -> pad to fixed seq/mel len
  - ASR w8a16, NOT w4a16: INT8 weights w/o QAT distort Indic phonemes (WER↑)
  - AI Hub transforms MHA->SHA, linear->1x1 conv, then QNN_CONTEXT_BINARY
  - custom LoRA-merged weights: pass local HF checkpoint dir to export
  - router is the one tuned model (SFT/LoRA on synthetic routing data)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROFILES = {
    # model_type -> compile profile
    "router": {                 # mmBERT multilingual encoder, RAM-resident gatekeeper
        "weights_dtype": "int8", "activations_dtype": "int16",   # w8a16; classification-safe
        "static_axes": {"input_ids": [1, 128], "attention_mask": [1, 128]},  # fixed seq=128
        "runtime": "onnxruntime+qnn (no genai)",
        "calib_floor": 32,
    },
    "asr": {                    # IndicConformer CTC/RNNT, streaming
        "weights_dtype": "int8", "activations_dtype": "int16",   # w8a16 mandatory (phonemes)
        "static_axes": {"audio_mels": [1, 80, 3000]},            # 30s mel window, fixed
        "runtime": "onnxruntime+qnn (CTC greedy decode)",
        "calib_floor": 48,      # ASR needs more acoustic diversity
    },
}


@dataclass
class SpikePlan:
    model_type: str
    hf_or_module: str          # qai_hub_models module OR local HF checkpoint dir (LoRA-merged)
    device: str
    out_dir: Path
    calib_dir: Path
    custom_weights: bool


def preflight(plan: SpikePlan):
    try:
        import ruamel.yaml  # noqa
    except ImportError:
        sys.exit("FATAL: pip install ruamel-yaml==0.18.10")
    try:
        import qai_hub as hub
    except ImportError:
        sys.exit("FATAL: x86-64 host only. pip install qai-hub qai-hub-models; qai-hub configure --api_token <T>")
    prof = PROFILES[plan.model_type]
    devs = [d for d in hub.get_devices() if plan.device.lower() in d.name.lower()]
    if not devs:
        sys.exit(f"FATAL: device '{plan.device}' not in catalog")
    print(f"[0] {plan.model_type}: device={devs[0].name} profile={prof['weights_dtype']}/"
          f"{prof['activations_dtype']} runtime='{prof['runtime']}'")
    return hub, devs[0], prof


def source_onnx(plan: SpikePlan) -> str:
    """Standard path: qai_hub_models export CLI (run separately):
        router:  python -m qai_hub_models.models.mmbert.export --target-runtime onnx --skip-compiling
        asr:     python -m qai_hub_models.models.indic_conformer.export --target-runtime onnx --skip-compiling
    Custom (LoRA-merged) path: trace your local HF checkpoint to ONNX yourself and
    point --hf at the .onnx. Either way we validate the artifact exists here."""
    onnx = Path(plan.hf_or_module)
    if onnx.suffix == ".onnx" and onnx.exists():
        print(f"[1] source ONNX: {onnx} (custom_weights={plan.custom_weights})")
        return str(onnx)
    sys.exit(f"FATAL: expected a traced .onnx at {plan.hf_or_module}. Run the "
             f"qai_hub_models export (see docstring) or trace your LoRA-merged checkpoint.")


def quantize(hub, dev, plan: SpikePlan, prof: dict, onnx: str):
    calib = sorted(plan.calib_dir.glob("*.npy")) if plan.calib_dir.exists() else []
    if len(calib) < prof["calib_floor"]:
        sys.exit(f"FATAL: {plan.model_type} needs ≥{prof['calib_floor']} calib samples, got {len(calib)}")
    key = "audio_mels" if plan.model_type == "asr" else "input_ids"
    print(f"[2] quantize {plan.model_type}: {prof['weights_dtype']}/{prof['activations_dtype']}, "
          f"{len(calib)} samples, static_axes={prof['static_axes']}")
    job = hub.submit_quantize_job(
        model=onnx, calibration_data={key: [str(p) for p in calib]},
        weights_dtype=prof["weights_dtype"], activations_dtype=prof["activations_dtype"],
        options="--per_channel --algorithm mse_minimizer",
    )
    return job.get_target_model()


def compile_qnn(hub, dev, plan: SpikePlan, qmodel):
    """Single-graph compile (no link step — these aren't multi-graph LLMs).
    embed_in_onnx wraps the QNN context binary for the ORT QNN EP."""
    print(f"[3] compile -> QNN_CONTEXT_BINARY (embed_in_onnx) device={dev.name}")
    job = hub.submit_compile_job(
        model=qmodel, device=dev,
        options="--target_runtime onnx --quantized_io",
    )
    target = job.get_target_model()
    plan.out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = plan.out_dir / "model.onnx"
    target.download(str(wrapper))
    print(f"[3] downloaded -> {wrapper}")
    return wrapper


def assemble(plan: SpikePlan, prof: dict, wrapper: Path):
    """No genai_config — just a tiny manifest the plain-ORT loader reads to set the
    QNN provider + the static input shapes the NPU was compiled against."""
    manifest = {
        "model_type": plan.model_type, "wrapper": wrapper.name,
        "provider_options": {"backend_path": "QnnHtp.dll",
                             "htp_performance_mode": "burst",
                             "htp_graph_finalization_optimization_mode": "3"},
        "static_axes": prof["static_axes"], "runtime": prof["runtime"],
        "decode": ("ctc_greedy" if plan.model_type == "asr" else "argmax_logits"),
    }
    out = plan.out_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[4] wrote {out}; copy tokenizer/vocab beside the wrapper")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=list(PROFILES))
    ap.add_argument("--hf", required=True, help="traced .onnx path (export output or LoRA-merged trace)")
    ap.add_argument("--device", default="Snapdragon X Elite CRD")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-dir", default="./calib")
    ap.add_argument("--custom-weights", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    plan = SpikePlan(a.type, a.hf, a.device, Path(a.out), Path(a.calib_dir), a.custom_weights)
    prof = PROFILES[a.type]
    if a.dry_run:
        print(json.dumps({"type": a.type, "profile": prof, "out": a.out,
                          "custom_weights": a.custom_weights}, indent=2))
        return
    hub, dev, prof = preflight(plan)
    onnx = source_onnx(plan)
    qmodel = quantize(hub, dev, plan, prof, onnx)
    wrapper = compile_qnn(hub, dev, plan, qmodel)
    assemble(plan, prof, wrapper)
    print(f"\nDONE. scp {a.out}/ to the QDC X Elite for the {a.type} spike gate.")


if __name__ == "__main__":
    main()
