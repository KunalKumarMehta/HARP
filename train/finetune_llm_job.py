"""Build (and, only with --confirm, launch) an NVIDIA LoRA SFT job via `hf jobs`.

PAID COMPUTE. Default behavior is dry-run: print the exact command and a cost
note, exit 0. Nothing runs on GPU without an explicit --confirm.
After the job: fuse the adapter, `mlx_lm.convert` the merged model, re-run
evals/eval_local_llm.py before adopting.
"""
from __future__ import annotations

import argparse
import subprocess

SFT_SCRIPT = "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py"


def build_command(model_id: str, dataset_id: str, flavor: str = "l4x1",
                  timeout: str = "2h") -> list[str]:
    return [
        "hf", "jobs", "uv", "run", SFT_SCRIPT,
        "--flavor", flavor, "--timeout", timeout,
        "--with", "trl>=0.12", "--with", "peft>=0.13",
        "--", "--model_name_or_path", model_id, "--dataset_name", dataset_id,
        "--use_peft", "--lora_r", "16", "--lora_alpha", "32",
        "--output_dir", "harp-chat-lora", "--push_to_hub",
    ]


def _run(cmd: list[str]) -> None:  # separated so tests can stub it
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--flavor", default="l4x1")
    ap.add_argument("--timeout", default="2h")
    ap.add_argument("--confirm", action="store_true",
                    help="actually launch the PAID GPU job")
    a = ap.parse_args(argv)
    cmd = build_command(a.model, a.dataset, a.flavor, a.timeout)
    print(" ".join(cmd))
    print(f"\ncost note: flavor {a.flavor}, ceiling {a.timeout} — billed to the HF account.")
    if not a.confirm:
        print("dry-run only. Re-run with --confirm to launch.")
        return 0
    _run(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
