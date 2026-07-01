#!/usr/bin/env python3
"""
HARP Mac demo — latency/throughput benchmark: LOCAL (Ollama on your Mac) vs
CLOUD (NVIDIA Nemotron via NIM). Streams both tiers and reports real TTFT +
tokens/sec, so the routing story has measured numbers behind it instead of claims.

The routing gate (harp_demo.py) decides WHERE a task runs; this shows WHY:
local wins first-token latency (no network, small model), cloud wins throughput
and hard-task quality (datacenter GPU, big model). Same prompts, both tiers.

Usage:
  python bench_demo.py            # measure both tiers on real models
  python bench_demo.py --local    # local (Ollama) only — no cloud key needed
  python bench_demo.py --mock     # print the plan; make no calls (no models/keys)

Env (same knobs as harp_demo.py):
  HARP_NIM_API_KEY   from build.nvidia.com (needed for the cloud tier; NVIDIA_API_KEY also accepted)
  HARP_OLLAMA_MODEL  default "nemotron-mini"  (an Ollama model you've pulled)
  HARP_CLOUD_MODEL   default "nvidia/llama-3.3-nemotron-super-49b-v1.5" (confirm id on build.nvidia.com)
"""
import os, sys, time, json, argparse
from pathlib import Path

BENCH_PATH  = Path(__file__).resolve().parent / "harp_bench.jsonl"
LOCAL_BASE  = os.getenv("HARP_LOCAL_BASE", "http://localhost:11434/v1")
LOCAL_MODEL = os.getenv("HARP_OLLAMA_MODEL", "nemotron-mini")
CLOUD_BASE  = os.getenv("HARP_CLOUD_BASE", "https://integrate.api.nvidia.com/v1")
# Default to the pragmatic planner id from cloud/model_registry.py (fits 1xH100).
CLOUD_MODEL = os.getenv("HARP_CLOUD_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
# HARP_NIM_API_KEY is the core-stack name; accept NVIDIA_API_KEY too (same key).
CLOUD_KEY   = os.getenv("HARP_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")

# One short turn (local's sweet spot) and one hard turn (where cloud earns its keep).
PROMPTS = [
    ("short", "Reply in one short sentence: what is drip irrigation?"),
    ("hard",  "Design a 3-day irrigation and fertilizer schedule across five wheat "
              "plots at different growth stages, accounting for a rainfall forecast. "
              "Give it step by step."),
]


def _client(base, key):
    from openai import OpenAI
    return OpenAI(base_url=base, api_key=key or "not-used")


def measure(where, text):
    """Stream one completion; return real metrics. TTFT = wall-clock to first
    content token; tok/s over the decode window. ponytail: one stream chunk ≈ one
    token on every OpenAI-compatible server we target — honest proxy, no tokenizer."""
    if where == "local":
        client, model = _client(LOCAL_BASE, "ollama"), LOCAL_MODEL
    else:
        if not CLOUD_KEY:
            raise RuntimeError("no cloud key — set HARP_NIM_API_KEY (get one at build.nvidia.com)")
        client, model = _client(CLOUD_BASE, CLOUD_KEY), CLOUD_MODEL
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    last = t0
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are HARP, a concise field assistant."},
                  {"role": "user", "content": text}],
        temperature=0.2, max_tokens=256, stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            ntok += 1
            last = now
    decode_s = last - (t0 + (ttft or 0.0))
    tok_s = (ntok - 1) / decode_s if decode_s > 0 and ntok > 1 else 0.0
    return {"ttft_ms": int((ttft or 0.0) * 1000), "tok_s": round(tok_s, 1),
            "tokens": ntok, "total_ms": int((last - t0) * 1000)}


def run(local_only=False, mock=False):
    tiers = ["local"] if local_only else ["local", "cloud"]
    print(f"\nHARP latency/throughput bench   local={LOCAL_MODEL}  cloud={CLOUD_MODEL}")
    print(f"{'mode: MOCK (no calls)' if mock else 'mode: live (streaming real models)'}\n")
    print(f"{'TIER':<11}{'PROMPT':<8}{'TTFT ms':<10}{'tok/s':<8}{'tokens':<8}{'total ms':<10}")
    print("-" * 55)
    rows = []
    for label, text in PROMPTS:
        for tier in tiers:
            if mock:
                m = {"ttft_ms": 0, "tok_s": 0.0, "tokens": 0, "total_ms": 0}
            else:
                try:
                    m = measure(tier, text)
                except Exception as e:
                    print(f"{tier:<11}{label:<8}[error: {e}]")
                    continue
            name = "on-device" if tier == "local" else "cloud"
            print(f"{name:<11}{label:<8}{m['ttft_ms']:<10}{m['tok_s']:<8}{m['tokens']:<8}{m['total_ms']:<10}")
            rows.append({"tier": name, "prompt": label, **m})
    with open(BENCH_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("-" * 55)
    print("Bench written to harp_bench.jsonl.  on-device wins TTFT; cloud wins tok/s + hard-task depth.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="local tier only (no cloud key)")
    ap.add_argument("--mock", action="store_true", help="print the plan, make no calls")
    a = ap.parse_args()
    run(local_only=a.local, mock=a.mock)
