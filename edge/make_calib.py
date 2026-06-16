"""
HARP · edge/make_calib.py · CEE-owned · MIT
Calibration-set generator for AI Hub w4a16/w8a16 quantize jobs.

WHY THIS EXISTS: a thin or uniform calibration set distorts deep-layer activation
scales → the linker ICE-crashes on the full model while a 4-layer slice compiles
clean (Deployment Walkthrough §"Calibration Breakdown"). compile_qwen3.py asserts
≥32 samples; this guarantees they are also DIVERSE — count alone does not prevent
the ICE, distribution coverage does.

Emits one .npy per sample (int64 input_ids) into --out, ready for
submit_quantize_job(calibration_data={"input_ids": [...]}).

Diversity is enforced on three axes the walkthrough flags:
  - DOMAIN  : code / math / prose / dialogue / structured / multilingual
  - LENGTH  : short / medium / long (covers prefill-shape variance)
  - both spreads asserted before write; dedup by token signature.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Multi-domain seed prompts. Indic + code-switch lines included because the router
# and IndicConformer calibration must see Hinglish, not just English (Indic STT doc).
SEEDS: dict[str, list[str]] = {
    "code": [
        "def quantize(w, bits): return round(w * (2**bits - 1))",
        "for i in range(n):\n    acc += a[i] * b[i]  # dot product",
        "SELECT user_id, SUM(amount) FROM tx GROUP BY user_id HAVING SUM(amount) > 1000;",
    ],
    "math": [
        "Prove that the sum of the first n odd integers equals n squared.",
        "Integrate x^2 e^{-x} dx from 0 to infinity using integration by parts.",
        "If A is 3x3 with det(A)=0, what can you conclude about its rank?",
    ],
    "prose": [
        "Summarize the tradeoffs of edge versus cloud inference in two sentences.",
        "The monsoon arrived early this year, soaking the terraced fields before dawn.",
        "Explain why unified memory bandwidth bounds autoregressive decode speed.",
    ],
    "dialogue": [
        "User: book me a cab to the airport. Assistant: For what time?",
        "Can you remind me what we decided about the router threshold yesterday?",
        "Yes, I agree — let's escalate that one to the cloud model.",
    ],
    "structured": [
        '{"intent": "transfer", "amount": 500, "currency": "INR", "to": "savings"}',
        "name,role,dept\nyash,engineer,edge\nasha,lead,cloud",
        "- prefill: compute-bound\n- decode: bandwidth-bound\n- escalate: network-bound",
    ],
    "multilingual": [
        "Mujhe kal ka weather batao please.",                 # Hinglish
        "नमस्ते, मेरा खाता शेष कितना है?",                      # Hindi
        "Transaction approve karne ke liye OTP bhejo.",        # code-switch
    ],
}

LENGTH_TIERS = {"short": (1, 12), "medium": (13, 48), "long": (49, 192)}


def _expand(prompt: str, target_tokens: int, tokenizer) -> list[int]:
    ids = tokenizer.encode(prompt)
    if len(ids) >= target_tokens:
        return ids[:target_tokens]
    # repeat-and-trim to reach a target length tier without changing domain content
    out = []
    while len(out) < target_tokens:
        out.extend(ids)
    return out[:target_tokens]


def build(out_dir: Path, tokenizer, per_domain_per_tier: int = 2) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    written, sigs, domains, lengths = 0, set(), set(), []
    for domain, prompts in SEEDS.items():
        for tier, (lo, hi) in LENGTH_TIERS.items():
            target = (lo + hi) // 2
            for k in range(per_domain_per_tier):
                prompt = prompts[(k) % len(prompts)]
                ids = _expand(prompt, target, tokenizer)
                sig = (domain, tier, tuple(ids[:8]))
                if sig in sigs:
                    continue
                sigs.add(sig)
                np.save(out_dir / f"calib_{domain}_{tier}_{k}.npy",
                        np.asarray(ids, dtype="int64"))
                written += 1
                domains.add(domain)
                lengths.append(len(ids))
    # DIVERSITY GATE — count is necessary, spread is what stops the ICE
    if written < 32:
        sys.exit(f"FATAL: only {written} samples; need ≥32. Raise --per-tier.")
    if len(domains) < 4:
        sys.exit(f"FATAL: only {len(domains)} domains; calibration not diverse.")
    if max(lengths) < 49 or min(lengths) > 12:
        sys.exit("FATAL: length tiers not covered (need short AND long samples).")
    return {"written": written, "domains": sorted(domains),
            "len_min": min(lengths), "len_max": max(lengths)}


def _load_tokenizer(model_dir: str | None):
    if model_dir:
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_dir)
        except Exception as e:
            print(f"[warn] HF tokenizer load failed ({e}); using byte fallback")
    class _ByteTok:                       # deterministic, dependency-free fallback
        def encode(self, s): return list(s.encode("utf-8"))
    return _ByteTok()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./calib")
    ap.add_argument("--tokenizer", default=None, help="HF model dir for real token ids")
    ap.add_argument("--per-tier", type=int, default=2)
    a = ap.parse_args()
    tok = _load_tokenizer(a.tokenizer)
    stats = build(Path(a.out), tok, a.per_tier)
    print(f"calibration set: {stats['written']} samples, domains={stats['domains']}, "
          f"len {stats['len_min']}–{stats['len_max']} tokens -> {a.out}")
    print("feed to compile_qwen3.py via --calib-dir " + a.out)


if __name__ == "__main__":
    main()
