"""Label a HARP trace with verified edge outcomes for router.drift. Stdlib-only.

For each row, a cloud judge (NIM, OpenAI-compatible) grades the edge answer
as adequate/not -> writes `edge_correct`. Rows the edge never answered
(escalated) are skipped unless --shadow, which generates the missing edge
answer with the local MLX model first — without it the labels only cover
locally-kept queries and the calibration is biased toward the easy side.

Usage:
  HARP_NIM_API_KEY=... python scripts/label_trace.py mac_demo/harp_trace.jsonl
  ... --shadow            # also label escalated rows (needs harp[apple])
  ... --out labeled.jsonl # default: <trace>.labeled.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

JUDGE_BASE = os.getenv("HARP_CLOUD_BASE", "https://integrate.api.nvidia.com/v1")
JUDGE_MODEL = os.getenv("HARP_CLOUD_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
JUDGE_PROMPT = (
    "You grade an on-device assistant. QUERY:\n{q}\n\nANSWER:\n{a}\n\n"
    "Is the answer correct and adequate for the query? Reply with exactly one word: YES or NO."
)


def cloud_judge(query: str, answer: str) -> bool:
    key = os.getenv("HARP_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if not key:
        raise SystemExit("HARP_NIM_API_KEY not set — get one at build.nvidia.com")
    body = json.dumps({
        "model": JUDGE_MODEL, "temperature": 0.0, "max_tokens": 4,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(q=query, a=answer)}],
    }).encode()
    req = urllib.request.Request(
        f"{JUDGE_BASE}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())["choices"][0]["message"]["content"]
    return out.strip().upper().startswith("YES")


def label(rows: list[dict], judge, shadow_gen=None) -> dict:
    """Set edge_correct in place; returns counts. judge: (q, a) -> bool.
    shadow_gen: (q) -> str for rows with no edge answer, or None to skip them."""
    n_judged = n_shadow = n_skipped = 0
    for r in rows:
        if isinstance(r.get("edge_correct"), bool):
            continue  # already labeled
        ans = r.get("answer") if r.get("decision") == "local" else None
        if not ans and shadow_gen is not None:
            ans = shadow_gen(r["query"])
            r["shadow_answer"] = ans
            n_shadow += 1
        if not ans:
            n_skipped += 1
            continue
        r["edge_correct"] = judge(r["query"], ans)
        n_judged += 1
    return {"judged": n_judged, "shadowed": n_shadow, "skipped": n_skipped}


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--shadow", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    rows = [json.loads(x) for x in a.trace.read_text().splitlines() if x.strip()]
    shadow_gen = None
    if a.shadow:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mac_demo"))
        import local_llm

        shadow_gen = local_llm.generate
    counts = label(rows, cloud_judge, shadow_gen)
    out = a.out or a.trace.with_suffix(".labeled.jsonl")
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"{counts['judged']} judged ({counts['shadowed']} shadow-generated), "
          f"{counts['skipped']} skipped (no edge answer) -> {out}")
    if counts["skipped"] and not a.shadow:
        print("hint: --shadow labels escalated rows too (unbiased calibration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
