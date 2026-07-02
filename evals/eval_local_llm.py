"""Chat/summarization eval on Apple silicon. Pass bar (spec): TTFT < 2 s,
>= 20 tok/s, >= 16/20 rubric-pass. Writes eval_report_local_llm.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mac_demo"))

PASS_TTFT_S, PASS_TPS, PASS_RUBRIC = 2.0, 20.0, 16


def rubric_pass(answer: str, expect: list[str]) -> bool:
    low = answer.lower()
    return all(k.lower() in low for k in expect)


def _main() -> int:
    import local_llm

    if not local_llm.available():
        print("mlx_lm not installed — pip install -e .[apple]; nothing to eval")
        return 1
    rows = [json.loads(x) for x in
            (ROOT / "mac_demo" / "prompts_eval.jsonl").read_text().splitlines() if x.strip()]

    model, tok = local_llm._load()

    # Try stream_generate first; fall back to generate if unavailable (API drift).
    try:
        from mlx_lm import stream_generate
        _use_stream = True
    except ImportError:
        _use_stream = False

    results, ttfts, tps_all = [], [], []
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"]}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        t0, first, ntok, out = time.perf_counter(), None, 0, []
        if _use_stream:
            for resp in stream_generate(model, tok, prompt=text, max_tokens=256):
                if first is None:
                    first = time.perf_counter() - t0
                ntok += 1
                # mlx_lm >=0.21 yields objects with .text; older yields plain str
                chunk = resp.text if hasattr(resp, "text") else str(resp)
                out.append(chunk)
        else:
            from mlx_lm import generate as _gen
            answer_raw = _gen(model, tok, prompt=text, max_tokens=256, verbose=False)
            first = time.perf_counter() - t0
            out = [answer_raw]
            ntok = len(answer_raw.split())
        dt = time.perf_counter() - t0
        answer = "".join(out)
        ok = rubric_pass(answer, r["expect"])
        ttfts.append(first or dt)
        tps_all.append(ntok / dt if dt > 0 else 0.0)
        results.append({"prompt": r["prompt"][:60], "rubric": ok,
                        "ttft_s": round(first or dt, 3), "tok_s": round(tps_all[-1], 1)})
    n_ok = sum(x["rubric"] for x in results)
    med = sorted(tps_all)[len(tps_all) // 2]
    worst_ttft = max(ttfts)
    passes = worst_ttft < PASS_TTFT_S and med >= PASS_TPS and n_ok >= PASS_RUBRIC
    report = {"model": local_llm.model_id(), "rubric_ok": n_ok, "median_tok_s": round(med, 1),
              "worst_ttft_s": round(worst_ttft, 3), "passes": passes, "results": results}
    (ROOT / "eval_report_local_llm.json").write_text(json.dumps(report, indent=2))
    print(f"{report['model']}: rubric {n_ok}/20, median {med:.1f} tok/s, "
          f"worst TTFT {worst_ttft:.2f}s -> passes={passes}")
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(_main())
