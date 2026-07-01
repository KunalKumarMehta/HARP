#!/usr/bin/env python3
"""
HARP — real calibration of the conformal gate (②).

Runs paired queries on a LOCAL model (Ollama / Apple Silicon, run TERSE/no-CoT —
the fast latency-bound on-device mode) and a CLOUD model (NVIDIA Nemotron via NIM,
run with full step-by-step reasoning — the heavy tier). Labels each with
programmatic ground truth (verifiable arithmetic word problems — no LLM judge,
fully reproducible). Scores each with the gate's ACTUAL score fn
(router.mock_score_fn), fits the *fixed* ConformalGate on a calibration split, and
reports the measured UNDER-route rate on a held-out test split with a bootstrap CI.

The label err_local=1 means "the on-device model, run in its fast mode, got it
wrong -> this query should have escalated." That is exactly the routing call HARP
makes, so it is the honest label for the routing use-case.

Usage:
  export HARP_NIM_API_KEY="nvapi-..."           # or NVIDIA_API_KEY
  python calibrate_real.py --n 300 --alpha 0.10 --cloud-max 80   # full overnight run
  python calibrate_real.py --n 80 --local-only                   # fast local tune
Writes harp_calibration.jsonl (per-query trace) + harp_calibration.summary.json.
"""
import os, sys, re, json, time, argparse, random, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root -> import router
from router.router_policy import mock_score_fn, ConformalGate

OUT = Path(__file__).resolve().parent / "harp_calibration.jsonl"
LOCAL_BASE  = os.getenv("HARP_LOCAL_BASE", "http://localhost:11434/v1")
LOCAL_MODEL = os.getenv("HARP_OLLAMA_MODEL", "nemotron-mini")
CLOUD_BASE  = os.getenv("HARP_CLOUD_BASE", "https://integrate.api.nvidia.com/v1")
CLOUD_MODEL = os.getenv("HARP_CLOUD_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
CLOUD_KEY   = os.getenv("HARP_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")

# on-device tier = fast, no chain-of-thought (latency-bound). cloud tier = reason hard.
LOCAL_SYS = ("You are a fast on-device assistant. Give ONLY the final answer as "
             "'ANSWER: <integer>'. Do not show any working.")
CLOUD_SYS = ("Solve the problem carefully, step by step. Then on the last line write "
             "'ANSWER: <integer>' with just the final integer.")


# ----------------------------------------------------------------- verifiable corpus
# Difficulty-balanced by band. Hard bands are longer AND carry reasoning markers, so
# the gate's length+marker score u rises with real task hardness — while ground
# truth stays exact. u is deliberately crude (mock_score_fn, not a trained encoder):
# a crude score keeps the UNDER-route bound valid but pays a higher over-route cost,
# which is exactly the tradeoff we want to measure and disclose.
def gen_problems(n, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        band = i % 4
        if band == 0:                                            # trivial add — easy & plain (low u)
            a, b = rng.randint(2, 9), rng.randint(2, 9)
            q = f"What is {a} + {b}?"
            truth = a + b
        elif band == 1:                                          # 2-digit x 1-digit — plain (low u)
            a, b = rng.randint(13, 49), rng.randint(4, 9)
            q = f"What is {a} times {b}?"
            truth = a * b
        elif band == 2:                                          # 3-digit x 2-digit — DECEPTIVE: short
            a, b = rng.randint(123, 899), rng.randint(17, 89)    # & plain (low u) but hard for a 4B ->
            q = f"What is {a} times {b}?"                        # a genuine low-u under-route trap
            truth = a * b
        else:                                                    # 4-step word problem
            a, b, c = rng.randint(30, 90), rng.randint(30, 90), rng.randint(9, 19)
            t, f = rng.randint(80, 300), rng.randint(2, 7)
            q = (f"A farmer harvests {a} kg from the north plot and {b} kg from the "
                 f"south plot. Each kg sells for {c} rupees. Transport costs {t} rupees "
                 f"in total, and a market fee of {f} rupees applies per kg sold. How "
                 f"many rupees remain? Reason step by step and derive the total.")
            truth = (a + b) * c - t - f * (a + b)
        out.append({"q": q, "truth": truth, "band": band})
    return out


def parse_num(text):
    if not text:
        return None
    m = re.findall(r'ANSWER:\s*(-?\d[\d,]*)', text, re.I) or re.findall(r'-?\d[\d,]*', text)
    return int(m[-1].replace(",", "")) if m else None


def call(client, model, q, system, max_tokens, retries=3):
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": q}],
                temperature=0.0, max_tokens=max_tokens,
            )
            return r.choices[0].message.content, int((time.time() - t0) * 1000)
        except Exception as e:
            if attempt == retries - 1:
                return f"[error: {e}]", 0
            time.sleep(1.5 * (attempt + 1))                      # backoff on rate-limit/hiccup


def bootstrap_ci(mask, iters=5000, seed=7):
    """90% CI on a rate via nonparametric bootstrap over the sample."""
    if not mask:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(mask)
    rates = sorted(sum(mask[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return (rates[int(0.05 * iters)], rates[int(0.95 * iters)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cloud-max", type=int, default=80, help="run cloud on at most this many (evenly spaced)")
    ap.add_argument("--local-only", action="store_true")
    a = ap.parse_args()

    from openai import OpenAI
    local = OpenAI(base_url=LOCAL_BASE, api_key="ollama")
    cloud = None
    if not a.local_only:
        if not CLOUD_KEY:
            sys.exit("HARP_NIM_API_KEY / NVIDIA_API_KEY not set (or pass --local-only)")
        cloud = OpenAI(base_url=CLOUD_BASE, api_key=CLOUD_KEY)

    probs = gen_problems(a.n, a.seed)
    cloud_ids = set()
    if cloud is not None and a.cloud_max > 0:
        stride = max(1, a.n // a.cloud_max)
        cloud_ids = set(range(1, a.n + 1, stride))

    rows = []
    print(f"running {a.n} queries  (local={LOCAL_MODEL} terse | "
          f"cloud={'-' if a.local_only else CLOUD_MODEL} on {len(cloud_ids)})\n")
    for i, p in enumerate(probs, 1):
        u = mock_score_fn(p["q"])
        la, lms = call(local, LOCAL_MODEL, p["q"], LOCAL_SYS, 160)
        err_local = int(parse_num(la) != p["truth"])
        err_cloud, cp, cms = None, None, 0
        if i in cloud_ids:
            ca, cms = call(cloud, CLOUD_MODEL, p["q"], CLOUD_SYS, 2048)
            cp = parse_num(ca)
            err_cloud = int(cp != p["truth"])
        rows.append({"i": i, "band": p["band"], "u": round(u, 3), "truth": p["truth"],
                     "local_ans": parse_num(la), "err_local": err_local, "cloud_ans": cp,
                     "err_cloud": err_cloud, "local_ms": lms, "cloud_ms": cms, "q": p["q"]})
        cflag = "" if err_cloud is None else (" C✗" if err_cloud else " C✓")
        print(f"  {i:3}/{a.n}  band{p['band']} u={u:.2f}  {'L✗' if err_local else 'L✓'}{cflag}"
              f"  (local {lms}ms{'' if err_cloud is None else f', cloud {cms}ms'})")

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # --- split, fit the FIXED gate on calibration, measure under-route on test
    order = list(range(len(rows)))
    random.Random(a.seed).shuffle(order)
    cut = int(0.7 * len(rows))
    cal = [rows[j] for j in order[:cut]]
    test = [rows[j] for j in order[cut:]]
    gate = ConformalGate(alpha=a.alpha).fit([r["u"] for r in cal], [r["err_local"] for r in cal])

    test_wrong = [r for r in test if r["err_local"] == 1]
    test_right = [r for r in test if r["err_local"] == 0]
    under_mask = [int(not gate.escalate(r["u"])) for r in test_wrong]           # 1 = under-routed
    under = (sum(under_mask) / len(test_wrong)) if test_wrong else 0.0
    over = (sum(1 for r in test_right if gate.escalate(r["u"])) / len(test_right)) if test_right else 0.0
    lo, hi = bootstrap_ci(under_mask)
    esc = sum(1 for r in test if gate.escalate(r["u"])) / len(test) if test else 0.0

    n_wrong = sum(r["err_local"] for r in rows)
    local_acc = 1 - n_wrong / len(rows)
    cl = [r for r in rows if r["err_cloud"] is not None]
    cloud_acc = (1 - sum(r["err_cloud"] for r in cl) / len(cl)) if cl else None

    print("\n" + "=" * 66)
    print(f"REAL CALIBRATION  (n={len(rows)}, cal={len(cal)}, test={len(test)}, alpha={a.alpha})")
    print(f"  local  ({LOCAL_MODEL}) accuracy : {local_acc:.3f}   [{n_wrong}/{len(rows)} wrong]")
    if cloud_acc is not None:
        print(f"  cloud  accuracy (n={len(cl)})            : {cloud_acc:.3f}   (escalation target)")
    print(f"  conformal delta                   : {gate.delta:.3f}")
    print(f"  UNDER-route Pr[kept local|wrong]  : {under:.3f}   90% CI [{lo:.3f}, {hi:.3f}]"
          f"   (bound <= {a.alpha}; test_wrong={len(test_wrong)})")
    print(f"  over-route  Pr[escalated|right]    : {over:.3f}   (disclosed cost of the crude score)")
    print(f"  escalation rate (test)             : {esc:.3f}   ({1-esc:.0%} stayed on-device)")
    print("=" * 66)
    print(f"trace -> {OUT.name}")
    json.dump({"n": len(rows), "alpha": a.alpha, "local_model": LOCAL_MODEL, "cloud_model": CLOUD_MODEL,
               "local_acc": local_acc, "cloud_acc": cloud_acc, "cloud_n": len(cl), "delta": gate.delta,
               "under_route": under, "under_ci": [lo, hi], "over_route": over, "escalation_rate": esc,
               "n_wrong_total": n_wrong, "test_wrong": len(test_wrong)},
              open(OUT.with_suffix(".summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
