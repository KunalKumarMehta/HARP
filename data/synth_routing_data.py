"""
HARP — Hardware-Aware Routing Platform
data/synth_routing_data.py  ·  MIT

Synthetic routing-decision corpus for the mmBERT-small ENCODER router.
Target: a binary sequence classifier {0=local, 1=escalate}. NOT a decoder.

Failure modes this corpus is designed to avoid:
    - PAIRED GENERATION: hold query constant, run edge-SLM and cloud-LLM, label
      from the correctness/quality DELTA.
    - CLASSIFIER COLLAPSE: artifact-laden corpora mark the small model "optimal"
      for ~79% of queries -> CE collapses to majority class (always-local) ->
      aggressive silent under-routing. Defenses, all applied below:
        (a) deliberately difficulty-BALANCED query corpus (no 79% easy skew),
        (b) inverse-frequency class WEIGHTS emitted per row (train with weighted
            or focal CE — never vanilla CE),
        (c) never trust argmax at inference (the conformal gate is the backstop).
    - VERBOSITY BIAS: strip <think>...</think> before any reward scoring.
    - CNA (Ceiling-Normalized Accuracy): score against the cloud-achievable
      ceiling, not raw accuracy. Reported in stats.

Labeling logic:
  VERIFIABLE (math/code, ground-truth y*):
      edge✓ cloud✓ -> local      (escalation would waste cloud)
      edge✓ cloud✗ -> local
      edge✗ cloud✓ -> escalate   (the high-value label)
      edge✗ cloud✗ -> escalate   (heuristic: bigger model is the only shot)
  OPEN-ENDED (reward-model scored, <think> stripped):
      escalate iff (r_cloud - r_edge) > epsilon

Solver interface: `edge_solve` / `cloud_solve` are pluggable. The mock oracle
below is a seeded difficulty model so the generator runs and the data is
learnable TODAY; swap in (Qwen3-4B on-device) and (Nemotron via NIM) unchanged.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Verbosity-bias defense: reward must not reward CoT length."""
    return THINK_RE.sub("", text).strip()


# ---------------------------------------------------------------- query corpus
# Difficulty-balanced by construction. Each template carries a latent difficulty
# d in [0,1] AND realistic lexical signal (a real router learns markers like
# 'prove'/'derive' correlate with hardness — we don't hand them the label, we
# let the surface form carry it). ~equal mass per band => no collapse skew.

@dataclass(frozen=True)
class Template:
    text: str
    difficulty: float
    task_type: str          # "verifiable" | "open"


_BANDS: list[Template] = [
    # --- trivial (d ~0.05) : chit-chat, ack, single fact lookup
    Template("hi", 0.02, "open"),
    Template("thanks, that helps", 0.03, "open"),
    Template("what time is it in Mumbai", 0.06, "verifiable"),
    Template("convert 10 km to miles", 0.08, "verifiable"),
    Template("what's the capital of Karnataka", 0.07, "verifiable"),
    Template("spell 'accommodation'", 0.05, "verifiable"),
    # --- simple (d ~0.25) : short transform, one-hop
    Template("summarize this paragraph in one line", 0.22, "open"),
    Template("rewrite this sentence more politely", 0.24, "open"),
    Template("what is 17 percent of 240", 0.20, "verifiable"),
    Template("list three synonyms for 'robust'", 0.18, "open"),
    Template("extract the date from: invoice dated 14 March 2026", 0.26, "verifiable"),
    Template("translate 'good morning' to Hindi", 0.23, "verifiable"),
    # --- moderate (d ~0.5) : multi-hop, light reasoning
    Template("compare two pricing plans and say which is cheaper at 30 units", 0.48, "verifiable"),
    Template("draft a two-sentence apology email for a late delivery", 0.46, "open"),
    Template("given these 5 numbers, find the median and explain", 0.50, "verifiable"),
    Template("what are the trade-offs between TCP and UDP for streaming", 0.55, "open"),
    Template("debug why this off-by-one loop misses the last element", 0.58, "verifiable"),
    Template("explain how a hash map handles collisions", 0.52, "open"),
    # --- hard (d ~0.78) : deep reasoning, multi-step proof/design
    Template("prove that the routing gate bounds under-routing at alpha", 0.82, "verifiable"),
    Template("derive the latency budget for a multi-agent planner step by step", 0.80, "verifiable"),
    Template("design a sharded cache with consistency guarantees and justify", 0.84, "open"),
    Template("optimize this dynamic-programming solution and prove correctness", 0.86, "verifiable"),
    Template("diagnose the root cause across these three interacting services", 0.79, "open"),
    Template("architect an offline-first sync protocol and analyze conflict cases", 0.83, "open"),
]

# light surface variation so the encoder sees lexical diversity, not 24 strings
_PREFIX = ["", "please ", "quick q: ", "hey, ", "can you ", "i need to "]
_SUFFIX = ["", "?", " — thanks", " for me", " now"]


def _vary(t: Template, rng: random.Random) -> str:
    s = rng.choice(_PREFIX) + t.text + rng.choice(_SUFFIX)
    return s[0].upper() + s[1:] if s else s


# ---------------------------------------------------------------- mock solver oracle
# Seeded difficulty model. edge correct ~ sigmoid(k(skill_edge - d)); cloud
# stronger. Swap for real (on-device Qwen3-4B) / (Nemotron NIM) via same sig.

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def make_mock_solvers(edge_skill: float, cloud_skill: float, k: float = 9.0):
    def edge_solve(d: float, rng: random.Random) -> tuple[bool, float]:
        p = _sigmoid(k * (edge_skill - d))
        correct = rng.random() < p
        reward = max(0.0, min(1.0, p + rng.gauss(0, 0.05)))   # quality proxy
        return correct, reward

    def cloud_solve(d: float, rng: random.Random) -> tuple[bool, float]:
        p = _sigmoid(k * (cloud_skill - d))
        correct = rng.random() < p
        reward = max(0.0, min(1.0, p + rng.gauss(0, 0.05)))
        return correct, reward

    return edge_solve, cloud_solve


# ---------------------------------------------------------------- labeling

def _label(task_type: str, e_ok: bool, c_ok: bool,
           e_rew: float, c_rew: float, epsilon: float) -> tuple[int, str]:
    if task_type == "verifiable":
        escalate = (not e_ok) and True        # edge wrong -> escalate (both branches)
        escalate = not e_ok
    else:  # open-ended reward margin (rewards already on think-stripped output)
        escalate = (c_rew - e_rew) > epsilon
    return (1, "escalate") if escalate else (0, "local")


# ---------------------------------------------------------------- record

@dataclass
class Record:
    id: str
    text: str
    label: int
    label_str: str
    task_type: str
    edge_correct: bool | None
    cloud_correct: bool | None
    reward_margin: float | None
    weight: float = 1.0
    # --- contention axis features (mirror router_policy.RoutingFeatures) ---
    # Emitted so the corpus schema matches the live feature contract. Sampled
    # independently of difficulty; they do NOT change the complexity label — the
    # contention shed is an inference-time gate over a LOCAL verdict, not a
    # supervised target. Keys are stable for the trainer's feature extractor.
    npu_inflight: bool = False
    npu_queue_depth: int = 0
    tools_present: bool = False
    offline: bool = False


# ---------------------------------------------------------------- generator

def generate(
    n: int,
    seed: int = 13,
    edge_skill: float = 0.55,
    cloud_skill: float = 0.92,
    epsilon: float = 0.12,
) -> list[Record]:
    rng = random.Random(seed)
    edge_solve, cloud_solve = make_mock_solvers(edge_skill, cloud_skill)
    recs: list[Record] = []
    for i in range(n):
        t = rng.choice(_BANDS)
        text = _vary(t, rng)
        e_ok, e_rew = edge_solve(t.difficulty, rng)
        c_ok, c_rew = cloud_solve(t.difficulty, rng)
        label, label_str = _label(t.task_type, e_ok, c_ok, e_rew, c_rew, epsilon)
        inflight = rng.random() < 0.25                    # lane busy ~1/4 of the time
        depth = rng.randint(1, 4) if inflight and rng.random() < 0.5 else (1 if inflight else 0)
        recs.append(Record(
            id=f"r{i:06d}",
            text=text,
            label=label,
            label_str=label_str,
            task_type=t.task_type,
            edge_correct=e_ok if t.task_type == "verifiable" else None,
            cloud_correct=c_ok if t.task_type == "verifiable" else None,
            reward_margin=round(c_rew - e_rew, 4) if t.task_type == "open" else None,
            npu_inflight=inflight,
            npu_queue_depth=depth,
            tools_present=rng.random() < 0.2,
            offline=rng.random() < 0.1,
        ))
    _apply_class_weights(recs)
    return recs


def _apply_class_weights(recs: list[Record]) -> None:
    """Inverse-frequency weights -> the minority (escalate) class is up-weighted
    so weighted/focal CE can't collapse to majority-local. Emitted PER ROW so the
    trainer just reads `weight`."""
    n1 = sum(r.label for r in recs)
    n0 = len(recs) - n1
    if n0 == 0 or n1 == 0:
        return
    w0 = len(recs) / (2.0 * n0)
    w1 = len(recs) / (2.0 * n1)
    for r in recs:
        r.weight = round(w1 if r.label == 1 else w0, 4)


# ---------------------------------------------------------------- stats / CNA

def stats(recs: list[Record]) -> dict:
    n = len(recs)
    n_esc = sum(r.label for r in recs)
    # CNA ceiling: fraction the cloud could solve (verifiable) — the achievable max
    verif = [r for r in recs if r.task_type == "verifiable"]
    cloud_ceiling = (sum(1 for r in verif if r.cloud_correct) / len(verif)) if verif else None
    edge_solo = (sum(1 for r in verif if r.edge_correct) / len(verif)) if verif else None
    return {
        "n": n,
        "escalate": n_esc,
        "local": n - n_esc,
        "escalate_frac": round(n_esc / n, 4),
        "imbalance_ratio": round(max(n_esc, n - n_esc) / max(1, min(n_esc, n - n_esc)), 2),
        "class_weights": {
            "local": next((r.weight for r in recs if r.label == 0), None),
            "escalate": next((r.weight for r in recs if r.label == 1), None),
        },
        "verifiable_frac": round(len(verif) / n, 4),
        "cna_cloud_ceiling": round(cloud_ceiling, 4) if cloud_ceiling is not None else None,
        "edge_solo_accuracy": round(edge_solo, 4) if edge_solo is not None else None,
        "collapse_guard": "PASS" if 0.2 <= (n_esc / n) <= 0.8 else "REVIEW (skewed corpus)",
    }


# ---------------------------------------------------------------- split + write

def split_write(recs: list[Record], out: Path, val_frac=0.1, test_frac=0.1,
                seed: int = 13) -> dict:
    rng = random.Random(seed)
    idx = list(range(len(recs)))
    rng.shuffle(idx)
    n_test = int(len(recs) * test_frac)
    n_val = int(len(recs) * val_frac)
    test_i = set(idx[:n_test])
    val_i = set(idx[n_test:n_test + n_val])
    out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, members: list[Record]) -> None:
        with open(out / name, "w") as fh:
            for r in members:
                fh.write(json.dumps(asdict(r), separators=(",", ":")) + "\n")

    train = [r for j, r in enumerate(recs) if j not in test_i and j not in val_i]
    val = [r for j, r in enumerate(recs) if j in val_i]
    test = [r for j, r in enumerate(recs) if j in test_i]
    dump("train.jsonl", train)
    dump("val.jsonl", val)
    dump("test.jsonl", test)

    meta = {
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "stats": stats(recs),
        "tokenizer": "mmbert-small",
        "max_len": 512,
        "label_map": {"local": 0, "escalate": 1},
        "train_recipe": {
            "head": "binary sequence classification, FP16 head",
            "loss": "weighted CE using per-row `weight` (or focal, gamma=2)",
            "peft": "LoRA r=16 a=32 on attention proj; encoder body W8A16 post-QAT",
            "never": "vanilla CE (collapses to majority-local)",
            "inference_backstop": "conformal gate in router_policy.py — never argmax",
        },
    }
    with open(out / "stats.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


# ---------------------------------------------------------------- cli / self-test

if __name__ == "__main__":
    recs = generate(n=4000, seed=13)
    out = Path("./routing_dataset")
    meta = split_write(recs, out)
    s = meta["stats"]
    print("== synth routing corpus (encoder classifier) ==")
    print(f"  n={s['n']}  local={s['local']}  escalate={s['escalate']}  "
          f"esc_frac={s['escalate_frac']}  imbalance={s['imbalance_ratio']}x")
    print(f"  class_weights: local={s['class_weights']['local']}  "
          f"escalate={s['class_weights']['escalate']}")
    print(f"  CNA cloud ceiling={s['cna_cloud_ceiling']}  "
          f"edge-solo acc={s['edge_solo_accuracy']}")
    print(f"  collapse_guard: {s['collapse_guard']}")
    print(f"  split: {meta['split']}  ->  {out}/")
    print("\n  sample rows:")
    for r in recs[:6]:
        print(f"    [{r.label_str:8} w={r.weight:<5}] {r.text[:60]}")
