"""Eval harness for routing score heads on the held-out test split.

Pass bar: beat the mock baseline AUC AND p95 latency < 10 ms per query.
Candidates: mock (baseline), the trained n-gram head.
Adopt/train decisions read this report.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS_P95_MS = 10.0


def evaluate(fns: dict[str, Callable[[str], float]], rows: list[dict]) -> dict:
    from router.ngram_head import auc  # noqa: E402
    if not rows:
        return {}
    labels = [r["label"] for r in rows]
    rep: dict[str, dict] = {}
    for name, fn in fns.items():
        lat: list[float] = []
        scores: list[float] = []
        for r in rows:
            t0 = time.perf_counter()
            scores.append(fn(r["text"]))
            lat.append((time.perf_counter() - t0) * 1000.0)
        acc = sum((s >= 0.5) == bool(y) for s, y in zip(scores, labels)) / len(rows)
        rep[name] = {
            "auc": round(auc(scores, labels), 4),
            "acc": round(acc, 4),
            "p95_ms": round(statistics.quantiles(lat, n=20)[18] if len(lat) > 1 else lat[0], 3),
        }
    base = rep.get("mock", {}).get("auc", 0.5)
    for name, r in rep.items():
        r["passes"] = bool(name != "mock" and r["auc"] > base and r["p95_ms"] < PASS_P95_MS)
    return rep


def _load_test_rows() -> list[dict]:
    p = ROOT / "routing_dataset" / "test.jsonl"
    if not p.exists():
        subprocess.run([sys.executable, "data/synth_routing_data.py"], cwd=ROOT, check=True)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _main() -> int:
    from router.ngram_head import TrainedScoreHead  # noqa: E402
    from router.router_policy import mock_score_fn  # noqa: E402
    fns: dict[str, Callable[[str], float]] = {"mock": mock_score_fn}
    try:
        fns["trained_ngram_head"] = TrainedScoreHead.load()
    except FileNotFoundError:
        print("n-gram weights missing — run `python -m router.ngram_head` first")
    rep = evaluate(fns, _load_test_rows())
    out = ROOT / "eval_report_score_head.json"
    out.write_text(json.dumps(rep, indent=2))
    for name, r in rep.items():
        print(f"{name:22} auc={r['auc']:.3f} acc={r['acc']:.3f} "
              f"p95={r['p95_ms']:.2f}ms passes={r['passes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
