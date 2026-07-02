"""Stdlib trained routing score head: hashed n-gram logistic regression.

The zero-dependency floor replacing the mock_score_fn heuristic. Trained on the
synthetic routing corpus (data/synth_routing_data.py). Keeps the score-fn
contract: (str) -> float in [0, 0.99], u(x) = P(escalate), deterministic, <10ms.
The MLX-trained encoder head (train/train_score_head.py) is the upgrade path.

Run `python -m router.ngram_head` to (re)train: regenerates the dataset if
absent, fits weights, writes score_head_weights.json + score_head_calibration.json
(u/err on the val split — the conformal gate must be calibrated on THIS score
axis, never on demo_calibration()'s synthetic one).
"""
from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

DIM = 16384
WEIGHTS_PATH = Path(__file__).parent / "score_head_weights.json"
CALIBRATION_PATH = Path(__file__).parent / "score_head_calibration.json"


def _h(token: str) -> int:
    # ponytail: crc32 hashing trick; builtin hash() is salted
    return zlib.crc32(token.encode("utf-8")) % DIM


def featurize(text: str) -> dict[int, float]:
    """Word unigrams + char 3-5 grams, hashed to DIM, L2-normalized."""
    t = text.lower()
    idx: dict[int, float] = {}
    for w in t.split():
        k = _h("w:" + w)
        idx[k] = idx.get(k, 0.0) + 1.0
    for n in (3, 4, 5):
        for i in range(len(t) - n + 1):
            k = _h(f"c{n}:" + t[i : i + n])
            idx[k] = idx.get(k, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in idx.values())) or 1.0
    return {k: v / norm for k, v in idx.items()}


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


class TrainedScoreHead:
    """Loads committed weights; callable with the score-fn contract."""

    __name__ = "trained_ngram_head"

    def __init__(self, weights: list[float], bias: float) -> None:
        if len(weights) != DIM:
            raise ValueError(f"weight dim {len(weights)} != {DIM}")
        self.w = weights
        self.b = bias

    @classmethod
    def load(cls, path: Path = WEIGHTS_PATH) -> "TrainedScoreHead":
        blob = json.loads(Path(path).read_text())
        return cls(blob["weights"], blob["bias"])

    def __call__(self, query: str) -> float:
        f = featurize(query)
        z = self.b + sum(self.w[i] * v for i, v in f.items())
        return min(max(_sigmoid(z), 0.0), 0.99)


def load_head_calibration(
    path: Path = CALIBRATION_PATH,
) -> tuple[list[float], list[int]] | None:
    try:
        blob = json.loads(Path(path).read_text())
        return list(blob["u"]), list(blob["err"])
    except (OSError, ValueError, KeyError):
        return None


def auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based (Mann-Whitney) AUC, ties get half credit."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    # ponytail: O(P*N) fine at 4k rows
    wins = sum((p > n_) + 0.5 * (p == n_) for p in pos for n_ in neg)
    return wins / (len(pos) * len(neg))


def train(
    rows: list[dict], epochs: int = 6, lr: float = 0.5
) -> tuple[list[float], float]:
    """Plain SGD logistic regression; uses per-record `weight` (anti-collapse)."""
    w = [0.0] * DIM
    b = 0.0
    feats = [(featurize(r["text"]), float(r["label"]), float(r.get("weight", 1.0))) for r in rows]
    for _ in range(epochs):
        for f, y, wt in feats:  # fixed order: deterministic training
            z = b + sum(w[i] * v for i, v in f.items())
            g = (_sigmoid(z) - y) * wt
            b -= lr * g
            for i, v in f.items():
                w[i] -= lr * g * v
    return w, b


def _load_split(dsdir: Path, name: str) -> list[dict]:
    p = dsdir / f"{name}.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _ensure_dataset(root: Path) -> Path:
    ds = root / "routing_dataset"
    if not (ds / "train.jsonl").exists():
        import subprocess
        import sys

        subprocess.run([sys.executable, "data/synth_routing_data.py"], cwd=root, check=True)
    return ds


def _main() -> int:
    root = Path(__file__).resolve().parents[1]
    ds = _ensure_dataset(root)
    tr, va, te = (_load_split(ds, n) for n in ("train", "val", "test"))
    w, b = train(tr)
    head = TrainedScoreHead([round(x, 5) for x in w], round(b, 5))

    from router.router_policy import mock_score_fn

    labels = [r["label"] for r in te]
    a_head = auc([head(r["text"]) for r in te], labels)
    a_mock = auc([mock_score_fn(r["text"]) for r in te], labels)
    print(f"test AUC: trained={a_head:.3f}  mock={a_mock:.3f}")
    assert a_head > a_mock, "trained head must beat the mock baseline"

    WEIGHTS_PATH.write_text(json.dumps({"dim": DIM, "weights": head.w, "bias": head.b}))
    # Calibration on the val split, VERIFIABLE rows only: err = 1 iff edge was wrong.
    cal_rows = [r for r in va if r.get("edge_correct") is not None]
    u = [head(r["text"]) for r in cal_rows]
    err = [0 if r["edge_correct"] else 1 for r in cal_rows]
    CALIBRATION_PATH.write_text(json.dumps({"u": u, "err": err}))
    print(f"wrote {WEIGHTS_PATH.name} ({WEIGHTS_PATH.stat().st_size // 1024} KB), "
          f"{CALIBRATION_PATH.name} ({len(u)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
