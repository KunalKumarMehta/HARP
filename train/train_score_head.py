"""Train the MLX score head: linear probe on frozen sentence embeddings.

Only runs when the eval harness says the n-gram head is not enough. Requires
`pip install -e .[apple]`. Embeds the synthetic corpus with an MLX embedding
backbone, fits a logistic head (stdlib SGD — the probe is tiny), and writes
train/mlx_score_head.json (not committed; regenerate on demand).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BACKBONE = "mlx-community/all-MiniLM-L6-v2-bf16"
OUT_PATH = Path(__file__).parent / "mlx_score_head.json"


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def train_dense(feats: list[list[float]], ys: list[float], wts: list[float],
                epochs: int = 8, lr: float = 0.2) -> tuple[list[float], float]:
    """Weighted online SGD logistic regression; wt scales each sample's gradient."""
    if not feats:
        return [], 0.0
    dim = len(feats[0])
    w, b = [0.0] * dim, 0.0
    for _ in range(epochs):
        for f, y, wt in zip(feats, ys, wts):
            g = (_sigmoid(b + sum(wi * xi for wi, xi in zip(w, f))) - y) * wt
            b -= lr * g
            for i, xi in enumerate(f):
                w[i] -= lr * g * xi
    return w, b


class MLXScoreHead:
    __name__ = "mlx_score_head"

    def __init__(self, backbone: str, weights: list[float], bias: float) -> None:
        self.backbone, self.w, self.b = backbone, weights, bias
        self._embedder = None

    @classmethod
    def load(cls, path: Path = OUT_PATH) -> "MLXScoreHead":
        blob = json.loads(Path(path).read_text())
        return cls(blob["backbone"], blob["weights"], blob["bias"])

    def _embed(self, texts: list[str]) -> list[list[float]]:
        from mlx_embeddings import generate, load

        if self._embedder is None:
            self._embedder = load(self.backbone)
        model, tok = self._embedder
        out = generate(model, tok, texts=texts)
        return [list(map(float, row)) for row in out.text_embeds.tolist()]

    def __call__(self, query: str) -> float:
        f = self._embed([query])[0]
        z = self.b + sum(wi * xi for wi, xi in zip(self.w, f))
        return min(max(_sigmoid(z), 0.0), 0.99)


def _load_split(name: str) -> list[dict]:
    p = ROOT / "routing_dataset" / f"{name}.jsonl"
    if not p.exists():
        subprocess.run([sys.executable, "data/synth_routing_data.py"], cwd=ROOT, check=True)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _main() -> int:
    head = MLXScoreHead(BACKBONE, [], 0.0)
    tr, te = _load_split("train"), _load_split("test")
    print(f"embedding {len(tr)} train rows with {BACKBONE} ...")
    feats = head._embed([r["text"] for r in tr])
    w, b = train_dense(feats, [float(r["label"]) for r in tr],
                       [float(r.get("weight", 1.0)) for r in tr])
    head.w, head.b = [round(x, 6) for x in w], round(b, 6)
    OUT_PATH.write_text(json.dumps({"backbone": BACKBONE, "weights": head.w, "bias": head.b}))

    from router.ngram_head import auc

    scores = [head(r["text"]) for r in te[:400]]
    print(f"test AUC (400 rows): {auc(scores, [r['label'] for r in te[:400]]):.3f}")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
