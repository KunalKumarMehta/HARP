# Apple Silicon Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mock routing score head and the placeholder LOCAL-tier answerer with real models: HF-scouted or trained, Apple silicon (MLX) for inference, NVIDIA (`hf jobs`) for heavy training.

**Architecture:** Eval-first pipeline. A stdlib n-gram head provides the zero-dep floor for routing scores; `scripts/hf_scout.py` finds Hub candidates; `evals/` harnesses measure candidates against pass bars on our own data; `train/` scripts train only when nobody passes. Wire points: `RoutingPolicy(score_fn=...)` and `mac_demo.call_model`'s local branch.

**Tech Stack:** Python ≥3.11 stdlib (core), optional extra `apple` = mlx-lm + mlx-embeddings + huggingface_hub, `hf` CLI (authed, user kkmp).

**Spec:** `docs/superpowers/specs/2026-07-02-apple-silicon-models-design.md`

## Global Constraints

- Core stays stdlib-only: `dependencies = []` in pyproject. MLX/HF deps ONLY in the `apple` optional extra.
- Score fn contract: `(str) -> float`, output in `[0.0, 0.99]`, deterministic. Head inference < 10 ms per query on M-series.
- Conformal gate delta is score-scale-dependent: any new score fn ships matching calibration arrays computed from labeled queries (u = score_fn(text), err = 1 - edge_correct on verifiable records).
- Datasets and model weights are NOT committed — except `router/score_head_weights.json` and `router/score_head_calibration.json` (small JSON artifacts, committed).
- Dataset regeneration is deterministic: `python data/synth_routing_data.py` (n=4000, seed=13, writes `./routing_dataset/`).
- Paid `hf jobs` launches require an explicit `--confirm` flag; never automatic.
- Chat pass bar: TTFT < 2 s, ≥ 20 tok/s, ≥ 16/20 rubric-pass. Score-head pass bar: beat mock AUC on test split + < 10 ms.
- Tests must run without network and without MLX installed (skip-guard MLX paths).
- Repo tests run via plain `python tests/test_*.py` (each has a `_main()` runner) — follow that pattern, no pytest dependency.
- Commit after every task. Line length 100 (ruff).

**Note (deviation from spec paths):** eval harnesses live in `evals/` not `eval/` (avoids shadowing the builtin name as a top-level package).

---

### Task 1: Stdlib n-gram score head

**Files:**
- Create: `router/ngram_head.py`
- Create: `tests/test_ngram_head.py`
- Generated then committed: `router/score_head_weights.json`, `router/score_head_calibration.json`

**Interfaces:**
- Consumes: `data/synth_routing_data.py` CLI (regenerates `routing_dataset/{train,val,test}.jsonl`; fields used: `text: str`, `label: int`, `weight: float`, `task_type: str`, `edge_correct: bool|None`); `router.router_policy.mock_score_fn`.
- Produces: `TrainedScoreHead` (callable, `__call__(query: str) -> float`, attr `__name__ == "trained_ngram_head"`, classmethod `load(path=WEIGHTS_PATH) -> TrainedScoreHead`), `load_head_calibration(path=CALIBRATION_PATH) -> tuple[list[float], list[int]] | None`, `auc(scores: list[float], labels: list[int]) -> float`, module `__main__` that trains + writes both JSON artifacts.

- [ ] **Step 1: Write the failing test**

`tests/test_ngram_head.py`:

```python
"""Contract tests for the stdlib n-gram score head. Stdlib-only, no network."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from router.ngram_head import (  # noqa: E402
    CALIBRATION_PATH, WEIGHTS_PATH, TrainedScoreHead, auc, load_head_calibration,
)
from router.router_policy import mock_score_fn  # noqa: E402


def _ensure_artifacts() -> None:
    if not WEIGHTS_PATH.exists() or not CALIBRATION_PATH.exists():
        subprocess.run([sys.executable, "-m", "router.ngram_head"], cwd=ROOT, check=True)


def _load_split(name: str) -> list[dict]:
    p = ROOT / "routing_dataset" / f"{name}.jsonl"
    if not p.exists():
        subprocess.run([sys.executable, "data/synth_routing_data.py"], cwd=ROOT, check=True)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_auc_rank_based() -> None:
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert abs(auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) - 0.5) < 1e-9


def test_contract_range_and_determinism() -> None:
    _ensure_artifacts()
    head = TrainedScoreHead.load()
    assert head.__name__ == "trained_ngram_head"
    for q in ("hi", "prove the gate bounds under-routing", "x" * 2000, ""):
        s1, s2 = head(q), head(q)
        assert s1 == s2, "must be deterministic"
        assert 0.0 <= s1 <= 0.99, s1


def test_beats_mock_auc_on_test_split() -> None:
    _ensure_artifacts()
    head = TrainedScoreHead.load()
    rows = _load_split("test")
    labels = [r["label"] for r in rows]
    a_head = auc([head(r["text"]) for r in rows], labels)
    a_mock = auc([mock_score_fn(r["text"]) for r in rows], labels)
    assert a_head > a_mock, f"trained head AUC {a_head:.3f} <= mock {a_mock:.3f}"


def test_calibration_artifact() -> None:
    _ensure_artifacts()
    cal = load_head_calibration()
    assert cal is not None
    u, err = cal
    assert len(u) == len(err) > 100
    assert all(0.0 <= x <= 0.99 for x in u)
    assert set(err) <= {0, 1} and 0 < sum(err) < len(err)  # non-separable, both classes


def test_load_missing_returns_none_or_raises() -> None:
    assert load_head_calibration(Path("/nonexistent/cal.json")) is None
    try:
        TrainedScoreHead.load(Path("/nonexistent/w.json"))
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  OK {fn.__name__}")
    print(f"test_ngram_head: {len(fns)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_ngram_head.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'router.ngram_head'`

- [ ] **Step 3: Write the implementation**

`router/ngram_head.py`:

```python
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
    return zlib.crc32(token.encode("utf-8")) % DIM  # ponytail: crc32 hashing trick; builtin hash() is salted


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
    wins = sum((p > n_) + 0.5 * (p == n_) for p in pos for n_ in neg)  # ponytail: O(P*N) fine at 4k rows
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
```

- [ ] **Step 4: Train and run tests**

Run: `python -m router.ngram_head && python tests/test_ngram_head.py`
Expected: training prints `test AUC: trained=0.xxx mock=0.yyy` with trained > mock; then `test_ngram_head: 5 checks passed`.
If AUC does not beat mock, raise epochs to 10 / try lr 0.3 — do not lower the bar.

- [ ] **Step 5: Ignore the dataset dir, commit**

Add `routing_dataset/` to `.gitignore` (if absent).

```bash
git add router/ngram_head.py router/score_head_weights.json router/score_head_calibration.json tests/test_ngram_head.py .gitignore
git commit -m "feat(router): stdlib n-gram trained score head + own-axis calibration artifacts"
```

---

### Task 2: Wire trained head as the endpoint default

**Files:**
- Modify: `router/router_policy.py` (add `default_score_fn()` after `mock_score_fn`, ~line 223)
- Modify: `serve/openai_endpoint.py:115-126` (`_default_policy`, `_classifier_name`)
- Modify: `tests/test_route_endpoint.py:92-95`

**Interfaces:**
- Consumes: `router.ngram_head.TrainedScoreHead`, `load_head_calibration` (Task 1).
- Produces: `router.router_policy.default_score_fn() -> Callable[[str], float]` — trained head if weights load, else warn + `mock_score_fn`. `RoutingPolicy.__init__` default stays `mock_score_fn` (its self-test calibration is coupled to that scale); the endpoint is what upgrades.

- [ ] **Step 1: Update the endpoint test to expect the trained head**

Replace `test_health_reports_classifier` in `tests/test_route_endpoint.py`:

```python
def test_health_reports_classifier() -> None:
    body = _online_client().get("/health").json()
    assert "route_classifier" in body
    assert "trained_ngram_head" in body["route_classifier"], body["route_classifier"]
    assert "placeholder" not in body["route_classifier"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python tests/test_route_endpoint.py`
Expected: FAIL on the new assertion (`mock_score_fn (placeholder for mmBERT-small head)` still reported).

- [ ] **Step 3: Add `default_score_fn` to router_policy.py**

Insert directly after `mock_score_fn` (keep `mock_score_fn` unchanged; update only its docstring line "swap in the QNN-EP encoder for production" → "swap in a trained head (router/ngram_head.py or the MLX encoder) for production"):

```python
def default_score_fn() -> Callable[[str], float]:
    """Best available score head: the trained n-gram head when its committed
    weights load, else the mock heuristic (with a warning). Callers using the
    trained head MUST calibrate on its own score axis (load_head_calibration),
    not on demo_calibration() — delta is scale-dependent."""
    try:
        from router.ngram_head import TrainedScoreHead

        return TrainedScoreHead.load()
    except Exception as e:  # missing/corrupt weights: degrade, never crash routing
        import warnings

        warnings.warn(f"trained score head unavailable ({e}); using mock_score_fn")
        return mock_score_fn
```

- [ ] **Step 4: Wire the endpoint**

In `serve/openai_endpoint.py`, extend the import at line 52 to include `default_score_fn`, then replace `_default_policy` and `_classifier_name`:

```python
def _default_policy() -> RoutingPolicy:
    fn = default_score_fn()
    cal = None
    if getattr(fn, "__name__", "") == "trained_ngram_head":
        from router.ngram_head import load_head_calibration

        cal = load_head_calibration()
    return RoutingPolicy(score_fn=fn).calibrate(*(cal or demo_calibration()))


def _classifier_name(state) -> str:
    """Name of the active complexity score_fn behind the AUTO gate."""
    fn = getattr(state.policy, "score_fn", None)
    name = getattr(fn, "__name__", "unknown")
    suffix = " (heuristic fallback; trained head unavailable)" if name == "mock_score_fn" else ""
    return f"{name}{suffix}"
```

- [ ] **Step 5: Run the endpoint + router tests**

Run: `python tests/test_route_endpoint.py && python router/router_policy.py`
Expected: all endpoint checks pass (including existing routing-behavior tests — if `test_busy_hint_sheds` or offline tests fail, the trained head + its calibration changed AUTO outcomes: verify the failing case's expected decision still holds semantically; guards should be score-independent) and the router self-test still passes (it uses mock + synth calibration, untouched).

- [ ] **Step 6: Commit**

```bash
git add router/router_policy.py serve/openai_endpoint.py tests/test_route_endpoint.py
git commit -m "feat(serve): endpoint gate runs the trained n-gram head on its own calibration axis"
```

---

### Task 3: HF scout script

**Files:**
- Create: `scripts/hf_scout.py`
- Create: `tests/test_hf_scout.py`

**Interfaces:**
- Consumes: HF Hub REST API `https://huggingface.co/api/models` (stdlib `urllib`; no huggingface_hub needed).
- Produces: CLI `python scripts/hf_scout.py --task {routing|chat} [--limit N] [--max-params-b F] [--require-mlx] [--out PATH]` writing `scout_report_<task>.json`; pure function `rank(models: list[dict], *, max_params_b: float | None, licenses: set[str], require_mlx: bool) -> list[dict]` (each ranked dict gains `"scout_score": float`).

- [ ] **Step 1: Write the failing test**

`tests/test_hf_scout.py`:

```python
"""Scout ranking is pure + deterministic; tested on fixtures, no network."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hf_scout import ALLOWED_LICENSES, rank  # noqa: E402

FIXTURE = [
    {"id": "mlx-community/Qwen2.5-3B-Instruct-4bit", "downloads": 50000, "likes": 100,
     "tags": ["mlx", "license:apache-2.0"], "lastModified": "2026-05-01T00:00:00.000Z",
     "safetensors": {"total": 3_000_000_000}},
    {"id": "bigco/agpl-model", "downloads": 900000, "likes": 5000,
     "tags": ["license:agpl-3.0"], "lastModified": "2026-06-01T00:00:00.000Z",
     "safetensors": {"total": 1_000_000_000}},
    {"id": "someone/huge-70b", "downloads": 800000, "likes": 900,
     "tags": ["license:apache-2.0"], "lastModified": "2026-06-01T00:00:00.000Z",
     "safetensors": {"total": 70_000_000_000}},
    {"id": "acme/router-classifier", "downloads": 1200, "likes": 10,
     "tags": ["license:mit"], "lastModified": "2025-01-01T00:00:00.000Z",
     "safetensors": {"total": 100_000_000}},
]


def test_filters_license_and_size() -> None:
    got = rank(FIXTURE, max_params_b=8.0, licenses=ALLOWED_LICENSES, require_mlx=False)
    ids = [m["id"] for m in got]
    assert "bigco/agpl-model" not in ids          # license filtered
    assert "someone/huge-70b" not in ids          # over param budget
    assert "acme/router-classifier" in ids


def test_mlx_required_and_deterministic() -> None:
    got = rank(FIXTURE, max_params_b=8.0, licenses=ALLOWED_LICENSES, require_mlx=True)
    assert [m["id"] for m in got] == ["mlx-community/Qwen2.5-3B-Instruct-4bit"]
    again = rank(list(reversed(FIXTURE)), max_params_b=8.0, licenses=ALLOWED_LICENSES,
                 require_mlx=True)
    assert [m["id"] for m in got] == [m["id"] for m in again]


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  OK {fn.__name__}")
    print(f"test_hf_scout: {len(fns)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
```

- [ ] **Step 2: Run to verify it fails**

Run: `python tests/test_hf_scout.py`
Expected: FAIL `ModuleNotFoundError: No module named 'hf_scout'`

- [ ] **Step 3: Implement**

`scripts/hf_scout.py`:

```python
"""Scout the HF Hub for candidate models per HARP task. Stdlib-only.

Usage:
  python scripts/hf_scout.py --task chat --require-mlx --max-params-b 8
  python scripts/hf_scout.py --task routing --max-params-b 1

Writes scout_report_<task>.json (ranked shortlist). Offline / API failure:
falls back to the last cached report and says so.
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://huggingface.co/api/models"
ALLOWED_LICENSES = {"apache-2.0", "mit", "bsd-3-clause", "llama3.2", "gemma", "qwen"}
QUERIES = {
    # (search terms, extra query params) per task; several angles per task.
    "routing": [
        ("prompt complexity classifier", {}),
        ("prompt router", {}),
        ("query difficulty", {}),
        ("routellm", {}),
    ],
    "chat": [
        ("instruct 4bit", {"author": "mlx-community", "pipeline_tag": "text-generation"}),
        ("instruct", {"pipeline_tag": "text-generation", "library": "mlx"}),
    ],
}


def _fetch(search: str, extra: dict, limit: int) -> list[dict]:
    params = {"search": search, "limit": str(limit), "sort": "downloads", "direction": "-1",
              "expand[]": "safetensors", **extra}
    url = f"{API}?{urllib.parse.urlencode(params, doseq=True)}"
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read())


def _license(m: dict) -> str | None:
    for t in m.get("tags", []):
        if t.startswith("license:"):
            return t.removeprefix("license:")
    return None


def _params_b(m: dict) -> float | None:
    total = (m.get("safetensors") or {}).get("total")
    return total / 1e9 if total else None


def _is_mlx(m: dict) -> bool:
    return m["id"].startswith("mlx-community/") or "mlx" in m.get("tags", [])


def _age_days(m: dict) -> float:
    ts = m.get("lastModified")
    if not ts:
        return 3650.0
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return max(0.0, (datetime.now(timezone.utc) - dt).days)


def rank(models: list[dict], *, max_params_b: float | None, licenses: set[str],
         require_mlx: bool) -> list[dict]:
    seen: set[str] = set()
    out = []
    for m in models:
        if m["id"] in seen:
            continue
        seen.add(m["id"])
        lic = _license(m)
        if lic is not None and lic not in licenses:
            continue
        pb = _params_b(m)
        if max_params_b is not None and pb is not None and pb > max_params_b:
            continue
        if require_mlx and not _is_mlx(m):
            continue
        score = (math.log10(1 + m.get("downloads", 0))
                 + 0.5 * math.log10(1 + m.get("likes", 0))
                 + (2.0 if _is_mlx(m) else 0.0)
                 - _age_days(m) / 365.0)
        out.append({**m, "scout_score": round(score, 4)})
    return sorted(out, key=lambda m: (-m["scout_score"], m["id"]))


def scout(task: str, limit: int, max_params_b: float | None, require_mlx: bool,
          out_path: Path) -> list[dict]:
    raw: list[dict] = []
    try:
        for search, extra in QUERIES[task]:
            raw.extend(_fetch(search, extra, limit))
    except OSError as e:
        if out_path.exists():
            print(f"HF API unreachable ({e}); using cached {out_path}")
            return json.loads(out_path.read_text())["candidates"]
        raise SystemExit(f"HF API unreachable and no cached report: {e}")
    ranked = rank(raw, max_params_b=max_params_b, licenses=ALLOWED_LICENSES,
                  require_mlx=require_mlx)[:25]
    out_path.write_text(json.dumps({"task": task, "candidates": ranked}, indent=2))
    print(f"wrote {out_path} ({len(ranked)} candidates)")
    for m in ranked[:10]:
        print(f"  {m['scout_score']:7.3f}  {m['id']}")
    return ranked


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=sorted(QUERIES), required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-params-b", type=float, default=None)
    ap.add_argument("--require-mlx", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or Path(f"scout_report_{a.task}.json")
    scout(a.task, a.limit, a.max_params_b, a.require_mlx, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 4: Run tests, then one live smoke**

Run: `python tests/test_hf_scout.py`
Expected: `test_hf_scout: 2 checks passed`
Run: `python scripts/hf_scout.py --task chat --require-mlx --max-params-b 8`
Expected: writes `scout_report_chat.json`, prints a ranked top-10 of `mlx-community/*` instruct models.

- [ ] **Step 5: Ignore reports, commit**

Add `scout_report_*.json` to `.gitignore`.

```bash
git add scripts/hf_scout.py tests/test_hf_scout.py .gitignore
git commit -m "feat(scripts): stdlib HF Hub scout with ranked, license-filtered shortlists"
```

---

### Task 4: Score-head eval harness

**Files:**
- Create: `evals/__init__.py` (empty)
- Create: `evals/eval_score_head.py`
- Create: `tests/test_eval_score_head.py`

**Interfaces:**
- Consumes: `router.ngram_head` (`TrainedScoreHead`, `auc`), `router.router_policy.mock_score_fn`, `routing_dataset/test.jsonl`.
- Produces: `evaluate(fns: dict[str, Callable[[str], float]], rows: list[dict]) -> dict` returning `{name: {"auc": float, "acc": float, "p95_ms": float, "passes": bool}}` (`passes` = beats `"mock"`'s AUC AND p95 < 10 ms; the `"mock"` entry itself gets `passes: False`); CLI writes `eval_report_score_head.json`.

- [ ] **Step 1: Write the failing test**

`tests/test_eval_score_head.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.eval_score_head import evaluate  # noqa: E402

ROWS = [{"text": t, "label": y} for t, y in [
    ("hi", 0), ("thanks", 0), ("what time is it", 0), ("ok", 0),
    ("prove the conformal bound step by step", 1),
    ("design a distributed planner and derive its latency budget", 1),
    ("why does the isotonic calibrator dominate platt scaling here", 1),
    ("summarize", 0),
]]


def test_evaluate_shapes_and_pass_bar() -> None:
    good = lambda q: 0.9 if len(q) > 20 else 0.1  # noqa: E731
    bad = lambda q: 0.5  # noqa: E731
    rep = evaluate({"mock": bad, "good": good}, ROWS)
    assert set(rep) == {"mock", "good"}
    for r in rep.values():
        assert {"auc", "acc", "p95_ms", "passes"} <= set(r)
    assert rep["good"]["auc"] > rep["mock"]["auc"]
    assert rep["good"]["passes"] is True
    assert rep["mock"]["passes"] is False


def _main() -> int:
    test_evaluate_shapes_and_pass_bar()
    print("test_eval_score_head: 1 check passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
```

- [ ] **Step 2: Run to verify it fails**

Run: `python tests/test_eval_score_head.py`
Expected: FAIL `ModuleNotFoundError: No module named 'evals'`

- [ ] **Step 3: Implement**

`evals/eval_score_head.py`:

```python
"""Eval harness for routing score heads on the held-out test split.

Pass bar: beat the mock baseline AUC AND p95 latency < 10 ms per query.
Candidates: mock (baseline), the trained n-gram head, and — when installed —
an MLX head via --mlx-weights. Adopt/train decisions read this report.
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

from router.ngram_head import TrainedScoreHead, auc  # noqa: E402
from router.router_policy import mock_score_fn  # noqa: E402

PASS_P95_MS = 10.0


def evaluate(fns: dict[str, Callable[[str], float]], rows: list[dict]) -> dict:
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
```

- [ ] **Step 4: Run test + harness**

Run: `python tests/test_eval_score_head.py && python evals/eval_score_head.py`
Expected: test passes; harness prints both rows with `trained_ngram_head ... passes=True`.

- [ ] **Step 5: Commit**

Add `eval_report_*.json` to `.gitignore`.

```bash
git add evals/ tests/test_eval_score_head.py .gitignore
git commit -m "feat(evals): score-head eval harness with measured adopt/train pass bar"
```

---

### Task 5: Local chat model — MLX runtime, eval, mac_demo wiring

**Files:**
- Modify: `pyproject.toml:30-35` (add `apple` extra), `pyproject.toml:42` (packages += `"evals"`)
- Create: `mac_demo/models.json`, `mac_demo/prompts_eval.jsonl`, `mac_demo/local_llm.py`
- Create: `evals/eval_local_llm.py`
- Modify: `mac_demo/harp_demo.py:82-105` (`call_model` local branch)
- Create: `tests/test_local_llm.py`

**Interfaces:**
- Consumes: scout report (Task 3) to choose the default model id; `mlx_lm` (optional).
- Produces: `mac_demo.local_llm.available() -> bool`; `mac_demo.local_llm.generate(prompt: str, max_tokens: int = 256) -> str` (raises `RuntimeError` if unavailable); `evals.eval_local_llm.rubric_pass(answer: str, expect: list[str]) -> bool`; manifest `mac_demo/models.json` `{"local_chat": "<hf model id>"}`.

- [ ] **Step 1: pyproject — `apple` extra and `evals` package**

In `[project.optional-dependencies]` add:

```toml
# Apple-silicon inference/training path (MLX). Core stays stdlib-only.
apple = ["mlx-lm>=0.21", "mlx-embeddings>=0.0.3", "huggingface_hub>=0.34"]
```

In `[tool.setuptools]` change packages line to:

```toml
packages = ["shared", "router", "cloud", "edge", "fabric", "demo", "serve", "evals"]
```

- [ ] **Step 2: Manifest + prompt set**

`mac_demo/models.json` — default from the Task 3 scout report's top chat candidate (expected shape):

```json
{"local_chat": "mlx-community/Qwen2.5-3B-Instruct-4bit"}
```

`mac_demo/prompts_eval.jsonl` — 20 prompts, each `{"prompt": ..., "expect": [keywords]}`. Mix: 8 summarization (give a paragraph inline, expect its key nouns), 6 short factual/chat, 6 rewrite/format tasks. Example rows (write all 20 in this style, self-contained, no external files):

```json
{"prompt": "Summarize in one sentence: The conformal gate bounds the probability that a hard query stays on-device by calibrating a threshold on queries the edge model got wrong.", "expect": ["conformal", "edge"]}
{"prompt": "List three risks of running an LLM only on-device.", "expect": ["battery", "memory"]}
{"prompt": "Rewrite as a polite one-line email: meeting moved to 3pm.", "expect": ["3"]}
```

- [ ] **Step 3: Write the failing test**

`tests/test_local_llm.py`:

```python
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mac_demo"))

import local_llm  # noqa: E402
from evals.eval_local_llm import rubric_pass  # noqa: E402


def test_rubric() -> None:
    assert rubric_pass("The conformal gate protects edge routing.", ["conformal", "edge"])
    assert not rubric_pass("no idea", ["conformal"])
    assert rubric_pass("anything", [])  # empty expectations always pass


def test_prompt_set_wellformed() -> None:
    rows = [json.loads(x) for x in
            (ROOT / "mac_demo" / "prompts_eval.jsonl").read_text().splitlines() if x.strip()]
    assert len(rows) == 20
    assert all(isinstance(r["prompt"], str) and isinstance(r["expect"], list) for r in rows)


def test_generate_guarded() -> None:
    if local_llm.available():
        out = local_llm.generate("Say the word ready.", max_tokens=8)
        assert isinstance(out, str) and out.strip()
    else:
        try:
            local_llm.generate("hi")
            raise AssertionError("expected RuntimeError when mlx_lm missing")
        except RuntimeError:
            pass


def _main() -> int:
    for fn in (test_rubric, test_prompt_set_wellformed, test_generate_guarded):
        fn()
        print(f"  OK {fn.__name__}")
    print("test_local_llm: 3 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
```

- [ ] **Step 4: Run to verify it fails** — `python tests/test_local_llm.py` → `ModuleNotFoundError: No module named 'local_llm'`

- [ ] **Step 5: Implement runtime + eval harness**

`mac_demo/local_llm.py`:

```python
"""LOCAL-tier answerer on Apple silicon via MLX. Model id from models.json.

Optional path: requires `pip install -e .[apple]` and a one-time
`hf download <model>` (mlx_lm downloads on first load otherwise).
Everything degrades gracefully when mlx_lm is missing — callers check
available() or catch RuntimeError.
"""
from __future__ import annotations

import json
from pathlib import Path

_MANIFEST = Path(__file__).parent / "models.json"
_model = None
_tokenizer = None


def model_id() -> str:
    return json.loads(_MANIFEST.read_text())["local_chat"]


def available() -> bool:
    try:
        import mlx_lm  # noqa: F401

        return True
    except ImportError:
        return False


def _load():
    global _model, _tokenizer
    if _model is None:
        from mlx_lm import load

        _model, _tokenizer = load(model_id())
    return _model, _tokenizer


def generate(prompt: str, max_tokens: int = 256) -> str:
    if not available():
        raise RuntimeError("mlx_lm not installed — pip install -e .[apple]")
    from mlx_lm import generate as _gen

    model, tok = _load()
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return _gen(model, tok, prompt=text, max_tokens=max_tokens, verbose=False)
```

`evals/eval_local_llm.py`:

```python
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
    from mlx_lm import stream_generate

    model, tok = local_llm._load()
    results, ttfts, tps_all = [], [], []
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"]}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        t0, first, ntok, out = time.perf_counter(), None, 0, []
        for resp in stream_generate(model, tok, prompt=text, max_tokens=256):
            if first is None:
                first = time.perf_counter() - t0
            ntok += 1
            out.append(resp.text)
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
```

- [ ] **Step 6: Wire `mac_demo/harp_demo.py`**

In `call_model` (line 82), before the OpenAI-client path for the local tier, insert an MLX-first branch (keep the existing client path as fallback):

```python
def call_model(where: str, text: str):
    """Return (answer, latency_ms). where in {local, cloud}."""
    if where == "local":
        try:
            import local_llm
            if local_llm.available():
                t0 = time.time()
                out = local_llm.generate(text)
                return out, int((time.time() - t0) * 1000)
        except Exception:
            pass  # fall through to the configured local endpoint
    # ... existing OpenAI-client implementation unchanged ...
```

- [ ] **Step 7: Install extra, run tests + eval**

```bash
pip install -e ".[apple]"
python tests/test_local_llm.py
python evals/eval_local_llm.py
```

Expected: 3 checks pass; eval prints per-model line ending `passes=True`. First eval run downloads the model (~2 GB) — that's expected. If `passes=False` on tok/s, try the 1.5B scout candidate before touching training.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml mac_demo/models.json mac_demo/prompts_eval.jsonl mac_demo/local_llm.py evals/eval_local_llm.py mac_demo/harp_demo.py tests/test_local_llm.py
git commit -m "feat(mac_demo): MLX local chat tier + measured eval harness (adopt path)"
```

---

### Task 6: MLX score-head trainer (train-if-needed path)

**Files:**
- Create: `train/__init__.py` (empty), `train/train_score_head.py`
- Modify: `pyproject.toml:42` (packages += `"train"`)
- Create: `tests/test_train_score_head.py`

**Interfaces:**
- Consumes: `routing_dataset/` splits; `mlx-embeddings` (optional) with backbone default `mlx-community/all-MiniLM-L6-v2-bf16`; stdlib SGD pattern from Task 1.
- Produces: `train/mlx_score_head.json` (`{"backbone": str, "weights": [...], "bias": float}`, NOT committed) and `MLXScoreHead` (callable score-fn contract, `__name__ == "mlx_score_head"`, `classmethod load(path)`); pure `train_dense(feats: list[list[float]], ys: list[float], wts: list[float], epochs: int = 8, lr: float = 0.2) -> tuple[list[float], float]` (importable without MLX).

- [ ] **Step 1: Write the failing test** (`tests/test_train_score_head.py` — tests the dense SGD + contract without MLX):

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.train_score_head import train_dense  # noqa: E402


def test_train_dense_separates() -> None:
    feats = [[1.0, 0.0]] * 20 + [[0.0, 1.0]] * 20
    ys = [0.0] * 20 + [1.0] * 20
    w, b = train_dense(feats, ys, [1.0] * 40)
    score = lambda f: 1 / (1 + 2.718281828 ** -(b + sum(x * y for x, y in zip(w, f))))  # noqa: E731
    assert score([0.0, 1.0]) > 0.8 > 0.2 > score([1.0, 0.0])


def _main() -> int:
    test_train_dense_separates()
    print("test_train_score_head: 1 check passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
```

- [ ] **Step 2: Run to verify it fails** — `python tests/test_train_score_head.py` → ModuleNotFoundError

- [ ] **Step 3: Implement** `train/train_score_head.py`:

```python
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
```

- [ ] **Step 4: Run test** — `python tests/test_train_score_head.py` → `1 check passed`. Full MLX training run (`python -m train.train_score_head`) only if the score-head eval report says the n-gram head fails its bar — record its AUC in the eval report either way if run. Note: MLXScoreHead p95 will include embedding time; it must still clear the < 10 ms bar via `evals/eval_score_head.py` before ever being adopted.

- [ ] **Step 5: Commit**

Add `train/mlx_score_head.json` to `.gitignore`.

```bash
git add train/ tests/test_train_score_head.py pyproject.toml .gitignore
git commit -m "feat(train): MLX linear-probe score head (train-if-needed path)"
```

---

### Task 7: NVIDIA fine-tune job launcher (chat, escape hatch)

**Files:**
- Create: `train/finetune_llm_job.py`
- Create: `tests/test_finetune_job.py`

**Interfaces:**
- Consumes: `hf jobs uv run` CLI (authed); TRL SFT.
- Produces: `build_command(model_id: str, dataset_id: str, flavor: str = "l4x1", timeout: str = "2h") -> list[str]`; CLI that PRINTS the command + cost note by default and only executes with `--confirm`.

- [ ] **Step 1: Write the failing test** (`tests/test_finetune_job.py`):

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.finetune_llm_job import build_command  # noqa: E402


def test_build_command_shape() -> None:
    cmd = build_command("Qwen/Qwen2.5-3B-Instruct", "user/harp-chat-sft")
    assert cmd[:4] == ["hf", "jobs", "uv", "run"]
    assert "--flavor" in cmd and "l4x1" in cmd
    assert "--timeout" in cmd
    joined = " ".join(cmd)
    assert "Qwen/Qwen2.5-3B-Instruct" in joined and "user/harp-chat-sft" in joined


def test_no_launch_without_confirm() -> None:
    import train.finetune_llm_job as m
    calls = []
    m._run = lambda cmd: calls.append(cmd)  # stub the executor
    m.main(["--model", "Qwen/Qwen2.5-3B-Instruct", "--dataset", "user/harp-chat-sft"])
    assert calls == [], "must not launch a paid job without --confirm"


def _main() -> int:
    test_build_command_shape()
    test_no_launch_without_confirm()
    print("test_finetune_job: 2 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
```

- [ ] **Step 2: Run to verify it fails** → ModuleNotFoundError

- [ ] **Step 3: Implement** `train/finetune_llm_job.py`:

```python
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
```

- [ ] **Step 4: Run test** — `python tests/test_finetune_job.py` → `2 checks passed`

- [ ] **Step 5: Commit**

```bash
git add train/finetune_llm_job.py tests/test_finetune_job.py
git commit -m "feat(train): confirm-gated hf-jobs LoRA launcher for the chat escape hatch"
```

---

### Task 8: ADR-0021, CI gates

**Files:**
- Modify: `docs/ADR.md` (append ADR-0021)
- Modify: `.github/workflows/ci.yml` (new gates)

**Interfaces:**
- Consumes: everything above.
- Produces: docs + CI coverage.

- [ ] **Step 1: Append ADR-0021 to `docs/ADR.md`** (match the file's existing ADR format — read two neighboring ADRs first and mirror their heading/section style):

> **ADR-0021 — Edge target pivots to Apple silicon; eval-first model adoption.**
> Context: ADR-0007/0009 targeted mmBERT-small on Hexagon via QAIRT; the compile spike needs an x86-64 host + AI Hub token and never ran. Decision: primary edge = Apple silicon (MLX); NVIDIA remains the cloud/training side (`hf jobs`). `edge/` retained as legacy, unmaintained. Models are adopted from the HF Hub when they beat measured pass bars on our own data (`evals/`), trained otherwise (`train/`). The AUTO gate's complexity axis is now the trained n-gram head (`router/ngram_head.py`), calibrated on its own score axis; the MLX linear-probe head is the upgrade path. Supersedes ADR-0007, ADR-0009; ADR-0020's swap-only claim is now demonstrated.

- [ ] **Step 2: CI** — in `.github/workflows/ci.yml`, add to the existing test job(s) (ubuntu, stdlib — mirror how existing gates are listed):

```yaml
      - name: gate 17 — n-gram score head contract
        run: python tests/test_ngram_head.py
      - name: gate 18 — hf scout ranking
        run: python tests/test_hf_scout.py
      - name: gate 19 — score-head eval harness
        run: python tests/test_eval_score_head.py
      - name: gate 20 — local llm guards + prompt set
        run: python tests/test_local_llm.py
      - name: gate 21 — mlx trainer dense SGD
        run: python tests/test_train_score_head.py
      - name: gate 22 — finetune job launcher confirm-gate
        run: python tests/test_finetune_job.py
```

All six are stdlib-safe (MLX paths are skip-guarded), so they run on ubuntu. Also add the new gates to the `demo-integration` `needs:` list — this closes the pre-existing gap where gates 10-16 didn't gate it; extend `needs:` to cover 10-22 while there.

- [ ] **Step 3: Run everything locally**

```bash
for t in tests/test_ngram_head.py tests/test_hf_scout.py tests/test_eval_score_head.py \
         tests/test_local_llm.py tests/test_train_score_head.py tests/test_finetune_job.py \
         tests/test_route_endpoint.py; do python "$t" || exit 1; done
python router/router_policy.py
```

Expected: every file reports all checks passed; router self-test prints `accuracy 5/5`.

- [ ] **Step 4: Commit**

```bash
git add docs/ADR.md .github/workflows/ci.yml
git commit -m "docs+ci: ADR-0021 apple-silicon pivot; gate new model pipeline, fix demo-integration needs"
```
