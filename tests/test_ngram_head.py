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
