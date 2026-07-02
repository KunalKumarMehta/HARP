"""Drift report math + trace loading + recalibrate guard. No network, no MLX."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from router.drift import MIN_RECAL_ROWS, _main, drift_report, load_verified  # noqa: E402


def test_drift_report_math() -> None:
    # 4 wrong rows, delta keeps 1 of them local -> under-route 0.25
    u = [0.1, 0.2, 0.6, 0.9, 0.3, 0.4]
    err = [1, 1, 1, 1, 0, 0]
    rep = drift_report(u, err, delta=0.15, alpha=0.05)
    assert rep["n_wrong"] == 4 and abs(rep["under_route"] - 0.25) < 1e-9
    assert rep["drifted"]
    assert not drift_report(u, err, delta=0.05, alpha=0.05)["drifted"]


def test_no_wrong_rows_is_zero() -> None:
    rep = drift_report([0.5, 0.6], [0, 0], delta=0.7, alpha=0.05)
    assert rep["under_route"] == 0.0 and not rep["drifted"]


def test_load_verified_skips_unlabeled() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "trace.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in [
            {"query": "a", "edge_correct": True},
            {"query": "b"},                          # unverified -> skipped
            {"query": "c", "edge_correct": None},    # explicit null -> skipped
            {"query": "d", "edge_correct": False},
        ]))
        got = load_verified(p)
        assert [r["query"] for r in got] == ["a", "d"]


def test_recalibrate_refuses_tiny_trace() -> None:
    rows = [{"query": f"q{i}", "edge_correct": i % 2 == 0} for i in range(MIN_RECAL_ROWS - 1)]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "trace.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows))
        rc = _main([str(p), "--recalibrate"])
        assert rc == 1
    # committed calibration untouched
    from router.ngram_head import load_head_calibration
    assert load_head_calibration() is not None


def _main_() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  OK {fn.__name__}")
    print(f"test_drift: {len(fns)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main_())
