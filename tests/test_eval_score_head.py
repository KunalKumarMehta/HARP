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
