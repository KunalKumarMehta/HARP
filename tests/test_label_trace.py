"""Trace labeler: judge wiring, shadow fill-in, skip logic. Stub judge, no network."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from label_trace import label  # noqa: E402


def test_labels_local_rows_and_skips_escalated() -> None:
    rows = [
        {"query": "easy", "decision": "local", "answer": "fine answer"},
        {"query": "hard", "decision": "escalate", "answer": "cloud answer"},
        {"query": "done", "decision": "local", "answer": "", "edge_correct": True},
    ]
    counts = label(rows, judge=lambda q, a: q == "easy")
    assert counts == {"judged": 1, "shadowed": 0, "skipped": 1}
    assert rows[0]["edge_correct"] is True
    assert "edge_correct" not in rows[1]           # escalated, no shadow
    assert rows[2]["edge_correct"] is True         # pre-labeled, untouched


def test_shadow_fills_escalated_rows() -> None:
    rows = [{"query": "hard", "decision": "escalate", "answer": "cloud answer"}]
    counts = label(rows, judge=lambda q, a: a == "shadow says hi",
                   shadow_gen=lambda q: "shadow says hi")
    assert counts == {"judged": 1, "shadowed": 1, "skipped": 0}
    assert rows[0]["edge_correct"] is True and rows[0]["shadow_answer"] == "shadow says hi"


def test_labeled_output_feeds_drift() -> None:
    from router.drift import drift_report
    rows = [{"query": f"q{i}", "decision": "local", "answer": "a"} for i in range(4)]
    label(rows, judge=lambda q, a: q != "q0")      # one wrong row
    u = [0.1, 0.5, 0.6, 0.7]
    err = [0 if r["edge_correct"] else 1 for r in rows]
    rep = drift_report(u, err, delta=0.2, alpha=0.05)
    assert rep["n_wrong"] == 1 and rep["drifted"]  # the wrong row stayed local


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  OK {fn.__name__}")
    print(f"test_label_trace: {len(fns)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
