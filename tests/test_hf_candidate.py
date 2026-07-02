"""HF candidate adapter: label mapping + clamping via injected stub pipeline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.hf_candidate import HFClassifierScoreFn  # noqa: E402


def _stub(scores: dict[str, float]):
    return lambda texts: [[{"label": k, "score": v} for k, v in scores.items()]] * len(texts)


def test_sums_escalate_labels() -> None:
    fn = HFClassifierScoreFn("stub/model", escalate_labels=("hard", "very_hard"),
                             pipe=_stub({"easy": 0.2, "hard": 0.5, "very_hard": 0.3}))
    assert abs(fn("q") - 0.8) < 1e-9


def test_clamps_to_contract_range() -> None:
    fn = HFClassifierScoreFn("stub/model", escalate_labels=("LABEL_1",),
                             pipe=_stub({"LABEL_1": 1.0}))
    assert fn("q") == 0.99
    lo = HFClassifierScoreFn("stub/model", escalate_labels=("missing",),
                             pipe=_stub({"LABEL_1": 1.0}))
    assert lo("q") == 0.0


def test_works_in_eval_harness() -> None:
    from evals.eval_score_head import evaluate
    fn = HFClassifierScoreFn("stub/model", pipe=_stub({"LABEL_1": 0.9, "LABEL_0": 0.1}))
    rep = evaluate({"stub/model": fn}, [{"text": "a", "label": 1}, {"text": "b", "label": 0}])
    assert "stub/model" in rep and 0.0 <= rep["stub/model"]["auc"] <= 1.0


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  OK {fn.__name__}")
    print(f"test_hf_candidate: {len(fns)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
