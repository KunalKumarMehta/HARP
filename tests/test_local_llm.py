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


def test_stream_guarded() -> None:
    if local_llm.available():
        chunks = list(local_llm.stream("Say the word ready.", max_tokens=8))
        assert chunks and all(isinstance(c, str) for c in chunks)
        assert "".join(chunks).strip()
    else:
        try:
            list(local_llm.stream("hi"))
            raise AssertionError("expected RuntimeError when mlx_lm missing")
        except RuntimeError:
            pass


def _main() -> int:
    for fn in (test_rubric, test_prompt_set_wellformed, test_generate_guarded,
               test_stream_guarded):
        fn()
        print(f"  OK {fn.__name__}")
    print("test_local_llm: 4 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
