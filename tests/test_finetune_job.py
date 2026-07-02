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
