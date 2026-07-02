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
