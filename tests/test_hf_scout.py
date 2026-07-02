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
    {"id": "someone/huge-70b", "downloads": 800000, "likes": 5000,
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
