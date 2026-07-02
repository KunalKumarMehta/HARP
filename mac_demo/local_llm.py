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


def stream(prompt: str, max_tokens: int = 256):
    """Yield answer text chunks as they are generated (TTFT = first yield)."""
    if not available():
        raise RuntimeError("mlx_lm not installed — pip install -e .[apple]")
    from mlx_lm import stream_generate

    model, tok = _load()
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    for r in stream_generate(model, tok, prompt=text, max_tokens=max_tokens):
        yield r.text


def generate(prompt: str, max_tokens: int = 256) -> str:
    return "".join(stream(prompt, max_tokens))
