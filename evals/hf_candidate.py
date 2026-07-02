"""Wrap a scouted HF text-classification model as a routing score fn.

Lets scout candidates (e.g. routellm/bert*, nvidia complexity classifiers)
run through evals/eval_score_head.py against the same pass bars as the
n-gram head. Needs `pip install transformers torch` — eval-time only, never
a core dependency. u(x) = summed probability of the escalate-side labels.
"""
from __future__ import annotations

from typing import Callable


class HFClassifierScoreFn:
    """Score-fn contract: (str) -> float in [0, 0.99]. Lazy model load."""

    __name__ = "hf_classifier"

    def __init__(self, model_id: str, escalate_labels: tuple[str, ...] = ("LABEL_1",),
                 pipe: Callable | None = None) -> None:
        self.model_id = model_id
        self.labels = set(escalate_labels)
        self._pipe = pipe  # injectable for tests

    def _pipeline(self) -> Callable:
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline("text-classification", model=self.model_id, top_k=None)
        return self._pipe

    def __call__(self, query: str) -> float:
        out = self._pipeline()([query])[0]  # [{"label": ..., "score": ...}, ...]
        p = sum(d["score"] for d in out if d["label"] in self.labels)
        return min(max(float(p), 0.0), 0.99)
