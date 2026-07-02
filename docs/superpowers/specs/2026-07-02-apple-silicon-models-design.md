# Apple Silicon Models: HF Scout + Eval-First Train Pipeline

Date: 2026-07-02
Status: approved

## Goal

Replace the two model placeholders with real models, targeting Apple silicon
for edge inference and NVIDIA for cloud training:

1. **Routing score head** — replace `mock_score_fn` in
   `router/router_policy.py` with a real model satisfying the
   `(str) -> float` contract (score in `[0, 0.99]`, u(x) = P(escalate)).
2. **Local chat/summarization** — a real LOCAL-tier answerer for `mac_demo`,
   running via MLX on Apple silicon.

The Qualcomm/Hexagon path (ADR-0007, ADR-0009) is deprioritized: primary edge
target pivots to Apple silicon. `edge/` stays in place, marked legacy.

Guiding rule: **eval-first**. A fitting model may already exist on the
Hugging Face Hub; adopt when a candidate beats the target metric on our own
data, train only when nobody passes.

## Architecture

```
scripts/hf_scout.py       → ranked candidate list per task (JSON)
eval/eval_score_head.py   → AUC/accuracy on routing_dataset/test.jsonl
eval/eval_local_llm.py    → latency + quality on fixed prompt set (MLX)
train/train_score_head.py → MLX fine-tune of small encoder (local Mac)
train/finetune_llm_job.py → LoRA via `hf jobs` on NVIDIA GPU (if eval fails)
```

Wire points (shape unchanged):

- Score head: `RoutingPolicy(score_fn=...)` at `router/router_policy.py:230`.
  Calibrator + conformal gate are backbone-agnostic; only the score axis
  changes. Conformal delta recalibrated on the edge-wrong set against the new
  score function.
- Chat model: `mac_demo` LOCAL-tier answerer.

## Components

### Scout — `scripts/hf_scout.py`

- Input: `--task routing|chat`, param budget, license allowlist,
  `--require-mlx` flag.
- Source: HF Hub API (`hf models list --format json`; `huggingface_hub` when
  installed).
- Ranking: MLX build exists (`mlx-community/*` mirror), size within budget,
  downloads, recency.
- For `routing`: also searches existing prompt-complexity / router
  classifiers (RouteLLM-style) — adoption may skip training entirely.
- Output: `scout_report_<task>.json` shortlist. Reusable, not a one-off.

### Eval harnesses

- **Score head** (`eval/eval_score_head.py`): run candidate over
  `routing_dataset/test.jsonl` (regenerated deterministically, seed 13, if
  absent). Metrics: AUC, accuracy, per-query latency.
  **Pass bar: beat mock baseline AUC and < 10 ms per query on M-series.**
- **Chat/summarization** (`eval/eval_local_llm.py`): fixed ~20-prompt set
  through the MLX runtime. Metrics: tokens/sec, TTFT, rubric quality check.
  **Pass bar: TTFT < 2 s, ≥ 20 tokens/sec on this Mac, and ≥ 16/20 prompts
  judged acceptable by the rubric.**
- Decision rule: best candidate passes → adopt + wire. Nobody passes →
  training path triggers.

### Training paths

- **Score head** (`train/train_score_head.py`): MLX fine-tune of a small
  encoder (backbone chosen by scout) on the synthetic routing corpus
  (`data/synth_routing_data.py`, 4000 records, uses per-record `weight`
  field). Minutes on-Mac. Exports weights plus a `TrainedScoreHead` loader
  that keeps the `(str) -> float` contract.
- **Stdlib floor**: a zero-dependency hashed n-gram logistic-regression head
  remains the fallback when MLX is not installed — trained on the same
  corpus, weights serialized to JSON, loaded by a stdlib-only class.
- **Chat** (`train/finetune_llm_job.py`): LoRA via `hf jobs uv run` on an
  NVIDIA flavor (t4/l4), explicit cost ceiling, **always confirm with the
  user before launching any paid job**. Then fuse → `mlx_lm.convert` →
  re-run local eval.

## Dependency policy

- Repo core stays stdlib-only (`dependencies = []`).
- `mlx`, `mlx-lm`, `huggingface_hub` live in an optional extra
  `harp[apple]`. Scripts fail fast with a clear install hint when missing.
- Model weights are not committed. Downloaded via `hf download`; paths
  recorded in a small `models.json` manifest.

## Error handling

- No HF network → scout falls back to the last cached `scout_report_*.json`.
- MLX missing → score head falls back to the stdlib n-gram head with an
  explicit warning; `mac_demo` falls back to its current answerer.
- Malformed/missing weights → warn + fall back, never crash the router.
- Paid `hf jobs` launches are always user-confirmed, never automatic.

## Testing

- Scout: deterministic ranking from fixture API responses (no network in CI).
- Score head contract test: output in `[0, 0.99]`, deterministic for a fixed
  query, beats mock baseline AUC on the held-out test split.
- `mac_demo` smoke with the MLX model behind a skip-if-not-installed guard.
- CI: new gates skip on non-Mac runners.

## Docs

- **ADR-0021**: edge target pivots Qualcomm → Apple silicon; ADR-0007 and
  ADR-0009 superseded; `edge/` retained as legacy.
- `serve/openai_endpoint.py` `/health`: `route_classifier` label updated to
  the adopted/trained head (drop "placeholder" wording);
  `tests/test_route_endpoint.py:95` assertion updated to match.

## Out of scope

- ASR and vision specialists (later phases).
- Deleting `edge/` or the Qualcomm scripts.
- Committing datasets or model weights to git.

## Risks

- Synthetic-corpus overfit: a head trained on synth data learns synth
  artifacts. Mitigated by honest labeling; the bar is beating the length
  heuristic, and the conformal gate remains the safety backstop.
- `hf jobs` cost: bounded by explicit ceiling + user confirmation.
