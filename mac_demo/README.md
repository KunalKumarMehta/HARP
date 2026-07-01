# HARP — on-device + cloud live demo (canonical)

The runnable proof of the core claim on hardware you already own: **HARP routes each
task between an on-device model (your Mac's Apple Silicon) and a cloud model (NVIDIA
Nemotron), and keeps working offline.** No Snapdragon required — any capable device
(Apple Silicon, Snapdragon X, an AI PC, a flagship phone) is the on-device tier.

```
  your task ──► HARP gate ──► on-device model (Ollama, local, private, offline-capable)
                     └──────► cloud model     (Nemotron via NIM, heavy reasoning)
```

## The demos (run from inside `mac_demo/`)

| Demo | Proves | Live command |
|---|---|---|
| `harp_demo.py` | **per-turn routing** — each turn goes on-device or cloud, offline fails closed; logs a field note (pinned on-device) and summarizes it from a real store | `python harp_demo.py` |
| `bench_demo.py` | **measured numbers** — real TTFT + tok/s, on-device vs cloud | `python bench_demo.py` |
| `plan_demo.py`  | **the full spine** — cloud plans a step-DAG, Mac + cloud execute it, outputs thread | `python plan_demo.py` |
| `calibrate_real.py` | **the gate's guarantee, measured** — fits the conformal gate on real paired runs, reports under-routing + CI | `python calibrate_real.py --n 300` |

The first three take `--mock` (routing only, no models/keys) and write a `*.jsonl` trace.

## The gate's guarantee, measured on real hardware

`calibrate_real.py` runs paired arithmetic problems (programmatic ground truth — no LLM
judge) on the on-device model vs the cloud model, scores each with the gate's score
function, fits the **calibrated conformal gate** ([`router/router_policy.py`](../router/router_policy.py)),
and reports the measured under-route rate on a held-out split. On one 300-query run
(`nemotron-mini` on Apple Silicon vs `llama-3.3-nemotron-super-49b`):

| | |
|---|---|
| on-device accuracy | 0.50 |
| cloud accuracy (escalation target) | **0.97** |
| **under-routing** `Pr[kept local \| edge wrong]` | **2.5%**, 90% CI [0%, 7.5%] — below α=0.05 |
| over-routing (cost of the crude score) | 4% |
| stayed on-device | 54% |

The gate escalates exactly the queries the on-device model gets wrong. Under-routing —
the *dangerous* direction — is bounded; over-routing is the disclosed cost of a
length/keyword score (a trained encoder lowers it). Trace: `harp_calibration.jsonl`.

## Run it in one minute (no models, no keys)

```bash
python harp_demo.py --mock      # per-turn routing table (writes harp_trace.jsonl)
python bench_demo.py --mock     # bench layout (real numbers on a live run -> harp_bench.jsonl)
python plan_demo.py --mock      # canned DAG routed on-device vs cloud (writes plan_trace.jsonl)
```

`harp_demo --mock` is the same routing decision the live demo makes — trivial turns stay
on-device, a hard multi-step plan escalates, and a quick lookup arriving while the local
lane is busy is *shed* to the cloud.

## Run it live (on-device + cloud, real answers)

### 1. On-device model — Ollama on your Mac
```bash
brew install ollama          # or download from ollama.com
ollama serve                 # leave running in one terminal
ollama pull nemotron-mini    # NVIDIA Nemotron-Mini 4B. Fallback: ollama pull llama3.2:3b
```
Ollama exposes an OpenAI-compatible endpoint at `http://localhost:11434/v1` — that is
the on-device tier.

### 2. Cloud model — Nemotron via NIM
1. On **build.nvidia.com**, open a Nemotron model page and create an **API key**.
2. Confirm the exact model id on that page; set it if it differs from the default.
```bash
export HARP_NIM_API_KEY="nvapi-...your-key..."   # core-stack name; NVIDIA_API_KEY also accepted
export HARP_CLOUD_MODEL="nvidia/llama-3.3-nemotron-super-49b-v1.5"   # confirm on build.nvidia.com
```
> `HARP_CLOUD_MODEL` is a convenience for these standalone demos. The core HARP stack
> selects cloud models by role — override those via `HARP_MODEL_<ROLE>` (see
> [`cloud/model_registry.py`](../cloud/model_registry.py)).

### 3. Run
```bash
pip install openai
python harp_demo.py            # per-turn routing: on-device + cloud, real answers + latencies
python harp_demo.py --offline  # simulate no network -> every task fails closed to on-device
python bench_demo.py           # latency/throughput: on-device vs cloud (--local skips the cloud tier)
python plan_demo.py            # cloud (Nemotron) plans a DAG; Mac + cloud execute it, outputs thread
python plan_demo.py "your own hard task here"
```

Each run prints a table and writes a `*.jsonl` trace (real per-turn decisions and latencies).

## What it proves

- **Per-task routing** — trivial turns (greeting, confirm, short summary) stay
  on-device; a hard multi-step plan escalates to Nemotron; a quick lookup arriving
  while the local lane is busy is *shed* to the cloud (two distinct escalation triggers).
- **Offline / fail-closed** — `--offline` routes every task on-device, no silent failure.
- **Honest gate** — the complexity score is a transparent heuristic in `complexity()`.
  The full calibrated gate lives in [`router/router_policy.py`](../router/router_policy.py)
  (isotonic + conformal); this standalone demo talks to Ollama and NIM directly so it
  runs with nothing but `openai` installed.

## Tuning

- `HARP_ALPHA` (default 0.55) — raise to keep more work on-device, lower to escalate sooner.
- `HARP_OLLAMA_MODEL` / `HARP_CLOUD_MODEL` — swap the on-device / cloud model pair.
  (These demos use `HARP_OLLAMA_MODEL`, not `HARP_LOCAL_MODEL`: the serve/Hermes path
  reads `HARP_LOCAL_MODEL` as a Genie NPU *bundle id*, a different thing.)

> This is the canonical demo. For the full HARP spine (cloud plan → validated wire →
> executor → router across mock backends), see [`demo/`](../demo/) and `make demo`.
