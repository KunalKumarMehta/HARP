# HARP — User & Operations Manual

How to run HARP on a laptop (mocks), on the Qualcomm Device Cloud (real NPU),
across devices, and against a cloud NIM. Plus troubleshooting the QDC pain points.

---

## 1. Prerequisites

- **Local dev:** Python 3.11+ (3.14 fine). Optional: `pip install websockets
  jsonschema numpy` for extra CI rigor — not required (codec + fabric fall back to
  stdlib).
- **On-device:** a Qualcomm Snapdragon X Elite (QDC interactive session or a
  Copilot+ PC), Windows on ARM64. QAIRT 2.45 (provides `genie-t2t-run`).
- **Cloud (optional):** `HARP_NIM_API_KEY` for a live Nemotron NIM endpoint.

---

## 2. Local quickstart (zero setup, runs on mocks)

```bash
python -m demo.run_demo                 # end-to-end: plan → wire → executor → routed
python -m demo.run_demo --offline       # network drop → everything fails closed to edge
python -m demo.run_demo --distributed   # edge tier runs on a separate fabric node
python -m demo.run_demo --genie         # use the Genie swarm edge path (stub off-device)
python -m demo.run_demo --live "Summarize this call and decide next action"  # needs NIM key
```

Run the core contract gates locally (CI runs the full 16 — see
`.github/workflows/ci.yml`, or `make check` for this subset):
```bash
for g in shared.conformance fabric.sync_queue tests.e2e_smoke tests.ws_roundtrip \
         shared.plan_codec tests.executor_smoke edge.genie_backend fabric.remote_backend; do
  python -m $g || exit 1
done
python -c "import asyncio; from shared.harp_contract import _smoke; asyncio.run(_smoke())"
```

To serve HARP as an OpenAI-compatible local model (the seam the Hermes integration
points at):
```bash
python -m serve.openai_endpoint            # :8765 — see integrations/hermes/README.md
```

---

## 3. Qualcomm Device Cloud (the painful part — one command)

A fresh QDC X Elite session is **barebone**: no Python, no deps, no `genie-t2t-run`.
Get the repo onto the box (scp or `git clone`), then from the repo root **in the
SSH/cmd session**:

```bat
edge\bootstrap_qdc.cmd
```

This is idempotent (safe to re-run) and detect-first. It:
1. Installs Python 3.12 ARM64 (if missing) and creates `.venv` with the deps.
2. Searches the machine for `genie-t2t-run.exe` (QDC images usually stage QAIRT),
   wires it onto PATH, and pins `HARP_GENIE_BIN`.
3. Runs the Risk-A gate over `build\qwen3-4b-w4a16` and writes
   `evidence_pack_genie.md` / `.json`.

Flags:
```bat
edge\bootstrap_qdc.cmd -SkipRun                      :: provision only
edge\bootstrap_qdc.cmd -WithOnnx                     :: also install onnxruntime-genai/-qnn (self-compile path)
edge\bootstrap_qdc.cmd -QairtZip C:\path\qairt-2.45.zip   :: if no QAIRT is staged
```

Connecting to QDC: SSH-tunnel via `ssh.qdc.qualcomm.com` using the host/credentials
from your QDC reservation, optionally RDP-forward `:3389` to reach the desktop.

---

## 4. On-device Risk-A (after bootstrap)

```bat
python run_test.py                 :: integrated power + latency gate over the bundle
python run_test.py --target csv    :: CSV power fallback (Free HWiNFO, 12h cap)
python run_test.py --target android:: ADB sysfs power (phone)
```
Output `evidence_pack_genie.md` captures the key on-device efficiency metrics
(prefill / TTFT / decode tok/s / energy-per-token, NPU-engagement assertion). The
gate **PASSES only on real silicon** with the NPU engaged (≥15 tok/s); off-device
it fails by design.

---

## 5. Adding more precompiled models (the swarm)

Drop any AI-Hub precompiled Genie bundle (a dir containing `genie_config.json` +
`*.bin`) into `build/`. `genie_swarm()` auto-discovers it — no code change:
```bash
build/
  qwen3-4b-w4a16/        # text  (ships)
  whisper-base/          # audio (drop in → ASR step runs on-device)
  qwen2.5-vl-3b/         # vision (drop in → vision step runs on-device)
```
Modality + canonical `model_id` are inferred from each bundle's `metadata.json`. To
override explicitly, add `build/swarm.json`:
```json
{"models": [
  {"model_id": "qwen3-4b",   "dir": "qwen3-4b-w4a16", "modality": "text"},
  {"model_id": "whisper-base","dir": "whisper-base",   "modality": "audio"}
]}
```
> The `model_id` must match what the cloud planner emits, or that step escalates to
> cloud by capability negotiation (graceful, not a crash).

Self-compiling a model **not** in the catalog instead? Use `edge/compile_qwen3.py`
(LLM) or `edge/compile_spike.py` (encoder/ASR), then `edge/bench.py` (QNNBackend).

---

## 6. Multi-device (phone + laptop)

**On the phone** (or any second node), serve a local backend:
```python
import asyncio
from fabric.remote_backend import serve_backend
from edge.genie_backend import genie_swarm
asyncio.run(serve_backend(genie_swarm(), host="0.0.0.0", port=8770))  # 0.0.0.0, not localhost
```
**On the laptop**, point the router at it:
```python
from fabric.remote_backend import RemoteBackend
edge = RemoteBackend("ws://<phone-ip>:8770")     # drops into PolicyRouter like any backend
```
A plan step assigned to the edge tier now executes on the phone's NPU and streams
back. Pull the network → the router's offline guard fails the step closed.
`python -m demo.run_demo --distributed` demonstrates this over a loopback node.

---

## 7. Cloud (NIM / Nemotron)

```bash
export HARP_NIM_API_KEY=<key>
python -m cloud.emit_first_plan "Analyze the call recording and screen scan, then decide"
python -m demo.run_demo --live "<your task>"     # plans live, routes across tiers
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `genie-t2t-run not found` / gate FAILS off-device | QAIRT not on PATH | run `edge\bootstrap_qdc.cmd`; or `set HARP_GENIE_BIN=C:\...\genie-t2t-run.exe` |
| bootstrap can't find QAIRT | not staged on the image | `edge\bootstrap_qdc.cmd -QairtZip <zip>`, or install QAIRT 2.45 via Qualcomm Package Manager then re-run |
| `python` missing after install | PATH not refreshed | open a new shell and re-run the bootstrap (idempotent) |
| `NPU NOT ENGAGED: <15 tok/s` | silent CPU fallback / arch mismatch | ensure `QnnHtp.dll` on PATH and the bundle's HTP arch (v73) matches the SoC |
| garbled/empty tokens on real device | QAIRT build frames stdout differently | adjust `_OUTPUT_BEGIN`/`_OUTPUT_END` in `edge/genie_backend.py` (the one build-specific knob) |
| vision/audio step went to cloud | no matching on-device bundle | drop the specialist bundle into `build/` (see §5) |
| `PlanWireError` | malformed/cyclic plan | inspect the planner output; the codec rejects it by design |
| WS `connection closed before completion` | peer/node died mid-inference | expected: the step is quarantined, not silently completed; check the node |

---

## 9. Repository map

See [SDD.md](SDD.md) §2 for the component table and the README for the architecture
diagram. Design rationale lives in [ADR.md](ADR.md); data shapes in
[DATA_SCHEMA.md](DATA_SCHEMA.md); scope in [PRD.md](PRD.md).
