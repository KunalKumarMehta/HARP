#!/usr/bin/env bash
# HARP · edge/run_session.sh · MIT
# One-shot QDC X Elite benchmark session. Each QDC reservation = ONE command.
# Minutes are scarce (5,000 total); this script does the slow prep on YOUR host
# and spends device time only on env-check + bench.py.
#
# Usage (from the compile host, after `python compile_qwen3.py`):
#   ./run_session.sh -p <ssh_fwd_port> -u <user> -m ./build/qwen3-4b-w4a16
#
# Prereqs on device (one-time, see RUNBOOK_risk_a.md B2): oga_setup.ps1 ran;
# HWiNFO64 running with Shared Memory enabled.
set -euo pipefail

PORT=22; USER_NAME="hexagon"; MODEL_DIR="./build/qwen3-4b-w4a16"
HOST="localhost"; REMOTE="."
while getopts "p:u:m:h:" o; do case "$o" in
  p) PORT="$OPTARG";; u) USER_NAME="$OPTARG";; m) MODEL_DIR="$OPTARG";; h) HOST="$OPTARG";;
esac; done

SSH="ssh -p ${PORT} ${USER_NAME}@${HOST}"
SCP="scp -P ${PORT}"
t0=$(date +%s)
echo "==> [0] session start ($(date)) — minute meter running"

# --- [1] stage artifacts + harness (small; seconds) ---
echo "==> [1] staging model + harness"
${SSH} "mkdir -p ${REMOTE}/models"
${SCP} -r "${MODEL_DIR}" "${USER_NAME}@${HOST}:${REMOTE}/models/$(basename "${MODEL_DIR}")"
${SCP} power.py qnn_backend.py bench.py harp_contract.py mutation.py \
       "${USER_NAME}@${HOST}:${REMOTE}/"

# --- [2] env preflight: fail FAST before burning minutes on a broken runtime ---
echo "==> [2] runtime preflight"
${SSH} "python -c 'import onnxruntime_genai, onnxruntime; print(\"ARM64 runtime OK\")'" \
  || { echo 'FATAL: native ARM64 runtime missing — run oga_setup.ps1 first'; exit 1; }
${SSH} "python -c \"import ctypes,mmap; mmap.mmap(-1,0,tagname='Global\\\\HWiNFO_SENS_SM2',access=mmap.ACCESS_READ); print('HWiNFO SM2 reachable')\"" \
  || echo 'WARN: HWiNFO SM2 not reachable — energy will be null; start HWiNFO + enable Shared Memory, or use CsvFallbackSampler'

# --- [3] RUN BENCHMARK (the only step that justifies device time) ---
echo "==> [3] bench.py — benchmark execution"
${SSH} "cd ${REMOTE} && python bench.py" | tee session_stdout.log

# --- [4] pull evidence pack back ---
echo "==> [4] retrieving evidence pack"
${SCP} "${USER_NAME}@${HOST}:${REMOTE}/evidence_pack.md" ./evidence_pack.md || true
${SCP} "${USER_NAME}@${HOST}:${REMOTE}/evidence_pack.json" ./evidence_pack.json || true

dt=$(( $(date +%s) - t0 ))
echo "==> [done] ${dt}s of device time used (~$(( dt/60 ))m of 5000)."
echo "    verdict: $(grep -m1 'Verdict' evidence_pack.md 2>/dev/null || echo 'see session_stdout.log')"
echo "    >>> END YOUR QDC INTERACTIVE SESSION NOW to stop the meter <<<"
