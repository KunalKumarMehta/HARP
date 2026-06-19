#!/usr/bin/env python3
"""
HARP · run_test.py · one-command Risk-A run on the QDC X Elite · MIT

Replaces the earlier two-process model (power.py + bench.py launched separately).
That sampled the NPU rail in a parallel process — a blind window that could miss
or precede the actual decode, giving misleading energy/token. This drives the
INTEGRATED gate (edge.bench_genie -> run_gate), which samples power *inside* the
inference workload, so energy = ∫P dt / N over exactly the tokens generated.

It benchmarks the PRECOMPILED Genie bundle (build/qwen3-4b-w4a16/) via
GenieBackend — the fast path that skips self-compilation. Path-robust: run it
from anywhere.

    python run_test.py                  # WoS HWiNFO NPU rail
    python run_test.py --target csv     # CSV power fallback (Free HWiNFO 12h cap)
    python run_test.py --target android # ADB sysfs (phone)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
target = sys.argv[sys.argv.index("--target") + 1] if "--target" in sys.argv else "wos"

print(f"--> HARP Risk-A · integrated power+latency gate · target={target}")
rc = subprocess.run(
    [sys.executable, "-m", "edge.bench_genie", "--target", target],
    cwd=str(ROOT),
).returncode
print(f"--> Testing cycle complete (exit {rc}).")
sys.exit(rc)
