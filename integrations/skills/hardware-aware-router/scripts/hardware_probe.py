#!/usr/bin/env python3
"""
HARP — hardware_probe.py  ·  MIT

Standalone NPU-presence probe. Emits a JSON verdict to stdout and ALWAYS exits 0 —
a missing Qualcomm SDK / onnxruntime is a "no NPU" answer, not an error. Reused by
the Hermes hook (probe) and the OpenClaw stdout path (parse stdout).

Verdict:
  {
    "npu_present": bool,
    "signals": {"onnxruntime_qnn": bool, "genie_t2t_run": bool, "qnn_sdk_root": bool},
    "providers": [...],          # onnxruntime EPs, if importable
    "detail": "<human note>"
  }

Dependency-light: stdlib + an OPTIONAL onnxruntime import. No HARP imports, so it
runs anywhere (CI, a bare laptop, an OpenClaw shell-out).
"""
from __future__ import annotations

import json
import os
import shutil
import sys


def _onnxruntime_qnn() -> tuple[bool, list[str]]:
    """True if onnxruntime exposes the QNN execution provider (Snapdragon NPU)."""
    try:
        import onnxruntime as ort  # type: ignore
        providers = list(ort.get_available_providers())
        return ("QNNExecutionProvider" in providers), providers
    except Exception:
        return False, []


def _genie_on_path() -> bool:
    """HARP's actual NPU fast path: the QAIRT genie-t2t-run tool. HARP_GENIE_BIN
    (a path or a name) wins, mirroring edge/genie_backend._genie_path()."""
    override = os.environ.get("HARP_GENIE_BIN")
    if override:
        if os.path.isfile(override) or shutil.which(override):
            return True
    return shutil.which("genie-t2t-run") is not None


def _qnn_sdk_root() -> bool:
    """A staged Qualcomm AI Runtime SDK (QAIRT/QNN) on common env vars."""
    for var in ("QNN_SDK_ROOT", "SNPE_ROOT", "QAIRT_SDK_ROOT"):
        root = os.environ.get(var)
        if root and os.path.isdir(root):
            return True
    return False


def probe() -> dict:
    ort_qnn, providers = _onnxruntime_qnn()
    genie = _genie_on_path()
    sdk = _qnn_sdk_root()
    npu_present = bool(ort_qnn or genie)
    if npu_present:
        detail = "NPU fast path available"
    elif sdk:
        detail = "Qualcomm SDK staged but no runtime entrypoint (genie-t2t-run / QNN EP)"
    else:
        detail = "no NPU signals — route LOCAL turns to the CPU stub or escalate to cloud"
    return {
        "npu_present": npu_present,
        "signals": {"onnxruntime_qnn": ort_qnn, "genie_t2t_run": genie,
                    "qnn_sdk_root": sdk},
        "providers": providers,
        "detail": detail,
    }


def main() -> int:
    try:
        verdict = probe()
    except Exception as e:  # never crash a caller's pipeline
        verdict = {"npu_present": False, "signals": {}, "providers": [],
                   "detail": f"probe error (treated as no-NPU): {type(e).__name__}: {e}"}
    print(json.dumps(verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
