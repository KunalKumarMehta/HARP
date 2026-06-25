"""
HARP — tests/test_skill_packaging.py  ·  MIT

agentskills.io compliance for integrations/skills/hardware-aware-router and the
Hermes hook plugin. Asserts: SKILL.md starts at byte 0 with YAML frontmatter whose
name == parent dir and description <= 1024 chars; the hook plugin.yaml declares
kind: hook; hardware_probe.py runs and emits valid JSON with npu_present, exit 0,
even with no Qualcomm SDK installed. Stdlib only (manual frontmatter parse — no
pyyaml dependency).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SKILL_DIR = os.path.join(_ROOT, "integrations", "skills", "hardware-aware-router")
_HOOK_DIR = os.path.join(_ROOT, "integrations", "hermes", "plugins", "hooks",
                        "hardware-aware-router")


def _frontmatter(path: str) -> dict:
    """Parse top-level `key: value` pairs from a leading `--- ... ---` block.
    Skips nested (indented) keys like metadata:. Good enough for compliance checks
    without a YAML dependency."""
    raw = open(path, "rb").read()
    assert raw.startswith(b"---"), "SKILL.md must start at byte 0 with frontmatter"
    text = raw.decode("utf-8")
    end = text.index("\n---", 3)
    block = text[3:end]
    out: dict = {}
    for line in block.splitlines():
        if not line.strip() or line.startswith((" ", "\t")) or ":" not in line:
            continue
        key, _, val = line.partition(":")
        out[key.strip()] = val.strip()
    return out


def test_skill_md_frontmatter() -> None:
    fm = _frontmatter(os.path.join(_SKILL_DIR, "SKILL.md"))
    assert fm["name"] == os.path.basename(_SKILL_DIR) == "hardware-aware-router"
    assert fm["name"].islower() and " " not in fm["name"] and len(fm["name"]) <= 64
    assert 0 < len(fm["description"]) <= 1024
    assert "route between on-device NPU and cloud planner" in fm["description"]
    assert fm.get("license") == "MIT"


def test_skill_has_scripts_and_references() -> None:
    assert os.path.isfile(os.path.join(_SKILL_DIR, "scripts", "hardware_probe.py"))
    assert os.path.isfile(
        os.path.join(_SKILL_DIR, "references", "routing_heuristics.md"))


def test_hook_plugin_yaml_kind() -> None:
    fm = {}
    for line in open(os.path.join(_HOOK_DIR, "plugin.yaml")):
        if ":" in line and not line.startswith((" ", "\t")):
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()
    assert fm["kind"] == "hook"
    assert fm["name"] == "hardware-aware-router"


def test_hardware_probe_runs_and_emits_json() -> None:
    probe = os.path.join(_SKILL_DIR, "scripts", "hardware_probe.py")
    proc = subprocess.run([sys.executable, probe], capture_output=True, text=True)
    assert proc.returncode == 0, f"probe must exit 0, got {proc.returncode}: {proc.stderr}"
    verdict = json.loads(proc.stdout)              # must be valid JSON
    assert "npu_present" in verdict
    assert isinstance(verdict["npu_present"], bool)
    assert "signals" in verdict


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  OK {fn.__name__}")
    print(f"test_skill_packaging: {len(fns)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
