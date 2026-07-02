"""Scout the HF Hub for candidate models per HARP task. Stdlib-only.

Usage:
  python scripts/hf_scout.py --task chat --require-mlx --max-params-b 8
  python scripts/hf_scout.py --task routing --max-params-b 1

Writes scout_report_<task>.json (ranked shortlist). Offline / API failure:
falls back to the last cached report and says so.
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://huggingface.co/api/models"
ALLOWED_LICENSES = {"apache-2.0", "mit", "bsd-3-clause", "llama3.2", "gemma", "qwen"}
QUERIES = {
    # (search terms, extra query params) per task; several angles per task.
    "routing": [
        ("prompt complexity classifier", {}),
        ("prompt router", {}),
        ("query difficulty", {}),
        ("routellm", {}),
    ],
    "chat": [
        ("instruct 4bit", {"author": "mlx-community", "pipeline_tag": "text-generation"}),
        ("instruct", {"pipeline_tag": "text-generation", "library": "mlx"}),
    ],
}


def _fetch(search: str, extra: dict, limit: int) -> list[dict]:
    params = {"search": search, "limit": str(limit), "sort": "downloads", "direction": "-1",
              "expand[]": "safetensors", **extra}
    url = f"{API}?{urllib.parse.urlencode(params, doseq=True)}"
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read())


def _license(m: dict) -> str | None:
    for t in m.get("tags", []):
        if t.startswith("license:"):
            return t.removeprefix("license:")
    return None


def _params_b(m: dict) -> float | None:
    total = (m.get("safetensors") or {}).get("total")
    return total / 1e9 if total else None


def _is_mlx(m: dict) -> bool:
    return m["id"].startswith("mlx-community/") or "mlx" in m.get("tags", [])


def _age_days(m: dict) -> float:
    ts = m.get("lastModified")
    if not ts:
        return 3650.0
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return max(0.0, (datetime.now(timezone.utc) - dt).days)


def rank(models: list[dict], *, max_params_b: float | None, licenses: set[str],
         require_mlx: bool) -> list[dict]:
    seen: set[str] = set()
    out = []
    for m in models:
        if m["id"] in seen:
            continue
        seen.add(m["id"])
        lic = _license(m)
        if lic is not None and lic not in licenses:
            continue
        pb = _params_b(m)
        if max_params_b is not None and pb is not None and pb > max_params_b:
            continue
        if require_mlx and not _is_mlx(m):
            continue
        score = (math.log10(1 + m.get("downloads", 0))
                 + 0.5 * math.log10(1 + m.get("likes", 0))
                 + (2.0 if _is_mlx(m) else 0.0)
                 - _age_days(m) / 365.0)
        out.append({**m, "scout_score": round(score, 4)})
    return sorted(out, key=lambda m: (-m["scout_score"], m["id"]))


def scout(task: str, limit: int, max_params_b: float | None, require_mlx: bool,
          out_path: Path) -> list[dict]:
    raw: list[dict] = []
    try:
        for search, extra in QUERIES[task]:
            raw.extend(_fetch(search, extra, limit))
    except OSError as e:
        if out_path.exists():
            print(f"HF API unreachable ({e}); using cached {out_path}")
            return json.loads(out_path.read_text())["candidates"]
        raise SystemExit(f"HF API unreachable and no cached report: {e}")
    ranked = rank(raw, max_params_b=max_params_b, licenses=ALLOWED_LICENSES,
                  require_mlx=require_mlx)[:25]
    out_path.write_text(json.dumps({"task": task, "candidates": ranked}, indent=2))
    print(f"wrote {out_path} ({len(ranked)} candidates)")
    for m in ranked[:10]:
        print(f"  {m['scout_score']:7.3f}  {m['id']}")
    return ranked


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=sorted(QUERIES), required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-params-b", type=float, default=None)
    ap.add_argument("--require-mlx", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or Path(f"scout_report_{a.task}.json")
    scout(a.task, a.limit, a.max_params_b, a.require_mlx, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
