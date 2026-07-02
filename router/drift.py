"""Close the loop from synthetic calibration to real traffic.

Reads a HARP trace (jsonl; rows need "query" and a verified "edge_correct"
bool — rows without it are skipped), rescores queries with the trained head,
and reports the empirical under-route rate Pr[kept local | edge wrong]
against the alpha bound. With --recalibrate, rebuilds
score_head_calibration.json on the live axis and prints old-vs-new delta.

Usage:
  python -m router.drift mac_demo/harp_trace.jsonl [--alpha 0.05] [--recalibrate]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from router.ngram_head import CALIBRATION_PATH, TrainedScoreHead, load_head_calibration
from router.router_policy import ConformalGate

MIN_RECAL_ROWS = 30  # conformal quantile is noise below this


def load_verified(path: Path) -> list[dict]:
    """Trace rows that carry a verified edge outcome."""
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return [r for r in rows if isinstance(r.get("edge_correct"), bool)]


def drift_report(u: list[float], err: list[int], delta: float, alpha: float) -> dict:
    """Empirical under-route rate on the wrong set vs the alpha bound."""
    wrong = [ui for ui, ei in zip(u, err) if ei == 1]
    under = sum(ui <= delta for ui in wrong) / len(wrong) if wrong else 0.0
    return {"n": len(u), "n_wrong": len(wrong), "under_route": under,
            "alpha": alpha, "drifted": under > alpha}


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--recalibrate", action="store_true")
    a = ap.parse_args(argv)

    rows = load_verified(a.trace)
    if not rows:
        print(f"{a.trace}: no rows with a verified edge_correct field; nothing to check")
        return 1
    head = TrainedScoreHead.load()
    u = [head(r["query"]) for r in rows]
    err = [0 if r["edge_correct"] else 1 for r in rows]

    cal = load_head_calibration()
    if cal is None:
        print("no committed head calibration found", file=sys.stderr)
        return 1
    old = ConformalGate(alpha=a.alpha).fit(*cal)
    rep = drift_report(u, err, old.delta, a.alpha)
    print(f"verified rows: {rep['n']}  edge-wrong: {rep['n_wrong']}  "
          f"under-route: {rep['under_route']:.3f}  bound: {rep['alpha']}")
    if rep["drifted"]:
        print("DRIFT: empirical under-route exceeds alpha — recalibrate (--recalibrate)")

    if a.recalibrate:
        if len(rows) < MIN_RECAL_ROWS:
            print(f"refusing to recalibrate on {len(rows)} rows (< {MIN_RECAL_ROWS})")
            return 1
        new = ConformalGate(alpha=a.alpha).fit(u, err)
        CALIBRATION_PATH.write_text(json.dumps({"u": u, "err": err}))
        print(f"recalibrated on {len(rows)} live rows: "
              f"delta {old.delta:.4f} -> {new.delta:.4f}  wrote {CALIBRATION_PATH.name}")
    return 1 if rep["drifted"] and not a.recalibrate else 0


if __name__ == "__main__":
    raise SystemExit(_main())
