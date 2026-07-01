#!/usr/bin/env python3
"""
HARP Mac demo — routes each task between a LOCAL model (Ollama on your Mac)
and a CLOUD model (NVIDIA Nemotron via NIM), through a transparent complexity gate.

No Snapdragon required: "on-device" here = your Mac's local model (Apple Silicon).
That is the whole point — HARP uses whatever capable hardware you actually have.

Usage:
  python harp_demo.py            # live: local (Ollama) + cloud (Nemotron via NIM)
  python harp_demo.py --offline  # simulate a dropped network -> everything fails closed to local
  python harp_demo.py --mock     # no models/keys needed; prints the routing logic only

Env (only needed for the live run):
  HARP_NIM_API_KEY   from build.nvidia.com   (required unless --offline / --mock;
                     NVIDIA_API_KEY also accepted — same key, same endpoint)
  HARP_OLLAMA_MODEL  default "nemotron-mini" (an Ollama model you've pulled)
  HARP_CLOUD_MODEL   default "nvidia/llama-3.3-nemotron-super-49b-v1.5" (confirm id on build.nvidia.com)
  HARP_ALPHA         escalation threshold in [0,1], default 0.55
"""
import os, sys, time, json, re, argparse
from pathlib import Path

TRACE_PATH = Path(__file__).resolve().parent / "harp_trace.jsonl"

LOCAL_BASE  = os.getenv("HARP_LOCAL_BASE", "http://localhost:11434/v1")   # Ollama OpenAI-compatible
# HARP_OLLAMA_MODEL (not HARP_LOCAL_MODEL): the serve/ + Hermes path reads
# HARP_LOCAL_MODEL as a Genie NPU *bundle id*, so this demo uses its own name.
LOCAL_MODEL = os.getenv("HARP_OLLAMA_MODEL", "nemotron-mini")
CLOUD_BASE  = os.getenv("HARP_CLOUD_BASE", "https://integrate.api.nvidia.com/v1")
CLOUD_MODEL = os.getenv("HARP_CLOUD_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
# HARP_NIM_API_KEY is the core-stack name; accept NVIDIA_API_KEY too (same key).
CLOUD_KEY   = os.getenv("HARP_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
# HARP_ALPHA here is a threshold on this demo's transparent complexity() heuristic
# (higher = keep more on-device). It is NOT the conformal miscoverage alpha of the
# calibrated gate in router/router_policy.py (default 0.05) — different knob, different
# meaning. This standalone demo deliberately uses the simple, inspectable version.
ALPHA       = float(os.getenv("HARP_ALPHA", "0.55"))

# A scripted session. `busy` marks a turn that arrives while the local lane is
# already working — HARP sheds it to the cloud instead of making you wait.
TURNS = [
    {"text": "Namaste — kaam shuru karein?"},
    {"text": "I need to log today's wheat field inspection.", "tool": "log",
     "note": "South plot inspected: wheat heading, minor rust on lower leaves, soil moisture ok."},
    {"text": "Summarize my last three field notes in one line.", "tool": "summarize"},
    {"text": "Quick — today's mandi rate for wheat?", "busy": True},
    {"text": "Design a 3-day irrigation and fertilizer schedule across my five plots, "
             "step by step, accounting for the rainfall forecast and each crop's growth stage."},
    {"text": "Haan, theek hai — save kar do."},
]

# ----------------------------------------------------------------------------- gate
def complexity(text: str) -> float:
    """Transparent, inspectable hardness score in [0,1]. This is a HEURISTIC gate;
    the full calibrated/conformal version lives in router/router_policy.py."""
    t = text.lower()
    n = len(text)
    score = min(n / 280.0, 0.5)                                  # longer -> harder
    steps = len(re.findall(r"\b(step|plan|schedule|design|compare|analy|optimi|"
                           r"across|each|forecast|calculat|reason)\w*", t))
    score += min(steps * 0.15, 0.45)                             # multi-step signals
    if "?" in text and n < 60:                                   # short lookup question
        score *= 0.6
    if re.search(r"\b(namaste|haan|theek|thik|save|hello|hi|ok|okay)\b", t) and n < 50:
        score = min(score, 0.15)                                 # greeting/confirm = trivial
    return max(0.0, min(score, 1.0))

def gate(turn: dict):
    c = complexity(turn["text"])
    decision = "escalate" if c >= ALPHA else "local"
    reason = "complexity_gate"
    if turn.get("busy") and decision == "local":                # lane busy -> shed to cloud
        decision, reason = "escalate", "contention_shed"
    return decision, reason, round(c, 2)

# ----------------------------------------------------------------------------- backends
def _client(base, key):
    from openai import OpenAI
    return OpenAI(base_url=base, api_key=key or "not-used")

def call_model(where: str, text: str):
    """Return (answer, latency_ms). where in {local, cloud}."""
    t0 = time.time()
    if where == "local":
        client, model = _client(LOCAL_BASE, "ollama"), LOCAL_MODEL
    else:
        if not CLOUD_KEY:
            raise RuntimeError("HARP_NIM_API_KEY not set — get one at build.nvidia.com")
        client, model = _client(CLOUD_BASE, CLOUD_KEY), CLOUD_MODEL
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are HARP, a concise field assistant. Answer in <=2 sentences."},
                  {"role": "user", "content": text}],
        temperature=0.2, max_tokens=220,
    )
    return r.choices[0].message.content.strip(), int((time.time() - t0) * 1000)

# ----------------------------------------------------------------------------- notes tool
# A real on-device store, so the assistant actually HAS the memory the turns imply.
# Note reads/writes are pinned LOCAL (privacy) — they never touch the cloud lane.
NOTES_PATH = Path(__file__).resolve().parent / "harp_notes.jsonl"
SEED_NOTES = [   # prior days' notes (in-memory seed; no file write in --mock)
    {"ts": "2026-06-29", "text": "North plot: wheat at flowering, soil slightly dry, aphids on three rows."},
    {"ts": "2026-06-30", "text": "Canal plot: irrigated 40 min, leaf colour good, no pests seen."},
]

def _load_notes():
    notes = list(SEED_NOTES)
    if NOTES_PATH.exists():
        with open(NOTES_PATH) as fh:
            notes += [json.loads(l) for l in fh if l.strip()]
    return notes

def _log_note(text):
    with open(NOTES_PATH, "a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%d"), "text": text}, ensure_ascii=False) + "\n")

def _route_call(gate_turn, text, offline, mock):
    """Gate `gate_turn` (keeps its busy/contention flag), then run `text` on the chosen tier."""
    decision, reason, score = gate(gate_turn)
    tier = "cloud" if decision == "escalate" else "on-device"
    if offline and decision == "escalate":                      # network down -> fail closed
        decision, tier, reason = "local", "on-device", "offline_failclosed"
    answer, ms = "", 0
    if not mock:
        try:
            answer, ms = call_model("local" if decision == "local" else "cloud", text)
        except Exception as e:
            answer, ms = f"[error: {e}]", 0
    return decision, tier, reason, score, answer, ms

# ----------------------------------------------------------------------------- run
def run(offline=False, mock=False, trace_path=None):
    trace = []
    print(f"\nHARP routing session   (alpha={ALPHA}  local={LOCAL_MODEL}  cloud={CLOUD_MODEL})")
    print(f"{'mode: OFFLINE (fail-closed to on-device)' if offline else 'mode: online (on-device + cloud)'}\n")
    print(f"{'#':<3}{'DECISION':<12}{'TIER':<11}{'TRIGGER':<20}{'score':<7}{'ms':<7}TURN")
    print("-" * 100)
    for i, turn in enumerate(TURNS, 1):
        tool = turn.get("tool")
        if tool == "log":                                       # privacy-pinned local tool
            decision, tier, reason, score, answer, ms = "local", "on-device", "privacy_pin", 0.0, "", 0
            if not mock:
                _log_note(turn["note"])
                answer = f"Noted — saved on-device ({len(_load_notes())} field notes)."
        elif tool == "summarize":                               # real memory: read notes, then route the summary
            ctx = "; ".join(f'{n["ts"]} {n["text"]}' for n in _load_notes()[-3:])
            prompt = f"Summarize these field notes in one line: {ctx}"
            decision, tier, reason, score, answer, ms = _route_call({"text": prompt}, prompt, offline, mock)
        else:
            decision, tier, reason, score, answer, ms = _route_call(turn, turn["text"], offline, mock)
        print(f"{i:<3}{decision.upper():<12}{tier:<11}{reason:<20}{score:<7}{ms:<7}{turn['text'][:46]}")
        trace.append({"turn": i, "decision": decision, "tier": tier, "reason": reason,
                      "complexity": score, "latency_ms": ms, "query": turn["text"], "answer": answer})
    out = Path(trace_path) if trace_path else TRACE_PATH
    with open(out, "w") as f:
        for r in trace:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_local = sum(1 for r in trace if r["decision"] == "local")
    print("-" * 100)
    print(f"{n_local}/{len(trace)} turns stayed on-device.  Trace written to {out.name}\n")
    return trace

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    run(offline=a.offline, mock=a.mock)

