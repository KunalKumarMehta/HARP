#!/usr/bin/env python3
"""
HARP Mac demo — the full spine on hardware you own. A CLOUD planner (NVIDIA
Nemotron) decomposes one hard task into a step DAG; HARP then ROUTES each step —
light/perception steps run on your Mac's LOCAL model (Ollama), the deep-reason
step escalates to Nemotron — and threads each step's output into the next.

This is the README architecture diagram (plan -> wire -> routed execution ->
threaded dataflow) with REAL models instead of mocks, and no Snapdragon: your
Mac's local model is the capable edge device.

Usage (from inside mac_demo/, or as `python -m mac_demo.plan_demo` from the repo root):
  python plan_demo.py                 # live: Nemotron plans, Mac + cloud execute
  python plan_demo.py "your task..."  # plan+route your own task
  python plan_demo.py --mock          # canned DAG, routing only, no models/keys

Env: same as harp_demo.py (HARP_NIM_API_KEY [NVIDIA_API_KEY accepted], HARP_OLLAMA_MODEL,
     HARP_CLOUD_MODEL, HARP_ALPHA). The routing gate is reused from harp_demo.py.
"""
import os, sys, re, json, time, argparse
from pathlib import Path
try:  # ponytail: reuse the one gate, don't fork it — works run as module OR in-folder
    from mac_demo.harp_demo import complexity, ALPHA
except ImportError:
    from harp_demo import complexity, ALPHA

TRACE_PATH  = Path(__file__).resolve().parent / "plan_trace.jsonl"
LOCAL_BASE  = os.getenv("HARP_LOCAL_BASE", "http://localhost:11434/v1")
LOCAL_MODEL = os.getenv("HARP_OLLAMA_MODEL", "nemotron-mini")
CLOUD_BASE  = os.getenv("HARP_CLOUD_BASE", "https://integrate.api.nvidia.com/v1")
CLOUD_MODEL = os.getenv("HARP_CLOUD_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
# HARP_NIM_API_KEY is the core-stack name; accept NVIDIA_API_KEY too (same key).
CLOUD_KEY   = os.getenv("HARP_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")

DEFAULT_TASK = ("Review my three wheat field notes, summarize the risks, then give a "
                "prioritized action plan for the week accounting for the rainfall forecast.")

# Canned DAG for --mock (and the shape the planner is asked to emit): each node is a
# sub-task with deps; 'hint' nudges the tier; '<id>_output' threads upstream results.
_MOCK_DAG = {"nodes": [
    {"id": "n1", "task": "Summarize field note 1 (north plot): standing water after rain.", "deps": [], "hint": "edge"},
    {"id": "n2", "task": "Summarize field note 2 (south plot): early yellowing on wheat tips.", "deps": [], "hint": "edge"},
    {"id": "n3", "task": "List the top risks from n1_output and n2_output.", "deps": ["n1", "n2"], "hint": "edge"},
    {"id": "n4", "task": "From n3_output, design a step-by-step prioritized 7-day action "
                          "plan across all plots, reasoning about the rainfall forecast.",
     "deps": ["n3"], "hint": "cloud"},
]}


def _client(base, key):
    from openai import OpenAI
    return OpenAI(base_url=base, api_key=key or "not-used")


def _call(where, text):
    if where == "cloud":
        if not CLOUD_KEY:
            raise RuntimeError("no cloud key — set NVIDIA_API_KEY (build.nvidia.com)")
        client, model = _client(CLOUD_BASE, CLOUD_KEY), CLOUD_MODEL
    else:
        client, model = _client(LOCAL_BASE, "ollama"), LOCAL_MODEL
    t0 = time.time()
    r = client.chat.completions.create(
        model=model, temperature=0.2, max_tokens=320,
        messages=[{"role": "system", "content": "You are HARP, a concise field assistant. Be brief and concrete."},
                  {"role": "user", "content": text}])
    return r.choices[0].message.content.strip(), int((time.time() - t0) * 1000)


def plan(task):
    """Ask the cloud planner (Nemotron) for a JSON step-DAG. Falls back to the
    canned DAG if there's no key or the output won't parse."""
    if not CLOUD_KEY:
        return _MOCK_DAG
    schema = ('{"nodes":[{"id":str,"task":str,"deps":[str],"hint":"edge"|"cloud"}]}. '
              'Mark only the final deep-reasoning/planning node hint="cloud"; the rest "edge". '
              'Dependent nodes reference upstream results as <id>_output inside task.')
    sys_msg = ("You are HARP's cloud planner. Decompose the user task into a minimal "
               "execution DAG. Output STRICT JSON only, no prose, no fences, matching: " + schema)
    try:
        client = _client(CLOUD_BASE, CLOUD_KEY)
        r = client.chat.completions.create(
            model=CLOUD_MODEL, temperature=0.1, max_tokens=1024,
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": task}])
        return _extract_json(r.choices[0].message.content)
    except Exception as e:
        print(f"  [planner fell back to canned DAG: {e}]")
        return _MOCK_DAG


def _extract_json(text):
    t = re.sub(r"```(?:json)?", "", text).strip()
    s = t.find("{")
    depth = 0
    for i in range(s, len(t)):
        depth += (t[i] == "{") - (t[i] == "}")
        if depth == 0:
            return json.loads(t[s:i + 1])
    raise ValueError("no balanced JSON in planner output")


def _topo(nodes):
    by_id = {n["id"]: n for n in nodes}
    done, order = set(), []
    while len(order) < len(nodes):
        ready = [n for n in nodes if n["id"] not in done and all(d in done for d in n.get("deps", []))]
        if not ready:
            raise ValueError("cyclic or dangling plan")
        for n in ready:
            order.append(n); done.add(n["id"])
    return order, by_id


def run(task, mock=False):
    dag = _MOCK_DAG if mock else plan(task)
    order, _ = _topo(dag["nodes"])
    print(f"\nHARP plan+route   alpha={ALPHA}  local={LOCAL_MODEL}  cloud={CLOUD_MODEL}")
    print(f"{'mode: MOCK (routing only, no calls)' if mock else 'mode: live'}   task: {task[:60]}\n")
    print(f"{'STEP':<5}{'TIER':<11}{'TRIGGER':<15}{'score':<7}TASK")
    print("-" * 90)
    outputs, trace = {}, []
    for n in order:
        # thread upstream outputs into this step's instruction (dataflow)
        args = n["task"]
        for dep in n.get("deps", []):
            args = args.replace(f"{dep}_output", outputs.get(dep, ""))
        c = complexity(args)
        escalate = n.get("hint") == "cloud" or c >= ALPHA
        reason = "planner_pin" if n.get("hint") == "cloud" else ("complexity_gate" if escalate else "on_device")
        tier = "cloud" if escalate else "local"
        out, ms = ("", 0)
        if not mock:
            try:
                out, ms = _call(tier, args)
            except Exception as e:
                out, ms = f"[error: {e}]", 0
        outputs[n["id"]] = out
        name = "cloud" if tier == "cloud" else "on-device"
        print(f"{n['id']:<5}{name:<11}{reason:<15}{round(c,2):<7}{n['task'][:52]}")
        trace.append({"step": n["id"], "tier": name, "reason": reason, "score": round(c, 2),
                      "ms": ms, "task": n["task"], "output": out})
    with open(TRACE_PATH, "w") as f:
        for r in trace:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    edge_n = sum(1 for r in trace if r["tier"] == "on-device")
    print("-" * 90)
    print(f"{edge_n}/{len(trace)} steps ran on-device, {len(trace)-edge_n} escalated. "
          f"Trace -> plan_trace.jsonl")
    final = trace[-1]["output"]
    if final and not mock:
        print(f"\nfinal ({trace[-1]['tier']}):\n{final[:400]}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="*", help="task to plan+route (default: wheat-farm example)")
    ap.add_argument("--mock", action="store_true", help="canned DAG, routing only, no calls")
    a = ap.parse_args()
    run(" ".join(a.task) or DEFAULT_TASK, mock=a.mock)
