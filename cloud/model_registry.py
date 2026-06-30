"""
HARP — hardware-aware edge↔cloud routing
cloud/model_registry.py  ·  MIT

Manager/Worker tier abstraction. Backend logic NEVER hardcodes a model string;
it resolves a ROLE through this registry. Swapping a model is a config edit
here, not a code change anywhere else.

Identifiers below are VERIFIED against the June-2026 build.nvidia.com catalog
(see NVIDIA Nemotron NIM Specifications doc). They are config defaults, not
literals embedded in dispatch logic — override per-deployment via env.

Tier mapping (HARP Manager-Worker == ReWOO Planner/Solver == Nemotron Super/Nano):
  MANAGER  -> cloud planner / orchestrator: reasoning-heavy, long-horizon, tool DAG synthesis
  WORKER   -> cloud sub-agent / solver: high-throughput, perception, factual synthesis
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    MANAGER_REASONING = "manager_reasoning"   # planner: deep CoT, DAG synthesis
    MANAGER_PRAGMATIC = "manager_pragmatic"   # planner that fits 1xH100, strong tool-calling
    WORKER_GENERAL = "worker_general"         # high-throughput sub-agent / solver
    WORKER_MULTIMODAL = "worker_multimodal"   # unified ASR+vision+text perception
    WORKER_LIGHT = "worker_light"             # thermal/VRAM-constrained, function-calling
    RAG_EMBED = "rag_embed"                   # /v1/embeddings
    RAG_RERANK = "rag_rerank"                 # /v1/ranking


@dataclass(frozen=True)
class ModelSpec:
    """One verified catalog entry. `reasoning` => emits delta.reasoning_content
    and honors enable_thinking/reasoning_budget. `endpoint` selects the API path."""
    model_id: str
    context_window: int
    reasoning: bool
    multimodal: bool
    endpoint: str = "chat"          # chat | embeddings | ranking
    note: str = ""


# VERIFIED build.nvidia.com strings (June 2026). Override any via HARP_MODEL_<ROLE>.
_DEFAULTS: dict[Role, ModelSpec] = {
    Role.MANAGER_REASONING: ModelSpec(
        "nvidia/nemotron-3-super-120b-a12b", 1_000_000, reasoning=True, multimodal=False,
        note="120B(12B) LatentMoE, 1M ctx, NVFP4. Apex planner; enable_thinking for DAG synthesis."),
    Role.MANAGER_PRAGMATIC: ModelSpec(
        "nvidia/llama-3.3-nemotron-super-49b-v1.5", 131_072, reasoning=True, multimodal=False,
        note="49B dense, fits 1xH100, strong tool-calling. Default cloud planner."),
    Role.WORKER_GENERAL: ModelSpec(
        "nvidia/nemotron-3-nano-30b-a3b", 128_000, reasoning=False, multimodal=False,
        note="30B(3.5B active), high-throughput sub-agent / Solver Node."),
    Role.WORKER_MULTIMODAL: ModelSpec(
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", 256_000, reasoning=True, multimodal=True,
        note="Collapses ASR+vision+text into one loop. One cloud worker serves all 3 swarm modalities."),
    Role.WORKER_LIGHT: ModelSpec(
        "nvidia/llama-3.1-nemotron-nano-4b-v1.1", 131_072, reasoning=False, multimodal=False,
        note="4B, FP8/NVFP4, function-calling. Lightweight model for edge deployment."),
    Role.RAG_EMBED: ModelSpec(
        "nvidia/llama-nemotron-embed-vl-1b-v2", 0, reasoning=False, multimodal=True,
        endpoint="embeddings", note="Multimodal PDF/chart embedding -> 2048-dim vectors."),
    Role.RAG_RERANK: ModelSpec(
        "nvidia/llama-nemotron-rerank-1b-v2", 0, reasoning=False, multimodal=False,
        endpoint="ranking", note="Cross-encoder rerank for retrieval ranking."),
}


def resolve(role: Role) -> ModelSpec:
    """Role -> concrete ModelSpec. Env override wins so deployments can repoint
    without touching code: HARP_MODEL_MANAGER_REASONING=nvidia/... """
    spec = _DEFAULTS[role]
    override = os.getenv(f"HARP_MODEL_{role.name}")
    if override:
        return ModelSpec(override, spec.context_window, spec.reasoning,
                         spec.multimodal, spec.endpoint, spec.note + " [env-override]")
    return spec


# Maps the planner's tool name -> the cloud Role that should serve it when a step
# escalates.
TOOL_TO_ROLE: dict[str, Role] = {
    "asr_transcribe": Role.WORKER_MULTIMODAL,
    "vision_screen":  Role.WORKER_MULTIMODAL,
    "text_summarize": Role.WORKER_GENERAL,
    "text_parse":     Role.WORKER_GENERAL,
    "deep_reason":    Role.MANAGER_PRAGMATIC,   # escalate target; swap to MANAGER_REASONING for hard plans
}


if __name__ == "__main__":
    for role in Role:
        s = resolve(role)
        flags = "".join(["R" if s.reasoning else "-", "M" if s.multimodal else "-"])
        print(f"{role.value:20} [{flags}] {s.endpoint:10} {s.model_id:42} ctx={s.context_window}")
