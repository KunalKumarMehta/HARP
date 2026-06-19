"""
HARP · fabric/remote_backend.py · MIT
Multi-device execution over the WebSocket fabric — the Qualcomm Multi-Device
Award made real in code (orchestration "impossible on a single device").

The trick: a remote node is just another Backend.

    laptop:  PolicyRouter(edge=RemoteBackend("ws://phone:8770"), cloud=NIM)
    phone:   serve_backend(GenieBackend(...))   # runs the real NPU inference

Because `RemoteBackend` satisfies the frozen shared.harp_contract.Backend, the
router dispatches a step to the phone exactly as it would to a local backend —
the executor, the codec, and the contract are untouched. `capabilities()` is an
RPC, so routing decisions reflect the REMOTE device's real NPU/modalities; a
plan step assigned to the edge tier executes on the phone and streams its tokens
back. Pull the network and the router's offline guard fails the step closed to
whatever local backend the laptop holds — the graceful-degradation story.

Transport: `websockets` (same lib + asyncio API as ws_node.py). One short-lived
connection per call keeps streams from interleaving and reconnect trivial.

Security: WSS + bearer token auth (ADR-010 production one-liner).
  - Configure via HARP_FABRIC_* env vars or AuthConfig object.
  - Token validated on connect; TLS context configurable.
  - Pass auth=AuthConfig(...) to both serve_backend() and RemoteBackend().
"""

from __future__ import annotations

import asyncio
import json

import websockets
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from fabric.auth import AuthConfig
from shared.harp_contract import (
    Backend, Capability, InferRequest, Metrics, Modality, Tier,
)

# ---------------------------------------------------------------- wire (de)serialization

def _cap_to_dict(c: Capability) -> dict:
    return {"backend_id": c.backend_id, "tier": c.tier.value,
            "npu_present": c.npu_present, "ram_gb": c.ram_gb,
            "max_context": c.max_context,
            "modalities": [m.value for m in c.modalities],
            "offline_capable": c.offline_capable,
            "supports_streaming": c.supports_streaming}


def _cap_from_dict(d: dict, *, backend_id: str | None = None) -> Capability:
    return Capability(
        backend_id=backend_id or d["backend_id"], tier=Tier(d["tier"]),
        npu_present=d["npu_present"], ram_gb=d["ram_gb"],
        max_context=d["max_context"],
        modalities=tuple(Modality(m) for m in d["modalities"]),
        offline_capable=d["offline_capable"], supports_streaming=d["supports_streaming"])


def _req_to_dict(r: InferRequest) -> dict:
    return {"messages": r.messages, "model_id": r.model_id,
            "modality": r.modality.value, "max_tokens": r.max_tokens, "stream": r.stream}


def _req_from_dict(d: dict) -> InferRequest:
    return InferRequest(messages=d["messages"], model_id=d["model_id"],
                        modality=Modality(d.get("modality", "text")),
                        max_tokens=int(d.get("max_tokens", 512)),
                        stream=bool(d.get("stream", True)))


def _metrics_to_dict(m: Metrics) -> dict:
    return {"backend_id": m.backend_id, "ttft_ms": m.ttft_ms,
            "tokens_per_s": m.tokens_per_s, "energy_mj_per_tok": m.energy_mj_per_tok,
            "thermal_c": m.thermal_c}


def _metrics_from_dict(d: dict) -> Metrics:
    return Metrics(backend_id=d["backend_id"], ttft_ms=d["ttft_ms"],
                   tokens_per_s=d["tokens_per_s"],
                   energy_mj_per_tok=d.get("energy_mj_per_tok"),
                   thermal_c=d.get("thermal_c"))


# ---------------------------------------------------------------- server (the remote node)

async def serve_backend(backend: Backend, host: str = "0.0.0.0", port: int = 8770,
                        ready: asyncio.Event | None = None,
                        auth: AuthConfig | None = None) -> None:
    """Expose a local Backend to peers. Bind 0.0.0.0 so the laptop can reach the
    phone on the LAN (never localhost — same rule as ws_node.serve_local).

    Security: pass `auth=AuthConfig(...)` to enable WSS + bearer token auth.
              Loads HARP_FABRIC_* env vars by default.
    """
    auth = auth or AuthConfig.from_env()
    server_ssl = auth.get_server_ssl_context()

    async def handler(ws):
        # Auth check on connect
        if not auth.validate_token(auth.extract_token_from_headers(ws.request.headers)):
            await ws.close(code=4001, reason="unauthorized")
            return
        async for raw in ws:
            try:
                msg = json.loads(raw)
                op = msg.get("op")
            except (json.JSONDecodeError, AttributeError) as e:
                await ws.send(json.dumps({"error": f"bad frame: {e}"}))
                continue
            try:
                if op == "capabilities":
                    cap = await backend.capabilities()
                    await ws.send(json.dumps({"capability": _cap_to_dict(cap)}))
                elif op == "profile":
                    m = await backend.profile(_req_from_dict(msg["req"]))
                    await ws.send(json.dumps({"metrics": _metrics_to_dict(m)}))
                elif op == "infer":
                    req = _req_from_dict(msg["req"])
                    gen = backend.infer(req)
                    try:
                        async for tok in gen:
                            await ws.send(json.dumps({"t": tok}))
                        await ws.send(json.dumps({"done": True}))
                    finally:
                        # Close the inference generator even if the client vanished
                        # mid-stream — on a real NPU backend this releases the
                        # in-flight decode session instead of leaking it.
                        aclose = getattr(gen, "aclose", None)
                        if aclose is not None:
                            await aclose()
                else:
                    await ws.send(json.dumps({"error": f"unknown op {op!r}"}))
            except websockets.ConnectionClosed:
                return                                  # client gone; end this handler quietly
            except Exception as e:                      # never kill the node on one bad request
                try:
                    await ws.send(json.dumps({"error": f"{type(e).__name__}: {e}"}))
                except websockets.ConnectionClosed:
                    return                              # can't report to a closed socket

    async with serve(handler, host, port, ssl=server_ssl) as server:
        if ready is not None:
            ready.set()
        await server.serve_forever()


# ---------------------------------------------------------------- client (a Backend proxy)

class RemoteBackend(Backend):
    """A Backend whose work happens on a peer node over the fabric. Drop it into a
    Router in place of any local backend; the router can't tell the difference.

    Security: pass `auth=AuthConfig(...)` to enable WSS + bearer token auth.
              Loads HARP_FABRIC_* env vars by default.
    """

    def __init__(self, uri: str, *, label: str | None = None,
                 open_timeout: float = 10.0, read_timeout: float = 60.0,
                 ping_interval: float = 20.0, ping_timeout: float = 20.0,
                 auth: AuthConfig | None = None):
        self.uri = uri
        self._label = label
        self._open_timeout = open_timeout
        self._read_timeout = read_timeout       # per-frame: catches an alive-but-silent peer
        self._ping_interval = ping_interval      # keepalive: catches a dead TCP link
        self._ping_timeout = ping_timeout
        self.auth = auth or AuthConfig.from_env()
        self._client_ssl = self.auth.get_client_ssl_context()
        self._cap: Capability | None = None

    def _connect(self):
        extra_headers = None
        if self.auth.token:
            extra_headers = [("Authorization", f"Bearer {self.auth.token}")]
        return connect(self.uri, open_timeout=self._open_timeout,
                       ssl=self._client_ssl, additional_headers=extra_headers,
                       ping_interval=self._ping_interval, ping_timeout=self._ping_timeout)

    async def _recv(self, ws) -> str:
        try:
            return await asyncio.wait_for(ws.recv(), timeout=self._read_timeout)
        except asyncio.TimeoutError as e:
            raise RuntimeError(f"remote peer silent for {self._read_timeout}s") from e

    async def _rpc(self, frame: dict) -> dict:
        async with self._connect() as ws:
            await ws.send(json.dumps(frame))
            return json.loads(await self._recv(ws))

    async def capabilities(self) -> Capability:
        # Cached: capabilities are static, and the router queries them per step.
        if self._cap is None:
            resp = await self._rpc({"op": "capabilities"})
            if "error" in resp:
                raise RuntimeError(f"remote capabilities failed: {resp['error']}")
            label = self._label or f"remote→{resp['capability']['backend_id']}"
            self._cap = _cap_from_dict(resp["capability"], backend_id=label)
        return self._cap

    async def infer(self, req: InferRequest):
        async with self._connect() as ws:
            await ws.send(json.dumps({"op": "infer", "req": _req_to_dict(req)}))
            done = False
            while True:
                try:
                    raw = await self._recv(ws)
                except websockets.ConnectionClosed:
                    break                            # closed mid-stream; handled below
                msg = json.loads(raw)
                if "t" in msg:
                    yield msg["t"]
                elif msg.get("done"):
                    done = True
                    break
                elif "error" in msg:
                    raise RuntimeError(f"remote infer failed: {msg['error']}")
            if not done:
                # Connection ended with no done/error frame: a truncated inference
                # MUST surface as an error so the executor quarantines the step —
                # never let a partial result masquerade as success.
                raise RuntimeError("remote infer: connection closed before completion")

    async def profile(self, req: InferRequest) -> Metrics:
        resp = await self._rpc({"op": "profile", "req": _req_to_dict(req)})
        if "error" in resp:
            raise RuntimeError(f"remote profile failed: {resp['error']}")
        return _metrics_from_dict(resp["metrics"])


# ---------------------------------------------------------------- self-test (CI Gate 9)

def _free_port() -> int:
    """An ephemeral free port. Avoids the fixed-port TIME_WAIT bind failures that
    made this gate flaky under rapid re-runs."""
    import socket
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


async def _selftest() -> None:
    from shared.harp_contract import (
        InferRequest, mock_cloud, mock_edge, PlanGraph, PlanStep, RouteDecision)
    from shared.conformance import assert_conforms
    from router.router_policy import PolicyRouter, RoutingPolicy
    from fabric.executor import PlanExecutor

    host, port = "127.0.0.1", _free_port()
    ready = asyncio.Event()
    # The "phone": serve a real (mock) edge backend over the fabric.
    server = asyncio.create_task(serve_backend(mock_edge(), host, port, ready))
    try:
        await asyncio.wait_for(ready.wait(), timeout=5.0)
        remote = RemoteBackend(f"ws://{host}:{port}")

        # 1) the proxy satisfies the frozen contract, over a real socket
        await assert_conforms(remote)

        # 2) capabilities RPC reflects the REMOTE device (drives routing correctly)
        cap = await remote.capabilities()
        assert cap.npu_present and Modality.AUDIO in cap.modalities
        assert cap.backend_id.startswith("remote→")

        # 3) the executor runs a plan with the edge tier executing ON the remote node
        cal_u = [i / 200.0 for i in range(200)]
        cal_err = [1 if (i % 100) / 100.0 < cal_u[i] else 0 for i in range(200)]
        pol = RoutingPolicy().calibrate(cal_u, cal_err)
        plan = PlanGraph("p-remote", [
            PlanStep("r1", Modality.AUDIO, RouteDecision.AUTO, "whisper-base",
                     "transcribe on the remote node"),
            PlanStep("r2", Modality.TEXT, RouteDecision.ESCALATE, "nemotron",
                     "escalate r1_output", depends_on=["r1"]),
        ])
        res = await PlanExecutor(
            PolicyRouter(remote, mock_cloud(), pol, online=True)).execute(plan)
        assert res.ok, "distributed plan must complete"
        assert res.by_id["r1"].tier == "edge" and \
            "[qnn-mock]" in res.by_id["r1"].output, "r1 must execute on the remote edge node"
        assert res.by_id["r2"].tier == "cloud", "r2 escalates to cloud"

        # 4) offline fails the remote-edge step closed (no network to the node)
        res_off = await PlanExecutor(
            PolicyRouter(remote, mock_cloud(), pol, online=False)).execute(plan)
        assert all(s.tier == "edge" for s in res_off.steps), "offline routes to edge tier"

        # 5) a TRUNCATED remote stream (peer drops mid-infer, no done/error frame)
        #    must RAISE, not masquerade as a successful step.
        async def _truncating(ws):
            async for _ in ws:
                await ws.send(json.dumps({"t": "partial "}))
                return                                # close without done/error
        tready = asyncio.Event()
        tport = _free_port()

        async def _run_trunc():
            async with serve(_truncating, host, tport) as s:
                tready.set()
                await s.serve_forever()
        tnode = asyncio.create_task(_run_trunc())
        try:
            await asyncio.wait_for(tready.wait(), timeout=5.0)
            rb_t = RemoteBackend(f"ws://{host}:{tport}")
            raised = False
            try:
                async for _ in rb_t.infer(InferRequest([{"role": "user", "content": "x"}], "m")):
                    pass
            except RuntimeError:
                raised = True
            assert raised, "truncated remote stream must raise, not silently succeed"
        finally:
            tnode.cancel()
            try:
                await tnode
            except (asyncio.CancelledError, Exception):
                pass

        print("fabric/remote_backend: conformance OK over real socket; plan executed "
              "across nodes (edge→remote, escalate→cloud); offline fails closed; "
              "truncated stream raises (no silent partial-success)")
    finally:
        server.cancel()
        try:
            await server
        except (asyncio.CancelledError, Exception):
            pass


if __name__ == "__main__":
    asyncio.run(_selftest())
