"""
HARP · edge/power.py · CEE-owned · MIT
Instantaneous power telemetry for energy-per-token.

Grounding (Snapdragon NPU Profiling Guide):
  - E_token = ∫P(t)dt / N_tokens, reported in mJ                     §"Mathematics of Energy-per-Token"
  - Android: P = V·I from BatteryManager CURRENT_NOW(µA) × voltage   §"Telemetry Acquisition via Android OS APIs"
  - WoS:     read Hexagon/CPU/total rails from HWiNFO64 shared mem   §"Telemetry on Windows on Snapdragon"
  - Subtract idle baseline (screen on, radios on, no inference) to
    attribute energy to the NPU + memory bus, not the whole SoC.     §"Advanced telemetry ... subtract this baseline"

Two real targets mirror the two-backend canon's edge tier:
  WoSHwinfoSampler   -> Snapdragon X Elite Copilot+ PC  (D4 candidate A)
  AndroidAdbSampler  -> Snapdragon 8 Elite phone/QRD     (D4 candidate B)

A sampler runs on a background thread at a fixed poll interval, timestamps
every reading, and returns the trace. bench.py integrates the trace.
"""
from __future__ import annotations

import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PowerSample:
    t: float          # perf_counter seconds
    watts: float


@dataclass
class PowerTrace:
    samples: list[PowerSample] = field(default_factory=list)

    def energy_joules(self, baseline_w: float = 0.0) -> float:
        """Trapezoidal ∫P(t)dt with idle baseline subtracted. Clamps at 0 so a
        noisy sub-baseline reading can't credit negative energy."""
        s = self.samples
        if len(s) < 2:
            return 0.0
        e = 0.0
        for a, b in zip(s, s[1:]):
            dt = b.t - a.t
            p = ((a.watts + b.watts) / 2.0) - baseline_w
            e += max(p, 0.0) * dt
        return e

    def avg_watts(self) -> float:
        return sum(x.watts for x in self.samples) / len(self.samples) if self.samples else 0.0

    def peak_watts(self) -> float:
        return max((x.watts for x in self.samples), default=0.0)


class PowerSampler(ABC):
    """Poll instantaneous system power on a background thread. Context-manager:
    `with sampler: <run inference>`; trace is available after exit."""

    def __init__(self, poll_hz: float = 5.0):
        self.poll_interval = 1.0 / poll_hz   # 5 Hz = 200 ms; guide cites 100–500 ms
        self.trace = PowerTrace()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @abstractmethod
    def read_watts(self) -> float: ...

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.trace.samples.append(PowerSample(time.perf_counter(), self.read_watts()))
            except Exception:
                pass  # never let a telemetry hiccup kill the inference being measured
            time.sleep(self.poll_interval)

    def __enter__(self) -> "PowerSampler":
        self.trace = PowerTrace()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def measure_baseline(self, seconds: float = 3.0) -> float:
        """Idle floor: screen on, no inference. Call BEFORE the gate run."""
        with self:
            time.sleep(seconds)
        return self.trace.avg_watts()


class AndroidAdbSampler(PowerSampler):
    """Snapdragon 8 Elite phone over ADB. Reads sysfs power_supply rails — works
    headless from the laptop host driving the QRD, no on-device app needed.

    current_now is µA, voltage_now is µV on mainline; sign convention varies by
    OEM (discharge may be negative). We take |I| since we only want magnitude of
    draw during a controlled inference burst.
    """
    CUR = "/sys/class/power_supply/battery/current_now"
    VOLT = "/sys/class/power_supply/battery/voltage_now"

    def __init__(self, serial: str | None = None, poll_hz: float = 5.0):
        super().__init__(poll_hz)
        self._pfx = ["adb"] + (["-s", serial] if serial else [])

    def _cat(self, path: str) -> int:
        out = subprocess.run(self._pfx + ["shell", "cat", path],
                             capture_output=True, text=True, timeout=2.0)
        return int(out.stdout.strip())

    def read_watts(self) -> float:
        i_a = abs(self._cat(self.CUR)) / 1e6   # µA -> A
        v_v = self._cat(self.VOLT) / 1e6       # µV -> V
        return i_a * v_v

    def thermal_c(self, zone: str = "thermal_zone0") -> float:
        out = subprocess.run(self._pfx + ["shell", "cat", f"/sys/class/thermal/{zone}/temp"],
                             capture_output=True, text=True, timeout=2.0)
        raw = int(out.stdout.strip())
        return raw / 1000.0 if raw > 1000 else float(raw)  # milli-°C on most kernels


class WoSHwinfoSampler(PowerSampler):
    """Snapdragon X Elite Copilot+ PC. HWiNFO64 publishes per-rail sensors into a
    shared-memory segment (Global\\HWiNFO_SENS_SM2). We match a sensor by label
    substring (e.g. 'NPU Power', 'CPU Power', 'System Power').

    GAP: the SM2 struct offset layout is not in either reference doc. Until the
    deep-research prompt below resolves the exact reading order, pass an explicit
    `reader` callable (e.g. a thin pybind over the HWiNFO SDK or a parsed CSV log
    export) so the harness is unblocked and the rail wiring is the only TODO.
    """
    def __init__(self, reader, label: str = "NPU Power", poll_hz: float = 5.0):
        super().__init__(poll_hz)
        self._reader = reader        # callable(label:str)->watts
        self._label = label

    def read_watts(self) -> float:
        return float(self._reader(self._label))


def get_sampler(target: str, **kw) -> PowerSampler:
    """target ∈ {'android','wos'} — keep call sites device-agnostic so bench.py
    runs identically across the D4 candidate devices."""
    if target == "android":
        return AndroidAdbSampler(**kw)
    if target == "wos":
        return WoSHwinfoSampler(**kw)
    raise ValueError(f"unknown power target: {target}")
