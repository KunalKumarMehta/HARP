"""
HARP · edge/power.py · MIT
Instantaneous power telemetry for energy-per-token.

Grounding:
  - E_token = ∫P(t)dt / N, mJ, idle baseline subtracted
  - WoS: HWiNFO_SENS_SM2 shared mem, two-pass NPU isolation
  - struct offsets / mutex / magic
  - CSV fallback (Free Edition 12h shm cap)
  - Android: P=V·I from sysfs current_now × voltage_now
"""
from __future__ import annotations

import ctypes
import mmap
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# ---- HWiNFO_SENS_SM2 binary layout -----------------------------------------
SM2_NAME = "Global\\HWiNFO_SENS_SM2"
SM2_MUTEX = "Global\\HWiNFO_SM2_MUTEX"
MAGIC = 0x53695748          # 'SiWH'
MAGIC_SWAP = 0x48576953     # endianness-inverted guard
SENSOR_TYPE_POWER = 5       # type enum: 1=temp 5=power 6=MHz 7=%


class HWiNFOHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),                 # 0x00
        ("version", ctypes.c_uint32),               # 0x04
        ("revision", ctypes.c_uint32),              # 0x08
        ("poll_time", ctypes.c_int64),              # 0x0C
        ("sensor_section_offset", ctypes.c_uint32), # 0x14
        ("sensor_element_size", ctypes.c_uint32),   # 0x18  (264)
        ("sensor_element_count", ctypes.c_uint32),  # 0x1C
        ("entry_section_offset", ctypes.c_uint32),  # 0x20
        ("entry_element_size", ctypes.c_uint32),    # 0x24  (316)
        ("entry_element_count", ctypes.c_uint32),   # 0x28
        ("polling_period", ctypes.c_uint32),        # 0x2C
    ]


class HWiNFOSensor(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("id", ctypes.c_uint32),                    # 0x00
        ("instance", ctypes.c_uint32),              # 0x04
        ("name_original", ctypes.c_char * 128),     # 0x08
        ("name_user", ctypes.c_char * 128),         # 0x88
    ]                                               # = 264 bytes


class HWiNFOEntry(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("type", ctypes.c_uint32),                  # 0x00  (5 = power)
        ("sensor_index", ctypes.c_uint32),          # 0x04  -> HWiNFOSensor[]
        ("id", ctypes.c_uint32),                    # 0x08
        ("name_original", ctypes.c_char * 128),     # 0x0C
        ("name_user", ctypes.c_char * 128),         # 0x8C
        ("unit", ctypes.c_char * 16),               # 0x10C
        ("value", ctypes.c_double),                 # 0x11C
        ("value_min", ctypes.c_double),             # 0x124
        ("value_max", ctypes.c_double),             # 0x12C
        ("value_avg", ctypes.c_double),             # 0x134
    ]                                               # = 316 bytes


def _cstr(b: bytes) -> str:
    return b.split(b"\x00", 1)[0].decode("utf-8", "replace").strip()


def parse_npu_power(buf, sensor_match=("Hexagon NPU", "Snapdragon"),
                    entry_match="Power") -> float:
    """Pure two-pass parser over a SENS_SM2 byte buffer. OS-independent and unit-
    tested off-target. Returns instantaneous NPU watts.

    Pass 1: find the sensor whose name_original substring-matches the NPU.
    Pass 2: entry where sensor_index==target AND type==POWER AND unit=='W'.
    value_max is also available for sub-poll-interval transient spikes.
    """
    hdr = HWiNFOHeader.from_buffer_copy(buf[: ctypes.sizeof(HWiNFOHeader)])
    if hdr.magic not in (MAGIC, MAGIC_SWAP):
        raise ValueError(f"bad SENS_SM2 magic 0x{hdr.magic:08X}")

    target = None
    for i in range(hdr.sensor_element_count):
        off = hdr.sensor_section_offset + i * hdr.sensor_element_size
        s = HWiNFOSensor.from_buffer_copy(buf[off: off + ctypes.sizeof(HWiNFOSensor)])
        name = _cstr(s.name_original)
        if any(m.lower() in name.lower() for m in sensor_match):
            target = i
            break
    if target is None:
        raise LookupError("no Hexagon NPU sensor in SENS_SM2")

    for j in range(hdr.entry_element_count):
        off = hdr.entry_section_offset + j * hdr.entry_element_size
        e = HWiNFOEntry.from_buffer_copy(buf[off: off + ctypes.sizeof(HWiNFOEntry)])
        if (e.sensor_index == target and e.type == SENSOR_TYPE_POWER
                and _cstr(e.unit) == "W" and entry_match.lower() in _cstr(e.name_original).lower()):
            return float(e.value)
    raise LookupError("no NPU power(W) entry for matched sensor")


@dataclass
class PowerSample:
    t: float
    watts: float


@dataclass
class PowerTrace:
    samples: list[PowerSample] = field(default_factory=list)

    def energy_joules(self, baseline_w: float = 0.0) -> float:
        s = self.samples
        if len(s) < 2:
            return 0.0
        return sum(max(((a.watts + b.watts) / 2.0) - baseline_w, 0.0) * (b.t - a.t)
                   for a, b in zip(s, s[1:]))

    def avg_watts(self) -> float:
        return sum(x.watts for x in self.samples) / len(self.samples) if self.samples else 0.0

    def peak_watts(self) -> float:
        return max((x.watts for x in self.samples), default=0.0)


class PowerSampler(ABC):
    def __init__(self, poll_hz: float = 5.0):
        self.poll_interval = 1.0 / poll_hz
        self.trace = PowerTrace()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @abstractmethod
    def read_watts(self) -> float: ...

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.trace.samples.append(PowerSample(time.perf_counter(), self.read_watts()))
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def __enter__(self):
        self.trace = PowerTrace()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def measure_baseline(self, seconds: float = 3.0) -> float:
        with self:
            time.sleep(seconds)
        return self.trace.avg_watts()


class WoSHwinfoSampler(PowerSampler):
    """Snapdragon X/X2 Elite Copilot+ PC. Real mmap + Win32 mutex.
    Mutex hold is mandatory: skipping it yields torn doubles (megawatt/kilo-°C
    garbage). Windows-only — raises elsewhere so the Linux dev box stays clean."""
    def __init__(self, poll_hz: float = 5.0, mutex_timeout_ms: int = 500):
        super().__init__(poll_hz)
        if not sys.platform.startswith("win"):
            raise RuntimeError("WoSHwinfoSampler requires Windows on Snapdragon; "
                               "use CsvFallbackSampler or AndroidAdbSampler off-target.")
        self._k32 = ctypes.windll.kernel32
        self._k32.OpenMutexW.restype = ctypes.c_void_p
        self._k32.OpenMutexW.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_wchar_p]
        self._timeout = mutex_timeout_ms
        SYNCHRONIZE = 0x00100000
        self._mutex = self._k32.OpenMutexW(SYNCHRONIZE, False, SM2_MUTEX)
        self._mm = mmap.mmap(-1, 0, tagname=SM2_NAME, access=mmap.ACCESS_READ)

    def read_watts(self) -> float:
        got = False
        if self._mutex:
            got = self._k32.WaitForSingleObject(ctypes.c_void_p(self._mutex), self._timeout) == 0
        try:
            self._mm.seek(0)
            buf = self._mm.read(self._mm.size())
            return parse_npu_power(buf)
        finally:
            if got:
                self._k32.ReleaseMutex(ctypes.c_void_p(self._mutex))


class CsvFallbackSampler(PowerSampler):
    """HWiNFO Free shm caps at 12h; tail the continuous CSV export as fallback.
    Fuzzy-matches the 'Hexagon NPU [Power]'-style column so user relabeling in the
    HWiNFO GUI doesn't silently break the mapping. Works on any OS for replay."""
    def __init__(self, csv_path: str, col_match=("hexagon", "npu", "power"), poll_hz: float = 2.0):
        super().__init__(poll_hz)
        self._path = csv_path
        self._col_match = col_match
        self._col = None

    def _resolve_col(self, header: list[str]) -> int:
        for i, h in enumerate(header):
            hl = h.lower()
            if all(m in hl for m in self._col_match):
                return i
        raise LookupError(f"no column matching {self._col_match} in CSV header")

    def read_watts(self) -> float:
        with open(self._path, "r", errors="ignore") as f:
            lines = f.readlines()
        if len(lines) < 2:
            return 0.0
        cols = lines[0].rstrip("\n").split(",")
        if self._col is None:
            self._col = self._resolve_col(cols)
        last = lines[-1].rstrip("\n").split(",")
        try:
            return float(last[self._col])
        except (IndexError, ValueError):
            return 0.0


class AndroidAdbSampler(PowerSampler):
    """Snapdragon 8 Elite phone over ADB sysfs — headless from the laptop host."""
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
        return (abs(self._cat(self.CUR)) / 1e6) * (self._cat(self.VOLT) / 1e6)

    def thermal_c(self, zone: str = "thermal_zone0") -> float:
        out = subprocess.run(self._pfx + ["shell", "cat", f"/sys/class/thermal/{zone}/temp"],
                             capture_output=True, text=True, timeout=2.0)
        raw = int(out.stdout.strip())
        return raw / 1000.0 if raw > 1000 else float(raw)


def get_sampler(target: str, **kw) -> PowerSampler:
    return {"android": AndroidAdbSampler, "wos": WoSHwinfoSampler,
            "csv": CsvFallbackSampler}[target](**kw)
