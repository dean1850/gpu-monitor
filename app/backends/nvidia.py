# SPDX-License-Identifier: MIT
# Copyright (c) 2026 gpu-monitor contributors
"""
NVIDIA backend — NVML via a self-contained ctypes binding.

FIX-P11-NV-1: NEW BACKEND.  Written against the NVML C ABI, NOT against
hardware — the reference host for this project has an Intel iGPU only.

Why ctypes rather than the ``nvidia-ml-py`` package: this project already
talks to the kernel through raw ``perf_event_open(2)`` with ctypes, so the
idiom is established; more importantly, adding a pip dependency that exists
solely for a vendor the operator may not have means every deployment pays
for it.  ``libnvidia-ml.so.1`` is loaded lazily, and if it is not present the
backend reports "unavailable" and nothing else in the app notices.

The proprietary NVIDIA driver does NOT populate DRM ``fdinfo`` engine
counters, so this backend cannot reuse the shared fdinfo scanner that Intel
and AMD share.  Per-process attribution instead comes from
``nvmlDeviceGetProcessUtilization()``, which returns per-PID SM / memory /
encoder / decoder percentages directly — already normalised, so there is no
delta arithmetic here at all.

CONTAINER PREREQUISITE: seeing an NVIDIA GPU from inside a container needs
the NVIDIA Container Toolkit on the host and a device reservation in
docker-compose.yml.  ``/dev/dri`` passthrough alone is NOT sufficient.  See
the commented block in docker-compose.yml.
"""

from __future__ import annotations

import ctypes
import logging
import time

from .base import GpuBackend, GpuDevice

logger = logging.getLogger(__name__)

NVML_ACTIVE_SOURCE = "nvml"

ENGINE_DISPLAY_ORDER: tuple[str, ...] = ("Graphics/SM", "Memory", "Encoder", "Decoder")
ENGINE_KEYS: tuple[str, ...] = ("sm", "memory", "encoder", "decoder")

_KEY_TO_DISPLAY: dict[str, str] = {
    "sm": "Graphics/SM",
    "memory": "Memory",
    "encoder": "Encoder",
    "decoder": "Decoder",
}

# ── NVML constants ───────────────────────────────────────────────────────

NVML_SUCCESS = 0
NVML_ERROR_UNINITIALIZED = 1
NVML_ERROR_INVALID_ARGUMENT = 2
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_NO_PERMISSION = 4
NVML_ERROR_NOT_FOUND = 6
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_ERROR_DRIVER_NOT_LOADED = 9
NVML_ERROR_LIBRARY_NOT_FOUND = 12
NVML_ERROR_FUNCTION_NOT_FOUND = 13

NVML_CLOCK_GRAPHICS = 0
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2

NVML_TEMPERATURE_GPU = 0

_NAME_BUFFER = 96
_UUID_BUFFER = 96
_PCI_BUS_ID_BUFFER = 32
_PCI_BUS_ID_BUFFER_V2 = 16

_LIB_CANDIDATES = ("libnvidia-ml.so.1", "libnvidia-ml.so")


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Memory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _PciInfo(ctypes.Structure):
    _fields_ = [
        ("busIdLegacy", ctypes.c_char * _PCI_BUS_ID_BUFFER_V2),
        ("domain", ctypes.c_uint),
        ("bus", ctypes.c_uint),
        ("device", ctypes.c_uint),
        ("pciDeviceId", ctypes.c_uint),
        ("pciSubSystemId", ctypes.c_uint),
        ("busId", ctypes.c_char * _PCI_BUS_ID_BUFFER),
    ]


class _ProcessUtilizationSample(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("timeStamp", ctypes.c_ulonglong),
        ("smUtil", ctypes.c_uint),
        ("memUtil", ctypes.c_uint),
        ("encUtil", ctypes.c_uint),
        ("decUtil", ctypes.c_uint),
    ]


class _ProcessInfo(ctypes.Structure):
    """``nvmlProcessInfo_v2_t`` — matches the _v3 getter's expectations."""

    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


class NvmlError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class _Nvml:
    """
    Thin ctypes wrapper over the handful of NVML entry points we use.

    FIX-P11-NV-2: every symbol is resolved through :meth:`_sym`, which walks
    a list of candidate names.  NVML versions its ABI by suffixing symbols
    (``nvmlInit`` -> ``nvmlInit_v2``, ``nvmlDeviceGetComputeRunningProcesses``
    -> ``_v2`` -> ``_v3``), and which suffixes exist depends on the installed
    driver.  Resolving optimistically newest-first and falling back keeps
    this working across driver generations instead of pinning to one.
    """

    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        self._initialised = False

    # ── loading ──

    def load(self) -> None:
        last_error: OSError | None = None
        for name in _LIB_CANDIDATES:
            try:
                self._lib = ctypes.CDLL(name)
                logger.info("nvidia: loaded %s", name)
                return
            except OSError as exc:
                last_error = exc
        raise NvmlError(
            NVML_ERROR_LIBRARY_NOT_FOUND,
            "libnvidia-ml.so.1 not found. On a container this normally means "
            "the NVIDIA Container Toolkit is not wired up — see the nvidia "
            f"block in docker-compose.yml. ({last_error})",
        )

    def _sym(self, *names: str):
        assert self._lib is not None
        for name in names:
            fn = getattr(self._lib, name, None)
            if fn is not None:
                return fn
        raise NvmlError(
            NVML_ERROR_FUNCTION_NOT_FOUND,
            f"NVML exports none of {names!r} — driver too old?",
        )

    def _check(self, code: int, context: str) -> None:
        if code == NVML_SUCCESS:
            return
        raise NvmlError(code, f"{context}: {self.error_string(code)}")

    def error_string(self, code: int) -> str:
        try:
            fn = self._sym("nvmlErrorString")
            fn.restype = ctypes.c_char_p
            return (fn(code) or b"").decode("utf-8", "replace") or f"NVML error {code}"
        except Exception:  # noqa: BLE001
            return f"NVML error {code}"

    # ── lifecycle ──

    def init(self) -> None:
        if self._lib is None:
            self.load()
        self._check(self._sym("nvmlInit_v2", "nvmlInit")(), "nvmlInit")
        self._initialised = True

    def shutdown(self) -> None:
        if not self._initialised:
            return
        try:
            self._sym("nvmlShutdown")()
        except Exception:  # noqa: BLE001
            pass
        self._initialised = False

    # ── enumeration ──

    def device_count(self) -> int:
        count = ctypes.c_uint()
        self._check(
            self._sym("nvmlDeviceGetCount_v2", "nvmlDeviceGetCount")(ctypes.byref(count)),
            "nvmlDeviceGetCount",
        )
        return int(count.value)

    def handle_by_index(self, index: int):
        handle = ctypes.c_void_p()
        self._check(
            self._sym("nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex")(
                ctypes.c_uint(index), ctypes.byref(handle)
            ),
            f"nvmlDeviceGetHandleByIndex({index})",
        )
        return handle

    def device_name(self, handle) -> str:
        buf = ctypes.create_string_buffer(_NAME_BUFFER)
        self._check(
            self._sym("nvmlDeviceGetName")(handle, buf, ctypes.c_uint(_NAME_BUFFER)),
            "nvmlDeviceGetName",
        )
        return buf.value.decode("utf-8", "replace")

    def device_uuid(self, handle) -> str:
        buf = ctypes.create_string_buffer(_UUID_BUFFER)
        self._check(
            self._sym("nvmlDeviceGetUUID")(handle, buf, ctypes.c_uint(_UUID_BUFFER)),
            "nvmlDeviceGetUUID",
        )
        return buf.value.decode("utf-8", "replace")

    def pci_bus_id(self, handle) -> str | None:
        info = _PciInfo()
        code = self._sym(
            "nvmlDeviceGetPciInfo_v3", "nvmlDeviceGetPciInfo_v2", "nvmlDeviceGetPciInfo"
        )(handle, ctypes.byref(info))
        if code != NVML_SUCCESS:
            return None
        return info.busId.decode("utf-8", "replace") or None

    # ── metrics (each returns None when unsupported) ──

    def utilization(self, handle) -> tuple[int, int] | None:
        util = _Utilization()
        if self._sym("nvmlDeviceGetUtilizationRates")(handle, ctypes.byref(util)) != NVML_SUCCESS:
            return None
        return int(util.gpu), int(util.memory)

    def memory(self, handle) -> tuple[int, int] | None:
        mem = _Memory()
        code = self._sym("nvmlDeviceGetMemoryInfo")(handle, ctypes.byref(mem))
        if code != NVML_SUCCESS:
            return None
        return int(mem.used), int(mem.total)

    def clock_mhz(self, handle, clock_type: int) -> int | None:
        value = ctypes.c_uint()
        code = self._sym("nvmlDeviceGetClockInfo")(
            handle, ctypes.c_uint(clock_type), ctypes.byref(value)
        )
        return int(value.value) if code == NVML_SUCCESS else None

    def max_clock_mhz(self, handle, clock_type: int) -> int | None:
        value = ctypes.c_uint()
        code = self._sym("nvmlDeviceGetMaxClockInfo")(
            handle, ctypes.c_uint(clock_type), ctypes.byref(value)
        )
        return int(value.value) if code == NVML_SUCCESS else None

    def temperature_c(self, handle) -> int | None:
        value = ctypes.c_uint()
        code = self._sym("nvmlDeviceGetTemperature")(
            handle, ctypes.c_uint(NVML_TEMPERATURE_GPU), ctypes.byref(value)
        )
        return int(value.value) if code == NVML_SUCCESS else None

    def power_watts(self, handle) -> float | None:
        value = ctypes.c_uint()
        code = self._sym("nvmlDeviceGetPowerUsage")(handle, ctypes.byref(value))
        return round(value.value / 1000.0, 2) if code == NVML_SUCCESS else None

    def encoder_util(self, handle) -> int | None:
        util = ctypes.c_uint()
        period = ctypes.c_uint()
        code = self._sym("nvmlDeviceGetEncoderUtilization")(
            handle, ctypes.byref(util), ctypes.byref(period)
        )
        return int(util.value) if code == NVML_SUCCESS else None

    def decoder_util(self, handle) -> int | None:
        util = ctypes.c_uint()
        period = ctypes.c_uint()
        code = self._sym("nvmlDeviceGetDecoderUtilization")(
            handle, ctypes.byref(util), ctypes.byref(period)
        )
        return int(util.value) if code == NVML_SUCCESS else None

    # ── per-process ──

    def process_utilization(
        self, handle, last_seen_us: int
    ) -> tuple[list[_ProcessUtilizationSample], int]:
        """
        Return (samples, newest_timestamp_us) since *last_seen_us*.

        FIX-P11-NV-3: the two-call size-probe dance is mandatory.  NVML tells
        you how many samples it has by returning INSUFFICIENT_SIZE from a
        call with a NULL buffer; calling once with a guessed buffer size
        silently truncates on a busy GPU.
        """
        fn = self._sym("nvmlDeviceGetProcessUtilization")

        count = ctypes.c_uint(0)
        code = fn(handle, None, ctypes.byref(count), ctypes.c_ulonglong(last_seen_us))

        if code == NVML_SUCCESS or count.value == 0:
            return [], last_seen_us
        if code != NVML_ERROR_INSUFFICIENT_SIZE:
            raise NvmlError(code, f"nvmlDeviceGetProcessUtilization: {self.error_string(code)}")

        # Head-room: processes can start between the two calls.
        capacity = count.value + 16
        buffer = (_ProcessUtilizationSample * capacity)()
        count = ctypes.c_uint(capacity)
        code = fn(handle, buffer, ctypes.byref(count), ctypes.c_ulonglong(last_seen_us))
        if code != NVML_SUCCESS:
            raise NvmlError(code, f"nvmlDeviceGetProcessUtilization: {self.error_string(code)}")

        samples = [buffer[i] for i in range(min(count.value, capacity))]
        newest = max((s.timeStamp for s in samples), default=last_seen_us)
        return samples, newest

    def running_processes(self, handle) -> dict[int, int]:
        """Return ``{pid: used_gpu_memory_bytes}`` across compute + graphics."""
        out: dict[int, int] = {}
        for names in (
            ("nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetComputeRunningProcesses_v2",
             "nvmlDeviceGetComputeRunningProcesses"),
            ("nvmlDeviceGetGraphicsRunningProcesses_v3", "nvmlDeviceGetGraphicsRunningProcesses_v2",
             "nvmlDeviceGetGraphicsRunningProcesses"),
        ):
            try:
                fn = self._sym(*names)
            except NvmlError:
                continue

            count = ctypes.c_uint(0)
            code = fn(handle, ctypes.byref(count), None)
            if code == NVML_SUCCESS or count.value == 0:
                continue
            if code != NVML_ERROR_INSUFFICIENT_SIZE:
                continue

            capacity = count.value + 8
            buffer = (_ProcessInfo * capacity)()
            count = ctypes.c_uint(capacity)
            if fn(handle, ctypes.byref(count), buffer) != NVML_SUCCESS:
                continue
            for i in range(min(count.value, capacity)):
                out[int(buffer[i].pid)] = int(buffer[i].usedGpuMemory)
        return out


class NvidiaBackend(GpuBackend):
    vendor = "nvidia"

    def __init__(self, forced: bool = False) -> None:
        super().__init__()
        self._forced = forced
        self._nvml = _Nvml()
        self._last_seen_us: dict[str, int] = {}
        self._attrib_error_logged = False

    # ── discovery ────────────────────────────────────────────────────────

    def probe(self) -> list[GpuDevice]:
        try:
            self._nvml.init()
        except NvmlError as exc:
            # Absent NVIDIA userspace is the NORMAL case on most hosts. Only
            # complain when the operator explicitly asked for NVIDIA.
            if self._forced:
                self._fail("nvml_unavailable", str(exc))
            else:
                logger.info("nvidia: not available (%s) — skipping.", exc)
            return []
        except Exception as exc:  # noqa: BLE001
            if self._forced:
                self._fail("nvml_unavailable", f"NVML init failed: {exc}")
            return []

        try:
            count = self._nvml.device_count()
        except NvmlError as exc:
            self._fail("nvml_enumeration_failed", str(exc))
            self._nvml.shutdown()
            return []

        if count == 0:
            if self._forced:
                self._fail(
                    "no_device",
                    "GPU_ENABLE_NVIDIA=true but NVML reports zero devices. In a "
                    "container this usually means the GPU was not reserved — "
                    "check the deploy.resources block in docker-compose.yml.",
                )
            self._nvml.shutdown()
            return []

        devices: list[GpuDevice] = []
        for index in range(count):
            try:
                handle = self._nvml.handle_by_index(index)
                name = self._nvml.device_name(handle)
                uuid = self._nvml.device_uuid(handle)
                bus_id = self._nvml.pci_bus_id(handle)
            except NvmlError as exc:
                logger.warning("nvidia: device %d unreadable: %s", index, exc)
                continue

            device = GpuDevice(
                # UUID beats PCI address here: it survives PCI renumbering
                # and matches what nvidia-smi reports.
                uid=f"nvidia:{uuid}",
                vendor="nvidia",
                name=name,
                index=index,
                pci_address=bus_id,
                driver="nvidia",
                engine_names=ENGINE_DISPLAY_ORDER,
                engine_keys=ENGINE_KEYS,
            )
            device.extra["handle"] = handle
            devices.append(device)
            logger.info("nvidia: found %s (%s)", name, uuid)

        self._devices = devices
        if devices:
            self._active_source = NVML_ACTIVE_SOURCE
        return devices

    def close(self) -> None:
        self._nvml.shutdown()

    # ── sampling ─────────────────────────────────────────────────────────

    def sample(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for device in self._devices:
            try:
                out[device.uid] = self._sample_device(device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("nvidia: sampling %s failed: %s", device.uid, exc)
                out[device.uid] = {"error": str(exc)}
        return out

    def _sample_device(self, device: GpuDevice) -> dict:
        handle = device.extra["handle"]
        block: dict = {}
        engines: dict[str, dict[str, float]] = {}

        rates = self._nvml.utilization(handle)
        if rates is not None:
            gpu_util, mem_util = rates
            block["utilisation"] = {"value": float(gpu_util)}
            engines["Graphics/SM"] = {"busy": float(gpu_util)}
            engines["Memory"] = {"busy": float(mem_util)}

        enc = self._nvml.encoder_util(handle)
        if enc is not None:
            engines["Encoder"] = {"busy": float(enc)}
        dec = self._nvml.decoder_util(handle)
        if dec is not None:
            engines["Decoder"] = {"busy": float(dec)}

        if engines:
            block["engines"] = engines

        mem = self._nvml.memory(handle)
        if mem is not None:
            used, total = mem
            memory: dict[str, float] = {"used_mib": round(used / (1024 * 1024), 1)}
            if total > 0:
                memory["total_mib"] = round(total / (1024 * 1024), 1)
                memory["used_pct"] = round((used / total) * 100.0, 2)
            block["memory"] = memory

        frequency: dict[str, float] = {}
        actual = self._nvml.clock_mhz(handle, NVML_CLOCK_GRAPHICS)
        if actual is not None:
            frequency["actual"] = float(actual)
        maximum = self._nvml.max_clock_mhz(handle, NVML_CLOCK_GRAPHICS)
        if maximum is not None:
            frequency["max"] = float(maximum)
        if frequency:
            block["frequency"] = frequency

        watts = self._nvml.power_watts(handle)
        if watts is not None:
            block["power"] = {"watts": watts}

        temp = self._nvml.temperature_c(handle)
        if temp is not None:
            block["temperature"] = {"celsius": float(temp)}

        return block

    # ── attribution ──────────────────────────────────────────────────────

    def attribution(self) -> dict[str, dict[int, dict[str, float]]] | None:
        if not self._devices:
            return None

        result: dict[str, dict[int, dict[str, float]]] = {}

        for device in self._devices:
            handle = device.extra["handle"]
            uid = device.uid

            # FIX-P11-NV-4: seed the watermark one interval into the past on
            # the first tick. Passing 0 would ask NVML for every sample it
            # has ever buffered, producing a large one-off burst of stale
            # PIDs on the first frame after startup.
            last_seen = self._last_seen_us.get(uid)
            if last_seen is None:
                last_seen = int((time.time() - 2.0) * 1_000_000)

            try:
                samples, newest = self._nvml.process_utilization(handle, last_seen)
            except NvmlError as exc:
                if exc.code == NVML_ERROR_NOT_SUPPORTED:
                    # Common on consumer cards under some driver branches.
                    if not self._attrib_error_logged:
                        logger.info(
                            "nvidia: nvmlDeviceGetProcessUtilization is not "
                            "supported by this driver — falling back to "
                            "per-process memory only."
                        )
                        self._attrib_error_logged = True
                    samples, newest = [], last_seen
                else:
                    if not self._attrib_error_logged:
                        logger.warning("nvidia: attribution failed: %s", exc)
                        self._attrib_error_logged = True
                    continue
            except Exception as exc:  # noqa: BLE001
                if not self._attrib_error_logged:
                    logger.warning("nvidia: attribution failed: %s", exc)
                    self._attrib_error_logged = True
                continue

            self._last_seen_us[uid] = newest

            per_pid: dict[int, dict[str, float]] = {}
            # NVML can return several samples per PID within the window;
            # keep the highest, which is the peak over the interval.
            for s in samples:
                pid = int(s.pid)
                bucket = per_pid.setdefault(
                    pid, {k: 0.0 for k in ENGINE_KEYS} | {"total": 0.0}
                )
                bucket["sm"] = max(bucket["sm"], float(s.smUtil))
                bucket["memory"] = max(bucket["memory"], float(s.memUtil))
                bucket["encoder"] = max(bucket["encoder"], float(s.encUtil))
                bucket["decoder"] = max(bucket["decoder"], float(s.decUtil))

            # Merge in processes holding VRAM but not currently computing, so
            # an idle-but-resident model server is still visible.
            try:
                for pid, used_bytes in self._nvml.running_processes(handle).items():
                    bucket = per_pid.setdefault(
                        pid, {k: 0.0 for k in ENGINE_KEYS} | {"total": 0.0}
                    )
                    bucket["memory_mib"] = round(used_bytes / (1024 * 1024), 1)
            except Exception:  # noqa: BLE001
                pass

            for bucket in per_pid.values():
                bucket["total"] = round(
                    sum(bucket.get(k, 0.0) for k in ENGINE_KEYS), 2
                )

            if per_pid:
                result[uid] = per_pid

        return result
