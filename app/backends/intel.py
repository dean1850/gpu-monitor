# SPDX-License-Identifier: MIT
# Copyright (c) 2026 gpu-monitor contributors
"""
Intel backend — i915 perf PMU via raw ``perf_event_open(2)`` + sysfs.

FIX-P11-INTEL-1: this is the Prompt-9 engine-counter code lifted out of
``gpu_reader.py`` and made per-device.  The behaviour is deliberately
unchanged — it works on this project's host and there was no reason to
rewrite it — but every path that used to be a module constant pinned to
``card0`` / ``/sys/devices/i915`` is now derived from the ``GpuDevice``.

Why the raw syscall and not ``intel_gpu_top``: on the reference host
(Proxmox/QEMU VM, Intel UHD 630 / Comet Lake) igt-gpu-tools 2.3's
``intel_gpu_top`` fails with "Failed to detect engines!" even as root.
That is a bug in that igt release, not a missing PMU — ``perf stat -e
i915/rcs0-busy/`` returns real nanosecond counters on the same box.  So we
talk to the PMU directly with ctypes: pure stdlib, no external binary.

FIX-P11-INTEL-2: multi-GPU PMU naming.  With one Intel GPU the kernel
registers the PMU as ``/sys/devices/i915``.  With more than one it uses
``/sys/devices/i915_0000_03_00.0`` (PCI address, dots and colons replaced
by underscores).  Both layouts are probed, per device.
"""

from __future__ import annotations

import ctypes
import errno
import logging
import os
import platform
import struct
import time

from .base import GpuBackend, GpuDevice
from .drm import (
    FdinfoScanner,
    device_display_name,
    enumerate_drm_cards,
    first_existing,
    read_int_file,
    read_text,
    wrap_delta,
)

logger = logging.getLogger(__name__)

PMU_ACTIVE_SOURCE = "i915_pmu_perf"
FALLBACK_ACTIVE_SOURCE = "sysfs_freq_rc6_only"

# PMU event name -> display key.
_PMU_EVENT_TO_DISPLAY: dict[str, str] = {
    "rcs0-busy": "Render/3D",
    "bcs0-busy": "Blitter",
    "vcs0-busy": "Video",
    "vecs0-busy": "VideoEnhance",
}
ENGINE_DISPLAY_ORDER: tuple[str, ...] = (
    "Render/3D", "Blitter", "Video", "VideoEnhance",
)

# fdinfo raw engine name -> internal per-container key.
FDINFO_ENGINE_MAP: dict[str, str] = {
    "render": "render",
    "copy": "blitter",
    "video": "video",
    "video-enhance": "video_enhance",
}
ENGINE_KEYS: tuple[str, ...] = ("render", "blitter", "video", "video_enhance")

PERF_FORMAT_TOTAL_TIME_ENABLED = 1 << 0
PERF_FORMAT_TOTAL_TIME_RUNNING = 1 << 1
PERF_FLAG_FD_CLOEXEC = 1 << 3

# FIX-P11-INTEL-3: load libc defensively.  On every realistic target it is
# present, but if it isn't we want a clean OSError(ENOSYS) from the wrapper
# rather than a module-import crash that takes the whole app down.
try:
    _libc: ctypes.CDLL | None = ctypes.CDLL("libc.so.6", use_errno=True)
    _libc.syscall.restype = ctypes.c_long
except OSError:  # pragma: no cover
    _libc = None

_SYS_PERF_EVENT_OPEN_BY_ARCH: dict[str, int] = {
    "x86_64": 298,
    "aarch64": 241,
}


class _PerfEventAttr(ctypes.Structure):
    """``struct perf_event_attr``, padded so ``sizeof() == 128``."""

    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32),
        ("config1", ctypes.c_uint64),
        ("config2", ctypes.c_uint64),
        ("branch_sample_type", ctypes.c_uint64),
        ("sample_regs_user", ctypes.c_uint64),
        ("sample_stack_user", ctypes.c_uint32),
        ("clockid", ctypes.c_int32),
        ("sample_regs_intr", ctypes.c_uint64),
        ("aux_watermark", ctypes.c_uint32),
        ("sample_max_stack", ctypes.c_uint16),
        ("__reserved_2", ctypes.c_uint16),
        ("_pad", ctypes.c_uint8 * 16),
    ]


assert ctypes.sizeof(_PerfEventAttr) == 128


def _perf_event_open(attr: _PerfEventAttr, pid: int, cpu: int, group_fd: int, flags: int) -> int:
    if _libc is None:
        raise OSError(errno.ENOSYS, "libc.so.6 unavailable; perf_event_open unsupported")

    arch = platform.machine()
    syscall_nr = _SYS_PERF_EVENT_OPEN_BY_ARCH.get(arch)
    if syscall_nr is None:
        raise OSError(errno.ENOSYS, f"perf_event_open syscall number unknown for arch {arch!r}")

    fd = _libc.syscall(
        ctypes.c_long(syscall_nr),
        ctypes.byref(attr),
        ctypes.c_int(pid),
        ctypes.c_int(cpu),
        ctypes.c_int(group_fd),
        ctypes.c_ulong(flags),
    )
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


def _read_perf_counter(fd: int) -> tuple[int, int, int]:
    """Read (value, time_enabled, time_running) — 3x u64."""
    return struct.unpack("<QQQ", os.read(fd, 24))


def _first_int(text: str) -> int:
    import re

    m = re.search(r"\d+", text)
    if not m:
        raise ValueError(f"no integer found in: {text!r}")
    return int(m.group())


def _parse_int_auto(text: str) -> int:
    text = text.strip()
    if text[:2].lower() == "0x":
        return int(text, 16)
    try:
        return int(text, 0)
    except ValueError:
        return int(text, 10)


def _parse_format_field(spec: str) -> tuple[str, int]:
    """Parse ``"config:0-7"`` / ``"config1:32-63"`` / ``"config:8"``."""
    field, _, bits = spec.strip().partition(":")
    lo_str = bits.partition("-")[0] if "-" in bits else bits
    return field.strip(), int(lo_str)


class IntelBackend(GpuBackend):
    vendor = "intel"

    def __init__(self, forced: bool = False) -> None:
        super().__init__()
        self._forced = forced
        self._scanner: FdinfoScanner | None = None
        # (uid, display_name) -> previous (value, enabled, running)
        self._perf_prev: dict[tuple[str, str], tuple[int, int, int]] = {}
        self._perf_prev_wall: dict[str, float] = {}
        self._rc6_prev: dict[str, tuple[int, float]] = {}
        self._range_warned: set[str] = set()
        self._read_error_logged: set[str] = set()
        # attribution state: uid -> {(pid, client): {engine: ns}}
        self._prev_clients: dict[str, dict[tuple[int, str], dict[str, int]]] = {}
        self._prev_attrib_wall: float | None = None

    # ── discovery ────────────────────────────────────────────────────────

    def probe(self) -> list[GpuDevice]:
        devices: list[GpuDevice] = []
        try:
            cards = enumerate_drm_cards()
        except Exception as exc:  # noqa: BLE001 — probe must never raise
            self._fail("probe_failed", f"DRM enumeration failed: {exc}")
            return []

        intel_cards = [c for c in cards if c["vendor"] == "intel"]

        if not intel_cards:
            if self._forced:
                self._fail(
                    "no_device",
                    "GPU_ENABLE_INTEL=true but no Intel DRM card was found. "
                    "Check that /dev/dri is passed through and that the i915 "
                    "(or xe) driver is bound.",
                )
            return []

        for index, card in enumerate(intel_cards):
            uid = f"intel:{card['pci_address'] or card['card']}"
            name = device_display_name(card["vendor_id"], card["device_id"], card["driver"])

            device = GpuDevice(
                uid=uid,
                vendor="intel",
                name=name,
                index=index,
                card_path=card["card_path"],
                render_node=card["render_node"],
                pci_address=card["pci_address"],
                driver=card["driver"],
                engine_names=ENGINE_DISPLAY_ORDER,
                engine_keys=ENGINE_KEYS,
            )
            device.extra["pmu_dir"] = self._find_pmu_dir(card["pci_address"])
            device.extra["freq_actual"] = first_existing(
                (
                    f"{card['card_path']}/gt_act_freq_mhz",
                    f"{card['card_path']}/gt/gt0/rps_act_freq_mhz",
                )
            )
            device.extra["freq_requested"] = first_existing(
                (
                    f"{card['card_path']}/gt_cur_freq_mhz",
                    f"{card['card_path']}/gt/gt0/rps_cur_freq_mhz",
                )
            )
            device.extra["freq_max"] = first_existing(
                (
                    f"{card['card_path']}/gt_max_freq_mhz",
                    f"{card['card_path']}/gt/gt0/rps_max_freq_mhz",
                )
            )
            device.extra["rc6"] = first_existing(
                (f"{card['card_path']}/power/rc6_residency_ms",)
            )
            device.extra["fds"] = {}
            devices.append(device)

        self._devices = devices
        return devices

    @staticmethod
    def _find_pmu_dir(pci_address: str | None) -> str | None:
        """
        Locate this device's i915 PMU directory.

        FIX-P11-INTEL-2: try the per-device name first so that on a
        multi-Intel-GPU host each device gets its own PMU rather than both
        silently reading the first one.
        """
        candidates: list[str] = []
        if pci_address:
            candidates.append("/sys/devices/i915_" + pci_address.replace(":", "_").replace(".", "_"))
            candidates.append("/sys/devices/i915_" + pci_address.replace(":", "_"))
        candidates.append("/sys/devices/i915")

        for path in candidates:
            if os.path.exists(os.path.join(path, "type")):
                return path
        return None

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open PMU fds for every device.  Never raises."""
        opened_any = False
        failures: list[str] = []

        for device in self._devices:
            pmu_dir = device.extra.get("pmu_dir")
            if pmu_dir is None:
                failures.append(f"{device.name}: no i915 PMU directory in /sys/devices")
                continue
            try:
                device.extra["fds"] = self._open_pmu_fds(pmu_dir)
                opened_any = True
                logger.info(
                    "intel: PMU counters opened for %s (%s) via %s.",
                    device.name, device.uid, pmu_dir,
                )
            except OSError as exc:
                device.extra["fds"] = {}
                if exc.errno in (errno.EACCES, errno.EPERM):
                    failures.append(
                        f"{device.name}: perf_event_open denied (errno "
                        f"{errno.errorcode.get(exc.errno, exc.errno)}). Needs "
                        "CAP_PERFMON and seccomp:unconfined — see docker-compose.yml."
                    )
                else:
                    failures.append(f"{device.name}: perf_event_open failed: {exc}")
            except Exception as exc:  # noqa: BLE001
                device.extra["fds"] = {}
                failures.append(f"{device.name}: i915 PMU unavailable: {exc}")

        if opened_any:
            self._active_source = PMU_ACTIVE_SOURCE
            self._error = None
            self._error_kind = None
        elif self._devices:
            self._active_source = FALLBACK_ACTIVE_SOURCE
            self._error_kind = "perf_open_failed"
            self._error = "; ".join(failures) or "i915 PMU counters unavailable."
            logger.error("intel: %s", self._error)

        if failures and opened_any:
            logger.warning("intel: some devices degraded: %s", "; ".join(failures))

        self._rebuild_scanner()

    def _rebuild_scanner(self) -> None:
        node_map = {
            d.render_node: d.uid for d in self._devices if d.render_node
        }
        self._scanner = FdinfoScanner(node_map) if node_map else None

    def _open_pmu_fds(self, pmu_dir: str) -> dict[str, int]:
        """All-or-nothing open of the four engine-busy counters."""
        pmu_type = int(read_text(os.path.join(pmu_dir, "type")).strip())
        cpu = _first_int(read_text(os.path.join(pmu_dir, "cpumask")))

        fds: dict[str, int] = {}
        try:
            for event_name, display_name in _PMU_EVENT_TO_DISPLAY.items():
                config, config1 = self._derive_pmu_config(pmu_dir, event_name)

                attr = _PerfEventAttr()
                attr.type = pmu_type
                attr.size = 128
                attr.config = config
                attr.config1 = config1
                attr.read_format = (
                    PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING
                )

                # i915 is a system-scoped PMU: pid=-1, cpu=<the PMU's cpu>.
                fds[display_name] = _perf_event_open(
                    attr, pid=-1, cpu=cpu, group_fd=-1, flags=PERF_FLAG_FD_CLOEXEC
                )
        except Exception:
            for fd in fds.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise
        return fds

    @staticmethod
    def _derive_pmu_config(pmu_dir: str, event_name: str) -> tuple[int, int]:
        """
        Derive (config, config1) using the standard perf sysfs encoding in
        ``<pmu>/events/<event>`` and ``<pmu>/format/<key>``.
        """
        spec = read_text(os.path.join(pmu_dir, "events", event_name)).strip()
        registers: dict[str, int] = {"config": 0, "config1": 0, "config2": 0}

        for term in spec.split(","):
            term = term.strip()
            if not term:
                continue
            key, _, val_str = term.partition("=")
            key = key.strip()
            value = _parse_int_auto(val_str) if val_str else 0

            if key in registers:
                registers[key] |= value
                continue

            register, low_bit = _parse_format_field(
                read_text(os.path.join(pmu_dir, "format", key))
            )
            registers[register] = registers.get(register, 0) | (value << low_bit)

        return registers["config"], registers["config1"]

    def close(self) -> None:
        for device in self._devices:
            for fd in (device.extra.get("fds") or {}).values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            device.extra["fds"] = {}

    # ── sampling ─────────────────────────────────────────────────────────

    def sample(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for device in self._devices:
            try:
                out[device.uid] = self._sample_device(device)
            except Exception as exc:  # noqa: BLE001 — one bad device must
                # not take down the others or the sample loop.
                logger.warning("intel: sampling %s failed: %s", device.uid, exc)
                out[device.uid] = {"error": str(exc)}
        return out

    def _sample_device(self, device: GpuDevice) -> dict:
        block: dict = {}

        engines = self._read_engine_busy(device)
        if engines:
            block["engines"] = engines
            # Overall utilisation for the summary row: the busiest engine,
            # not the sum — engines run concurrently, so summing can exceed
            # 100% and would be meaningless as a "GPU busy" figure.
            block["utilisation"] = {"value": round(max(e["busy"] for e in engines.values()), 2)}

        frequency = self._read_frequency(device)
        if frequency:
            block["frequency"] = frequency

        rc6 = self._read_rc6(device)
        if rc6:
            block["rc6"] = rc6

        return block

    def _read_engine_busy(self, device: GpuDevice) -> dict[str, dict[str, float]]:
        fds: dict[str, int] = device.extra.get("fds") or {}
        if not fds:
            return {}

        now = time.monotonic()
        uid = device.uid
        result: dict[str, dict[str, float]] = {}

        prev_wall = self._perf_prev_wall.get(uid)
        if prev_wall is None:
            # First tick: establish the baseline only.
            for name in ENGINE_DISPLAY_ORDER:
                fd = fds.get(name)
                if fd is None:
                    continue
                try:
                    self._perf_prev[(uid, name)] = _read_perf_counter(fd)
                except OSError as exc:
                    self._log_read_error_once(f"{uid}:{name}", exc)
                    continue
                result[name] = {"busy": 0.0}
            self._perf_prev_wall[uid] = now
            return result

        delta_wall_ns = (now - prev_wall) * 1e9
        self._perf_prev_wall[uid] = now

        for name in ENGINE_DISPLAY_ORDER:
            fd = fds.get(name)
            if fd is None:
                continue
            try:
                value, time_enabled, time_running = _read_perf_counter(fd)
            except OSError as exc:
                self._log_read_error_once(f"{uid}:{name}", exc)
                continue

            prev = self._perf_prev.get((uid, name))
            self._perf_prev[(uid, name)] = (value, time_enabled, time_running)
            if prev is None:
                result[name] = {"busy": 0.0}
                continue

            prev_value, prev_enabled, prev_running = prev
            d_value = wrap_delta(value, prev_value)
            d_enabled = wrap_delta(time_enabled, prev_enabled)
            d_running = wrap_delta(time_running, prev_running)

            # Scale for PMU multiplexing (time_running < time_enabled).
            scaled = d_value * (d_enabled / d_running) if d_running > 0 else 0.0
            busy = (scaled / delta_wall_ns) * 100.0 if delta_wall_ns > 0 else 0.0

            if (busy < -0.5 or busy > 100.5) and name not in self._range_warned:
                logger.warning(
                    "intel: %s %s busy%% out of range (%.2f) — clamping.",
                    uid, name, busy,
                )
                self._range_warned.add(name)

            result[name] = {"busy": round(max(0.0, min(100.0, busy)), 2)}

        return result

    def _log_read_error_once(self, key: str, exc: OSError) -> None:
        if key not in self._read_error_logged:
            logger.warning("intel: failed to read perf counter %s: %s", key, exc)
            self._read_error_logged.add(key)

    def _read_frequency(self, device: GpuDevice) -> dict[str, float]:
        freq: dict[str, float] = {}
        actual = read_int_file(device.extra.get("freq_actual"))
        if actual is not None:
            freq["actual"] = float(actual)
        requested = read_int_file(device.extra.get("freq_requested"))
        if requested is not None:
            freq["requested"] = float(requested)
        maximum = read_int_file(device.extra.get("freq_max"))
        if maximum is not None:
            freq["max"] = float(maximum)
        return freq

    def _read_rc6(self, device: GpuDevice) -> dict[str, float]:
        raw = read_int_file(device.extra.get("rc6"))
        if raw is None:
            return {}

        now = time.monotonic()
        prev = self._rc6_prev.get(device.uid)
        self._rc6_prev[device.uid] = (raw, now)
        if prev is None:
            return {"value": 0.0}

        prev_ms, prev_wall = prev
        delta_ms = raw - prev_ms
        delta_wall_ms = (now - prev_wall) * 1000.0
        if delta_wall_ms <= 0 or delta_ms < 0:
            return {"value": 0.0}
        return {"value": round(max(0.0, min(100.0, (delta_ms / delta_wall_ms) * 100.0)), 2)}

    # ── attribution ──────────────────────────────────────────────────────

    def attribution(self) -> dict[str, dict[int, dict[str, float]]] | None:
        if self._scanner is None:
            return None
        return compute_fdinfo_attribution(
            scanner=self._scanner,
            engine_map=FDINFO_ENGINE_MAP,
            engine_keys=ENGINE_KEYS,
            prev_clients=self._prev_clients,
            prev_wall_holder=self,
        )


# ── shared attribution maths (used by Intel and AMD) ─────────────────────


def compute_fdinfo_attribution(
    scanner: FdinfoScanner,
    engine_map: dict[str, str],
    engine_keys: tuple[str, ...],
    prev_clients: dict[str, dict[tuple[int, str], dict[str, int]]],
    prev_wall_holder: object,
) -> dict[str, dict[int, dict[str, float]]] | None:
    """
    Turn two consecutive fdinfo scans into per-PID engine percentages.

    ``prev_clients`` is mutated in place (it is the caller's state) and
    ``prev_wall_holder._prev_attrib_wall`` carries the Δwall clock.

    FIX-P11-DRM-5: DELTA SANITY GUARD.  A counter that goes backwards (a
    client that closed and had its id reused, or a genuine u64 wrap) used to
    surface as a full-scale 100% spike after wraparound arithmetic turned the
    negative delta into ~1.8e19 ns.  Any delta implying more than 100% of one
    engine over the elapsed wall time is now treated as a counter reset and
    contributes zero for that tick instead of a false spike.
    """
    clients = scanner.scan()
    if clients is None:
        return None

    now = time.monotonic()
    prev_wall = getattr(prev_wall_holder, "_prev_attrib_wall", None)
    result: dict[str, dict[int, dict[str, float]]] = {}

    if prev_wall is not None and now > prev_wall:
        delta_wall_ns = (now - prev_wall) * 1e9

        for uid, uid_clients in clients.items():
            prev_uid = prev_clients.get(uid, {})
            per_pid: dict[int, dict[str, float]] = {}

            for key, engines in uid_clients.items():
                pid, _client = key
                prev_engines = prev_uid.get(key)
                if prev_engines is None:
                    continue  # first sighting — no delta yet

                bucket = per_pid.setdefault(
                    pid, {k: 0.0 for k in engine_keys} | {"total": 0.0}
                )

                for raw_name, ns_value in engines.items():
                    internal = engine_map.get(raw_name)
                    if internal is None:
                        continue
                    prev_ns = prev_engines.get(raw_name)
                    if prev_ns is None:
                        continue

                    delta_ns = wrap_delta(ns_value, prev_ns)
                    # FIX-P11-DRM-5: reject implausible deltas outright.
                    if delta_ns > delta_wall_ns:
                        continue
                    pct = (delta_ns / delta_wall_ns) * 100.0
                    bucket[internal] += max(0.0, min(100.0, pct))

            for bucket in per_pid.values():
                for k in engine_keys:
                    bucket[k] = round(bucket[k], 2)
                bucket["total"] = round(sum(bucket[k] for k in engine_keys), 2)

            if per_pid:
                result[uid] = per_pid

    prev_clients.clear()
    prev_clients.update(clients)
    setattr(prev_wall_holder, "_prev_attrib_wall", now)
    return result
