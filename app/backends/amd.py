# SPDX-License-Identifier: MIT
# Copyright (c) 2026 gpu-monitor contributors
"""
AMD backend — amdgpu sysfs + hwmon, with fdinfo-derived engine breakdown.

FIX-P11-AMD-1: NEW BACKEND.  Written against the amdgpu kernel driver's
documented sysfs ABI, NOT against hardware — the reference host for this
project has an Intel iGPU only.  Every read is therefore treated as
optional: a metric whose sysfs file is absent or unreadable is simply not
emitted, and the dashboard shows "n/a" instead of a fabricated zero.  If
you are the first person to run this against a real Radeon, the fastest way
to see what did and didn't bind is ``GET /health`` — the amd backend block
lists exactly which sysfs paths resolved per device.

Metric sources (all under ``/sys/class/drm/cardN/device/``):

    gpu_busy_percent      overall GPU busy %          (authoritative)
    mem_busy_percent      memory-controller busy %
    mem_info_vram_used    VRAM in use, bytes
    mem_info_vram_total   VRAM fitted, bytes
    pp_dpm_sclk           shader-clock DPM table; the active state is
                          flagged with a trailing '*'
    hwmon/hwmonN/         power1_average (µW), temp1_input (m°C),
                          freq1_input (Hz)

Engine breakdown comes from ``/proc/<pid>/fdinfo`` (``drm-engine-gfx``,
``-compute``, ``-dec``, ``-enc``) summed across clients, because amdgpu
exposes no aggregate per-engine counter the way i915's PMU does.  That sum
is the same number the per-container rows are built from, so the two views
are guaranteed consistent with each other.
"""

from __future__ import annotations

import glob
import logging
import os
import re

from .base import GpuBackend, GpuDevice
from .drm import (
    FdinfoScanner,
    device_display_name,
    enumerate_drm_cards,
    first_existing,
    read_int_file,
    read_text,
)
from .intel import compute_fdinfo_attribution

logger = logging.getLogger(__name__)

SYSFS_ACTIVE_SOURCE = "amdgpu_sysfs"

# Display names, in the order they appear in the legend.
ENGINE_DISPLAY_ORDER: tuple[str, ...] = ("Graphics", "Compute", "Decode", "Encode")

# fdinfo raw engine name -> internal per-container key.
# FIX-P11-AMD-2: amdgpu spells multi-instance engines "dec", "dec_1",
# "enc_1", "vcn_dec" etc. depending on kernel version and ASIC.  All the
# variants of one class are folded onto a single key so a two-VCN card
# doesn't produce two legend entries that mean the same thing.
FDINFO_ENGINE_MAP: dict[str, str] = {
    "gfx": "graphics",
    "compute": "compute",
    "dec": "decode",
    "dec_1": "decode",
    "vcn_dec": "decode",
    "jpeg": "decode",
    "enc": "encode",
    "enc_1": "encode",
    "vcn_enc": "encode",
}
ENGINE_KEYS: tuple[str, ...] = ("graphics", "compute", "decode", "encode")

_KEY_TO_DISPLAY: dict[str, str] = {
    "graphics": "Graphics",
    "compute": "Compute",
    "decode": "Decode",
    "encode": "Encode",
}

_SCLK_ACTIVE_RE = re.compile(r"^\s*\d+:\s*(\d+)\s*Mhz\s*\*", re.IGNORECASE | re.MULTILINE)


class AmdBackend(GpuBackend):
    vendor = "amd"

    def __init__(self, forced: bool = False) -> None:
        super().__init__()
        self._forced = forced
        self._scanner: FdinfoScanner | None = None
        self._prev_clients: dict[str, dict[tuple[int, str], dict[str, int]]] = {}
        self._prev_attrib_wall: float | None = None
        # Last attribution result, reused to synthesise aggregate engine %.
        self._last_attrib: dict[str, dict[int, dict[str, float]]] = {}

    # ── discovery ────────────────────────────────────────────────────────

    def probe(self) -> list[GpuDevice]:
        try:
            cards = enumerate_drm_cards()
        except Exception as exc:  # noqa: BLE001
            self._fail("probe_failed", f"DRM enumeration failed: {exc}")
            return []

        amd_cards = [c for c in cards if c["vendor"] == "amd"]

        if not amd_cards:
            if self._forced:
                self._fail(
                    "no_device",
                    "GPU_ENABLE_AMD=true but no AMD DRM card was found. Check "
                    "that /dev/dri is passed through and the amdgpu driver is "
                    "bound (lsmod | grep amdgpu).",
                )
            return []

        devices: list[GpuDevice] = []
        for index, card in enumerate(amd_cards):
            uid = f"amd:{card['pci_address'] or card['card']}"
            device_dir = os.path.join(card["card_path"], "device")

            device = GpuDevice(
                uid=uid,
                vendor="amd",
                name=device_display_name(card["vendor_id"], card["device_id"], card["driver"]),
                index=index,
                card_path=card["card_path"],
                render_node=card["render_node"],
                pci_address=card["pci_address"],
                driver=card["driver"],
                engine_names=ENGINE_DISPLAY_ORDER,
                engine_keys=ENGINE_KEYS,
            )

            device.extra["busy"] = first_existing((os.path.join(device_dir, "gpu_busy_percent"),))
            device.extra["mem_busy"] = first_existing(
                (os.path.join(device_dir, "mem_busy_percent"),)
            )
            device.extra["vram_used"] = first_existing(
                (os.path.join(device_dir, "mem_info_vram_used"),)
            )
            device.extra["vram_total"] = first_existing(
                (os.path.join(device_dir, "mem_info_vram_total"),)
            )
            device.extra["sclk"] = first_existing((os.path.join(device_dir, "pp_dpm_sclk"),))
            device.extra.update(self._find_hwmon(device_dir))
            devices.append(device)

            logger.info(
                "amd: found %s (%s) — busy=%s vram=%s hwmon_power=%s hwmon_temp=%s",
                device.name, uid,
                bool(device.extra["busy"]), bool(device.extra["vram_total"]),
                bool(device.extra.get("power")), bool(device.extra.get("temp")),
            )

            if device.extra["busy"] is None and self._forced:
                self._fail(
                    "sysfs_incomplete",
                    f"{device.name}: gpu_busy_percent is missing. This usually "
                    "means an older amdgpu build or an APU that does not export "
                    "utilisation. Frequency/VRAM may still work.",
                )

        self._devices = devices
        self._active_source = SYSFS_ACTIVE_SOURCE
        return devices

    @staticmethod
    def _find_hwmon(device_dir: str) -> dict[str, str | None]:
        """Resolve the hwmon instance attached to this GPU, if any."""
        out: dict[str, str | None] = {"power": None, "temp": None, "hwmon_freq": None}
        for hwmon_dir in sorted(glob.glob(os.path.join(device_dir, "hwmon", "hwmon*"))):
            out["power"] = out["power"] or first_existing(
                (
                    os.path.join(hwmon_dir, "power1_average"),
                    os.path.join(hwmon_dir, "power1_input"),
                )
            )
            out["temp"] = out["temp"] or first_existing(
                (os.path.join(hwmon_dir, "temp1_input"),)
            )
            out["hwmon_freq"] = out["hwmon_freq"] or first_existing(
                (os.path.join(hwmon_dir, "freq1_input"),)
            )
        return out

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        node_map = {d.render_node: d.uid for d in self._devices if d.render_node}
        self._scanner = FdinfoScanner(node_map) if node_map else None
        if self._devices and self._scanner is None:
            logger.warning(
                "amd: no render node resolved for any AMD card — per-container "
                "attribution and the engine breakdown will be unavailable. "
                "Overall utilisation from gpu_busy_percent still works."
            )

    # ── sampling ─────────────────────────────────────────────────────────

    def sample(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for device in self._devices:
            try:
                out[device.uid] = self._sample_device(device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("amd: sampling %s failed: %s", device.uid, exc)
                out[device.uid] = {"error": str(exc)}
        return out

    def _sample_device(self, device: GpuDevice) -> dict:
        block: dict = {}

        busy = read_int_file(device.extra.get("busy"))
        if busy is not None:
            block["utilisation"] = {"value": float(max(0, min(100, busy)))}

        # Engine breakdown, summed from the last fdinfo attribution pass.
        engines = self._aggregate_engines(device.uid)
        if engines:
            block["engines"] = engines

        frequency = self._read_frequency(device)
        if frequency:
            block["frequency"] = frequency

        memory = self._read_memory(device)
        if memory:
            block["memory"] = memory

        mem_busy = read_int_file(device.extra.get("mem_busy"))
        if mem_busy is not None:
            block.setdefault("memory", {})["busy_pct"] = float(max(0, min(100, mem_busy)))

        power_uw = read_int_file(device.extra.get("power"))
        if power_uw is not None:
            block["power"] = {"watts": round(power_uw / 1_000_000.0, 2)}

        temp_mc = read_int_file(device.extra.get("temp"))
        if temp_mc is not None:
            block["temperature"] = {"celsius": round(temp_mc / 1000.0, 1)}

        return block

    def _aggregate_engines(self, uid: str) -> dict[str, dict[str, float]]:
        """
        Sum the per-PID engine percentages into a whole-GPU engine breakdown.

        FIX-P11-AMD-3: this is derived, not measured.  amdgpu has no
        aggregate per-engine counter, so the honest options were (a) omit the
        engine chart for AMD entirely, or (b) sum what fdinfo reports per
        client.  (b) wins because it is exactly what the per-container rows
        already show, so the two views can never disagree — and the
        authoritative whole-GPU number (gpu_busy_percent) is still displayed
        separately as "utilisation".
        """
        per_pid = self._last_attrib.get(uid)
        if not per_pid:
            return {}

        totals = {key: 0.0 for key in ENGINE_KEYS}
        for bucket in per_pid.values():
            for key in ENGINE_KEYS:
                totals[key] += bucket.get(key, 0.0)

        return {
            _KEY_TO_DISPLAY[key]: {"busy": round(max(0.0, min(100.0, value)), 2)}
            for key, value in totals.items()
        }

    def _read_frequency(self, device: GpuDevice) -> dict[str, float]:
        freq: dict[str, float] = {}

        # Preferred: hwmon freq1_input, in Hz.
        hz = read_int_file(device.extra.get("hwmon_freq"))
        if hz is not None and hz > 0:
            freq["actual"] = round(hz / 1_000_000.0, 1)

        # Fallback / supplement: the starred entry in the DPM table.
        sclk_path = device.extra.get("sclk")
        if sclk_path:
            try:
                text = read_text(sclk_path)
            except OSError:
                text = ""
            match = _SCLK_ACTIVE_RE.search(text)
            if match:
                freq.setdefault("actual", float(match.group(1)))
            levels = [int(m) for m in re.findall(r":\s*(\d+)\s*Mhz", text, re.IGNORECASE)]
            if levels:
                freq["max"] = float(max(levels))

        return freq

    def _read_memory(self, device: GpuDevice) -> dict[str, float]:
        used = read_int_file(device.extra.get("vram_used"))
        total = read_int_file(device.extra.get("vram_total"))
        memory: dict[str, float] = {}
        if used is not None:
            memory["used_mib"] = round(used / (1024 * 1024), 1)
        if total is not None and total > 0:
            memory["total_mib"] = round(total / (1024 * 1024), 1)
            if used is not None:
                memory["used_pct"] = round((used / total) * 100.0, 2)
        return memory

    # ── attribution ──────────────────────────────────────────────────────

    def attribution(self) -> dict[str, dict[int, dict[str, float]]] | None:
        if self._scanner is None:
            return None
        result = compute_fdinfo_attribution(
            scanner=self._scanner,
            engine_map=FDINFO_ENGINE_MAP,
            engine_keys=ENGINE_KEYS,
            prev_clients=self._prev_clients,
            prev_wall_holder=self,
        )
        # Cached so _aggregate_engines() can reuse it on the same tick
        # without a second /proc walk.
        self._last_attrib = result or {}
        return result
