# SPDX-License-Identifier: MIT
# Copyright (c) 2026 gpu-monitor contributors
"""
Vendor backend contract.

FIX-P11-BASE-1: NEW MODULE.  Before Prompt 9, ``GpuReader`` *was* the Intel
implementation — PMU fds, i915 sysfs paths, and the i915 fdinfo engine names
were all hard-coded into the one class.  Supporting NVIDIA and AMD alongside
Intel means splitting that into:

    GpuReader  — orchestration: the sample loop, history, health aggregation.
    GpuBackend — one per vendor: discovery, per-device metrics, attribution.

Every backend obeys the same three rules, because the dashboard has to keep
working on a host where two of the three vendors are absent:

  1. ``probe()`` NEVER raises.  Missing hardware, missing sysfs, missing
     shared library, denied permission — all return an empty device list or
     an unavailable backend, never an exception.
  2. ``sample()`` NEVER raises.  A backend that breaks mid-run degrades to
     emitting its ``error`` string; the other backends keep sampling.
  3. Every metric is OPTIONAL.  A backend emits only the keys it can
     actually measure on this host.  The frontend renders "n/a" for absent
     keys rather than showing a misleading zero.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GpuDevice:
    """
    One physical GPU, as discovered by a backend's ``probe()``.

    ``uid`` must be STABLE across restarts — the frontend keys its chart
    datasets and its per-GPU DOM sections on it, and a uid that changed on
    every restart would orphan chart state and duplicate sections.  The PCI
    address is the natural choice; NVML's UUID is used for NVIDIA because
    it survives PCI renumbering.
    """

    uid: str
    vendor: str
    name: str
    index: int

    # DRM/sysfs handles — present for Intel and AMD, usually None for the
    # NVIDIA proprietary driver (which exposes no useful DRM sysfs).
    card_path: str | None = None          # /sys/class/drm/card0
    render_node: str | None = None        # /dev/dri/renderD128
    pci_address: str | None = None        # 0000:00:02.0
    driver: str | None = None             # i915 / amdgpu / nvidia

    # Ordered display names of this device's engines, e.g.
    # ("Render/3D", "Blitter", "Video", "VideoEnhance").
    engine_names: tuple[str, ...] = ()

    # Internal (snake_case) engine keys used in the per-container payload,
    # e.g. ("render", "blitter", "video", "video_enhance").
    engine_keys: tuple[str, ...] = ()

    # Vendor-specific handles the backend needs on every tick (perf fds,
    # NVML device handle, hwmon paths). Never serialised.
    extra: dict = field(default_factory=dict, repr=False)

    def describe(self) -> dict:
        """Static, JSON-serialisable identity — sent once, not per sample."""
        return {
            "uid": self.uid,
            "vendor": self.vendor,
            "name": self.name,
            "index": self.index,
            "pci_address": self.pci_address,
            "driver": self.driver,
            "render_node": self.render_node,
            "engine_names": list(self.engine_names),
            "engine_keys": list(self.engine_keys),
        }


class GpuBackend(ABC):
    """
    One vendor's metric source.

    Lifecycle::

        backend = XBackend()
        devices = backend.probe()      # never raises
        backend.start()                # never raises; sets error on failure
        while running:
            per_device = backend.sample()      # {uid: {metrics...}}
            attrib     = backend.attribution() # {uid: {pid: {engine: ns}}}
        backend.close()
    """

    vendor: str = "unknown"

    def __init__(self) -> None:
        self._devices: list[GpuDevice] = []
        self._error: str | None = None
        self._error_kind: str | None = None
        self._active_source: str = "unavailable"

    # ── discovery ────────────────────────────────────────────────────────

    @abstractmethod
    def probe(self) -> list[GpuDevice]:
        """
        Detect this vendor's GPUs.  MUST NOT raise — return [] instead.
        Implementations should set ``self._error`` when they detect
        hardware but cannot read it (e.g. amdgpu present but
        gpu_busy_percent unreadable), and leave it None when the vendor is
        simply not present on this host.
        """

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open long-lived handles.  MUST NOT raise."""

    def close(self) -> None:
        """Release handles.  MUST NOT raise."""

    # ── sampling ─────────────────────────────────────────────────────────

    @abstractmethod
    def sample(self) -> dict[str, dict]:
        """
        Return ``{device_uid: {metric_block: value, ...}}`` for this tick.

        Recognised metric blocks (all optional):
            engines      {"Render/3D": {"busy": 0-100}, ...}
            utilisation  {"value": 0-100}     overall busy %
            frequency    {"actual": MHz, "requested": MHz}
            memory       {"used_mib": n, "total_mib": n, "used_pct": 0-100}
            rc6          {"value": 0-100}     Intel idle residency
            power        {"watts": n}
            temperature  {"celsius": n}

        MUST NOT raise.
        """

    def attribution(self) -> dict[str, dict[int, dict[str, float]]] | None:
        """
        Per-PID GPU usage for this vendor's devices, as PERCENTAGES already
        normalised to the sampling interval::

            {device_uid: {pid: {engine_key: pct, ..., "total": pct}}}

        Return None when this backend cannot do attribution at all.  The
        default implementation returns None; Intel/AMD override it with the
        shared DRM fdinfo scanner, NVIDIA with NVML process utilisation.
        """
        return None

    # ── introspection ────────────────────────────────────────────────────

    @property
    def devices(self) -> list[GpuDevice]:
        return self._devices

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def error_kind(self) -> str | None:
        return self._error_kind

    @property
    def active_source(self) -> str:
        return self._active_source

    def status(self) -> dict:
        """JSON-serialisable backend status for /health."""
        return {
            "vendor": self.vendor,
            "device_count": len(self._devices),
            "active_source": self._active_source,
            "error": self._error,
            "error_kind": self._error_kind,
            "devices": [d.describe() for d in self._devices],
        }

    # ── helpers shared by concrete backends ──────────────────────────────

    def _fail(self, kind: str, message: str) -> None:
        """Record a backend-level failure without raising."""
        self._error_kind = kind
        self._error = message
        self._active_source = "unavailable"
        logger.warning("%s backend: %s", self.vendor, message)
