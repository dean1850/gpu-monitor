# SPDX-License-Identifier: MIT
# Copyright (c) 2026 gpu-monitor contributors
"""
Multi-GPU metrics orchestrator.

FIX-P11-GPU-1: REWRITTEN AS AN ORCHESTRATOR.  Up to Prompt 9 this module
*was* the Intel implementation — perf fds, i915 sysfs paths and i915 fdinfo
engine names were all inlined here.  Prompt 9 adds NVIDIA and AMD, so the
vendor-specific code moved into ``backends/`` and what remains here is the
part that was never vendor-specific:

    • the 1 Hz sample loop and its worker-thread offload
    • rolling history
    • PID -> container-name resolution and bucketing
    • health/status aggregation across backends

Emitted sample shape::

    {
      "timestamp": "<ISO8601>",
      "gpus": [
        {
          "uid": "intel:0000:00:02.0",
          "vendor": "intel",
          "name": "Intel UHD Graphics 630",
          "index": 0,
          "engines":     {"Render/3D": {"busy": 12.3}, ...},   # optional
          "utilisation": {"value": 12.3},                      # optional
          "frequency":   {"actual": 350.0, "requested": 350.0},# optional
          "rc6":         {"value": 88.1},                      # optional
          "memory":      {"used_mib": ..., "total_mib": ...},  # optional
          "power":       {"watts": 12.4},                      # optional
          "temperature": {"celsius": 43.0, "sensor": "edge",   # optional
                          "channels": {"edge": 43.0, "junction": 51.0}},
          "fan":         {"rpm": 1100.0, "pct": 31.4},         # optional
          "voltage":     {"gfx_mv": 806},                      # optional
          "power_state": "active" | "suspended",               # optional
          "containers":  {"jellyfin": {...}, "host": {...}}    # optional
        }
      ],

      # FIX-P11-GPU-2: pre-Prompt-9 flat mirror of gpus[0], emitted unless
      # GPU_LEGACY_SAMPLE_FIELDS=false. It costs a few hundred bytes and
      # means a stale browser tab, a bookmarked script, or a half-rolled-back
      # deploy keeps working instead of silently rendering nothing.
      "engines": ..., "frequency": ..., "rc6": ..., "containers": ...
    }
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from backends import GpuBackend, build_backends
from config import Config, load_config

if TYPE_CHECKING:
    from container_resolver import ContainerResolver

logger = logging.getLogger(__name__)

# Retained for backwards compatibility with main.py's /health import.
PMU_ACTIVE_SOURCE = "i915_pmu_perf"
FALLBACK_ACTIVE_SOURCE = "sysfs_freq_rc6_only"

# The bucket PIDs land in when the resolver cannot map them to a container.
HOST_BUCKET = "host"


class GpuReader:
    """
    Background reader across every enabled GPU backend.

    Usage::

        reader = GpuReader(container_resolver=resolver)
        await reader.start()
        sample = reader.get_latest()
        await reader.stop()
    """

    def __init__(
        self,
        container_resolver: ContainerResolver | None = None,
        config: Config | None = None,
    ) -> None:
        self._config = config or load_config()
        self._resolver = container_resolver
        self._backends: list[GpuBackend] = []
        self._latest: dict = {}
        self._history: deque[dict] = deque(maxlen=self._config.history_size)
        self._task: asyncio.Task[None] | None = None
        self._running = False

    # ── public API ───────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Probe backends, open handles, launch the sample loop.

        FIX-P11-GPU-3: never raises.  A host with no supported GPU at all
        still gets a running app that serves the dashboard and explains the
        situation through /health, rather than a container that crash-loops.
        """
        try:
            self._backends = await asyncio.to_thread(build_backends, self._config)
        except Exception as exc:  # noqa: BLE001
            logger.error("GpuReader: backend construction failed: %s", exc)
            self._backends = []

        for backend in self._backends:
            try:
                await asyncio.to_thread(backend.start)
            except Exception as exc:  # noqa: BLE001
                logger.error("GpuReader: %s backend start failed: %s", backend.vendor, exc)

        total = sum(len(b.devices) for b in self._backends)
        if total == 0:
            logger.warning(
                "GpuReader: no GPUs are being monitored. Check /health for the "
                "per-vendor reason, and confirm /dev/dri is passed through."
            )
        else:
            logger.info(
                "GpuReader: monitoring %d GPU(s) across %d backend(s).",
                total, len(self._backends),
            )

        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        for backend in self._backends:
            try:
                await asyncio.to_thread(backend.close)
            except Exception as exc:  # noqa: BLE001
                logger.warning("GpuReader: %s backend close failed: %s", backend.vendor, exc)

    def get_latest(self) -> dict:
        return self._latest

    def get_history(self) -> list[dict]:
        return list(self._history)

    def get_devices(self) -> list[dict]:
        """Static per-GPU identity, sent once on WebSocket connect."""
        devices: list[dict] = []
        for backend in self._backends:
            for device in backend.devices:
                entry = device.describe()
                entry["backend_error"] = backend.error
                devices.append(entry)
        return devices

    @property
    def config(self) -> Config:
        return self._config

    # ── health ───────────────────────────────────────────────────────────

    def get_active_source(self) -> str:
        """
        Aggregate source label.

        Single GPU: that backend's source (e.g. "i915_pmu_perf") — identical
        to pre-Prompt-9 behaviour.  Multiple: a comma-joined summary.
        """
        sources = [b.active_source for b in self._backends if b.devices]
        if not sources:
            return FALLBACK_ACTIVE_SOURCE
        if len(sources) == 1:
            return sources[0]
        return ",".join(sorted(set(sources)))

    def get_gpu_error(self) -> str | None:
        errors = [f"{b.vendor}: {b.error}" for b in self._backends if b.error]
        return "; ".join(errors) if errors else None

    def get_gpu_error_kind(self) -> str | None:
        kinds = [b.error_kind for b in self._backends if b.error_kind]
        return ",".join(sorted(set(kinds))) if kinds else None

    def get_backend_status(self) -> list[dict]:
        return [b.status() for b in self._backends]

    def has_devices(self) -> bool:
        return any(b.devices for b in self._backends)

    # ── sample loop ──────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        interval = self._config.sample_interval_s
        while self._running:
            try:
                # FIX-P11-GPU-4: the whole sample build runs in a worker
                # thread. With pid:host the /proc walk can touch hundreds of
                # PIDs, and with several backends that cost multiplies —
                # blocking the event loop would stall /health and every open
                # WebSocket for the duration.
                sample = await asyncio.to_thread(self._build_sample)
                self._latest = sample
                self._history.append(sample)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error("GpuReader: failed to build sample: %s", exc)

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise

    def _build_sample(self) -> dict:
        """Build one JSON-serialisable sample. Runs in a worker thread."""
        gpu_blocks: list[dict] = []

        for backend in self._backends:
            if not backend.devices:
                continue

            # Attribution BEFORE sample: the AMD backend derives its engine
            # breakdown from this tick's fdinfo pass.  In the old order the
            # engine chart showed the previous tick's attribution while the
            # container rows showed this one's, so the two views — documented
            # as guaranteed-consistent — were always one interval apart.
            attribution = self._attribution_for(backend)

            try:
                metrics = backend.sample()
            except Exception as exc:  # noqa: BLE001
                logger.warning("GpuReader: %s sample() failed: %s", backend.vendor, exc)
                metrics = {}

            for device in backend.devices:
                block: dict = {
                    "uid": device.uid,
                    "vendor": device.vendor,
                    "name": device.name,
                    "index": device.index,
                }
                block.update(metrics.get(device.uid) or {})

                if attribution is not None:
                    containers = self._bucket_by_container(
                        attribution.get(device.uid) or {},
                        tuple(device.engine_keys),
                        device.vendor,
                    )
                    if containers is not None:
                        block["containers"] = containers

                gpu_blocks.append(block)

        sample: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gpus": gpu_blocks,
        }

        if self._config.legacy_sample_fields and gpu_blocks:
            primary = gpu_blocks[0]
            for key in ("engines", "frequency", "rc6", "containers"):
                if key in primary:
                    sample[key] = primary[key]

        return sample

    def _attribution_for(
        self, backend: GpuBackend
    ) -> dict[str, dict[int, dict[str, float]]] | None:
        if not self._config.container_attribution:
            return None
        try:
            return backend.attribution()
        except Exception as exc:  # noqa: BLE001
            logger.warning("GpuReader: %s attribution() failed: %s", backend.vendor, exc)
            return None

    # ── PID -> container bucketing ───────────────────────────────────────

    def _bucket_by_container(
        self,
        per_pid: dict[int, dict[str, float]],
        engine_keys: tuple[str, ...],
        vendor: str,
    ) -> dict[str, dict[str, float]] | None:
        """
        Fold per-PID usage into per-container buckets, then merge in every
        GPU-capable container at 0% so idle ones stay visible.

        Answering "which of my containers *can* use this GPU?" is as useful
        as "which one is using it right now?" — especially on a homelab where
        only jellyfin transcodes, and only sometimes.
        """
        gpu_containers = self._gpu_containers_for(vendor)

        if not per_pid and not gpu_containers:
            return None

        containers: dict[str, dict[str, float]] = {}

        for pid, engines in per_pid.items():
            name = HOST_BUCKET
            if self._resolver is not None:
                try:
                    name = self._resolver.resolve_pid(pid) or HOST_BUCKET
                except Exception:  # noqa: BLE001
                    name = HOST_BUCKET

            bucket = containers.setdefault(
                name, {k: 0.0 for k in engine_keys} | {"total": 0.0}
            )
            for key in engine_keys:
                bucket[key] += float(engines.get(key, 0.0))
            if "memory_mib" in engines:
                bucket["memory_mib"] = round(
                    bucket.get("memory_mib", 0.0) + engines["memory_mib"], 1
                )

        for bucket in containers.values():
            for key in engine_keys:
                bucket[key] = round(min(100.0, bucket[key]), 2)
            bucket["total"] = round(sum(bucket[k] for k in engine_keys), 2)

        for name in gpu_containers:
            containers.setdefault(
                name, {k: 0.0 for k in engine_keys} | {"total": 0.0}
            )

        return containers

    def _gpu_containers_for(self, vendor: str) -> set[str]:
        """GPU-capable container names relevant to *vendor*."""
        if self._resolver is None:
            return set()
        try:
            by_vendor = self._resolver.list_gpu_containers_by_vendor()
        except Exception:  # noqa: BLE001
            return set()

        names: set[str] = set()
        for name, vendors in by_vendor.items():
            # "drm" means the container has /dev/dri, which serves both Intel
            # and AMD; NVIDIA containers are flagged separately.
            if vendor in vendors or ("drm" in vendors and vendor in ("intel", "amd")):
                names.add(name)
        return names
