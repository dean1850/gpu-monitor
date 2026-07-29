# SPDX-License-Identifier: MIT
# Copyright (c) 2026 gpu-monitor contributors
"""
Backend registry.

FIX-P11-REG-1: NEW MODULE.  ``build_backends()`` is the single place that
turns configuration into live backend objects.  It exists so that
``GpuReader`` never imports a vendor module directly — which matters because
importing ``nvidia`` on a host with no NVIDIA userspace must be free, and it
is: the ctypes ``CDLL`` load happens inside ``probe()``, not at import time.
"""

from __future__ import annotations

import logging

from config import VENDOR_AMD, VENDOR_INTEL, VENDOR_NVIDIA, Config

from .base import GpuBackend, GpuDevice

logger = logging.getLogger(__name__)

__all__ = ["GpuBackend", "GpuDevice", "build_backends"]


def build_backends(config: Config) -> list[GpuBackend]:
    """
    Instantiate and probe every enabled backend, in display order.

    Order is Intel, AMD, NVIDIA — deliberately fixed rather than
    discovery-ordered so the dashboard's GPU sections don't reshuffle
    between restarts on a mixed-vendor host.

    Never raises: a backend whose import or probe explodes is logged and
    skipped, and the remaining backends still come up.
    """
    backends: list[GpuBackend] = []

    for vendor in (VENDOR_INTEL, VENDOR_AMD, VENDOR_NVIDIA):
        if not config.is_enabled(vendor):
            logger.info("backends: %s disabled by configuration — not probing.", vendor)
            continue

        forced = config.is_forced(vendor)

        try:
            backend = _instantiate(vendor, forced)
        except Exception as exc:  # noqa: BLE001
            logger.error("backends: could not construct %s backend: %s", vendor, exc)
            continue

        try:
            devices = backend.probe()
        except Exception as exc:  # noqa: BLE001
            # probe() is contractually non-raising; this is belt and braces
            # so a bug in one vendor cannot stop the app from starting.
            logger.error("backends: %s probe raised unexpectedly: %s", vendor, exc)
            continue

        if devices:
            logger.info(
                "backends: %s enabled with %d device(s): %s",
                vendor, len(devices), ", ".join(d.name for d in devices),
            )
            backends.append(backend)
        elif backend.error:
            # Forced-on but absent: keep the backend so /health can explain.
            logger.warning("backends: %s reported: %s", vendor, backend.error)
            backends.append(backend)
        else:
            logger.info("backends: %s found no devices — skipping.", vendor)

    return backends


def _instantiate(vendor: str, forced: bool) -> GpuBackend:
    if vendor == VENDOR_INTEL:
        from .intel import IntelBackend

        return IntelBackend(forced=forced)
    if vendor == VENDOR_AMD:
        from .amd import AmdBackend

        return AmdBackend(forced=forced)
    if vendor == VENDOR_NVIDIA:
        from .nvidia import NvidiaBackend

        return NvidiaBackend(forced=forced)
    raise ValueError(f"unknown vendor {vendor!r}")
