# SPDX-License-Identifier: MIT
# Copyright (c) 2026 gpu-monitor contributors
"""
FastAPI GPU monitor backend.

Serves:
  GET  /health     — liveness, per-backend GPU status, resolver status
  GET  /api/gpus   — static per-GPU identity (uid, vendor, name, engines)
  WS   /ws         — live + historical metric stream
  GET  /  …        — static dashboard from app/static/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from config import load_config
from container_resolver import ContainerResolver
from gpu_reader import FALLBACK_ACTIVE_SOURCE, PMU_ACTIVE_SOURCE, GpuReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_reader: GpuReader | None = None
_resolver: ContainerResolver | None = None
_config = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _reader, _resolver

    # ContainerResolver starts BEFORE GpuReader so the reader has a working
    # (or at least attempted) resolver from its very first sample.
    if _config.container_attribution:
        _resolver = ContainerResolver()
        try:
            await _resolver.start()
            logger.info("ContainerResolver started.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ContainerResolver.start() raised unexpectedly: %s — "
                "continuing without per-container attribution.",
                exc,
            )
            _resolver = None
    else:
        logger.info(
            "GPU_CONTAINER_ATTRIBUTION=false — skipping Docker resolver and "
            "the /proc scan entirely."
        )
        _resolver = None

    _reader = GpuReader(container_resolver=_resolver, config=_config)
    # GpuReader.start() probes every enabled backend and never raises, so
    # /health is truthful from the first request regardless of what's present.
    await _reader.start()
    logger.info("GpuReader started (active_source=%s).", _reader.get_active_source())

    try:
        yield
    finally:
        # Reader stops first: it depends on the resolver, not the reverse.
        if _reader is not None:
            await _reader.stop()
            logger.info("GpuReader stopped.")
        if _resolver is not None:
            await _resolver.stop()
            logger.info("ContainerResolver stopped.")


app = FastAPI(title="GPU Monitor", lifespan=lifespan)

# ─────────────────────────────────────────────────────────────── REST ────────


# Browsers auto-request /favicon.ico. Without an explicit route it falls
# through to StaticFiles and 404s on every page load. Must precede the mount.
@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/gpus")
async def api_gpus() -> dict:
    """
    FIX-P11-MAIN-1: NEW ENDPOINT.  Static per-GPU identity, split out from
    the sample stream because it never changes.  The frontend needs engine
    names and colours to build each GPU's chart *before* the first sample
    arrives — otherwise the page shows an empty grey box for up to a second
    on every load, and on a host where a backend is failing it would show
    nothing at all with no explanation.
    """
    return {
        "gpus": _reader.get_devices() if _reader is not None else [],
        "config": _config.describe(),
    }


@app.get("/health")
async def health() -> dict:
    """
    Liveness plus which metric source is active per vendor.

    FIX-P11-MAIN-2: ``gpu_detected`` no longer means "the i915 PMU opened".
    It now means "at least one enabled backend found at least one device",
    which is the only definition that makes sense once a host can have an
    NVIDIA card and no Intel iGPU at all.  The old Intel-specific detail is
    still available — it just moved into ``backends[]``.
    """
    device_present = os.path.exists("/dev/dri/renderD128")
    has_data = _reader is not None and bool(_reader.get_latest())

    if _reader is not None:
        active_source = _reader.get_active_source()
        gpu_error = _reader.get_gpu_error()
        gpu_error_kind = _reader.get_gpu_error_kind()
        backends = _reader.get_backend_status()
        gpu_detected = _reader.has_devices()
        gpu_count = sum(b["device_count"] for b in backends)
    else:
        active_source = FALLBACK_ACTIVE_SOURCE
        gpu_error = "GpuReader not initialised."
        gpu_error_kind = "reader_unavailable"
        backends = []
        gpu_detected = False
        gpu_count = 0

    if _resolver is not None:
        resolver_status = _resolver.get_status()
    else:
        resolver_status = {
            "available": False,
            "degraded": False,
            "container_count": 0,
            "gpu_container_count": 0,
            "gpu_containers_by_access": {"drm": [], "nvidia": []},
            "pid_count": 0,
            "last_refresh_ago_s": None,
            "last_error": None,
        }

    return {
        "status": "ok",
        "gpu_detected": gpu_detected,
        "gpu_count": gpu_count,
        "device_present": device_present,
        "has_data": has_data,
        "active_source": active_source,
        "gpu_error": gpu_error,
        "gpu_error_kind": gpu_error_kind,
        "backends": backends,
        "config": _config.describe(),
        "container_resolver": resolver_status,
        # Retained so anything scraping the old field keeps parsing.
        "pmu_source_name": PMU_ACTIVE_SOURCE,
    }


# ───────────────────────────────────────────────────────────── WebSocket ─────


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Per-connection handler.

    Protocol:
      → {"type": "devices", "data": {...}}          (on connect, first)
      → {"type": "history", "data": [...samples]}   (on connect, second)
      → {"type": "sample",  "data": {...sample}}    (every interval)

    FIX-P11-MAIN-3: the ``devices`` frame is sent BEFORE ``history`` so the
    client can build its per-GPU sections and then replay history into them.
    Sending history first would mean replaying samples for GPUs the client
    doesn't know how to render yet.
    """
    await websocket.accept()
    logger.info("WebSocket client connected: %s", websocket.client)

    try:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "devices",
                    "data": {
                        "gpus": _reader.get_devices() if _reader is not None else [],
                        "config": _config.describe(),
                    },
                }
            )
        )

        history = _reader.get_history() if _reader is not None else []
        await websocket.send_text(json.dumps({"type": "history", "data": history}))

        while True:
            await asyncio.sleep(_config.sample_interval_s)
            sample = _reader.get_latest() if _reader is not None else {}
            await websocket.send_text(json.dumps({"type": "sample", "data": sample}))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", websocket.client)
    except Exception as exc:  # noqa: BLE001
        logger.warning("WebSocket send error (%s): %s", websocket.client, exc)


# ──────────────────────────────────────── Static files — MUST be last! ───────
#
# StaticFiles uses a catch-all route. Mounting it BEFORE the WebSocket or REST
# decorators would shadow /ws, /health and /api/gpus.

app.mount("/", StaticFiles(directory="static", html=True), name="static")

# ────────────────────────────────────────────────────────── CLI entry-point ──

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8088, log_level="info")
