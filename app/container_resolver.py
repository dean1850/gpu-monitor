# SPDX-License-Identifier: MIT
# Copyright (c) 2026 gpu-monitor contributors
"""
Async Docker container → PID resolver.

Maintains a background-refreshed map { host_pid: container_name } by polling
the Docker daemon via the local Unix socket.  Used by GpuReader to attribute
GPU client PIDs (always host-namespace PIDs, since the container runs with
`pid: host`) to the container they belong to.

FIX-P10-CR-1: also maintains a set of container names that have /dev/dri
passed through (via HostConfig.Devices) — exposed via :meth:`list_gpu_containers`
so GpuReader can list *all* GPU-capable containers in row 3, not only the
ones with a currently open render-node fd.  This is the source of truth for
"any container that could use the GPU" so that idle containers (e.g. jellyfin
when no transcode is running) still appear at 0% rather than disappearing.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

_REFRESH_INTERVAL: float = 3.0
_MAX_CONSECUTIVE_FAILURES: int = 10


class ContainerResolver:
    """
    Async resolver that maps host-namespace PIDs to Docker container names.

    asyncio is single-threaded; Python dict assignment is atomic in CPython,
    so the single-assignment replacement of ``_pid_map`` is safe for readers
    without an explicit asyncio.Lock.

    Usage::

        resolver = ContainerResolver()
        await resolver.start()          # no-op (just logs) if socket absent
        name = resolver.resolve_pid(12345)
        status = resolver.get_status()
        await resolver.stop()
    """

    def __init__(
        self,
        socket_path: str = "/var/run/docker.sock",
        refresh_interval: float = _REFRESH_INTERVAL,
    ) -> None:
        self._socket_path = socket_path
        self._refresh_interval = refresh_interval

        # FIX-P6-CR-1: always replaced via a single atomic assignment so
        # resolve_pid() never observes a half-built map.
        # FIX-P6-CR-8: stores only str values (container names) keyed by int
        # PIDs — no container objects or aiodocker handles leak in here.
        self._pid_map: dict[int, str] = {}

        # FIX-P10-CR-1: set of container names that have GPU access of any
        # kind.  Replaced atomically alongside _pid_map.  Drives the "show
        # every GPU-capable container at 0% if idle" behaviour.
        self._gpu_containers: set[str] = set()

        # FIX-P11-CR-1: the same information, split by which GPU family the
        # container can actually reach.  Values are tags, not vendor names:
        #   "drm"    -> has /dev/dri, so it can use an Intel or AMD GPU
        #   "nvidia" -> reserved an NVIDIA device through the container runtime
        # A container can hold both.  Without this split, a multi-vendor host
        # would list every NVIDIA-only container under the Intel GPU and vice
        # versa, which is worse than not listing them at all.
        self._gpu_containers_by_vendor: dict[str, set[str]] = {}

        self._available: bool = False
        self._degraded: bool = False
        self._consecutive_failures: int = 0
        self._last_refresh_time: float | None = None
        self._last_error: str | None = None
        self._container_count: int = 0

        self._task: asyncio.Task[None] | None = None
        self._docker: object | None = None   # aiodocker.Docker instance

    # ─────────────────────────────────────────────────────── public API ──────

    async def start(self) -> None:
        """
        Connect to Docker and start the background refresh task.

        FIX-P6-CR-6: if the socket is absent or unreadable this method
        logs a WARNING and returns silently — it does NOT raise.  The resolver
        stays permanently unavailable so the rest of the app continues without
        per-container data.
        """
        import os

        if not os.path.exists(self._socket_path):
            logger.warning(
                "ContainerResolver: docker socket not found at %s — "
                "per-container GPU attribution disabled.",
                self._socket_path,
            )
            return

        try:
            import aiodocker  # local import keeps module loadable even without package

            self._docker = aiodocker.Docker(url=f"unix://{self._socket_path}")
            logger.info("ContainerResolver: connected to Docker socket at %s.", self._socket_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ContainerResolver: failed to connect to Docker socket: %s — "
                "per-container GPU attribution disabled.",
                exc,
            )
            self._docker = None
            return

        self._task = asyncio.create_task(
            self._refresh_loop(), name="container-resolver"
        )

    async def stop(self) -> None:
        """
        FIX-P6-CR-2: cancel the refresh task BEFORE closing the aiodocker
        client.  try/finally guarantees the client is closed even if the
        cancellation or task teardown raises.
        """
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._docker is not None:
            try:
                await self._docker.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("ContainerResolver: error closing Docker client: %s", exc)
            finally:
                self._docker = None

    def resolve_pid(self, pid: int) -> str:
        """
        Look up *pid* in the in-memory map.

        Returns:
            container name  — if the PID belongs to a running container
            "host"          — if no container claims this PID (the process is
                              running directly on the host)
            "unknown"       — if the resolver is currently degraded (too many
                              consecutive refresh failures)

        The caller is responsible for converting string PIDs to int before
        calling this method.
        """
        if self._degraded:
            return "unknown"
        return self._pid_map.get(pid, "host")

    def is_available(self) -> bool:
        """True once at least one successful refresh has completed."""
        return self._available

    def list_gpu_containers(self) -> set[str]:
        """
        FIX-P10-CR-1: return the set of currently-running container names that
        have ``/dev/dri`` passed through via HostConfig.Devices.

        Returns an empty set before the first successful refresh, after a
        connection failure, or when no running container has /dev/dri.

        Single-assignment replacement in _do_refresh() means readers always
        observe a complete, consistent set — never a half-built one.  Callers
        receive a copy so they can mutate freely without racing the refresh
        loop.
        """
        return set(self._gpu_containers)

    def list_gpu_containers_by_vendor(self) -> dict[str, set[str]]:
        """
        FIX-P11-CR-1: ``{container_name: {"drm", "nvidia"}}`` for every
        running container with some form of GPU access.

        Tags describe the ACCESS MECHANISM, not the silicon: "drm" means the
        container can open a DRM render node (Intel or AMD — which one
        depends on what fds it actually opens), "nvidia" means the NVIDIA
        container runtime handed it a device.  Callers receive a deep copy.
        """
        return {name: set(tags) for name, tags in self._gpu_containers_by_vendor.items()}

    def get_status(self) -> dict:
        """
        Return a fully JSON-serialisable status dict for the /health endpoint.

        FIX-P6-MAIN-4: ``last_refresh_ago_s`` is ``float | None``; it is
        ``None`` before the first successful refresh and serialises as JSON
        ``null`` without error.

        FIX-P10-CR-1: also reports ``gpu_container_count`` so /health shows
        how many containers have /dev/dri passed through — useful for
        diagnosing "expected jellyfin to appear but it didn't".
        """
        last_refresh_ago: float | None = None
        if self._last_refresh_time is not None:
            last_refresh_ago = round(time.monotonic() - self._last_refresh_time, 1)
        return {
            "available": self._available,
            "degraded": self._degraded,
            "container_count": self._container_count,
            "gpu_container_count": len(self._gpu_containers),
            # FIX-P11-CR-1: split so "expected jellyfin under the Intel GPU
            # but it isn't there" is diagnosable from /health alone.
            "gpu_containers_by_access": {
                tag: sorted(
                    name
                    for name, tags in self._gpu_containers_by_vendor.items()
                    if tag in tags
                )
                for tag in ("drm", "nvidia")
            },
            "pid_count": len(self._pid_map),
            "last_refresh_ago_s": last_refresh_ago,
            "last_error": self._last_error,
        }

    # ──────────────────────────────────────────────────────── refresh loop ───

    async def _refresh_loop(self) -> None:
        """Background loop: rebuild the PID map every ``_refresh_interval`` seconds."""
        while True:
            try:
                await self._do_refresh()
            except asyncio.CancelledError:
                # FIX-P6-CR-3: let CancelledError propagate so stop() can
                # cleanly await the task — never swallow it here.
                raise
            except Exception as exc:  # noqa: BLE001
                self._consecutive_failures += 1
                self._last_error = str(exc)
                logger.warning(
                    "ContainerResolver: refresh error (failure #%d): %s",
                    self._consecutive_failures,
                    exc,
                )
                # FIX-P6-CR-7: on reaching the threshold, mark degraded but
                # keep looping so the resolver can recover when Docker comes
                # back up.  We do NOT break or return here.
                if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    if not self._degraded:
                        logger.warning(
                            "ContainerResolver: %d consecutive failures — "
                            "marking degraded; resolve_pid() returns 'unknown' "
                            "until a successful refresh.",
                            _MAX_CONSECUTIVE_FAILURES,
                        )
                    self._degraded = True

            await asyncio.sleep(self._refresh_interval)

    async def _do_refresh(self) -> None:
        """
        List running containers and atomically replace ``_pid_map`` and
        ``_gpu_containers``.

        FIX-P6-CR-1: builds a fresh dict + set then assigns each in one
                     statement so readers never see a half-built state.
        FIX-P6-CR-4: per-container top()/show() errors are caught individually
                     so one exiting container does not blank the whole map.
        FIX-P6-CR-5: PID column located by case-insensitive header-name match,
                     not by index (varies across host distributions).
        FIX-P6-CR-8: only ``int`` keys and ``str`` values are stored.
        FIX-P10-CR-1: while inspecting each container, also check its
                     HostConfig.Devices for any entry whose PathOnHost
                     contains "/dev/dri".  If found, the container name is
                     added to the new ``_gpu_containers`` set — which becomes
                     the authoritative list of "containers that can use the
                     GPU" regardless of whether they currently have a
                     render-node fd open.
        """
        docker = self._docker
        if docker is None:
            return

        # FIX-P11-CR-3: the `filter=` kwarg was a no-op — aiodocker spells it
        # `filters` and takes a JSON string, so this call was never actually
        # filtering and simply returned running containers by default.
        # Removed rather than corrected: the default is what we want, and a
        # parameter that looks like it does something but doesn't is worse
        # than no parameter.
        containers = await docker.containers.list()

        new_map: dict[int, str] = {}
        new_gpu_containers: set[str] = set()
        new_gpu_by_vendor: dict[str, set[str]] = {}
        seen_containers: int = 0

        for container in containers:
            try:
                info = await container.show()
                # Container names from the Docker API start with "/"; strip it.
                name: str = info["Name"].lstrip("/")

                tags = _detect_gpu_access(info)
                if tags:
                    new_gpu_containers.add(name)
                    new_gpu_by_vendor[name] = tags

                top_result = await container.top()
                titles: list[str] = top_result.get("Titles", [])

                # FIX-P6-CR-5: locate PID column by case-insensitive header name.
                pid_col: int | None = None
                for idx, title in enumerate(titles):
                    if title.upper() == "PID":
                        pid_col = idx
                        break

                if pid_col is None:
                    logger.debug(
                        "ContainerResolver: no PID column in top() for %s "
                        "(titles=%s) — skipping.",
                        name,
                        titles,
                    )
                    continue

                for row in top_result.get("Processes", []):
                    try:
                        # FIX-P6-CR-8: only int PID key + str name value stored.
                        new_map[int(row[pid_col])] = name
                    except (ValueError, IndexError):
                        pass

                seen_containers += 1

            except asyncio.CancelledError:
                raise  # never swallow CancelledError
            except Exception as exc:  # noqa: BLE001
                # FIX-P6-CR-4: one dead/exiting container must not abort the
                # whole refresh — skip it and continue.
                logger.debug(
                    "ContainerResolver: skipping container during refresh: %s", exc
                )

        # FIX-P6-CR-1 / FIX-P10-CR-1: single atomic assignment for BOTH
        # collections — no reader ever sees a partially populated state.
        self._pid_map = new_map
        self._gpu_containers = new_gpu_containers
        self._gpu_containers_by_vendor = new_gpu_by_vendor
        self._container_count = seen_containers
        self._last_refresh_time = time.monotonic()

        # FIX-P6-CR-7: reset failure tracking on success so recovery works
        # after the resolver has been in degraded state.
        if self._consecutive_failures > 0:
            logger.info(
                "ContainerResolver: recovered after %d consecutive failures.",
                self._consecutive_failures,
            )
        self._consecutive_failures = 0
        self._degraded = False
        self._last_error = None
        self._available = True


# ════════════════════════════════════════════════════════════════════════════
# GPU access detection
# ════════════════════════════════════════════════════════════════════════════

# Values of NVIDIA_VISIBLE_DEVICES that explicitly mean "no GPU".
_NVIDIA_ENV_DISABLED = {"", "void", "none"}


def _detect_gpu_access(info: dict) -> set[str]:
    """
    Return the set of GPU access tags a container holds: ``{"drm"}``,
    ``{"nvidia"}``, both, or empty.

    FIX-P11-CR-2: THE DETECTION FIX.  The previous implementation looked at
    ``HostConfig.Devices`` and nothing else, which misses more cases than it
    catches:

      * ``volumes: - /dev/dri:/dev/dri`` in a Compose file becomes a BIND
        MOUNT, not a device — it lands in ``HostConfig.Binds`` and in the
        top-level ``Mounts`` array, never in ``Devices``.  This is a very
        common way to give a container the iGPU, and it was invisible.
      * ``privileged: true`` grants every device without listing any.
      * NVIDIA containers never touch ``/dev/dri`` at all.  The NVIDIA
        Container Toolkit injects devices at runtime, so the only evidence in
        the inspect payload is a device *request*, the runtime name, or the
        ``NVIDIA_VISIBLE_DEVICES`` environment variable.

    Every signal is checked defensively: any field may be absent or null in
    the inspect payload, and a malformed entry must not abort the refresh for
    the other containers.
    """
    tags: set[str] = set()
    host_config = info.get("HostConfig") or {}

    # ── DRM (/dev/dri) — serves Intel and AMD ────────────────────────────

    # 1. Explicit device passthrough: devices: - /dev/dri:/dev/dri
    for dev in host_config.get("Devices") or []:
        if "/dev/dri" in ((dev or {}).get("PathOnHost") or ""):
            tags.add("drm")
            break

    # 2. Bind mount: volumes: - /dev/dri:/dev/dri
    if "drm" not in tags:
        for bind in host_config.get("Binds") or []:
            if isinstance(bind, str) and bind.startswith("/dev/dri"):
                tags.add("drm")
                break

    # 3. The resolved Mounts array — catches long-form mount syntax.
    if "drm" not in tags:
        for mount in info.get("Mounts") or []:
            if "/dev/dri" in ((mount or {}).get("Source") or ""):
                tags.add("drm")
                break

    # 4. Privileged containers get every host device implicitly.
    if "drm" not in tags and host_config.get("Privileged"):
        tags.add("drm")

    # ── NVIDIA — injected by the container runtime, never via /dev/dri ───

    # 1. `deploy.resources.reservations.devices` / `--gpus` becomes a
    #    DeviceRequest with driver "nvidia" or capability "gpu".
    for request in host_config.get("DeviceRequests") or []:
        request = request or {}
        if (request.get("Driver") or "").lower() == "nvidia":
            tags.add("nvidia")
            break
        capabilities = request.get("Capabilities") or []
        if any("gpu" in [str(c).lower() for c in (group or [])] for group in capabilities):
            tags.add("nvidia")
            break

    # 2. `runtime: nvidia` (the older, pre-DeviceRequests style).
    if (host_config.get("Runtime") or "").lower() == "nvidia":
        tags.add("nvidia")

    # 3. NVIDIA_VISIBLE_DEVICES in the image or container environment.
    if "nvidia" not in tags:
        for entry in (info.get("Config") or {}).get("Env") or []:
            if not isinstance(entry, str) or "=" not in entry:
                continue
            key, _, value = entry.partition("=")
            if key.strip() == "NVIDIA_VISIBLE_DEVICES":
                if value.strip().lower() not in _NVIDIA_ENV_DISABLED:
                    tags.add("nvidia")
                break

    return tags
