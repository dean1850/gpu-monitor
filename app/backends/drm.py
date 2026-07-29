# SPDX-License-Identifier: MIT
# Copyright (c) 2026 gpu-monitor contributors
"""
Shared DRM plumbing: card enumeration + the ``/proc/<pid>/fdinfo`` scanner.

FIX-P11-DRM-1: NEW MODULE.  Both the Intel (i915) and AMD (amdgpu) backends
enumerate GPUs through ``/sys/class/drm`` and attribute per-process usage
through ``/proc/<pid>/fdinfo`` — the *mechanism* is identical, only the
engine key names differ.  That shared code lives here so there is exactly
one implementation of the tricky parts (dedup, wraparound, multi-GPU fd
routing) instead of two that drift apart.

NVIDIA's proprietary driver does not populate DRM fdinfo engine counters, so
the NVIDIA backend does not use this module — it goes through NVML instead.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

DRM_CLASS_DIR = "/sys/class/drm"

# PCI vendor IDs.
PCI_VENDOR_INTEL = 0x8086
PCI_VENDOR_AMD = 0x1002
PCI_VENDOR_NVIDIA = 0x10DE
PCI_VENDOR_ATI = 0x1002  # AMD retained ATI's vendor ID

_VENDOR_ID_TO_NAME: dict[int, str] = {
    PCI_VENDOR_INTEL: "intel",
    PCI_VENDOR_AMD: "amd",
    PCI_VENDOR_NVIDIA: "nvidia",
}

_U64_MOD = 1 << 64

_CARD_RE = re.compile(r"^card(\d+)$")
_RENDER_RE = re.compile(r"^renderD(\d+)$")

# FIX-P11-DRM-2: a small built-in PCI-ID -> marketing-name table.  Resolving
# a device ID to a human name normally means parsing /usr/share/hwdata/pci.ids,
# which is a 1.5 MB file that ubuntu:24.04-slim does not ship and which would
# add an apt dependency for cosmetics only.  Instead: cover the parts a
# homelab transcoding/monitoring box actually contains, and fall back to
# "<Vendor> GPU [8086:3e92]" for anything else — always correct, occasionally
# less pretty.  pci.ids IS consulted first when the host happens to have it.
_KNOWN_DEVICE_NAMES: dict[tuple[int, int], str] = {
    # Intel Coffee Lake / Comet Lake UHD 630 family (this project's host).
    (0x8086, 0x3E92): "Intel UHD Graphics 630",
    (0x8086, 0x3E98): "Intel UHD Graphics 630",
    (0x8086, 0x3E9B): "Intel UHD Graphics 630",
    (0x8086, 0x9BC5): "Intel UHD Graphics 630",
    (0x8086, 0x9BC8): "Intel UHD Graphics 630",
    (0x8086, 0x3E91): "Intel UHD Graphics 655",
    (0x8086, 0x5912): "Intel HD Graphics 630",
    (0x8086, 0x5916): "Intel HD Graphics 620",
    (0x8086, 0x1912): "Intel HD Graphics 530",
    # Intel Gen11/12 iGPU.
    (0x8086, 0x8A56): "Intel Iris Plus Graphics",
    (0x8086, 0x9A49): "Intel Iris Xe Graphics",
    (0x8086, 0x4680): "Intel UHD Graphics 770",
    (0x8086, 0x4692): "Intel UHD Graphics 730",
    (0x8086, 0x46A6): "Intel Iris Xe Graphics",
    # Intel Arc discrete.
    (0x8086, 0x56A0): "Intel Arc A770",
    (0x8086, 0x56A1): "Intel Arc A750",
    (0x8086, 0x56A5): "Intel Arc A380",
    # AMD parts common in homelabs.
    (0x1002, 0x164E): "AMD Raphael iGPU",
    (0x1002, 0x1636): "AMD Renoir iGPU",
    (0x1002, 0x15E7): "AMD Barcelo iGPU",
    (0x1002, 0x731F): "AMD Radeon RX 5700 XT",
    (0x1002, 0x73BF): "AMD Radeon RX 6900 XT",
    (0x1002, 0x73DF): "AMD Radeon RX 6700 XT",
    (0x1002, 0x744C): "AMD Radeon RX 7900 XTX",
    (0x1002, 0x747E): "AMD Radeon RX 7700 XT",
    (0x1002, 0x67DF): "AMD Radeon RX 480/580",
}

_PCI_IDS_PATHS = (
    "/usr/share/hwdata/pci.ids",
    "/usr/share/misc/pci.ids",
)


def read_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def read_int_file(path: str | None) -> int | None:
    """Read an int from *path*; None on any failure. Never raises."""
    if path is None:
        return None
    try:
        return int(read_text(path).strip())
    except (OSError, ValueError):
        return None


def read_hex_file(path: str) -> int | None:
    """Read a '0x8086'-style sysfs value; None on any failure."""
    try:
        return int(read_text(path).strip(), 16)
    except (OSError, ValueError):
        return None


def first_existing(paths: tuple[str, ...] | list[str]) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def wrap_delta(new: int, old: int) -> int:
    """Δ between two monotonic u64 counters, tolerating wraparound."""
    delta = new - old
    if delta < 0:
        delta += _U64_MOD
    return delta


# ── device naming ────────────────────────────────────────────────────────


def _lookup_pci_ids_file(vendor_id: int, device_id: int) -> str | None:
    """
    Best-effort lookup in the system pci.ids database, if present.

    Format is tab-indented::

        8086  Intel Corporation
        \t3e92  CoffeeLake-S GT2 [UHD Graphics 630]
    """
    path = first_existing(_PCI_IDS_PATHS)
    if path is None:
        return None

    vendor_key = f"{vendor_id:04x}"
    device_key = f"{device_id:04x}"

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            in_vendor = False
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                if not line.startswith("\t"):
                    # Vendor-level line: "8086  Intel Corporation"
                    in_vendor = line[:4].lower() == vendor_key
                    continue
                if not in_vendor or line.startswith("\t\t"):
                    continue
                # Device-level line: "\t3e92  CoffeeLake-S GT2 [...]"
                stripped = line.lstrip("\t")
                if stripped[:4].lower() == device_key:
                    return stripped[4:].strip() or None
    except OSError:
        return None
    return None


def device_display_name(vendor_id: int, device_id: int, driver: str | None) -> str:
    """Human-readable GPU name, with graceful degradation to the raw IDs."""
    known = _KNOWN_DEVICE_NAMES.get((vendor_id, device_id))
    if known:
        return known

    from_db = _lookup_pci_ids_file(vendor_id, device_id)
    if from_db:
        vendor_label = _VENDOR_ID_TO_NAME.get(vendor_id, "GPU").upper()
        if vendor_label.lower() in from_db.lower():
            return from_db
        return f"{vendor_label} {from_db}"

    vendor_label = {
        PCI_VENDOR_INTEL: "Intel",
        PCI_VENDOR_AMD: "AMD",
        PCI_VENDOR_NVIDIA: "NVIDIA",
    }.get(vendor_id, "Unknown")
    suffix = f" ({driver})" if driver else ""
    return f"{vendor_label} GPU [{vendor_id:04x}:{device_id:04x}]{suffix}"


# ── card enumeration ─────────────────────────────────────────────────────


def _driver_of(card_path: str) -> str | None:
    """Resolve the kernel driver bound to this card via the device symlink."""
    try:
        return os.path.basename(
            os.path.realpath(os.path.join(card_path, "device", "driver"))
        )
    except OSError:
        return None


def _pci_address_of(card_path: str) -> str | None:
    try:
        real = os.path.realpath(os.path.join(card_path, "device"))
    except OSError:
        return None
    base = os.path.basename(real)
    # PCI addresses look like 0000:00:02.0
    if re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d", base):
        return base
    return None


def _render_nodes_by_pci() -> dict[str, str]:
    """Map PCI address -> /dev/dri/renderDN for every render node present."""
    mapping: dict[str, str] = {}
    try:
        entries = os.listdir(DRM_CLASS_DIR)
    except OSError:
        return mapping

    for entry in entries:
        if not _RENDER_RE.match(entry):
            continue
        node_path = os.path.join(DRM_CLASS_DIR, entry)
        pci = _pci_address_of(node_path)
        if pci:
            mapping[pci] = f"/dev/dri/{entry}"
    return mapping


def enumerate_drm_cards() -> list[dict]:
    """
    Enumerate every DRM card on the host.

    Returns a list of dicts::

        {"card": "card0", "card_path": "/sys/class/drm/card0",
         "vendor_id": 0x8086, "device_id": 0x3e92, "vendor": "intel",
         "driver": "i915", "pci_address": "0000:00:02.0",
         "render_node": "/dev/dri/renderD128"}

    FIX-P11-DRM-3: cards are sorted by PCI address, not by the cardN number.
    cardN indices are assigned in probe order and can change across reboots
    when a discrete card is added; the PCI address does not.  Stable
    ordering matters because it decides the order GPUs appear on screen.

    Never raises.
    """
    cards: list[dict] = []
    try:
        entries = sorted(os.listdir(DRM_CLASS_DIR))
    except OSError:
        logger.info("drm: %s not present — no DRM cards to enumerate.", DRM_CLASS_DIR)
        return cards

    render_map = _render_nodes_by_pci()

    for entry in entries:
        if not _CARD_RE.match(entry):
            continue

        card_path = os.path.join(DRM_CLASS_DIR, entry)
        device_dir = os.path.join(card_path, "device")

        vendor_id = read_hex_file(os.path.join(device_dir, "vendor"))
        device_id = read_hex_file(os.path.join(device_dir, "device"))
        if vendor_id is None:
            # Not a PCI-backed DRM device (vkms, virtio in some configs).
            continue

        pci = _pci_address_of(card_path)
        driver = _driver_of(card_path)

        cards.append(
            {
                "card": entry,
                "card_path": card_path,
                "vendor_id": vendor_id,
                "device_id": device_id if device_id is not None else 0,
                "vendor": _VENDOR_ID_TO_NAME.get(vendor_id),
                "driver": driver,
                "pci_address": pci,
                "render_node": render_map.get(pci) if pci else None,
            }
        )

    cards.sort(key=lambda c: (c["pci_address"] or "", c["card"]))
    return cards


# ── /proc fdinfo scanning ────────────────────────────────────────────────


def parse_ns_value(text: str) -> int | None:
    """Parse an fdinfo value of the form ``"1234567 ns"`` -> 1234567."""
    parts = text.split()
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


_DRM_ENGINE_PREFIX = "drm-engine-"


class FdinfoScanner:
    """
    Scans ``/proc/*/fdinfo`` for DRM engine counters, routed per GPU.

    Construct with a mapping of render-node/card device paths to device uids
    so that on a multi-GPU host each process's usage lands against the GPU it
    actually used::

        scanner = FdinfoScanner({"/dev/dri/renderD128": "intel:0000:00:02.0",
                                 "/dev/dri/renderD129": "amd:0000:03:00.0"})

    Returns cumulative nanosecond counters; converting to percentages is the
    caller's job (it owns the Δwall clock).
    """

    def __init__(self, node_to_uid: dict[str, str]) -> None:
        # Match on basename so both /dev/dri/renderD128 and the card node
        # resolve, regardless of how the symlink target is spelled.
        self._node_to_uid: dict[str, str] = {}
        for node, uid in node_to_uid.items():
            self._node_to_uid[os.path.basename(node)] = uid
        self._noid_logged: set[int] = set()

    @property
    def node_basenames(self) -> tuple[str, ...]:
        return tuple(self._node_to_uid)

    def scan(self) -> dict[str, dict[tuple[int, str], dict[str, int]]] | None:
        """
        Return ``{device_uid: {(pid, client_key): {raw_engine: ns}}}``.

        ``None`` only if ``/proc`` itself cannot be listed.

        FIX-P11-DRM-4: THE DEDUP FIX.  A process can hold several fds against
        the same DRM client (dup(), or a library opening the render node more
        than once).  Those fds report IDENTICAL cumulative counters, so naive
        summing multiplies a transcode's usage by the fd count.

        The old approach keyed dedup on ``drm-client-id`` and, when that line
        was absent, on ``os.stat(fd).st_ino``.  The inode fallback is WRONG:
        stat() on /proc/<pid>/fd/<n> follows through to the DRM *device node*,
        whose inode is identical for every process on the system.  Every
        client-id-less context therefore collapsed into a single bucket.

        The fix keys client-id-less fds on their EXACT ENGINE COUNTER VALUES
        instead.  Within one PID, two fds showing byte-identical counters are
        by definition the same underlying client and are counted once; two
        fds showing different counters are genuinely different contexts and
        are summed.  The unique value-sets are then folded into one synthetic
        ``(pid, "noid")`` client so the key stays stable tick to tick — the
        counters themselves keep moving, so they cannot be part of the key.
        """
        try:
            pid_entries = os.listdir("/proc")
        except OSError:
            return None

        result: dict[str, dict[tuple[int, str], dict[str, int]]] = {}

        for entry in pid_entries:
            if not entry.isdigit():
                continue
            pid = int(entry)

            fd_dir = f"/proc/{entry}/fd"
            try:
                fd_names = os.listdir(fd_dir)
            except OSError:
                # PIDs vanish mid-scan (ESRCH/ENOENT) or we lack permission
                # (EACCES). Both are expected and skipped silently.
                continue

            # uid -> client_id -> engines            (identified clients)
            identified: dict[str, dict[str, dict[str, int]]] = {}
            # uid -> set of frozen engine value-sets (client-id-less)
            anonymous: dict[str, set[tuple[tuple[str, int], ...]]] = {}

            for fd_name in fd_names:
                fd_path = os.path.join(fd_dir, fd_name)
                try:
                    target = os.readlink(fd_path)
                except OSError:
                    continue

                uid = self._uid_for_target(target)
                if uid is None:
                    continue

                try:
                    with open(f"/proc/{entry}/fdinfo/{fd_name}", "r") as f:
                        info_text = f.read()
                except OSError:
                    continue

                client_id, engines = _parse_fdinfo(info_text)
                if not engines:
                    # A DRM fd with no engine counters tells us nothing.
                    continue

                if client_id is not None:
                    identified.setdefault(uid, {})[client_id] = engines
                else:
                    frozen = tuple(sorted(engines.items()))
                    anonymous.setdefault(uid, set()).add(frozen)
                    if pid not in self._noid_logged:
                        logger.info(
                            "drm: PID %d has a DRM fd with no drm-client-id — "
                            "deduplicating by engine value-set instead "
                            "(FIX-P11-DRM-4). Normal for some libdrm/VA-API "
                            "builds; attribution still works.",
                            pid,
                        )
                        self._noid_logged.add(pid)

            for uid, clients in identified.items():
                bucket = result.setdefault(uid, {})
                for client_id, engines in clients.items():
                    bucket[(pid, client_id)] = engines

            for uid, value_sets in anonymous.items():
                merged: dict[str, int] = {}
                for frozen in value_sets:
                    for engine, ns in frozen:
                        merged[engine] = merged.get(engine, 0) + ns
                if merged:
                    result.setdefault(uid, {})[(pid, "noid")] = merged

        return result

    def _uid_for_target(self, target: str) -> str | None:
        base = os.path.basename(target)
        return self._node_to_uid.get(base)


def _parse_fdinfo(info_text: str) -> tuple[str | None, dict[str, int]]:
    """
    Extract ``(drm-client-id, {raw_engine_name: ns})`` from one fdinfo blob.

    Raw engine names keep their driver spelling (``render``, ``copy``,
    ``video`` for i915; ``gfx``, ``compute``, ``dec``, ``enc`` for amdgpu) —
    each backend maps them onto its own display keys.
    """
    client_id: str | None = None
    engines: dict[str, int] = {}

    for line in info_text.splitlines():
        if line.startswith("drm-client-id:"):
            client_id = line.split(":", 1)[1].strip()
        elif line.startswith(_DRM_ENGINE_PREFIX):
            key, _, value = line.partition(":")
            raw_name = key[len(_DRM_ENGINE_PREFIX):].strip()
            ns = parse_ns_value(value.strip())
            if ns is not None and raw_name:
                engines[raw_name] = ns

    return client_id, engines
