# SPDX-License-Identifier: MIT
# Copyright (c) 2026 gpu-monitor contributors
"""
Prompt 9 verification suite.

Covers the logic that CHANGED this session, using mocked kernel/Docker data
so the whole suite runs on a machine with no GPU at all:

  1. FdinfoScanner dedup — the value-set fix that replaced inode dedup
  2. Multi-GPU fd routing — usage lands against the right card
  3. Delta sanity guard — counter resets don't become 100% spikes
  4. Container GPU detection — Devices / Binds / Mounts / privileged / NVIDIA
  5. Config tri-state parsing and clamping
  6. DRM card enumeration + PCI-ordered sorting
  7. Backend contract — probe() never raises with no hardware present

Run:  python3 tests/test_gpu_monitor.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import config as cfgmod  # noqa: E402
from backends import drm  # noqa: E402
from backends.intel import compute_fdinfo_attribution  # noqa: E402
from container_resolver import _detect_gpu_access  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# 1 + 2. FdinfoScanner
# ══════════════════════════════════════════════════════════════════════════


class FakeProc:
    """Builds a fake /proc tree on disk so the scanner runs unmodified."""

    def __init__(self, root: str) -> None:
        self.root = root

    def add(self, pid: int, fds: list[tuple[str, str]]) -> None:
        """fds: list of (symlink_target, fdinfo_text)."""
        fd_dir = os.path.join(self.root, str(pid), "fd")
        fdinfo_dir = os.path.join(self.root, str(pid), "fdinfo")
        os.makedirs(fd_dir, exist_ok=True)
        os.makedirs(fdinfo_dir, exist_ok=True)
        for i, (target, text) in enumerate(fds):
            os.symlink(target, os.path.join(fd_dir, str(i)))
            with open(os.path.join(fdinfo_dir, str(i)), "w") as f:
                f.write(text)


def fdinfo(client_id: str | None, **engines: int) -> str:
    lines = []
    if client_id is not None:
        lines.append(f"drm-client-id:\t{client_id}")
    lines.append("drm-driver:\ti915")
    for name, ns in engines.items():
        lines.append(f"drm-engine-{name.replace('_', '-')}:\t{ns} ns")
    return "\n".join(lines) + "\n"


class TestFdinfoScanner(unittest.TestCase):
    def _scan(self, builder, node_map=None):
        node_map = node_map or {"/dev/dri/renderD128": "intel:0000:00:02.0"}
        with tempfile.TemporaryDirectory() as tmp:
            builder(FakeProc(tmp))
            scanner = drm.FdinfoScanner(node_map)
            # Capture the real functions BEFORE patching, or the fakes
            # below recurse into themselves.
            real_listdir = os.listdir
            real_readlink = os.readlink
            real_open = open

            def fake_listdir(path):
                if path == "/proc":
                    return real_listdir(tmp)
                if path.startswith("/proc/"):
                    return real_listdir(os.path.join(tmp, path[len("/proc/"):]))
                return real_listdir(path)

            def fake_readlink(path):
                return real_readlink(os.path.join(tmp, path[len("/proc/"):]))

            def fake_open(path, *a, **kw):
                if isinstance(path, str) and path.startswith("/proc/"):
                    path = os.path.join(tmp, path[len("/proc/"):])
                return real_open(path, *a, **kw)

            with mock.patch("os.listdir", fake_listdir), \
                 mock.patch("os.readlink", fake_readlink), \
                 mock.patch("builtins.open", fake_open):
                return scanner.scan()

    def test_duplicate_fds_same_client_id_counted_once(self):
        """Two fds sharing a drm-client-id must not double-count."""
        result = self._scan(lambda p: p.add(100, [
            ("/dev/dri/renderD128", fdinfo("7", video=5_000_000)),
            ("/dev/dri/renderD128", fdinfo("7", video=5_000_000)),
        ]))
        clients = result["intel:0000:00:02.0"]
        self.assertEqual(len(clients), 1)
        self.assertEqual(clients[(100, "7")]["video"], 5_000_000)

    def test_clientidless_identical_counters_deduped(self):
        """
        THE BUG FIX. Three fds, no drm-client-id, identical counters — this
        is one client seen three times, not three clients. The old inode
        fallback collapsed these correctly only by accident; the real
        failure was that it collapsed DIFFERENT clients too.
        """
        result = self._scan(lambda p: p.add(200, [
            ("/dev/dri/renderD128", fdinfo(None, video=9_000_000)),
            ("/dev/dri/renderD128", fdinfo(None, video=9_000_000)),
            ("/dev/dri/renderD128", fdinfo(None, video=9_000_000)),
        ]))
        clients = result["intel:0000:00:02.0"]
        self.assertEqual(len(clients), 1)
        self.assertEqual(clients[(200, "noid")]["video"], 9_000_000)

    def test_clientidless_distinct_counters_summed(self):
        """
        The regression the inode fallback caused: two genuinely different
        contexts with different counters must SUM, not collapse to one.
        os.stat() returned the same device-node inode for both, so the old
        code kept only the first and under-reported by half.
        """
        result = self._scan(lambda p: p.add(300, [
            ("/dev/dri/renderD128", fdinfo(None, video=4_000_000)),
            ("/dev/dri/renderD128", fdinfo(None, video=6_000_000)),
        ]))
        clients = result["intel:0000:00:02.0"]
        self.assertEqual(len(clients), 1, "must fold into one stable synthetic key")
        self.assertEqual(clients[(300, "noid")]["video"], 10_000_000)

    def test_multi_gpu_routing(self):
        """Each fd's usage lands against the GPU whose node it points at."""
        node_map = {
            "/dev/dri/renderD128": "intel:0000:00:02.0",
            "/dev/dri/renderD129": "amd:0000:03:00.0",
        }
        result = self._scan(lambda p: p.add(400, [
            ("/dev/dri/renderD128", fdinfo("1", video=1_000_000)),
            ("/dev/dri/renderD129", fdinfo("2", gfx=2_000_000)),
        ]), node_map)
        self.assertEqual(result["intel:0000:00:02.0"][(400, "1")]["video"], 1_000_000)
        self.assertEqual(result["amd:0000:03:00.0"][(400, "2")]["gfx"], 2_000_000)
        self.assertNotIn((400, "2"), result["intel:0000:00:02.0"])

    def test_non_drm_fds_ignored(self):
        result = self._scan(lambda p: p.add(500, [
            ("/dev/null", "not drm\n"),
            ("/dev/dri/renderD128", fdinfo("9", render=3_000_000)),
        ]))
        self.assertEqual(len(result["intel:0000:00:02.0"]), 1)


# ══════════════════════════════════════════════════════════════════════════
# 3. Delta sanity guard
# ══════════════════════════════════════════════════════════════════════════


class FakeScanner:
    def __init__(self, payloads):
        self._payloads = list(payloads)

    def scan(self):
        return self._payloads.pop(0)


class Holder:
    _prev_attrib_wall = None


class TestDeltaGuard(unittest.TestCase):
    ENGINE_MAP = {"video": "video"}
    KEYS = ("video",)

    def _run(self, first, second, elapsed=1.0):
        uid = "intel:0"
        holder = Holder()
        prev: dict = {}
        scanner = FakeScanner([{uid: first}, {uid: second}])

        t = [1000.0]
        with mock.patch("time.monotonic", side_effect=lambda: t[0]):
            compute_fdinfo_attribution(scanner, self.ENGINE_MAP, self.KEYS, prev, holder)
            t[0] += elapsed
            return compute_fdinfo_attribution(
                scanner, self.ENGINE_MAP, self.KEYS, prev, holder
            )

    def test_normal_delta_becomes_percentage(self):
        """0.5 s of engine time over a 1 s wall interval = 50%."""
        out = self._run(
            {(1, "a"): {"video": 0}},
            {(1, "a"): {"video": 500_000_000}},
        )
        self.assertAlmostEqual(out["intel:0"][1]["video"], 50.0, places=1)

    def test_counter_reset_does_not_spike(self):
        """
        A counter going backwards used to wrap to ~1.8e19 ns and render as a
        full-scale 100% spike. It must now contribute zero.
        """
        out = self._run(
            {(1, "a"): {"video": 900_000_000}},
            {(1, "a"): {"video": 1_000}},
        )
        self.assertEqual(out["intel:0"][1]["video"], 0.0)

    def test_first_sighting_yields_no_delta(self):
        out = self._run({}, {(1, "a"): {"video": 500_000_000}})
        self.assertEqual(out.get("intel:0", {}), {})


# ══════════════════════════════════════════════════════════════════════════
# 4. Container GPU detection
# ══════════════════════════════════════════════════════════════════════════


class TestGpuDetection(unittest.TestCase):
    def test_devices_passthrough(self):
        info = {"HostConfig": {"Devices": [{"PathOnHost": "/dev/dri"}]}}
        self.assertEqual(_detect_gpu_access(info), {"drm"})

    def test_specific_render_node_device(self):
        info = {"HostConfig": {"Devices": [{"PathOnHost": "/dev/dri/renderD128"}]}}
        self.assertEqual(_detect_gpu_access(info), {"drm"})

    def test_bind_mount_syntax(self):
        """`volumes: - /dev/dri:/dev/dri` — the case the old code missed."""
        info = {"HostConfig": {"Binds": ["/dev/dri:/dev/dri:rw"]}}
        self.assertEqual(_detect_gpu_access(info), {"drm"})

    def test_mounts_array(self):
        info = {"HostConfig": {}, "Mounts": [{"Source": "/dev/dri", "Type": "bind"}]}
        self.assertEqual(_detect_gpu_access(info), {"drm"})

    def test_privileged(self):
        info = {"HostConfig": {"Privileged": True}}
        self.assertEqual(_detect_gpu_access(info), {"drm"})

    def test_nvidia_device_request_driver(self):
        info = {"HostConfig": {"DeviceRequests": [{"Driver": "nvidia", "Count": -1}]}}
        self.assertEqual(_detect_gpu_access(info), {"nvidia"})

    def test_nvidia_device_request_capability(self):
        info = {"HostConfig": {"DeviceRequests": [{"Capabilities": [["gpu"]]}]}}
        self.assertEqual(_detect_gpu_access(info), {"nvidia"})

    def test_nvidia_runtime(self):
        info = {"HostConfig": {"Runtime": "nvidia"}}
        self.assertEqual(_detect_gpu_access(info), {"nvidia"})

    def test_nvidia_env_var(self):
        info = {"HostConfig": {}, "Config": {"Env": ["NVIDIA_VISIBLE_DEVICES=all"]}}
        self.assertEqual(_detect_gpu_access(info), {"nvidia"})

    def test_nvidia_env_void_is_not_gpu(self):
        info = {"HostConfig": {}, "Config": {"Env": ["NVIDIA_VISIBLE_DEVICES=void"]}}
        self.assertEqual(_detect_gpu_access(info), set())

    def test_both_vendors(self):
        info = {
            "HostConfig": {
                "Devices": [{"PathOnHost": "/dev/dri"}],
                "Runtime": "nvidia",
            }
        }
        self.assertEqual(_detect_gpu_access(info), {"drm", "nvidia"})

    def test_no_gpu(self):
        info = {"HostConfig": {"Devices": [], "Binds": ["/etc/localtime:/etc/localtime:ro"]}}
        self.assertEqual(_detect_gpu_access(info), set())

    def test_malformed_payload_does_not_raise(self):
        for info in ({}, {"HostConfig": None}, {"HostConfig": {"Devices": [None]}},
                     {"Mounts": [None]}, {"Config": {"Env": [None, "X"]}}):
            self.assertIsInstance(_detect_gpu_access(info), set)


# ══════════════════════════════════════════════════════════════════════════
# 5. Config
# ══════════════════════════════════════════════════════════════════════════


class TestConfig(unittest.TestCase):
    def test_defaults_are_auto(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = cfgmod.load_config()
        for vendor in cfgmod.ALL_VENDORS:
            self.assertEqual(cfg.mode(vendor), cfgmod.MODE_AUTO)
            self.assertTrue(cfg.is_enabled(vendor))
            self.assertFalse(cfg.is_forced(vendor))

    def test_explicit_false_disables(self):
        with mock.patch.dict(os.environ, {"GPU_ENABLE_NVIDIA": "false"}, clear=True):
            cfg = cfgmod.load_config()
        self.assertFalse(cfg.is_enabled("nvidia"))
        self.assertTrue(cfg.is_enabled("intel"))

    def test_explicit_true_forces(self):
        with mock.patch.dict(os.environ, {"GPU_ENABLE_AMD": "yes"}, clear=True):
            cfg = cfgmod.load_config()
        self.assertTrue(cfg.is_forced("amd"))

    def test_typo_falls_back_to_auto(self):
        with mock.patch.dict(os.environ, {"GPU_ENABLE_INTEL": "ture"}, clear=True):
            cfg = cfgmod.load_config()
        self.assertEqual(cfg.mode("intel"), cfgmod.MODE_AUTO)

    def test_interval_clamped(self):
        with mock.patch.dict(os.environ, {"GPU_SAMPLE_INTERVAL_S": "0"}, clear=True):
            self.assertEqual(cfgmod.load_config().sample_interval_s, 0.2)
        with mock.patch.dict(os.environ, {"GPU_SAMPLE_INTERVAL_S": "-5"}, clear=True):
            self.assertEqual(cfgmod.load_config().sample_interval_s, 0.2)

    def test_garbage_numeric_uses_default(self):
        with mock.patch.dict(os.environ, {"GPU_HISTORY_SIZE": "banana"}, clear=True):
            self.assertEqual(cfgmod.load_config().history_size, 300)

    def test_describe_is_serialisable(self):
        import json

        with mock.patch.dict(os.environ, {}, clear=True):
            json.dumps(cfgmod.load_config().describe())


# ══════════════════════════════════════════════════════════════════════════
# 6. DRM enumeration
# ══════════════════════════════════════════════════════════════════════════


class TestDrmEnumeration(unittest.TestCase):
    def _build_sysfs(self, tmp, cards):
        drm_dir = os.path.join(tmp, "drm")
        os.makedirs(drm_dir)
        for name, (vendor, device, pci, driver) in cards.items():
            device_target = os.path.join(tmp, "pci", pci)
            os.makedirs(device_target, exist_ok=True)
            with open(os.path.join(device_target, "vendor"), "w") as f:
                f.write(f"0x{vendor:04x}\n")
            with open(os.path.join(device_target, "device"), "w") as f:
                f.write(f"0x{device:04x}\n")
            driver_dir = os.path.join(tmp, "drivers", driver)
            os.makedirs(driver_dir, exist_ok=True)
            link = os.path.join(device_target, "driver")
            if not os.path.exists(link):
                os.symlink(driver_dir, link)
            card_dir = os.path.join(drm_dir, name)
            os.makedirs(card_dir)
            os.symlink(device_target, os.path.join(card_dir, "device"))
        return drm_dir

    def test_sorted_by_pci_not_card_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm_dir = self._build_sysfs(tmp, {
                "card1": (0x8086, 0x3E92, "0000:00:02.0", "i915"),
                "card0": (0x1002, 0x73DF, "0000:03:00.0", "amdgpu"),
            })
            with mock.patch.object(drm, "DRM_CLASS_DIR", drm_dir):
                cards = drm.enumerate_drm_cards()

            self.assertEqual([c["vendor"] for c in cards], ["intel", "amd"])
            self.assertEqual(cards[0]["pci_address"], "0000:00:02.0")
            self.assertEqual(cards[0]["driver"], "i915")
            self.assertEqual(cards[1]["driver"], "amdgpu")

    def test_missing_drm_dir_returns_empty(self):
        with mock.patch.object(drm, "DRM_CLASS_DIR", "/nonexistent/drm"):
            self.assertEqual(drm.enumerate_drm_cards(), [])

    def test_device_naming(self):
        self.assertEqual(
            drm.device_display_name(0x8086, 0x3E92, "i915"),
            "Intel UHD Graphics 630",
        )
        name = drm.device_display_name(0x8086, 0xFFFF, "i915")
        self.assertIn("8086", name)
        self.assertIn("ffff", name)

    def test_wrap_delta(self):
        self.assertEqual(drm.wrap_delta(100, 40), 60)
        self.assertEqual(drm.wrap_delta(5, (1 << 64) - 5), 10)


# ══════════════════════════════════════════════════════════════════════════
# 7. Backend contract
# ══════════════════════════════════════════════════════════════════════════


class TestBackendContract(unittest.TestCase):
    def test_probe_never_raises_without_hardware(self):
        from backends.amd import AmdBackend
        from backends.intel import IntelBackend
        from backends.nvidia import NvidiaBackend

        with mock.patch.object(drm, "DRM_CLASS_DIR", "/nonexistent/drm"):
            for cls in (IntelBackend, AmdBackend):
                for forced in (False, True):
                    backend = cls(forced=forced)
                    self.assertEqual(backend.probe(), [])
                    backend.start()
                    self.assertEqual(backend.sample(), {})
                    backend.close()

        # NVML absent: must be silent when auto, explicit when forced.
        quiet = NvidiaBackend(forced=False)
        self.assertEqual(quiet.probe(), [])
        self.assertIsNone(quiet.error)

        loud = NvidiaBackend(forced=True)
        self.assertEqual(loud.probe(), [])
        self.assertIsNotNone(loud.error)
        self.assertIn("nvidia", loud.status()["vendor"])

    def test_build_backends_respects_disable(self):
        from backends import build_backends

        with mock.patch.dict(os.environ, {
            "GPU_ENABLE_INTEL": "false",
            "GPU_ENABLE_AMD": "false",
            "GPU_ENABLE_NVIDIA": "false",
        }, clear=True):
            self.assertEqual(build_backends(cfgmod.load_config()), [])

    def test_status_is_json_serialisable(self):
        import json

        from backends.intel import IntelBackend

        with mock.patch.object(drm, "DRM_CLASS_DIR", "/nonexistent/drm"):
            backend = IntelBackend()
            backend.probe()
            json.dumps(backend.status())


if __name__ == "__main__":
    unittest.main(verbosity=2)
