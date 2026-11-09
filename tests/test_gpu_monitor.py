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
  8. AMD sysfs/hwmon — deep-sleep clock, every temperature sensor, fan,
     power fallbacks, runtime-suspend, idle engine zeros (RX 7400 layout)
  9. Intel iGPU sensors — RAPL GPU-plane power, CPU package temperature,
     fdinfo shared-memory totals, and "—" (absent) inside a VM

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


# ══════════════════════════════════════════════════════════════════════════
# 8. AMD backend against a fake amdgpu sysfs tree
# ══════════════════════════════════════════════════════════════════════════

# pp_dpm_sclk as printed by smu_cmn_print_dpm_clk_levels() on an idle
# RDNA3 card: the deep-sleep pseudo-level carries the star, not a number.
SCLK_DEEP_SLEEP = "S: 12Mhz *\n1: 500Mhz \n2: 2600Mhz \n"
SCLK_ACTIVE = "0: 500Mhz \n1: 1873Mhz *\n2: 2600Mhz \n"


class FakeAmdCard:
    """A /sys/class/drm/cardN tree laid out the way amdgpu exports it."""

    def __init__(self, root: str) -> None:
        self.card_path = os.path.join(root, "card1")
        self.device = os.path.join(self.card_path, "device")
        self.hwmon = os.path.join(self.device, "hwmon", "hwmon3")
        os.makedirs(self.hwmon)
        os.makedirs(os.path.join(self.device, "power"))

    def write(self, rel: str, value) -> None:
        path = os.path.join(self.device, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(f"{value}\n")

    def hw(self, name: str, value) -> None:
        self.write(os.path.join("hwmon", "hwmon3", name), value)

    def remove_hw(self, name: str) -> None:
        os.remove(os.path.join(self.hwmon, name))

    def rx7400(self) -> "FakeAmdCard":
        """Sensors a Navi 33 (1002:7499) exposes, per amdgpu_pm.c."""
        self.write("gpu_busy_percent", 0)
        self.write("mem_busy_percent", 8)
        self.write("mem_info_vram_used", 263 * 1024 * 1024)
        self.write("mem_info_vram_total", 8176 * 1024 * 1024)
        self.write("pp_dpm_sclk", SCLK_DEEP_SLEEP.rstrip("\n"))
        self.write(os.path.join("power", "runtime_status"), "active")
        self.hw("temp1_input", 63000)
        self.hw("temp1_label", "edge")
        self.hw("temp2_input", 71000)
        self.hw("temp2_label", "junction")
        self.hw("temp3_input", 66000)
        self.hw("temp3_label", "mem")
        # SMU13 on recent kernels: instantaneous power only.
        self.hw("power1_input", 18_000_000)
        self.hw("power1_cap", 55_000_000)
        self.hw("freq1_input", 0)          # idle: GFXOFF / deep sleep
        self.hw("freq1_label", "sclk")
        self.hw("freq2_input", 96_000_000)
        self.hw("freq2_label", "mclk")
        self.hw("fan1_input", 0)           # zero-RPM fan stop
        self.hw("pwm1", 0)
        self.hw("pwm1_max", 255)
        self.hw("in0_input", 6)
        self.hw("in0_label", "vddgfx")
        return self

    def as_card(self) -> dict:
        return {
            "card": "card1",
            "card_path": self.card_path,
            "vendor_id": 0x1002,
            "device_id": 0x7499,
            "vendor": "amd",
            "driver": "amdgpu",
            "pci_address": "0000:03:00.0",
            "render_node": "/dev/dri/renderD129",
        }


class TestAmdBackend(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.card = FakeAmdCard(self._tmp.name).rx7400()

    def tearDown(self):
        self._tmp.cleanup()

    def _backend(self):
        from backends import amd

        with mock.patch.object(amd, "enumerate_drm_cards",
                               return_value=[self.card.as_card()]):
            backend = amd.AmdBackend()
            devices = backend.probe()
        self.assertEqual(len(devices), 1)
        return backend, devices[0]

    def _sample(self, backend, device):
        block = backend.sample()[device.uid]
        self.assertNotIn("error", block)
        return block

    def test_name_for_rx7400(self):
        _backend, device = self._backend()
        self.assertEqual(device.name, "AMD Radeon RX 7400 / 7300 / Pro W7400")

    def test_sensor_inventory(self):
        _backend, device = self._backend()
        self.assertEqual(
            [t["key"] for t in device.sensors["temperatures"]],
            ["edge", "junction", "mem"],
        )
        self.assertEqual(
            [t["label"] for t in device.sensors["temperatures"]],
            ["Edge", "Hotspot", "VRAM"],
        )
        self.assertTrue(device.sensors["fan"])
        self.assertTrue(device.sensors["voltage"])
        # /health lists the resolved files; the per-sample identity doesn't.
        self.assertIn("sources", device.describe(include_sources=True))
        self.assertNotIn("sources", device.describe())

    def test_all_temperatures_reported(self):
        backend, device = self._backend()
        temp = self._sample(backend, device)["temperature"]
        self.assertEqual(temp["celsius"], 63.0)
        self.assertEqual(temp["sensor"], "edge")
        self.assertEqual(temp["channels"], {"edge": 63.0, "junction": 71.0, "mem": 66.0})

    def test_primary_temperature_falls_back_when_edge_unreadable(self):
        self.card.hw("temp1_input", "")  # read fails (driver returned error)
        backend, device = self._backend()
        temp = self._sample(backend, device)["temperature"]
        self.assertEqual(temp["sensor"], "junction")
        self.assertEqual(temp["celsius"], 71.0)
        self.assertNotIn("edge", temp["channels"])

    def test_unlabelled_channels_use_amdgpu_order(self):
        for n in (1, 2, 3):
            self.card.remove_hw(f"temp{n}_label")
        _backend, device = self._backend()
        self.assertEqual(
            [t["key"] for t in device.sensors["temperatures"]],
            ["edge", "junction", "mem"],
        )

    def test_idle_clock_is_zero_not_missing(self):
        backend, device = self._backend()
        freq = self._sample(backend, device)["frequency"]
        self.assertEqual(freq["actual"], 0.0)
        self.assertEqual(freq["max"], 2600.0)
        self.assertEqual(freq["memory"], 96.0)

    def test_deep_sleep_dpm_line_parsed_without_hwmon_freq(self):
        self.card.remove_hw("freq1_input")
        self.card.remove_hw("freq1_label")
        backend, device = self._backend()
        freq = self._sample(backend, device)["frequency"]
        self.assertEqual(freq["actual"], 12.0)
        # The "S:" reading must not be mistaken for a DPM level.
        self.assertEqual(freq["max"], 2600.0)

    def test_active_dpm_line_parsed(self):
        self.card.remove_hw("freq1_input")
        self.card.write("pp_dpm_sclk", SCLK_ACTIVE.rstrip("\n"))
        backend, device = self._backend()
        self.assertEqual(self._sample(backend, device)["frequency"]["actual"], 1873.0)

    def test_power_input_used_when_no_average(self):
        backend, device = self._backend()
        power = self._sample(backend, device)["power"]
        self.assertEqual(power, {"watts": 18.0, "cap_watts": 55.0})

    def test_power_falls_back_when_average_unreadable(self):
        self.card.hw("power1_average", "")  # exists, but the read fails
        backend, device = self._backend()
        self.assertEqual(self._sample(backend, device)["power"]["watts"], 18.0)

    def test_fan_and_voltage(self):
        self.card.hw("fan1_input", 1250)
        self.card.hw("pwm1", 64)
        backend, device = self._backend()
        block = self._sample(backend, device)
        self.assertEqual(block["fan"], {"rpm": 1250.0, "pct": 25.1})
        self.assertEqual(block["voltage"], {"gfx_mv": 6})

    def test_runtime_suspended_card(self):
        # While runtime-suspended amdgpu refuses hwmon reads (EPERM); an
        # unreadable file stands in for that here.
        self.card.write(os.path.join("power", "runtime_status"), "suspended")
        for name in ("temp1_input", "temp2_input", "temp3_input", "power1_input"):
            self.card.hw(name, "")
        backend, device = self._backend()
        block = self._sample(backend, device)
        self.assertEqual(block["power_state"], "suspended")
        self.assertNotIn("temperature", block)
        self.assertNotIn("power", block)
        # Sensors that exist must stay listed so the tiles return on wake.
        self.assertEqual(len(device.sensors["temperatures"]), 3)

    def test_idle_engines_are_zero_after_a_scan(self):
        backend, device = self._backend()
        # No scan yet: the breakdown is genuinely unknown.
        self.assertNotIn("engines", self._sample(backend, device))
        backend._scanner = FakeScanner([{}, {}])
        backend.attribution()
        engines = self._sample(backend, device)["engines"]
        self.assertEqual(set(engines), {"Graphics", "Compute", "Decode", "Encode"})
        self.assertTrue(all(e["busy"] == 0.0 for e in engines.values()))


class TestFdinfoParsing(unittest.TestCase):
    def test_duplicate_engine_keys_are_summed(self):
        text = (
            "drm-client-id:\t7\n"
            "drm-engine-dec:\t100 ns\n"
            "drm-engine-dec:\t50 ns\n"
            "drm-engine-gfx:\t10 ns\n"
        )
        client, engines = drm._parse_fdinfo(text)
        self.assertEqual(client, "7")
        self.assertEqual(engines, {"dec": 150, "gfx": 10})

    def test_pci_ids_bracket_name_preferred(self):
        with mock.patch.object(
            drm, "_lookup_pci_ids_file",
            return_value="Navi 99 [Radeon RX 9999 XT]",
        ):
            self.assertEqual(
                drm.device_display_name(0x1002, 0xFFFF, "amdgpu"),
                "AMD Radeon RX 9999 XT",
            )


# ══════════════════════════════════════════════════════════════════════════
# 9. Intel iGPU power / temperature / memory
# ══════════════════════════════════════════════════════════════════════════


def _write(path: str, value) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{value}\n")


class TestIntelSensors(unittest.TestCase):
    """UHD 630 (8086:3e92): no i915 hwmon, so host-level sources are used."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.card_path = os.path.join(self.root, "drm", "card0")
        os.makedirs(os.path.join(self.card_path, "device"))
        self.hwmon = os.path.join(self.root, "hwmon")
        self.thermal = os.path.join(self.root, "thermal")
        self.powercap = os.path.join(self.root, "powercap")
        self.meminfo = os.path.join(self.root, "meminfo")
        for d in (self.hwmon, self.thermal, self.powercap):
            os.makedirs(d)
        with open(self.meminfo, "w") as f:
            f.write("MemTotal:       16384000 kB\nMemFree: 1 kB\n")

    def tearDown(self):
        self._tmp.cleanup()

    def _bare_metal(self):
        coretemp = os.path.join(self.hwmon, "hwmon2")
        _write(os.path.join(coretemp, "name"), "coretemp")
        _write(os.path.join(coretemp, "temp1_label"), "Package id 0")
        _write(os.path.join(coretemp, "temp1_input"), 54000)
        _write(os.path.join(coretemp, "temp2_label"), "Core 0")
        _write(os.path.join(coretemp, "temp2_input"), 50000)
        zone = os.path.join(self.powercap, "intel-rapl:0:1")
        _write(os.path.join(zone, "name"), "uncore")
        _write(os.path.join(zone, "energy_uj"), 1_000_000)
        _write(os.path.join(zone, "max_energy_range_uj"), 262143328850)
        return zone

    def _backend(self, device_id=0x3E92):
        from backends import intel

        card = {
            "card": "card0", "card_path": self.card_path,
            "vendor_id": 0x8086, "device_id": device_id, "vendor": "intel",
            "driver": "i915", "pci_address": "0000:00:02.0",
            "render_node": "/dev/dri/renderD128",
        }
        patches = [
            mock.patch.object(intel, "enumerate_drm_cards", return_value=[card]),
            mock.patch.object(intel, "HWMON_CLASS_DIR", self.hwmon),
            mock.patch.object(intel, "THERMAL_CLASS_DIR", self.thermal),
            mock.patch.object(intel, "POWERCAP_DIR", self.powercap),
            mock.patch.object(intel, "MEMINFO_PATH", self.meminfo),
            mock.patch.object(intel, "POWER_PMU_DIR", os.path.join(self.root, "nopmu")),
            mock.patch.object(intel.IntelBackend, "_find_pmu_dir", return_value=None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        backend = intel.IntelBackend()
        devices = backend.probe()
        backend.start()
        self.addCleanup(backend.close)
        return backend, devices[0]

    def test_bare_metal_igpu_gets_power_temp_and_memory(self):
        zone = self._bare_metal()
        backend, device = self._backend()

        self.assertTrue(device.extra["integrated"])
        self.assertEqual(device.sensors["temperatures"], [{"key": "cpu_pkg", "label": "CPU pkg"}])

        first = backend.sample()[device.uid]
        self.assertNotIn("power", first)  # baseline tick
        self.assertEqual(first["temperature"]["celsius"], 54.0)
        self.assertEqual(first["temperature"]["sensor"], "cpu_pkg")

        # Advance the RAPL counter by a known amount over a known interval.
        prev_raw, prev_wall = backend._energy_prev[device.uid]
        backend._energy_prev[device.uid] = (prev_raw, prev_wall - 2.0)
        _write(os.path.join(zone, "energy_uj"), 1_000_000 + 7_000_000)
        power = backend.sample()[device.uid]["power"]
        self.assertAlmostEqual(power["watts"], 3.5, delta=0.1)
        self.assertIn("RAPL", power["source"])

        backend._scanner = mock.Mock(memory_bytes={device.uid: 512 * 1024 * 1024})
        memory = backend.sample()[device.uid]["memory"]
        self.assertEqual(memory["used_mib"], 512.0)
        self.assertEqual(memory["total_mib"], 16000.0)
        self.assertTrue(memory["shared"])

        # Clients went away: a supported kernel now means a real 0.
        backend._scanner = mock.Mock(memory_bytes={})
        self.assertEqual(backend.sample()[device.uid]["memory"]["used_mib"], 0.0)

    def test_powercap_wraparound(self):
        zone = self._bare_metal()
        backend, device = self._backend()
        backend.sample()
        prev_raw, prev_wall = backend._energy_prev[device.uid]
        backend._energy_prev[device.uid] = (262143328850 - 1_000_000, prev_wall - 1.0)
        _write(os.path.join(zone, "energy_uj"), 1_000_000)
        power = backend.sample()[device.uid]["power"]
        self.assertAlmostEqual(power["watts"], 2.0, delta=0.1)

    def test_vm_without_rapl_or_coretemp_reports_nothing(self):
        backend, device = self._backend()
        backend._scanner = mock.Mock(memory_bytes={})
        for _ in range(2):
            block = backend.sample()[device.uid]
        self.assertNotIn("power", block)
        self.assertNotIn("temperature", block)
        self.assertNotIn("memory", block)  # kernel never showed memory stats
        self.assertEqual(device.sensors["temperatures"], [])

    def test_x86_pkg_temp_thermal_zone_fallback(self):
        zone = os.path.join(self.thermal, "thermal_zone3")
        _write(os.path.join(zone, "type"), "x86_pkg_temp")
        _write(os.path.join(zone, "temp"), 61000)
        backend, device = self._backend()
        self.assertEqual(backend.sample()[device.uid]["temperature"]["celsius"], 61.0)

    def test_discrete_arc_does_not_get_host_sensors(self):
        self._bare_metal()
        backend, device = self._backend(device_id=0x56A0)
        self.assertFalse(device.extra["integrated"])
        self.assertIsNone(device.extra["energy"])
        self.assertEqual(device.sensors["temperatures"], [])


class TestFdinfoMemory(unittest.TestCase):
    I915_68 = (
        "drm-driver:\ti915\n"
        "drm-client-id:\t42\n"
        "drm-total-system0:\t2048 KiB\n"
        "drm-shared-system0:\t0\n"
        "drm-active-system0:\t0\n"
        "drm-resident-system0:\t1024 KiB\n"
        "drm-total-stolen-system0:\t4 MiB\n"
        "drm-engine-render:\t100 ns\n"
    )

    def test_resident_preferred_total_as_fallback(self):
        client, engines, mem = drm.parse_fdinfo_full(self.I915_68)
        self.assertEqual(client, "42")
        self.assertEqual(engines, {"render": 100})
        # system0 resident 1 MiB + stolen-system0 total 4 MiB
        self.assertEqual(mem, 1024 * 1024 + 4 * 1024 * 1024)

    def test_old_kernel_reports_none_not_zero(self):
        _client, _engines, mem = drm.parse_fdinfo_full(fdinfo("1", render=5))
        self.assertIsNone(mem)

    def test_scanner_sums_clients_and_dedups_forked_fd(self):
        node = "/dev/dri/renderD128"
        other = self.I915_68.replace("\t42", "\t43")
        holder: dict = {}

        def build(proc):
            proc.add(100, [(node, self.I915_68), (node, other)])
            proc.add(101, [(node, self.I915_68)])  # client 42 again, inherited

        real_ctor = drm.FdinfoScanner

        def capture(node_map):
            holder["scanner"] = real_ctor(node_map)
            return holder["scanner"]

        with mock.patch.object(drm, "FdinfoScanner", capture):
            TestFdinfoScanner._scan(None, build, {node: "intel:x"})
        # Two distinct clients x 5 MiB each; the inherited fd is not re-counted.
        self.assertEqual(holder["scanner"].memory_bytes, {"intel:x": 2 * 5 * 1024 * 1024})


if __name__ == "__main__":
    unittest.main(verbosity=2)
