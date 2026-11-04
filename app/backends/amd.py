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
    hwmon/hwmonN/         every sensor the driver exposes:
                            temp{1,2,3}_input   edge / junction / mem (m°C)
                            power1_average|power1_input, power1_cap (µW)
                            freq1_input (sclk), freq2_input (mclk)   (Hz)
                            fan1_input (RPM), pwm1 / pwm1_max
                            in0_input (vddgfx, mV)
    power/runtime_status  "suspended" while the card is runtime-powered-off

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

# FIX-AMD-CLK-1: the active DPM level is the line carrying a '*'.  Modern
# kernels (smu_cmn_print_dpm_clk_levels) print a deep-sleep pseudo-level
# labelled "S:" rather than a number when the clock is below half the lowest
# DPM state — which is where an idle RDNA2/3/4 card spends almost all its
# time:
#
#     S: 12Mhz *
#     1: 500Mhz
#     2: 2600Mhz
#
# The old pattern only accepted numeric labels, so an idle card reported no
# clock at all and the tile showed "—" except during brief bursts of load.
_SCLK_ACTIVE_RE = re.compile(
    r"^\s*(?:\d+|S):\s*(\d+)\s*Mhz\s*\*", re.IGNORECASE | re.MULTILINE
)
_SCLK_LEVEL_RE = re.compile(r"^\s*\d+:\s*(\d+)\s*Mhz", re.IGNORECASE | re.MULTILINE)

# hwmon temperature channel -> (key, display label).  amdgpu fixes the
# channel order (temp1 edge, temp2 junction, temp3 mem) and also exports a
# tempN_label file, which wins when present.
_TEMP_DEFAULT_KEYS: dict[int, str] = {1: "edge", 2: "junction", 3: "mem"}
_TEMP_LABELS: dict[str, str] = {
    "edge": "Edge",
    "junction": "Hotspot",
    "mem": "VRAM",
}
# Order the primary "temperature" value is chosen in.  Edge is what every
# other tool calls "GPU temperature"; APUs and some server parts only
# expose one channel, so fall through to whatever exists.
_TEMP_PRIMARY_ORDER: tuple[str, ...] = ("edge", "junction", "mem")


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
        self._attrib_ran = False

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
            device.extra["runtime_status"] = first_existing(
                (os.path.join(device_dir, "power", "runtime_status"),)
            )
            device.extra.update(self._find_hwmon(device_dir))

            temps = device.extra["temps"]
            device.sensors = {
                "temperatures": [
                    {"key": key, "label": _TEMP_LABELS.get(key, key.title())}
                    for key, _path in temps
                ],
                "fan": bool(device.extra["fan_rpm"]),
                "voltage": bool(device.extra["vddgfx"]),
            }
            device.sources = {
                name: device.extra[name]
                for name in (
                    "busy", "mem_busy", "vram_used", "vram_total", "sclk",
                    "runtime_status", "hwmon_freq", "hwmon_mclk", "power_cap",
                    "fan_rpm", "pwm", "vddgfx",
                )
            }
            device.sources["power"] = list(device.extra["power"])
            device.sources["temps"] = {key: path for key, path in temps}
            devices.append(device)

            logger.info(
                "amd: found %s (%s) — busy=%s vram=%s power=%s temps=%s fan=%s "
                "sclk=%s runtime_pm=%s",
                device.name, uid,
                bool(device.extra["busy"]), bool(device.extra["vram_total"]),
                bool(device.extra["power"]),
                ",".join(key for key, _ in temps) or "none",
                bool(device.extra["fan_rpm"]),
                "hwmon" if device.extra["hwmon_freq"]
                else ("pp_dpm_sclk" if device.extra["sclk"] else "none"),
                _read_word(device.extra["runtime_status"]) or "n/a",
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
    def _find_hwmon(device_dir: str) -> dict:
        """
        Resolve every hwmon sensor attached to this GPU.

        FIX-AMD-HWMON-1: previously only temp1_input (edge) was read, so the
        junction/hotspot and VRAM temperatures, fan and voltage that
        ``sensors`` shows for the same card never reached the dashboard, and
        a card whose temp1 read failed showed no temperature at all.

        Paths are resolved by EXISTENCE only, never by a trial read: a card
        that is runtime-suspended at startup refuses every hwmon read with
        EPERM, and must not be permanently written off for it.
        """
        out: dict = {
            "power": [],            # candidates, tried in order every tick
            "power_cap": None,
            "temps": [],            # [(key, path)] in channel order
            "hwmon_freq": None,
            "hwmon_mclk": None,
            "fan_rpm": None,
            "pwm": None,
            "pwm_max": None,
            "vddgfx": None,
        }
        seen_keys: set[str] = set()

        for hwmon_dir in sorted(glob.glob(os.path.join(device_dir, "hwmon", "hwmon*"))):
            def p(name: str, _d: str = hwmon_dir) -> str | None:
                return first_existing((os.path.join(_d, name),))

            # Not all products support both average and instantaneous power
            # (SMU13 parts dropped power1_average in newer kernels), and the
            # one that exists can still fail transiently — so keep both and
            # take the first that actually reads on each tick.
            for name in ("power1_average", "power1_input"):
                path = p(name)
                if path and path not in out["power"]:
                    out["power"].append(path)
            out["power_cap"] = out["power_cap"] or p("power1_cap")

            for channel in range(1, 10):
                path = p(f"temp{channel}_input")
                if path is None:
                    continue
                key = _hwmon_label(hwmon_dir, f"temp{channel}_label")
                key = key or _TEMP_DEFAULT_KEYS.get(channel, f"temp{channel}")
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                out["temps"].append((key, path))

            for channel in (1, 2):
                path = p(f"freq{channel}_input")
                if path is None:
                    continue
                label = _hwmon_label(hwmon_dir, f"freq{channel}_label")
                label = label or ("sclk" if channel == 1 else "mclk")
                slot = "hwmon_freq" if label == "sclk" else "hwmon_mclk"
                out[slot] = out[slot] or path

            out["fan_rpm"] = out["fan_rpm"] or p("fan1_input")
            out["pwm"] = out["pwm"] or p("pwm1")
            out["pwm_max"] = out["pwm_max"] or p("pwm1_max")

            for channel in range(0, 3):
                path = p(f"in{channel}_input")
                if path and _hwmon_label(hwmon_dir, f"in{channel}_label") in (None, "vddgfx"):
                    out["vddgfx"] = out["vddgfx"] or path
                    break

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

        # FIX-AMD-RPM-1: a headless dGPU is runtime-powered-off (BOCO/BACO)
        # after a few idle seconds, and while it is, the driver refuses
        # every sensor read rather than wake the card.  Without saying so the
        # card just turns into a wall of dashes that looks like a bug.
        state = _read_word(device.extra.get("runtime_status"))
        if state:
            block["power_state"] = "suspended" if state == "suspended" else "active"

        busy = read_int_file(device.extra.get("busy"))
        if busy is not None:
            block["utilisation"] = {"value": float(max(0, min(100, busy)))}

        # Engine breakdown, summed from this tick's fdinfo attribution pass.
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

        power = self._read_power(device)
        if power:
            block["power"] = power

        temperature = self._read_temperatures(device)
        if temperature:
            block["temperature"] = temperature

        fan = self._read_fan(device)
        if fan:
            block["fan"] = fan

        vddgfx = read_int_file(device.extra.get("vddgfx"))
        if vddgfx is not None:
            block["voltage"] = {"gfx_mv": vddgfx}

        return block

    @staticmethod
    def _read_power(device: GpuDevice) -> dict[str, float]:
        power: dict[str, float] = {}
        for path in device.extra.get("power") or ():
            power_uw = read_int_file(path)
            if power_uw is not None:
                power["watts"] = round(power_uw / 1_000_000.0, 2)
                break
        if "watts" in power:
            cap_uw = read_int_file(device.extra.get("power_cap"))
            if cap_uw:
                power["cap_watts"] = round(cap_uw / 1_000_000.0, 1)
        return power

    @staticmethod
    def _read_temperatures(device: GpuDevice) -> dict:
        channels: dict[str, float] = {}
        for key, path in device.extra.get("temps") or ():
            milli_c = read_int_file(path)
            # amdgpu reports an unsupported/absent sensor as an error, but
            # some firmware returns 0 or a large negative sentinel instead;
            # neither is a temperature a running GPU can have.
            if milli_c is None or milli_c <= 0 or milli_c > 150_000:
                continue
            channels[key] = round(milli_c / 1000.0, 1)

        if not channels:
            return {}

        primary = next(
            (k for k in _TEMP_PRIMARY_ORDER if k in channels), next(iter(channels))
        )
        return {
            "celsius": channels[primary],
            "sensor": primary,
            "channels": channels,
        }

    @staticmethod
    def _read_fan(device: GpuDevice) -> dict[str, float]:
        fan: dict[str, float] = {}
        rpm = read_int_file(device.extra.get("fan_rpm"))
        if rpm is not None and rpm >= 0:
            fan["rpm"] = float(rpm)
        pwm = read_int_file(device.extra.get("pwm"))
        if pwm is not None:
            pwm_max = read_int_file(device.extra.get("pwm_max")) or 255
            fan["pct"] = round(max(0.0, min(100.0, pwm * 100.0 / pwm_max)), 1)
        return fan

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
        # FIX-AMD-ENG-1: an idle GPU is 0%, not "unknown".  amdgpu only
        # prints drm-engine-* lines for engines a client has actually used,
        # so on an idle card there is nothing to sum — which used to blank
        # the whole engine chart and grey out its legend.  Zeros are honest
        # here because the scan DID run; only when attribution is off or
        # failed (no scan this tick) is the breakdown truly unknown.
        if not self._attrib_ran:
            return {}
        per_pid = self._last_attrib.get(uid) or {}

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
        # FIX-AMD-CLK-2: 0 Hz is a real reading — the shader clock of an
        # idle card in GFXOFF/deep sleep — not "missing".  Dropping it made
        # an idle AMD card show "—" while an idle Intel iGPU shows "0".
        hz = read_int_file(device.extra.get("hwmon_freq"))
        if hz is not None and hz >= 0:
            freq["actual"] = round(hz / 1_000_000.0, 1)

        # Fallback / supplement: the starred entry in the DPM table, plus the
        # table's top level as the maximum.
        sclk_path = device.extra.get("sclk")
        if sclk_path:
            try:
                text = read_text(sclk_path)
            except OSError:
                text = ""
            match = _SCLK_ACTIVE_RE.search(text)
            if match:
                freq.setdefault("actual", float(match.group(1)))
            # Numbered levels only: the "S:" deep-sleep line is a current
            # reading, not a DPM state, and must not define the maximum.
            levels = [int(m) for m in _SCLK_LEVEL_RE.findall(text)]
            if levels:
                freq["max"] = float(max(levels))

        mclk_hz = read_int_file(device.extra.get("hwmon_mclk"))
        if mclk_hz is not None and mclk_hz >= 0:
            freq["memory"] = round(mclk_hz / 1_000_000.0, 1)

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
            self._attrib_ran = False
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
        self._attrib_ran = result is not None
        return result


# ── module helpers ───────────────────────────────────────────────────────


def _read_word(path: str | None) -> str | None:
    """First whitespace-delimited token of a small sysfs file, lower-cased."""
    if path is None:
        return None
    try:
        text = read_text(path).strip()
    except OSError:
        return None
    return text.split()[0].lower() if text else None


def _hwmon_label(hwmon_dir: str, name: str) -> str | None:
    """Read an hwmon ``*_label`` file (e.g. temp2_label -> "junction")."""
    return _read_word(os.path.join(hwmon_dir, name))
