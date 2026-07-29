# SPDX-License-Identifier: MIT
# Copyright (c) 2026 gpu-monitor contributors
"""
Runtime configuration — environment variables only.

FIX-P11-CFG-1: NEW MODULE.  Prompt 9 adds multi-vendor GPU support, which
needs per-vendor enable/disable switches.  Rather than introduce a config
file format (and a YAML dependency), every knob is an environment variable
read from ``.env`` (which compose passes through via ``env_file``) — that is
already the single place this deployment is configured, and it needs no new
parser, no volume mount, and no "did you rebuild after editing config.yaml?"
class of bug.

Every default below is duplicated nowhere: ``.env`` is optional, so anything
absent falls back to the value in this module.

Vendor switches are TRI-STATE, not boolean:

    auto   (default) enable the backend if, and only if, matching hardware
           is actually detected.  Absent hardware is silent — not an error.
    true   force the backend on.  If the hardware or its userspace library
           is missing, the backend reports an explicit error via /health
           and the dashboard shows why.  Use this when you expect a card to
           be there and want to be told loudly if it isn't.
    false  never probe.  No sysfs walk, no dlopen of libnvidia-ml.so.
           Use this to keep the logs and /health output clean on a host
           that will never have that vendor.

"auto" is the default because it is the only value that behaves correctly
on an unknown machine, which matters for a public release.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Vendor identifiers used as dict keys, sample fields, and CSS hooks.
VENDOR_INTEL = "intel"
VENDOR_AMD = "amd"
VENDOR_NVIDIA = "nvidia"
ALL_VENDORS: tuple[str, ...] = (VENDOR_INTEL, VENDOR_AMD, VENDOR_NVIDIA)

# Tri-state values.
MODE_AUTO = "auto"
MODE_ON = "true"
MODE_OFF = "false"

_TRUEISH = {"true", "1", "yes", "on", "enable", "enabled"}
_FALSEISH = {"false", "0", "no", "off", "disable", "disabled"}


def _tristate(name: str, default: str = MODE_AUTO) -> str:
    """
    Parse a tri-state env var into MODE_AUTO / MODE_ON / MODE_OFF.

    FIX-P11-CFG-2: unrecognised values fall back to the default and log a
    warning rather than raising.  A typo in .env ("GPU_ENABLE_INTEL=ture")
    must not prevent the container from starting — it should start, monitor
    what it can, and tell you what it ignored.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in _TRUEISH:
        return MODE_ON
    if value in _FALSEISH:
        return MODE_OFF
    if value == MODE_AUTO:
        return MODE_AUTO

    logger.warning(
        "config: %s=%r is not one of auto/true/false — falling back to %r.",
        name, raw, default,
    )
    return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUEISH:
        return True
    if value in _FALSEISH:
        return False
    logger.warning(
        "config: %s=%r is not a boolean — falling back to %r.", name, raw, default
    )
    return default


def _float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("config: %s=%r is not a number — using %s.", name, raw, default)
        return default
    if not (minimum <= value <= maximum):
        logger.warning(
            "config: %s=%s is outside [%s, %s] — clamping.",
            name, value, minimum, maximum,
        )
        return max(minimum, min(maximum, value))
    return value


def _int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("config: %s=%r is not an integer — using %s.", name, raw, default)
        return default
    if not (minimum <= value <= maximum):
        logger.warning(
            "config: %s=%s is outside [%s, %s] — clamping.",
            name, value, minimum, maximum,
        )
        return max(minimum, min(maximum, value))
    return value


@dataclass(frozen=True)
class Config:
    """Immutable snapshot of the environment, read once at startup."""

    # Per-vendor tri-state switches.
    vendors: dict[str, str] = field(default_factory=dict)

    # Sampling.
    sample_interval_s: float = 1.0
    history_size: int = 300

    # Per-container attribution (the /proc fdinfo + NVML process scan).
    container_attribution: bool = True

    # Frontend: how many seconds of history the rolling charts show.
    chart_window_s: int = 60

    # Emit the pre-Prompt-9 flat top-level engines/frequency/rc6/containers
    # keys alongside the new "gpus" list.
    legacy_sample_fields: bool = True

    def is_enabled(self, vendor: str) -> bool:
        """True unless the vendor is explicitly switched off."""
        return self.vendors.get(vendor, MODE_AUTO) != MODE_OFF

    def is_forced(self, vendor: str) -> bool:
        """True when the operator demanded this vendor be present."""
        return self.vendors.get(vendor, MODE_AUTO) == MODE_ON

    def mode(self, vendor: str) -> str:
        return self.vendors.get(vendor, MODE_AUTO)

    def describe(self) -> dict:
        """JSON-serialisable view for /health."""
        return {
            "vendors": dict(self.vendors),
            "sample_interval_s": self.sample_interval_s,
            "history_size": self.history_size,
            "container_attribution": self.container_attribution,
            "chart_window_s": self.chart_window_s,
            "legacy_sample_fields": self.legacy_sample_fields,
        }


def load_config() -> Config:
    """Read every supported environment variable exactly once."""
    vendors = {
        VENDOR_INTEL: _tristate("GPU_ENABLE_INTEL"),
        VENDOR_AMD: _tristate("GPU_ENABLE_AMD"),
        VENDOR_NVIDIA: _tristate("GPU_ENABLE_NVIDIA"),
    }

    cfg = Config(
        vendors=vendors,
        # FIX-P11-CFG-3: the lower bound is 0.2 s, not 0.  A zero or
        # negative interval would spin the sample loop at 100% CPU and,
        # worse, make every Δcounter/Δwall division unstable.
        sample_interval_s=_float("GPU_SAMPLE_INTERVAL_S", 1.0, 0.2, 60.0),
        history_size=_int("GPU_HISTORY_SIZE", 300, 10, 5000),
        container_attribution=_bool("GPU_CONTAINER_ATTRIBUTION", True),
        chart_window_s=_int("GPU_CHART_WINDOW_S", 60, 10, 3600),
        legacy_sample_fields=_bool("GPU_LEGACY_SAMPLE_FIELDS", True),
    )

    logger.info(
        "config: intel=%s amd=%s nvidia=%s interval=%.2fs history=%d "
        "attribution=%s chart_window=%ds",
        vendors[VENDOR_INTEL], vendors[VENDOR_AMD], vendors[VENDOR_NVIDIA],
        cfg.sample_interval_s, cfg.history_size,
        cfg.container_attribution, cfg.chart_window_s,
    )
    return cfg
