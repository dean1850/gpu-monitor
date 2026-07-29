# gpu-monitor

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A self-hosted, real-time GPU dashboard that tells you *which container* is
using the GPU — not just that something is.**

Intel, AMD and NVIDIA, concurrently, on the same host. One section per card.
Dark mode, modelled on GNOME System Monitor's Resources tab.

![gpu-monitor dashboard showing an Intel, an AMD and an NVIDIA GPU](docs/screenshot.png)

---

## About

Every GPU monitoring tool for Linux answers "how busy is the GPU?". Almost
none answer the question you actually have on a homelab box: **who is using
it right now?**

If Jellyfin, Frigate, Immich and Ollama all share one card, `intel_gpu_top`
tells you the video engine is at 60% and leaves you to guess which of the
four is responsible. `nvidia-smi` gives you PIDs, but those PIDs are
meaningless from outside the container that owns them.

gpu-monitor closes that gap. It reads the GPU's engine counters straight from
the kernel, walks `/proc/<pid>/fdinfo` for every process on the host holding a
DRM render-node file descriptor, and maps each PID back through the Docker
socket to the container that owns it. The result is a live percentage
breakdown per container, per engine, per GPU.

Some specifics that shaped the design:

- **Per-engine, not one number.** A GPU isn't one thing. An Intel iGPU has
  Render/3D, Blitter, Video and VideoEnhance, and a QSV transcode lights up
  the last two while leaving the first two idle. Collapsing that into a single
  "GPU: 34%" throws away the part that tells you what's actually happening.
  Every engine the card exposes gets its own line, in that card's own
  terminology — Intel's Render/3D and VideoEnhance, AMD's Graphics and
  Compute, NVIDIA's Encode and Decode.
- **Mixed vendors are a first-class case,** not a bolt-on. An iGPU handling
  transcodes alongside a discrete card doing inference is an ordinary homelab
  arrangement. Each card gets its own titled section with a coloured vendor
  badge, its own engine chart, and its own container breakdown, so it's
  obvious at a glance which card is doing the work.
- **No fabricated numbers.** An Intel iGPU has no dedicated VRAM, no power
  sensor and no temperature sensor exposed through i915. Those fields render
  as `—`, never as a plausible-looking zero.
- **No shelling out.** No `intel_gpu_top`, no `nvidia-smi` subprocess, no
  vendor Python packages. Metrics come from `perf_event_open(2)`, sysfs,
  `/proc`, and NVML through `ctypes`. The whole thing is stdlib plus FastAPI.
- **Degrades instead of dying.** A backend that can't find its hardware
  returns an empty list. A backend that breaks mid-run reports a reason
  through `/health` while the others keep sampling. With no GPU at all, the
  app still starts and tells you why it found nothing.

Built for a LAN-only homelab host. Read the [security model](#security-model)
before putting it anywhere else — it needs real privileges to do what it does.

---

## Quick start

```bash
git clone https://github.com/<your-user>/gpu-monitor.git
cd gpu-monitor

# 1. Create your config. Defaults work as-is; edit if you want to.
cp .env.example .env

# 2. Build and start.
docker compose up -d --build
```

Open **`http://<your-host>:8088`**.

> **`--build` is not optional.** `docker compose up -d` on its own silently
> reuses the cached image, and every change you just made will appear to do
> nothing. This catches everyone exactly once.

That's the whole install. With the default `auto` vendor switches, whatever
GPUs the host actually has show up and the rest stay silent — no inventory
step, no per-vendor setup, with one exception: **NVIDIA needs host-side
toolkit setup first**, covered [below](#nvidia-setup).

**Verify it's working:**

```bash
docker logs gpu-monitor          # one line per backend probe result
curl -s localhost:8088/health | python3 -m json.tool
```

Then put load on the GPU — start a transcode, run an inference job — and
watch the relevant engine line climb with the responsible container named
underneath it.

---

## Configuration

Everything a normal user needs is in **`.env`**, which Docker Compose reads
automatically from the project directory. It is git-ignored, so your settings
survive `git pull` and never end up in a commit.

```bash
cp .env.example .env
$EDITOR .env
docker compose up -d --build
```

| Variable | Default | Meaning |
|---|---|---|
| `TZ` | `UTC` | Any IANA name, e.g. `Europe/Berlin`, `America/New_York`. Affects displayed clock times only |
| `GPU_ENABLE_INTEL` | `auto` | `auto` \| `true` \| `false` |
| `GPU_ENABLE_AMD` | `auto` | `auto` \| `true` \| `false` |
| `GPU_ENABLE_NVIDIA` | `auto` | `auto` \| `true` \| `false` |

Every value has a working default baked into `docker-compose.yml`, so the
stack still starts correctly if you skip `.env` entirely.

### The vendor switches are tri-state

This matters more than it sounds:

- **`auto`** — enable the backend only if matching hardware is actually
  found. Absent hardware is silent, not an error. Correct on a machine you
  haven't inventoried, which is why it's the default.
- **`true`** — force it on. If the card or its userspace library is missing,
  `/health` says so explicitly and the dashboard shows why. Use this when you
  *expect* a card to be there and want to be told loudly if it isn't — the
  fastest way to debug a passthrough or driver problem.
- **`false`** — never probe. No sysfs walk, no `dlopen`. Keeps logs and
  `/health` clean on a host that will never have that vendor.

All three can be on at once.

### Tuning (edit `docker-compose.yml`)

Deliberately kept out of `.env` so that file stays short enough to read. The
defaults are right for essentially every host.

| Variable | Default | Meaning |
|---|---|---|
| `GPU_SAMPLE_INTERVAL_S` | `1.0` | Sampling period, clamped to 0.2–60 |
| `GPU_HISTORY_SIZE` | `300` | Samples retained for replay to new clients |
| `GPU_CHART_WINDOW_S` | `60` | Seconds visible on each chart |
| `GPU_CONTAINER_ATTRIBUTION` | `true` | `false` skips Docker and `/proc` entirely |
| `GPU_LEGACY_SAMPLE_FIELDS` | `true` | Also emit the flat pre-multi-GPU sample keys |

---

## What it measures

| Metric | Intel | AMD | NVIDIA |
|---|---|---|---|
| Per-engine utilisation | i915 perf PMU | summed from fdinfo | NVML |
| Overall utilisation | busiest engine | `gpu_busy_percent` | NVML |
| Clock | sysfs | hwmon / `pp_dpm_sclk` | NVML |
| Idle residency (RC6) | sysfs | — | — |
| VRAM | — (shared system RAM) | `mem_info_vram_*` | NVML |
| Power | — | hwmon | NVML |
| Temperature | — | hwmon | NVML |
| Per-container attribution | `/proc/*/fdinfo` | `/proc/*/fdinfo` | NVML process utilisation |

Anything a host can't report shows as `—`, never as a fabricated zero. This
is why the Intel section below has three empty stat fields and the AMD one
doesn't — that's correct behaviour, not a bug:

![gpu-monitor with a single Intel iGPU](docs/screenshot-intel.png)

> Both screenshots are the real dashboard rendered headlessly against
> synthetic samples, so the values are representative rather than measured.

---

## NVIDIA setup

NVIDIA GPUs are **not** reachable through `/dev/dri` — the proprietary driver
exposes no useful DRM sysfs, so passing the device through does nothing for
it. They need the NVIDIA Container Toolkit on the host and a device
reservation in Compose.

```bash
# 1. Install the toolkit for your distro, then:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 2. Verify BEFORE touching this project:
docker run --rm --gpus all ubuntu nvidia-smi
```

Once that prints a device table, uncomment the `deploy:` block at the bottom
of `docker-compose.yml` and redeploy. Compose can't conditionally include a
YAML block from an environment variable, which is why this one step isn't in
`.env`.

Leaving `GPU_ENABLE_NVIDIA=auto` without doing any of this is harmless — the
backend finds no `libnvidia-ml.so.1`, logs one line, and skips itself.

---

## Security model

This container is deliberately privileged. It is designed for a **LAN-only
homelab host**, not for exposure to the internet or an untrusted network.

| Grant | Why it's needed | Risk if you're uncomfortable |
|---|---|---|
| `CAP_PERFMON`, `CAP_SYS_ADMIN` | `perf_event_open()` for i915 PMU engine counters | Drop both → Intel engine charts go blank; clock/RC6/attribution still work |
| `CAP_SYS_PTRACE` | Read `/proc/<pid>/fdinfo` of processes owned by other users | Drop → containers running as non-root (linuxserver.io images) silently never appear |
| `seccomp:unconfined` | Docker's default profile blocks `perf_event_open()` even with the caps above | Drop → same as dropping `CAP_PERFMON` |
| `pid: host` | Attribution scans every host PID for DRM render-node fds | Drop → per-container rows disappear entirely |
| `/var/run/docker.sock:ro` | Map PIDs to container names; list GPU-capable containers | Drop → containers show as `host`; everything else works |
| `/dev/dri` | Read the GPU | Required for Intel/AMD |

The socket is mounted **read-only**. Without `:ro` this container could
control the Docker daemon, which is a full host compromise — if you fork
this, don't remove it.

To run in a reduced-privilege mode, set `GPU_CONTAINER_ATTRIBUTION=false` in
`docker-compose.yml` and remove `pid: host`, `CAP_SYS_PTRACE` and the socket
mount. You keep whole-GPU metrics and lose the per-container breakdown.

---

## Endpoints

| Path | Purpose |
|---|---|
| `/` | Dashboard |
| `/health` | Per-backend status, detected devices, resolver state, active config |
| `/api/gpus` | Static per-GPU identity (uid, vendor, name, engine names) |
| `/ws` | WebSocket: `devices` frame, then `history`, then `sample` every interval |

`/health` is the first place to look when something is missing — it reports,
per vendor, exactly which probe failed and why.

---

## Troubleshooting

**Engine charts are empty but clock and RC6 work.**
`perf_event_open()` is being blocked. Check `cap_add` and
`security_opt: seccomp:unconfined` in `docker-compose.yml`, then confirm with
`curl -s localhost:8088/health | grep error`.

**A container that uses the GPU isn't listed.**
Check `/health` → `container_resolver.gpu_containers_by_access`. If it isn't
under `drm`, the container's GPU access isn't being detected — confirm it has
`/dev/dri` via `devices:`, `volumes:`, or `privileged: true`. If it *is*
listed but shows 0%, it has access but isn't currently using the GPU.

**A GPU I know is present doesn't appear.**
Set that vendor's switch to `true` in `.env` and redeploy. `auto` stays quiet
about hardware it can't find; `true` makes the failure explicit in `/health`
and in the dashboard.

**Nothing appears at all.**
`docker logs gpu-monitor` reports each backend's probe result on startup.

**Changes to `.env` didn't take effect.**
You forgot `--build`. Everyone does, once.

---

## Architecture

```
app/
├── main.py                FastAPI: /health, /api/gpus, /ws, static
├── config.py              Environment parsing (tri-state vendor switches)
├── gpu_reader.py          Sample loop, history, PID→container bucketing
├── container_resolver.py  Docker socket: PID map + GPU-capable containers
├── backends/
│   ├── base.py            GpuBackend contract, GpuDevice
│   ├── drm.py             Shared: card enumeration + /proc fdinfo scanner
│   ├── intel.py           i915 perf PMU via perf_event_open(2)
│   ├── amd.py             amdgpu sysfs + hwmon
│   └── nvidia.py          NVML via ctypes
└── static/index.html      Dashboard (Chart.js bundled locally, no CDN)
```

Three rules every backend follows, because the dashboard has to work on a
host where two of the three vendors are absent:

1. `probe()` never raises — missing hardware returns an empty list.
2. `sample()` never raises — a broken backend degrades, others keep going.
3. Every metric is optional — emit only what you can actually measure.

---

## Tests

```bash
python3 tests/test_gpu_monitor.py
```

Runs on a machine with no GPU at all — kernel and Docker data are mocked.

---

## License

Released under the **MIT License** — see [LICENSE](LICENSE) for the full text.

In short: you may use, copy, modify, merge, publish, distribute, sublicense
and sell this software, including commercially, provided the copyright notice
and permission notice are retained. It is provided as-is, with no warranty.

Every source file carries an `SPDX-License-Identifier: MIT` header so the
licensing is machine-readable to tooling like [REUSE](https://reuse.software/)
and GitHub's licence detection.

Copyright (c) 2026 gpu-monitor contributors.

The bundled `app/static/chart.umd.min.js` is [Chart.js](https://www.chartjs.org/),
also MIT licensed, and retains its own copyright notice.
