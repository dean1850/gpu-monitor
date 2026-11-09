# Quick start

Everything you need to run gpu-monitor, and every setting it has.
For what it does and why, see the [README](README.md).

---

## Requirements

- Linux host (the metrics come from the kernel's DRM/i915/amdgpu interfaces)
- Docker Engine with Compose v2.24 or newer — `docker compose version`
- An Intel, AMD or NVIDIA GPU. NVIDIA needs [extra host setup](#nvidia-hosts);
  Intel and AMD work out of the box.

No clone, no build toolchain, no Python on the host. The image is published to
GitHub Container Registry and pulled ready to run.

---

## Install

```bash
mkdir gpu-monitor && cd gpu-monitor

# 1. Fetch the two files you need.
curl -O https://raw.githubusercontent.com/dean1850/gpu-monitor/main/docker-compose.yml
curl -o .env https://raw.githubusercontent.com/dean1850/gpu-monitor/main/.env.example

# 2. Optional: edit .env. The defaults work as-is.
$EDITOR .env

# 3. Pull and start.
docker compose up -d
```

Open **`http://<your-host>:8088`**.

That's the whole install. With the default `auto` vendor switches, whatever
GPUs the host actually has show up and the rest stay silent — no inventory
step and no per-vendor setup, with one exception: NVIDIA needs the host-side
toolkit first, covered [below](#nvidia-hosts).

> The `.env` file is optional. Skip it entirely and the stack still comes up
> correctly — every setting has the same default baked into the image.

### Verify

```bash
docker logs gpu-monitor          # one line per backend probe result
curl -s localhost:8088/health | python3 -m json.tool
```

Then put load on the GPU — start a transcode, run an inference job — and watch
the relevant engine line climb with the responsible container named
underneath it.

### Update

```bash
docker compose pull && docker compose up -d
```

`:latest` follows the `main` branch. To pin a release instead, set
`image: ghcr.io/dean1850/gpu-monitor:1.2.3` in `docker-compose.yml`.

---

## Configuration

All configuration lives in **`.env`**, next to `docker-compose.yml`. Compose
reads it automatically and passes it into the container, so you should never
need to edit the compose file itself. `.env` is git-ignored, so your settings
never end up in a commit.

After any change:

```bash
docker compose up -d
```

Compose notices the changed environment and recreates the container. There is
no `--build` step any more — the image is pulled, not built.

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `TZ` | `UTC` | Timezone for displayed clock times |
| `GPU_ENABLE_INTEL` | `auto` | `auto` \| `true` \| `false` |
| `GPU_ENABLE_AMD` | `auto` | `auto` \| `true` \| `false` |
| `GPU_ENABLE_NVIDIA` | `auto` | `auto` \| `true` \| `false` |
| `GPU_MONITOR_PORT` | `8088` | Host port for the dashboard |
| `DOCKER_RUNTIME` | `runc` | Set to `nvidia` on NVIDIA hosts |
| `NVIDIA_VISIBLE_DEVICES` | `void` | Set to `all` on NVIDIA hosts |

### Timezone

`TZ` affects the wall-clock times in the dashboard header and in
`docker logs gpu-monitor`. It does **not** affect sampling or the charts —
those are all relative ("60 seconds ago" … "0").

Use any [IANA timezone name](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).
To match the Docker host rather than pick one by hand:

```bash
timedatectl show -p Timezone --value      # or: cat /etc/timezone
```

A few for reference:

```
UTC                 Europe/London       Europe/Berlin       Europe/Madrid
America/New_York    America/Chicago     America/Denver      America/Bogota
America/Los_Angeles America/Sao_Paulo   Africa/Lagos        Africa/Nairobi
Asia/Dubai          Asia/Kolkata        Asia/Shanghai       Asia/Tokyo
Australia/Perth     Australia/Sydney    Pacific/Auckland
```

### The vendor switches are tri-state

One switch per vendor, and all three can be on at once — a host with an Intel
iGPU and an NVIDIA card gets a section for each, side by side in the UI.

Each switch takes three values, not two:

- **`auto`** — enable this backend only if matching hardware is actually
  found. Absent hardware is silent, not an error. This is the default and the
  right answer for almost everyone: leave it alone and the dashboard shows
  exactly the GPUs you have.
- **`true`** — force the backend on. If the hardware or its userspace library
  is missing, the backend reports an explicit reason via `/health` and the
  dashboard tells you why instead of quietly showing nothing. Use this when
  you *expect* a card to be there and want to be told loudly if it isn't —
  the fastest way to debug a passthrough or driver problem.
- **`false`** — never probe. Skips the sysfs walk and library load entirely.
  Use this to keep logs and `/health` output clean on a host that will never
  have that vendor.

Anything else — a typo like `ture` — logs a warning and falls back to the
default rather than refusing to start.

Per vendor:

- **Intel** — integrated graphics (UHD / Iris Xe) and Arc discrete cards, via
  the `i915` or `xe` driver. Uses the `/dev/dri` passthrough already in the
  compose file. This is the fully-tested path.
- **AMD** — Radeon discrete and integrated graphics via `amdgpu`. Also
  `/dev/dri`, so no extra host setup.
- **NVIDIA** — read through NVML, **not** `/dev/dri`. The proprietary driver
  exposes no useful DRM sysfs, so passing `/dev/dri` through does nothing for
  it. See [below](#nvidia-hosts).

### Tuning

These are not in `.env.example`, deliberately — the defaults are correct for
essentially every host, and keeping them out keeps that file short enough
that people actually read it. Add any of them to `.env` if you need to.

| Variable | Default | Range | Meaning |
|---|---|---|---|
| `GPU_SAMPLE_INTERVAL_S` | `1.0` | 0.2–60 | Sampling period in seconds |
| `GPU_HISTORY_SIZE` | `300` | 10–5000 | Samples retained for replay to new clients |
| `GPU_CHART_WINDOW_S` | `60` | 10–3600 | Seconds visible on each chart |
| `GPU_CONTAINER_ATTRIBUTION` | `true` | | `false` skips the Docker socket and `/proc` scan entirely |
| `GPU_LEGACY_SAMPLE_FIELDS` | `true` | | Also emit the flat pre-multi-GPU sample keys |

Out-of-range numbers are clamped and logged, not rejected.

---

## NVIDIA hosts

NVIDIA GPUs are not reachable through `/dev/dri`. The container has to be
handed the device by the NVIDIA Container Toolkit, which means host setup
first.

```bash
# 1. Install the NVIDIA Container Toolkit for your distro, then:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 2. Verify BEFORE touching this project:
docker run --rm --gpus all ubuntu nvidia-smi
```

Once that prints a device table, set both of these in `.env`:

```ini
DOCKER_RUNTIME=nvidia
NVIDIA_VISIBLE_DEVICES=all
```

and `docker compose up -d`. That's all — there is no YAML to uncomment.

Both lines are needed. `DOCKER_RUNTIME` selects the runtime that can inject
the GPU; `NVIDIA_VISIBLE_DEVICES` tells it which devices to inject. The
defaults (`runc` / `void`) mean "no NVIDIA", and `void` is left in place on
Intel/AMD hosts on purpose: it is the sentinel the container resolver reads as
"this container has no NVIDIA access", so gpu-monitor doesn't list itself as
an NVIDIA consumer on a host that has no NVIDIA card.

Leaving `GPU_ENABLE_NVIDIA=auto` without doing any of the above is harmless —
the backend finds no `libnvidia-ml.so.1`, logs one line, and skips itself.

### A host with no `/dev/dri` at all

The compose file passes `/dev/dri` through unconditionally, and Docker refuses
to start a container whose device is missing. On an NVIDIA-only host without a
DRM render node, delete these two lines from `docker-compose.yml`:

```yaml
    devices:
      - /dev/dri:/dev/dri
```

Set `GPU_ENABLE_INTEL=false` and `GPU_ENABLE_AMD=false` while you're there, to
keep `/health` clean.

---

## Reduced-privilege mode

gpu-monitor is deliberately privileged — see the
[security model](README.md#security-model) for what each grant buys and what
breaks without it. To run it with the sharp edges off, set this in `.env`:

```ini
GPU_CONTAINER_ATTRIBUTION=false
```

and remove `pid: host`, `SYS_PTRACE`, and the `/var/run/docker.sock` mount
from `docker-compose.yml`. You keep whole-GPU metrics and lose the
per-container breakdown, which is most of the point of the tool — so do this
only if the trade-off is one you actually want.

---

## Troubleshooting

`/health` is the first place to look when something is missing. It reports,
per vendor, exactly which probe failed and why.

**Engine charts are empty but clock and RC6 work.**
`perf_event_open()` is being blocked. The `cap_add` entries (`PERFMON`,
`SYS_ADMIN`) and `security_opt: seccomp:unconfined` in `docker-compose.yml`
are both required — Docker's default seccomp profile blocks that syscall even
when the capability is granted. Confirm with
`curl -s localhost:8088/health | grep error`.

**Intel iGPU shows `—` for Power, Temp or Memory.**
Hover the tile: it says why it's empty. The startup log says which source
each one resolved to (`docker logs gpu-monitor | grep intel:`). Power needs the RAPL GPU plane and temperature
the CPU package sensor; check on the Docker host with
`ls /sys/devices/power/events/ /sys/class/powercap/` and
`cat /sys/class/hwmon/*/name`. Neither exists inside a VM, so a Proxmox guest
with the iGPU passed through will keep showing `—` for those two. Memory
needs kernel 6.8+ (`uname -r`) and `GPU_CONTAINER_ATTRIBUTION` left on, and
appears once something has opened the GPU.

**A container that uses the GPU isn't listed.**
Check `/health` → `container_resolver.gpu_containers_by_access`. If it isn't
under `drm`, that container's GPU access isn't being detected — confirm it has
`/dev/dri` via `devices:`, `volumes:`, or `privileged: true`. If it *is*
listed but shows 0%, it has access but isn't currently using the GPU.

**Containers show up as `host` instead of by name.**
The Docker socket isn't readable. It is mounted read-only on purpose; check
the `volumes:` entry is present.

**Containers running as a non-root user never appear.**
That needs `CAP_SYS_PTRACE` to read `/proc/<pid>/fdinfo` for processes owned
by other users — linuxserver.io images are the common case.

**A GPU I know is present doesn't appear.**
Set that vendor's switch to `true` in `.env` and redeploy. `auto` stays quiet
about hardware it can't find; `true` makes the failure explicit in `/health`
and in the dashboard.

**Nothing appears at all.**
`docker logs gpu-monitor` reports each backend's probe result on startup.

**`docker compose up` fails on the `env_file` block.**
Compose is older than v2.24. Either upgrade, or create an empty `.env` and
change the block to `env_file: [.env]`.

**Changes to `.env` didn't take effect.**
Run `docker compose up -d` again — Compose only re-reads `.env` when it
recreates the container.

---

## Building from source

Only needed if you're changing the code; normal use pulls the image.

```bash
git clone https://github.com/dean1850/gpu-monitor.git
cd gpu-monitor
cp .env.example .env

docker build -t ghcr.io/dean1850/gpu-monitor:latest .
docker compose up -d
```

Tagging your local build with the same name means `docker-compose.yml` picks
it up with no edits. Run `docker compose pull` to go back to the published
image.

Tests need no GPU and no dependencies — kernel and Docker access are mocked:

```bash
python3 tests/test_gpu_monitor.py
```
