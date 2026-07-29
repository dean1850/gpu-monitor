# FIX-P9-DOCK-1 (was FIX-P7-DOCK-1): intel-gpu-tools is no longer installed —
# see FIX-P9-GPU-1 in gpu_reader.py for why (intel_gpu_top is not used at
# all; engine-busy counters come from the i915 perf PMU via the raw
# perf_event_open() syscall, pure Python/ctypes, no external binary).
#
# ubuntu:24.04 is kept as the base image regardless: it's a current,
# well-supported glibc target for the ctypes syscall wrapper, and gives us
# python3/python3-venv + ca-certificates from a single, familiar apt feed.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# python3/venv     → run the FastAPI app.
# ca-certificates  → TLS roots for any outbound calls (e.g. aiodocker images).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 \
      python3-venv \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# FIX-P7-DOCK-2: isolated virtualenv keeps us clear of Ubuntu's PEP 668
# "externally managed environment" guard without resorting to
# --break-system-packages.  Putting the venv first on PATH means `python`
# and `pip` resolve to it for the rest of the build and at runtime.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install Python deps before copying app source so the Docker layer cache is
# preserved when only source files change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app/ contents to WORKDIR (/app/).
#   app/main.py              → /app/main.py          (uvicorn main:app)
#   app/gpu_reader.py        → /app/gpu_reader.py
#   app/container_resolver.py→ /app/container_resolver.py
#   app/static/              → /app/static/          (StaticFiles(directory="static"))
COPY app/ .

EXPOSE 8088

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088"]
