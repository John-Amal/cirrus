# Inference image for the cirrus service.
#
# Two stages: the first installs dependencies into a virtual environment, the
# second copies that environment into a clean image. The build toolchain and
# pip's caches stay behind, which matters here because the CPU-only torch
# wheel is already large and the default CUDA one would add gigabytes nobody
# can use in this container.

FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# CPU-only torch: the default wheel pulls ~2.5 GB of CUDA libraries that an
# inference container will never execute.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[serve]"


FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
# The exported model and the statistics it was trained with. The statistics
# are not optional: the service normalises inputs with them, and different
# ones would silently produce wrong answers rather than an error.
COPY serve/ ./serve/
COPY configs/ ./configs/
COPY data/stats/ ./data/stats/

# Run as a non-root user; nothing here needs write access at runtime.
RUN useradd --create-home --uid 10001 cirrus && chown -R cirrus /app
USER cirrus

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# 0.0.0.0 inside the container so the port can be published; the CLI default
# of 127.0.0.1 is the right one for a local run and the wrong one here.
CMD ["cirrus", "serve", "--host", "0.0.0.0", "--port", "8000"]
