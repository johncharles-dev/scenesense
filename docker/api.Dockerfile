# API image: orchestration only (agent loop, tools, Chroma, CPU embedder). Models are served by vLLM.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy YOLO_AUTOINSTALL=False PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Dependencies first (cached layer). TensorRT / ModelOpt are only used for the detector speed benchmarks.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project \
    --no-install-package tensorrt-cu13 --no-install-package nvidia-modelopt
COPY src ./src
RUN uv sync --frozen --no-dev \
    --no-install-package tensorrt-cu13 --no-install-package nvidia-modelopt

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080
CMD ["uvicorn", "scenesense.serve.api:app", "--host", "0.0.0.0", "--port", "8080"]
