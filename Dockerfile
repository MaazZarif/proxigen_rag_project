# Proxigen RAG backend image (CPU only).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/models/huggingface \
    FASTEMBED_CACHE_PATH=/models/fastembed

WORKDIR /app

RUN pip install --no-cache-dir uv

# Install the exact versions from uv.lock, but WITHOUT the NVIDIA/CUDA packages
# and with CPU-only PyTorch. This keeps the image a few GB smaller. The app only
# runs small models on CPU, so no GPU is needed.
COPY pyproject.toml uv.lock README.md ./
# If the CPU-only download is unavailable, it falls back to the standard
# (larger) packages from uv.lock so the build still succeeds.
RUN uv export --frozen --no-hashes --no-emit-project > /tmp/full.txt \
 && grep -Ev '^(nvidia-|cuda-|triton)' /tmp/full.txt > /tmp/cpu.txt \
 && ( uv pip install --system --no-cache --no-deps --torch-backend cpu -r /tmp/cpu.txt \
      || uv pip install --system --no-cache --no-deps -r /tmp/full.txt ) \
 && rm /tmp/full.txt /tmp/cpu.txt

COPY src ./src
COPY data/cleaned ./data/cleaned

# Run as a normal user. /models holds downloaded AI models (kept in a volume).
RUN useradd --create-home app \
 && mkdir -p /models \
 && chown -R app:app /models /app
USER app

EXPOSE 8000

# 1) Index the knowledge base (safe to repeat: chunk IDs are deterministic, so
#    existing chunks are overwritten, not duplicated).
# 2) Start the API.
CMD ["sh", "-c", "python -m proxigen_rag.ingestion.ingestion && uvicorn proxigen_rag.main:app --host 0.0.0.0 --port 8000"]
