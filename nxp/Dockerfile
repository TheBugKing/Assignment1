# RAG NXP — self-contained image (FastAPI + Ollama + a small model).
#
# Build:
#   docker build -t rag-app .
#
# Run (uses the baked-in llama3.2:3b — best accuracy/latency on CPU):
#   docker run -p 8000:8000 rag-app
#
# Persist uploads + the vector index between runs:
#   docker run -p 8000:8000 \
#       -v rag-data:/app/data \
#       -v rag-chroma:/app/chroma_db \
#       rag-app
#
# Override the model at runtime (will pull on first start, needs internet):
#   docker run -p 8000:8000 -e OLLAMA_MODEL=llama3.2:1b rag-app
#
# On startup the container launches Ollama in the background and serves the
# FastAPI app on :8000. The default model (llama3.2:3b, ~2 GB) is baked
# into the image so the first query works offline.

FROM python:3.12-slim-bookworm

# These ENV defaults mirror the host .env we ship for local development so
# that `docker run rag-app` and `uvicorn` on the host produce identical
# behaviour — same model, same retrieval depth, same timeout.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    OLLAMA_MODEL=llama3.2:3b \
    OLLAMA_TIMEOUT_S=300 \
    OLLAMA_KEEP_ALIVE=24h \
    CHUNKING_STRATEGY=logical \
    RETRIEVAL_K=8 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        zstd \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache the embedding model so the first query doesn't have to download it.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Pre-pull the LLM weights so `docker run` works offline / on first request.
RUN set -eux; \
    ollama serve & OPID=$!; \
    for i in $(seq 1 60); do \
        if curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi; \
        sleep 1; \
    done; \
    ollama pull "$OLLAMA_MODEL"; \
    kill "$OPID" 2>/dev/null || true; \
    wait "$OPID" 2>/dev/null || true

COPY rag_nxp ./rag_nxp
COPY static ./static
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh && mkdir -p /app/data /app/chroma_db

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
