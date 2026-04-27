#!/bin/sh
# Start Ollama in the background, wait until its HTTP API is ready, then
# launch the FastAPI app. Ollama keeps running for the lifetime of the
# container as a child process.
set -e

ollama serve &
OLLAMA_PID=$!

echo "[entrypoint] waiting for Ollama at ${OLLAMA_BASE_URL} ..."
for i in $(seq 1 60); do
    if curl -fs "${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1; then
        echo "[entrypoint] Ollama is ready."
        break
    fi
    sleep 1
done

# Best-effort: make sure the configured model is present. Pulled at build
# time, but if a different OLLAMA_MODEL was supplied at `docker run` we try
# to pull it now (requires internet). Failures are non-fatal - the API will
# still start and surface a clear error on /query.
if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx "${OLLAMA_MODEL}"; then
    echo "[entrypoint] model '${OLLAMA_MODEL}' not found locally, attempting pull ..."
    ollama pull "${OLLAMA_MODEL}" || \
        echo "[entrypoint] WARNING: could not pull '${OLLAMA_MODEL}'. /query will fail until the model is available."
fi

# Forward signals so docker stop is graceful.
trap 'kill -TERM "$OLLAMA_PID" 2>/dev/null; exit 0' INT TERM

echo "[entrypoint] starting uvicorn on 0.0.0.0:8000"
exec uvicorn rag_nxp.app:app --host 0.0.0.0 --port 8000
