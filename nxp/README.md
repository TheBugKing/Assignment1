# RAG NXP

Local, offline-capable document Q&A. Upload a PDF and ask questions grounded
in its content. The Docker image is fully self-contained — it bundles the
FastAPI app, a local LLM runtime (Ollama), the chat model weights
(`llama3.2:3b`), and the embedding model. No API keys, no external services.

- FastAPI backend (`rag_nxp/app.py`)
- ChromaDB vector store + `sentence-transformers` embeddings
- Ollama local LLM, baked into the image
- Single-file static UI (`static/index.html`)
- In-memory multi-turn chat with follow-up rewriting

---

## Quick start (Docker)

```bash
docker build -t rag-app .
docker run -p 8000:8000 rag-app
```

The first build is slow (it downloads the LLM weights into the image, ~2 GB
for `llama3.2:3b`). Subsequent builds are cached. Once the container is up:

1. Open [http://localhost:8000](http://localhost:8000).
2. Upload your PDF in the "Upload PDF" row.
3. Ask questions.

The container does **not** ship with any document — you ingest your own.

### API endpoints

| Method | Path             | Description                                                                |
| ------ | ---------------- | -------------------------------------------------------------------------- |
| `GET`  | `/`              | Serves the chat UI.                                                        |
| `GET`  | `/health`        | Status, configured model, and current index size.                          |
| `POST` | `/ingest/upload` | Multipart PDF upload + index (used by the UI).                             |
| `POST` | `/ingest`        | Re-index a PDF already inside the container at `/app/data/<pdf_filename>`. |
| `POST` | `/query`         | `{"question": "...", "session_id": "..."}` → grounded JSON answer.         |

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ingest/upload \
    -F "file=@./your.pdf"

curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"question": "What are the main safety rules?"}'
```

### Persisting state across container runs (optional)

The chat index lives in `/app/chroma_db` and uploaded PDFs in `/app/data`.
Mount volumes if you want them to survive `docker rm`:

```bash
docker run -p 8000:8000 \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/chroma_db:/app/chroma_db" \
    rag-app
```

### Using a different model

`llama3.2:3b` is the default and is baked into the image. To run with a
different model (e.g. a smaller/faster one on a low-RAM machine), override
`OLLAMA_MODEL` at run time. The first query will trigger a pull from
ollama.com, which requires internet access:

```bash
docker run -p 8000:8000 -e OLLAMA_MODEL=llama3.2:1b rag-app
docker run -p 8000:8000 -e OLLAMA_MODEL=qwen2.5:1.5b rag-app
docker run -p 8000:8000 -e OLLAMA_MODEL=phi3:mini    rag-app
```

---

## Local development (no Docker)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # macOS / Linux

pip install -r requirements.txt
copy .env.example .env         # cp on macOS / Linux

# Make sure Ollama is running on the host:
ollama pull llama3.2:3b
ollama serve

uvicorn rag_nxp.app:app --port 8000 --reload
```

---

## Configuration

All settings can be supplied as environment variables (or via `.env` for local
dev). Most useful ones:

| Variable             | Default (Docker)           | Purpose                                  |
| -------------------- | -------------------------- | ---------------------------------------- |
| `OLLAMA_BASE_URL`    | `http://127.0.0.1:11434`   | Ollama HTTP endpoint inside the image.   |
| `OLLAMA_MODEL`       | `llama3.2:3b`              | Any model Ollama can serve.              |
| `OLLAMA_NUM_PREDICT` | `512`                      | Max tokens generated per reply.          |
| `OLLAMA_TIMEOUT_S`   | `300`                      | HTTP timeout for the LLM call (seconds). |
| `OLLAMA_KEEP_ALIVE`  | `24h`                      | How long to keep the model warm.         |
| `CHUNKING_STRATEGY`  | `logical`                  | `fixed` \| `logical` \| `semantic`.      |
| `RETRIEVAL_K`        | `8`                        | Number of chunks returned per query.     |

The host `.env` shipped for local development uses the same defaults so
that `uvicorn` on the host and `docker run` produce identical behaviour.

---

## How the answers stay clean

- The model is told to return a single JSON object: `{"answer": "..."}`.
- `json_answer.py` tolerates fences, prose before/after JSON, and truncated
  JSON.
- If JSON parsing fails, a reasoning-aware fallback skips lines that look
  like chain-of-thought (e.g. "Wait, let me check…", "From passage 2…") and
  only surfaces real answer text. If every paragraph looks like reasoning,
  the service returns "I could not find this in the document" rather than
  leak the model's scratchpad to the UI.

---

## Troubleshooting

**`No indexed text yet`** — upload a PDF via the UI before querying.

**Slow first build** — the build step pulls the LLM (~2 GB for the default
`llama3.2:3b`) into the image so `docker run` works without internet.
Subsequent builds reuse the layer.

**Out of memory** — switch to a smaller model with `-e OLLAMA_MODEL=phi3:mini`
or `-e OLLAMA_MODEL=llama3.2:1b`.
