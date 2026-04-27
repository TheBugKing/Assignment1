"""RAG NXP FastAPI app — ingest, query, static UI, in-memory chat sessions."""

from __future__ import annotations

import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_nxp.config import settings
from rag_nxp.greetings import greeting_reply
from rag_nxp.rag_core import RagNxp, pdf_path_default

rag = RagNxp()

# session_id -> [{"role": "user"|"assistant", "content": str}, ...]
CHAT_SESSIONS: dict[str, list[dict[str, str]]] = {}
MAX_MESSAGES_PER_SESSION = 40
HISTORY_MESSAGES_FOR_PROMPT = 14


def _trim_session(sid: str) -> None:
    if sid in CHAT_SESSIONS and len(CHAT_SESSIONS[sid]) > MAX_MESSAGES_PER_SESSION:
        CHAT_SESSIONS[sid] = CHAT_SESSIONS[sid][-MAX_MESSAGES_PER_SESSION:]


def _format_chat_history(messages: list[dict[str, str]], max_chars: int = 6000) -> str:
    lines: list[str] = []
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _merge_retrieval_query(question: str, prior_messages: list[dict[str, str]]) -> str:
    """Combine prior user turn with short follow-ups so embeddings still match."""
    q = question.strip()
    if not prior_messages:
        return q
    last_user = None
    for m in reversed(prior_messages):
        if m["role"] == "user":
            last_user = m["content"]
            break
    if not last_user:
        return q
    low = q.lower().rstrip("?!. ")
    brief = len(q.split()) <= 8
    vague_starts = (
        "where",
        "what ",
        "how ",
        "why ",
        "when ",
        "who ",
        "which ",
        "what about",
        "how about",
        "and the",
        "same ",
        "is that",
        "explain",
    )
    vague_word = low in {
        "where",
        "what",
        "how",
        "why",
        "when",
        "who",
        "which",
        "ok",
        "yes",
        "no",
        "thanks",
        "and",
        "so",
    }
    if brief or vague_word or any(low.startswith(v) for v in vague_starts):
        return f"{last_user}\nFollow-up: {q}"
    return q


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = pdf_path_default()
    if path.is_file() and rag.index_size() == 0:
        try:
            rag.ingest_pdf(path, replace=True)
        except Exception:
            pass
    yield


app = FastAPI(title="RAG NXP API", version="0.1.0", lifespan=lifespan)

static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class QueryBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(None, description="Chat session id; omit to start a new session")


class RetrievedChunkOut(BaseModel):
    rank: int
    content: str
    source: str | None = None
    page_hint: str | None = None
    section: str | None = None
    page_number: int | None = None
    distance: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunkOut]
    model: str
    model_metadata: dict = Field(default_factory=dict)
    session_id: str


class IngestResponse(BaseModel):
    chunks_indexed: int
    pdf_path: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm": settings.ollama_model,
        "ollama_url": settings.ollama_base_url,
        "index_size": rag.index_size(),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(pdf: str | None = None):
    path = Path(pdf) if pdf else pdf_path_default()
    try:
        n = rag.ingest_pdf(path, replace=True)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return IngestResponse(chunks_indexed=n, pdf_path=str(path.resolve()))


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.data_dir / Path(file.filename).name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        n = rag.ingest_pdf(dest, replace=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return IngestResponse(chunks_indexed=n, pdf_path=str(dest.resolve()))


def _chunks_to_sources(chunks) -> list[RetrievedChunkOut]:
    return [
        RetrievedChunkOut(
            rank=i,
            content=ch.text,
            source=ch.source,
            page_hint=ch.page_hint,
            section=getattr(ch, "section", None),
            page_number=getattr(ch, "page_number", None),
            distance=ch.distance,
        )
        for i, ch in enumerate(chunks, start=1)
    ]


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryBody):
    q = body.question.strip()
    sid = (body.session_id or "").strip() or str(uuid.uuid4())
    CHAT_SESSIONS.setdefault(sid, [])
    prior = CHAT_SESSIONS[sid]

    canned = greeting_reply(q)
    if canned:
        prior.extend([{"role": "user", "content": q}, {"role": "assistant", "content": canned}])
        _trim_session(sid)
        return QueryResponse(
            answer=canned,
            sources=[],
            model="greeting",
            model_metadata={"reply_type": "greeting"},
            session_id=sid,
        )

    retrieval_q = _merge_retrieval_query(q, prior)
    chunks = rag.retrieve(retrieval_q)
    if not chunks:
        raise HTTPException(
            status_code=503,
            detail="No indexed text yet. Upload a PDF via the UI or drop one in data/ and call POST /ingest.",
        )

    prior_tail = prior[-HISTORY_MESSAGES_FOR_PROMPT:] if prior else []
    history_text = _format_chat_history(prior_tail) if prior_tail else None

    try:
        answer, llm_meta = await rag.generate_grounded_answer(
            q, chunks, chat_history=history_text
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    prior.extend([{"role": "user", "content": q}, {"role": "assistant", "content": answer}])
    _trim_session(sid)

    return QueryResponse(
        answer=answer,
        sources=_chunks_to_sources(chunks),
        model=settings.ollama_model,
        model_metadata={
            "embedding_model": settings.embedding_model,
            "chunks_used": len(chunks),
            **llm_meta,
        },
        session_id=sid,
    )


@app.get("/")
def root():
    index = static_dir / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"message": 'POST /query with {"question": "...", "session_id": "..."}'}
