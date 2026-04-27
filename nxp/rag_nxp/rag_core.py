"""PDF indexing, retrieval, and grounded generation via local Ollama.

Design:
- The model is instructed to reply with a single JSON object: {"answer": "..."}.
- Any reasoning / scratchpad / <think> block is discarded during parsing.
- A reasoning-aware fallback handles models that ignore the JSON instruction,
  but refuses to leak obvious chain-of-thought prose into the UI.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chromadb
import httpx
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from rag_nxp.chunking import chunk_fixed, chunk_logical, chunk_semantic
from rag_nxp.config import settings
from rag_nxp.json_answer import decode_model_json_reply

NOT_FOUND_MESSAGE = "This information is not present in the document."

LLM_SYSTEM_DOCUMENT_RAG = (
    "You are a document question-answering assistant.\n"
    "Your reply MUST be exactly one JSON object and nothing else, with this shape:\n"
    '{"answer": "<final answer here>"}\n'
    "Rules for the answer string:\n"
    "- Use only the CONTEXT the user supplies. If the retrieved context does "
    "not contain a direct answer to the question, respond with exactly: "
    f'"{NOT_FOUND_MESSAGE}". Do not infer or construct answers from loosely '
    "related content.\n"
    "- Match length to the question:\n"
    "    * Factoid questions (who/when/how many): 1-2 short sentences.\n"
    "    * Summary or list questions ('what are', 'list', 'summarize', 'describe'): "
    "be comprehensive — include every relevant fact present in CONTEXT, "
    "not just the first one. Up to ~8 sentences, one paragraph, OR a "
    "newline-separated list using '- ' bullets when listing discrete items.\n"
    "- Do not invent or generalise beyond CONTEXT.\n"
    "- No preamble, no reasoning, no scratch work, no <think> block.\n"
    "- Do not reference passages: no 'From passage', 'Passage N', 'According to "
    "the passage', no bracket citations like [1].\n"
    "- No meta phrases like 'I need to look through', 'Let me analyze', "
    "'This implies', 'Looking at the context', 'Draft:', 'Refined:'.\n"
    "- Output no text before or after the JSON object. No markdown fences.\n"
    "- Inside the JSON string, escape newlines as \\n and quotes as \\\"."
)

_LOG_PROMPT = os.environ.get("RAG_LOG_PROMPT", "1") not in {"0", "false", "False", ""}


def _log_prompt_to_console(system_prompt: str, user_prompt: str, backend: str) -> None:
    if not _LOG_PROMPT:
        return
    bar = "=" * 78
    print(f"\n{bar}\n[RAG] LLM call ({backend}) - SYSTEM PROMPT\n{bar}", file=sys.stderr, flush=True)
    print(system_prompt, file=sys.stderr, flush=True)
    print(f"{bar}\n[RAG] LLM call ({backend}) - USER PROMPT\n{bar}", file=sys.stderr, flush=True)
    print(user_prompt, file=sys.stderr, flush=True)
    print(f"{bar}\n", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Answer cleanup
# ---------------------------------------------------------------------------

_THINK_BLOCK = re.compile(
    r"(?is)<\s*(?:think|reasoning|thought|scratchpad)\s*>.*?<\s*/\s*(?:think|reasoning|thought|scratchpad)\s*>"
)
_LEFTOVER_THINK_OPEN = re.compile(
    r"(?is)<\s*(?:think|reasoning|thought|scratchpad)\s*>.*$"
)
_CODE_FENCE = re.compile(r"^```(?:\w+)?\s*([\s\S]*?)```\s*$")

_REASONING_LINE_STARTS = (
    "wait",
    "let me",
    "looking at",
    "i need to",
    "i should",
    "i'll",
    "i will",
    "first,",
    "okay,",
    "ok,",
    "so,",
    "hmm",
    "actually,",
    "let's",
    "thinking",
    "reasoning",
    "draft:",
    "refined:",
    "final:",
    "answer:",
    "step ",
    "from passage",
    "passage ",
    "according to the passage",
    "based on the passage",
    "based on the context",
    "from the context",
    "the context says",
    "the passage says",
    '- "',
    "- '",
    '* "',
    "* '",
)


def _looks_like_reasoning(line: str) -> bool:
    s = line.strip().lower()
    if not s:
        return True
    return s.startswith(_REASONING_LINE_STARTS)


def _strip_think_blocks(text: str) -> str:
    if not text:
        return text
    t = _THINK_BLOCK.sub("", text)
    t = _LEFTOVER_THINK_OPEN.sub("", t)
    return t.strip()


def _normalize_not_found(text: str) -> str:
    s = text.strip()
    if not s:
        return ""
    if s == NOT_FOUND_MESSAGE:
        return NOT_FOUND_MESSAGE
    if len(s) > 240:
        return s
    # All of these legacy / paraphrased refusal forms collapse to the
    # canonical NOT_FOUND_MESSAGE so callers can compare on equality.
    variants = (
        r"^(?:"
        r"I\s+(?:could\s+not|cannot|can't|am\s+unable\s+to)\s+find\s+"
        r"(?:this|that|it|the\s+answer|any\s+information(?:\s+about\s+that)?)"
        r"\s+in\s+(?:the\s+)?(?:document|provided\s+passages|the\s+passages|context)"
        r"|"
        r"This\s+(?:information|answer|content)\s+is\s+not\s+"
        r"(?:present|available|found|contained|mentioned|provided)\s+"
        r"in\s+(?:the\s+)?(?:document|provided\s+passages|context)"
        r")\.?\s*$"
    )
    if re.match(variants, s, re.IGNORECASE):
        return NOT_FOUND_MESSAGE
    return s


def _fallback_extract_answer(raw: str) -> str:
    """When the model did NOT emit valid JSON, recover safely.

    Walk paragraphs from the END and return the first one whose lines don't
    look like reasoning/scratchpad. If every paragraph looks like reasoning
    (e.g. the output got truncated mid-thought), return NOT_FOUND_MESSAGE
    rather than leaking the scratchpad to the UI.
    """
    if not raw:
        return NOT_FOUND_MESSAGE
    text = _strip_think_blocks(raw)
    m = _CODE_FENCE.match(text)
    if m:
        text = m.group(1).strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return NOT_FOUND_MESSAGE

    for para in reversed(paragraphs):
        lines = [ln for ln in para.splitlines() if ln.strip()]
        if not lines:
            continue
        non_reasoning = [ln for ln in lines if not _looks_like_reasoning(ln)]
        if non_reasoning:
            return "\n".join(non_reasoning).strip()

    return NOT_FOUND_MESSAGE


_BRACED_STRING_WRAPPER = re.compile(r'^\s*\{\s*"?([\s\S]*?)"?\s*\}\s*$')


def _strip_botched_json_wrapper(text: str) -> str:
    """Peel a stray `{...}` (or `{"..."}`) wrapper that small models emit.

    Only strips when the brace pair surrounds the entire string and the
    contents do not themselves look like a real JSON object (no internal
    `:` separating a key from a value).
    """
    if not text:
        return text
    m = _BRACED_STRING_WRAPPER.match(text)
    if not m:
        return text
    inner = m.group(1).strip()
    if not inner:
        return text
    # Refuse to strip if it looks like a real key:value object.
    if re.search(r'^\s*"[^"]+"\s*:', inner):
        return text
    return inner


def _finalize_answer_text(answer: str) -> str:
    if not answer:
        return ""
    t = _strip_think_blocks(answer)
    t = _strip_botched_json_wrapper(t)
    t = "\n".join(re.sub(r"[ \t]{2,}", " ", line).strip() for line in t.splitlines()).strip()
    if not t:
        return ""
    return _normalize_not_found(t)


# ---------------------------------------------------------------------------
# PDF / chunking
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    text: str
    source: str
    page_hint: str | None
    section: str | None = None
    page_number: int | None = None
    distance: float | None = None


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_line_whitespace(text: str) -> str:
    """Collapse horizontal whitespace and stray blank lines, but keep line breaks.

    The 'logical' and 'semantic' chunkers rely on line structure to detect
    section headings. The 'fixed' chunker is unaffected since newlines just
    count as one character within its sliding window.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", ln).strip() for ln in text.split("\n")]
    out: list[str] = []
    blank = False
    for ln in lines:
        if not ln:
            if not blank and out:
                out.append("")
            blank = True
        else:
            out.append(ln)
            blank = False
    return "\n".join(out).strip()


def extract_text_from_pdf(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        pages.append((i + 1, _normalize_line_whitespace(raw)))
    return pages


def stable_id(text: str, idx: int) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"chunk_{idx}_{h}"


# ---------------------------------------------------------------------------
# Main RAG class
# ---------------------------------------------------------------------------

class RagNxp:
    def __init__(self) -> None:
        self._embedder: SentenceTransformer | None = None
        self._collection = None
        self._client: chromadb.PersistentClient | None = None
        # Cache of all distinct `section` values currently in the index.
        # Used by section-aware retrieval to expand "Article I" into the
        # specific Section labels actually present (e.g. "Article I",
        # "Article I - Section 1", ... "Article I - Section 10").
        # Invalidated on every ingest_pdf().
        self._section_cache: set[str] | None = None

    def _embedder_model(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(settings.embedding_model)
        return self._embedder

    def _get_collection(self):
        if self._client is None:
            settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name="rag_nxp_chunks",
                metadata={"description": "Uploaded PDF chunks"},
            )
        return self._collection

    def index_size(self) -> int:
        return self._get_collection().count()

    def _build_chunks(
        self, pages: list[tuple[int, str]]
    ) -> list[tuple[str, str, str | None]]:
        """Dispatch to the configured chunking strategy."""
        strategy = (settings.chunking_strategy or "logical").lower()
        if strategy == "fixed":
            return chunk_fixed(pages, settings.chunk_size, settings.chunk_overlap)
        if strategy == "semantic":
            embedder = self._embedder_model()

            def embed_fn(texts: list[str]):
                return embedder.encode(texts, normalize_embeddings=True)

            return chunk_semantic(
                pages,
                target_chars=settings.chunk_size_target,
                max_chars=settings.chunk_size_max,
                overlap_sentences=settings.chunk_overlap_sentences,
                breakpoint_percentile=settings.semantic_breakpoint_percentile,
                embed_fn=embed_fn,
            )
        if strategy != "logical":
            print(
                f"[RAG] unknown CHUNKING_STRATEGY={strategy!r}; falling back to 'logical'.",
                file=sys.stderr,
                flush=True,
            )
        return chunk_logical(
            pages,
            target_chars=settings.chunk_size_target,
            max_chars=settings.chunk_size_max,
            overlap_sentences=settings.chunk_overlap_sentences,
        )

    def ingest_pdf(self, pdf_path: Path, replace: bool = True) -> int:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages = extract_text_from_pdf(pdf_path)
        chunks_meta = self._build_chunks(pages)
        if not chunks_meta:
            raise ValueError("No text extracted from PDF; check the file.")

        texts = [c.text for c in chunks_meta]
        ids = [stable_id(t, i) for i, t in enumerate(texts)]
        metadatas = [
            {
                "source": c.source,
                "page_hint": c.page_hint or "",
                "section": c.section or "",
                "page_number": c.page_number if c.page_number is not None else 0,
            }
            for c in chunks_meta
        ]

        embedder = self._embedder_model()
        embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()

        coll = self._get_collection()
        if replace:
            existing = coll.get(include=[])
            if existing["ids"]:
                coll.delete(ids=existing["ids"])

        batch = 64
        for i in range(0, len(ids), batch):
            coll.add(
                ids=ids[i : i + batch],
                embeddings=embeddings[i : i + batch],
                documents=texts[i : i + batch],
                metadatas=metadatas[i : i + batch],
            )
        # Section list changed; force a rebuild on the next retrieval that
        # needs it.
        self._section_cache = None
        return len(ids)

    # ------------------------------------------------------------------
    # Section-aware retrieval helpers
    # ------------------------------------------------------------------

    def _all_sections(self) -> set[str]:
        """List of unique non-empty `section` values currently indexed.

        Cached; invalidated on every ingest_pdf().
        """
        if self._section_cache is not None:
            return self._section_cache
        coll = self._get_collection()
        try:
            metas = coll.get(include=["metadatas"]).get("metadatas") or []
        except Exception:
            metas = []
        secs = {(m or {}).get("section") or "" for m in metas}
        secs.discard("")
        self._section_cache = secs
        return secs

    def _detect_section_filter(self, query: str) -> list[str]:
        """Return all canonical section labels the query likely refers to.

        Empty list means "no section reference detected, do plain
        semantic retrieval". Recognised forms:

          - "Preamble"                          -> ["Preamble"]
          - "Article X Section Y"               -> ["Article X - Section Y"]
          - "Article X" (no section spelled)    -> all "Article X*" present
          - "Amendment X" / "Xth Amendment"     -> all "Amendment X*" present
        """
        q = query.lower()

        if re.search(r"\bpreamble\b", q):
            return ["Preamble"] if "Preamble" in self._all_sections() else []

        m = re.search(
            r"\barticle\s+([ivxlcdm]+|\d+)\b[\s,\.\-]+section\s+(\d+[a-z]?)\b",
            q,
        )
        if m:
            art = m.group(1).upper()
            sec = m.group(2)
            target = f"Article {art} - Section {sec}"
            return [target] if target in self._all_sections() else []

        m = re.search(r"\barticle\s+([ivxlcdm]+|\d+)\b", q)
        if m:
            art = m.group(1).upper()
            return self._sections_with_top(f"Article {art}")

        m = re.search(
            r"\b(?:amendment\s+([ivxlcdm]+|\d+)|"
            r"([ivxlcdm]+|\d+)(?:st|nd|rd|th)?\s+amendment)\b",
            q,
        )
        if m:
            token = (m.group(1) or m.group(2)).upper()
            return self._sections_with_top(f"Amendment {token}")

        return []

    def _sections_with_top(self, top: str) -> list[str]:
        """All indexed sections that equal `top` or start with `top - `."""
        all_secs = self._all_sections()
        return sorted(
            s for s in all_secs
            if s == top or s.startswith(f"{top} - ")
        )

    def _result_to_chunks(self, res: dict) -> list[RetrievedChunk]:
        out: list[RetrievedChunk] = []
        docs = res["documents"][0] if res.get("documents") else []
        metas = res["metadatas"][0] if res.get("metadatas") else []
        dists = res["distances"][0] if res.get("distances") else []
        for j, (doc, meta) in enumerate(zip(docs, metas)):
            meta = meta or {}
            hint = meta.get("page_hint") or None
            if hint == "":
                hint = None
            section = meta.get("section") or None
            if section == "":
                section = None
            page_num = meta.get("page_number")
            if page_num in ("", 0, None):
                page_num = None
            src = meta.get("source") or "document"
            dist = (
                float(dists[j])
                if j < len(dists) and dists[j] is not None
                else None
            )
            out.append(RetrievedChunk(
                text=doc,
                source=src,
                page_hint=hint,
                section=section,
                page_number=page_num,
                distance=dist,
            ))
        return out

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Section-aware retrieval.

        If the query references a section we have indexed, run a metadata-
        filtered query first so chunks tagged with that section get the
        top slots in the prompt. Then run an unfiltered semantic query
        and use it to backfill any remaining slots, deduplicating against
        the boosted set. If no section reference is detected, behaviour
        is identical to plain semantic retrieval.
        """
        k = k or settings.retrieval_k
        embedder = self._embedder_model()
        q_emb = embedder.encode([query], normalize_embeddings=True).tolist()[0]

        coll = self._get_collection()
        n_total = coll.count()
        if n_total == 0:
            return []

        section_filter = self._detect_section_filter(query)
        boosted: list[RetrievedChunk] = []
        if section_filter:
            try:
                where = (
                    {"section": section_filter[0]}
                    if len(section_filter) == 1
                    else {"section": {"$in": section_filter}}
                )
                res_f = coll.query(
                    query_embeddings=[q_emb],
                    n_results=min(k, n_total),
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
                boosted = self._result_to_chunks(res_f)
                if boosted:
                    print(
                        f"[RAG] section filter hit: query references "
                        f"{section_filter!r}; boosted {len(boosted)} chunk(s).",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception as e:
                print(f"[RAG] section filter failed: {e}", file=sys.stderr, flush=True)
                boosted = []

        res_main = coll.query(
            query_embeddings=[q_emb],
            n_results=min(k, n_total),
            include=["documents", "metadatas", "distances"],
        )
        main_chunks = self._result_to_chunks(res_main)

        seen: set[tuple[str, str | None]] = set()
        out: list[RetrievedChunk] = []
        for c in boosted + main_chunks:
            key = (c.text[:120], c.page_hint)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
            if len(out) >= k:
                break
        return out

    def _build_prompt(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        chat_history: str | None = None,
    ) -> str:
        context_blocks = [ch.text for ch in chunks]
        context = "\n\n".join(context_blocks) if context_blocks else "(No retrieved passages.)"

        history_block = ""
        if chat_history and chat_history.strip():
            history_block = (
                "CONVERSATION (for resolving short follow-ups; facts still must come from CONTEXT):\n"
                f"{chat_history.strip()}\n\n"
            )

        return (
            "Use the CONTEXT below to answer the QUESTION.\n"
            "- For a factoid question, answer in 1-2 short sentences.\n"
            "- For a summary/list/'what are…' question, be comprehensive: "
            "include EVERY relevant item from the CONTEXT (not just the first "
            "one), as a single paragraph or as '- ' bullet lines separated by "
            "\\n inside the JSON string.\n"
            "If the CONTEXT truly does not contain the answer, the JSON answer "
            f'must be exactly: "{NOT_FOUND_MESSAGE}".\n\n'
            "CONTEXT:\n"
            f"{context}\n\n"
            f"{history_block}"
            "QUESTION:\n"
            f"{query}\n\n"
            "Reply with exactly one JSON object and NOTHING else: "
            '{"answer": "<your answer here>"}'
        )

    async def generate_grounded_answer(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        chat_history: str | None = None,
    ) -> tuple[str, dict]:
        """Return (answer, metadata). Answer is always clean text ready for UI."""
        prompt = self._build_prompt(query, chunks, chat_history)

        _log_prompt_to_console(
            LLM_SYSTEM_DOCUMENT_RAG, prompt, backend=f"ollama:{settings.ollama_model}"
        )
        url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": settings.ollama_model,
            "system": LLM_SYSTEM_DOCUMENT_RAG,
            "prompt": prompt,
            "stream": False,
            "keep_alive": settings.ollama_keep_alive,
            # Do NOT set "format": "json" — some models return an empty JSON
            # body in that mode. We do our own robust JSON parsing.
            "options": {
                "temperature": 0,
                "num_predict": settings.ollama_num_predict,
            },
        }
        async with httpx.AsyncClient(timeout=settings.ollama_timeout_s) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        raw = (data.get("response") or "").strip()
        llm_meta = {"llm": settings.ollama_model, "backend": "ollama"}

        cleaned_raw = _strip_think_blocks(raw)
        parsed, _cited = decode_model_json_reply(cleaned_raw)

        parse_mode = "json"
        answer = _finalize_answer_text(parsed or "")
        if not answer:
            parse_mode = "fallback"
            answer = _finalize_answer_text(_fallback_extract_answer(raw))

        if not answer:
            answer = NOT_FOUND_MESSAGE
            parse_mode = "empty"

        if _LOG_PROMPT:
            bar = "-" * 78
            print(f"{bar}\n[RAG] RAW MODEL OUTPUT ({parse_mode})\n{bar}", file=sys.stderr, flush=True)
            print(raw[:2000], file=sys.stderr, flush=True)
            print(f"{bar}\n[RAG] FINAL ANSWER -> {answer}\n{bar}", file=sys.stderr, flush=True)

        return answer, {
            **llm_meta,
            "parse_mode": parse_mode,
            "raw_model_output_preview": raw[:500],
        }


# ---------------------------------------------------------------------------
# Helpers for the FastAPI app
# ---------------------------------------------------------------------------

def _pdfs_in_data_dir() -> list[Path]:
    d = settings.data_dir
    if not d.is_dir():
        return []
    return sorted(
        (p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda path: path.name.lower(),
    )


def pdf_path_default() -> Path:
    """Prefer data/{pdf_filename}; otherwise the first .pdf in data/."""
    preferred = settings.data_dir / settings.pdf_filename
    if preferred.is_file():
        return preferred
    pdfs = _pdfs_in_data_dir()
    return pdfs[0] if pdfs else preferred
