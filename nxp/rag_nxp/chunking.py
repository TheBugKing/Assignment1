"""Chunking strategies for RAG NXP.

Three strategies, selectable via Settings.chunking_strategy:

- "fixed":    legacy character-window with overlap, page-bounded.
- "logical":  detect structural boundaries (Preamble / Article / Section /
              Amendment / numbered or ALL-CAPS headings), sentence-split
              within each section, and greedily pack sentences up to a soft
              size target with N-sentence overlap. Never cuts mid-sentence.
- "semantic": "logical" plus extra cuts inside long sections where
              consecutive-sentence embedding similarity drops sharply, i.e.
              at topic shifts. Requires the embedder.

Every strategy emits a `ChunkRecord(text, source, page_hint, section,
page_number)` named-tuple. The `section` field is the canonical section
label used for metadata-aware retrieval; values look like:

    "Preamble"
    "Article I"
    "Article I - Section 1"
    "Amendment I"

Subordinate headings (Section N, numbered headings, ALL-CAPS headings)
are joined to the active top-level heading with " - " so that the
Preamble is never confused with content from any Article, and chunks
inside an Article carry their parent label even when only "Section N"
appears literally on the page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Public chunk record
# ---------------------------------------------------------------------------

class ChunkRecord(NamedTuple):
    """One emitted chunk plus the metadata RagNxp needs to index/retrieve."""
    text: str
    source: str
    page_hint: str | None     # display string, e.g. "Article I - Section 1, page 4"
    section: str | None       # canonical section, e.g. "Article I - Section 1"
    page_number: int | None   # physical page number of the chunk


# ---------------------------------------------------------------------------
# Sentence splitting (abbreviation-aware, dependency-free)
# ---------------------------------------------------------------------------

_ABBREV_PATTERN = re.compile(
    r"\b("
    r"U\.S\.A|U\.S\.C|U\.S|e\.g|i\.e|"
    r"Mr|Mrs|Ms|Dr|Prof|Jr|Sr|St|Mt|"
    r"No|Inc|Ltd|Co|Corp|Ave|Blvd|"
    r"vs|etc|Art|Sec|Sect|Ch|Vol|pp"
    r")\.",
    re.IGNORECASE,
)
_DOT_SENTINEL = "\u0001"


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that respects common abbreviations."""
    if not text or not text.strip():
        return []
    protected = _ABBREV_PATTERN.sub(
        lambda m: m.group(0).replace(".", _DOT_SENTINEL), text
    )
    parts = re.split(r"(?<=[.!?])[\"\')\]]?\s+(?=[A-Z(\[\"'\d])", protected)
    return [p.replace(_DOT_SENTINEL, ".").strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Heading detection (top-level vs. subordinate)
# ---------------------------------------------------------------------------

# Each rule = (pattern, kind, label_fn). The pattern is matched at the
# start of a stripped line. The label_fn turns the regex match into a
# canonical label like "Article I" / "Section 8" / "Preamble".
#
# Many real PDFs (including the US Constitution PDF used as the test
# corpus) write headings as "Article. I." and "Section. 1." with a
# period BETWEEN the word and the numeral. The patterns therefore allow
# an optional "." (and surrounding whitespace) on either side.
_WS = r"[ \t]*"
_DOT = r"\.?"
_NUM = r"(?:[IVXLCDM]+|\d+)"

_HEADING_RULES: tuple[tuple[re.Pattern[str], str, Callable[[re.Match[str]], str]], ...] = (
    (
        re.compile(rf"^Preamble{_DOT}\b[\s.:\-]*", re.IGNORECASE),
        "preamble",
        lambda m: "Preamble",
    ),
    (
        re.compile(rf"^Article{_DOT}{_WS}\s+({_NUM}){_DOT}[\s.:\-]*", re.IGNORECASE),
        "article",
        lambda m: f"Article {m.group(1).upper()}",
    ),
    (
        re.compile(rf"^Amendment{_DOT}{_WS}\s+({_NUM}){_DOT}[\s.:\-]*", re.IGNORECASE),
        "amendment",
        lambda m: f"Amendment {m.group(1).upper()}",
    ),
    (
        re.compile(rf"^Chapter{_DOT}{_WS}\s+({_NUM}){_DOT}[\s.:\-]*", re.IGNORECASE),
        "chapter",
        lambda m: f"Chapter {m.group(1).upper()}",
    ),
    (
        re.compile(rf"^Section{_DOT}{_WS}\s+(\d+[A-Za-z]?){_DOT}[\s.:\-]*", re.IGNORECASE),
        "section",
        lambda m: f"Section {m.group(1)}",
    ),
    (
        re.compile(r"^(\d+(?:\.\d+){1,3})\s+(?=[A-Z])"),
        "numbered",
        lambda m: m.group(1),
    ),
    (
        re.compile(r"^([A-Z][A-Z0-9 \-/&]{5,80})\s*$"),
        "caps",
        lambda m: _normalize_heading(m.group(1)),
    ),
)

_TOP_KINDS = frozenset({"preamble", "article", "amendment", "chapter"})


def _normalize_heading(label: str) -> str:
    """Tidy capitalisation on heading labels.

    "ARTICLE I" -> "Article I"
    "preamble"  -> "Preamble"
    Roman numerals and uppercase tokens are kept upper-case.
    """
    s = label.strip()
    if not s:
        return s
    parts = s.split()

    def fix(part: str) -> str:
        if not part:
            return part
        # All-letters and matches a Roman numeral pattern -> keep uppercase.
        if re.fullmatch(r"[IVXLCDMivxlcdm]+", part):
            return part.upper()
        if part.isdigit():
            return part
        return part[:1].upper() + part[1:].lower()

    return " ".join(fix(p) for p in parts)


def _heading_at_line_start(line: str) -> tuple[str, str, str] | None:
    """If `line` begins with a recognised heading, return (kind, label, rest_of_line).

    `kind` is one of the strings registered in `_HEADING_RULES`; `label`
    is the canonical heading text (e.g. "Article I", "Section 8");
    `rest_of_line` is content that follows the heading on the same line
    and belongs UNDER the heading.
    """
    s = line.strip()
    if not s:
        return None
    for pat, kind, label_fn in _HEADING_RULES:
        m = pat.match(s)
        if m:
            label = label_fn(m)
            rest = s[m.end():].lstrip(" .:-—–")
            return (kind, label, rest)
    return None


# ---------------------------------------------------------------------------
# Sub-section model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SubSection:
    """A piece of one page that belongs to a single (top, sub) heading pair.

    `top_label`  carries across page breaks until a new top-level heading
                 (Preamble / Article / Amendment / Chapter) is found.
    `sub_label`  carries within the current top until a new subordinate
                 heading is found, and is reset whenever the top changes.
    """
    top_label: str | None
    sub_label: str | None
    page: int
    text: str

    @property
    def canonical_section(self) -> str | None:
        if self.top_label and self.sub_label:
            return f"{self.top_label} - {self.sub_label}"
        return self.top_label  # may be None


_PREAMBLE_TEXT_HINT = re.compile(r"\bwe\s+the\s+people\b", re.IGNORECASE)


def _split_pages_to_subsections(
    pages: Sequence[tuple[int, str]]
) -> list[_SubSection]:
    """Per-page splitting that carries (top, sub) headings across page breaks.

    Top-level headings reset the sub heading. Sub headings only change
    when a new sub heading line is hit. Each emitted _SubSection is a
    contiguous stretch of one page belonging to the same (top, sub)
    pair, so retrieval can attribute every chunk to the page that
    physically contains it.

    Implicit Preamble: many constitutional PDFs start the body with
    "We the People..." without an explicit "Preamble" heading. To
    guarantee the Preamble lands in its own chunk and is never merged
    with Article I content, we promote the first un-labelled subsection
    that starts with "We the People" to top_label="Preamble".
    """
    out: list[_SubSection] = []
    top: str | None = None
    sub: str | None = None

    for page_num, page_text in pages:
        if not page_text:
            continue

        cur_top = top
        cur_sub = sub
        cur_lines: list[str] = []

        def flush_local() -> None:
            nonlocal cur_lines
            body = "\n".join(cur_lines).strip()
            cur_lines = []
            if body:
                out.append(_SubSection(
                    top_label=cur_top,
                    sub_label=cur_sub,
                    page=page_num,
                    text=body,
                ))

        for line in page_text.split("\n"):
            hit = _heading_at_line_start(line)
            if hit:
                kind, label, rest = hit
                flush_local()
                if kind in _TOP_KINDS:
                    cur_top = label
                    cur_sub = None
                else:
                    cur_sub = label
                if rest:
                    cur_lines.append(rest)
                continue
            cur_lines.append(line)

        flush_local()
        top = cur_top
        sub = cur_sub

    if not out:
        # No content at all; per-page fallback to keep ingestion alive.
        for page_num, page_text in pages:
            if page_text.strip():
                out.append(_SubSection(
                    top_label=None, sub_label=None,
                    page=page_num, text=page_text,
                ))
        return out

    # Implicit Preamble promotion (guarantees a standalone Preamble chunk
    # even when the PDF has no literal "Preamble" heading). Many sources
    # carry a title line such as "US Constitution" before the Preamble
    # body, so we search the first ~200 characters of the first
    # un-labelled subsection for "we the people".
    for i, s in enumerate(out):
        if s.top_label is None and _PREAMBLE_TEXT_HINT.search(s.text[:200]):
            out[i] = _SubSection(
                top_label="Preamble", sub_label=None,
                page=s.page, text=s.text,
            )
            break

    return out


# ---------------------------------------------------------------------------
# Sentence packing
# ---------------------------------------------------------------------------

def _pack_sentences(
    sentences: Sequence[str],
    target_chars: int,
    max_chars: int,
    overlap_sentences: int,
) -> list[str]:
    """Greedy pack adjacent sentences up to ~target_chars (never exceeding
    max_chars). Each emitted chunk shares its trailing `overlap_sentences`
    with the next chunk.
    """
    if not sentences:
        return []
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append(" ".join(buf).strip())
            if overlap_sentences > 0:
                tail = buf[-overlap_sentences:]
                buf = list(tail)
                buf_len = sum(len(s) + 1 for s in buf)
            else:
                buf = []
                buf_len = 0

    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        if len(s) > max_chars:
            flush()
            for i in range(0, len(s), max_chars):
                chunks.append(s[i : i + max_chars])
            buf = []
            buf_len = 0
            continue
        prospective = buf_len + len(s) + (1 if buf else 0)
        if buf and prospective > target_chars and buf_len >= target_chars // 2:
            flush()
            prospective = buf_len + len(s) + (1 if buf else 0)
        if prospective > max_chars:
            flush()
        buf.append(s)
        buf_len += len(s) + (1 if len(buf) > 1 else 0)

    if buf:
        chunks.append(" ".join(buf).strip())
    return [c for c in chunks if c]


def _display_label(sub: _SubSection) -> str | None:
    parts: list[str] = []
    if sub.canonical_section:
        parts.append(sub.canonical_section)
    parts.append(f"page {sub.page}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Public chunkers
# ---------------------------------------------------------------------------

def chunk_fixed(
    pages: Sequence[tuple[int, str]],
    chunk_size: int,
    overlap: int,
) -> list[ChunkRecord]:
    """Legacy character-window strategy. Carries page metadata only — no
    structural section field, since a fixed window may straddle headings."""
    out: list[ChunkRecord] = []
    for page_num, text in pages:
        if not text:
            continue
        flat = re.sub(r"\s+", " ", text).strip()
        if not flat:
            continue
        start = 0
        while start < len(flat):
            end = min(start + chunk_size, len(flat))
            piece = flat[start:end].strip()
            if piece:
                out.append(ChunkRecord(
                    text=piece,
                    source="Uploaded PDF",
                    page_hint=f"page {page_num}",
                    section=None,
                    page_number=page_num,
                ))
            if end >= len(flat):
                break
            start = max(0, end - overlap)
    return out


def chunk_logical(
    pages: Sequence[tuple[int, str]],
    target_chars: int,
    max_chars: int,
    overlap_sentences: int,
) -> list[ChunkRecord]:
    """Structure-aware + sentence-aware chunking with section metadata.

    Every chunk carries the canonical section label of its parent
    sub-section (e.g. "Article I - Section 1"). The Preamble is always a
    standalone chunk because it lives in its own _SubSection and is
    packed independently from any Article content.
    """
    subs = _split_pages_to_subsections(pages)
    out: list[ChunkRecord] = []
    for sub in subs:
        flat = re.sub(r"\s+", " ", sub.text).strip()
        if not flat:
            continue
        sentences = split_sentences(flat) or [flat]
        for piece in _pack_sentences(
            sentences,
            target_chars=target_chars,
            max_chars=max_chars,
            overlap_sentences=overlap_sentences,
        ):
            out.append(ChunkRecord(
                text=piece,
                source="Uploaded PDF",
                page_hint=_display_label(sub),
                section=sub.canonical_section,
                page_number=sub.page,
            ))
    return out


def chunk_semantic(
    pages: Sequence[tuple[int, str]],
    target_chars: int,
    max_chars: int,
    overlap_sentences: int,
    breakpoint_percentile: int,
    embed_fn: Callable[[list[str]], np.ndarray],
) -> list[ChunkRecord]:
    """Logical chunking + extra cuts at semantic topic shifts.

    Each emitted chunk still belongs to exactly one (top, sub) heading
    pair, so the section metadata is identical in shape to chunk_logical.
    """
    subs = _split_pages_to_subsections(pages)
    out: list[ChunkRecord] = []
    for sub in subs:
        flat = re.sub(r"\s+", " ", sub.text).strip()
        if not flat:
            continue
        sentences = split_sentences(flat) or [flat]
        if len(sentences) <= 3:
            packed = _pack_sentences(
                sentences,
                target_chars=target_chars,
                max_chars=max_chars,
                overlap_sentences=overlap_sentences,
            )
        else:
            embs = embed_fn(sentences)
            embs = np.asarray(embs, dtype=np.float32)
            sims = np.einsum("ij,ij->i", embs[:-1], embs[1:])
            distances = 1.0 - sims
            cutoff = float(np.percentile(distances, breakpoint_percentile))
            groups: list[list[str]] = [[]]
            for i, s in enumerate(sentences):
                groups[-1].append(s)
                if i < len(sentences) - 1 and distances[i] >= cutoff:
                    groups.append([])
            packed = []
            for group in groups:
                packed.extend(
                    _pack_sentences(
                        group,
                        target_chars=target_chars,
                        max_chars=max_chars,
                        overlap_sentences=overlap_sentences,
                    )
                )
        for piece in packed:
            out.append(ChunkRecord(
                text=piece,
                source="Uploaded PDF",
                page_hint=_display_label(sub),
                section=sub.canonical_section,
                page_number=sub.page,
            ))
    return out
