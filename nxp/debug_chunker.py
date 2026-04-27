"""Direct test of the chunker on the user's PDF — no Ollama, no FastAPI.

Prints:
- raw page-1 text (so we can see what headings, if any, the PDF has)
- distinct (top, sub) labels found by _split_pages_to_subsections
- the first chunk per distinct section emitted by chunk_logical
- whether a Preamble chunk exists
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from pathlib import Path

from rag_nxp.chunking import (
    _split_pages_to_subsections,
    chunk_logical,
)
from rag_nxp.config import settings
from rag_nxp.rag_core import extract_text_from_pdf


def main() -> None:
    pdf = Path("data") / "US Constitution.pdf"
    if not pdf.is_file():
        raise SystemExit(f"PDF not found at {pdf!s}")
    print(f"PDF: {pdf!s}")

    pages = extract_text_from_pdf(pdf)
    print(f"Pages: {len(pages)}")

    print("\n" + "=" * 78)
    print("RAW page 1 text (first 1500 chars):")
    print("=" * 78)
    print(pages[0][1][:1500])
    print("=" * 78)

    subs = _split_pages_to_subsections(pages)
    print(f"\nSubsections detected: {len(subs)}")
    pair_counts = Counter((s.top_label, s.sub_label) for s in subs)
    print(f"Distinct (top, sub) pairs: {len(pair_counts)}")
    for (top, sub), c in pair_counts.most_common(40):
        print(f"  {c:3d}  top={top!r}  sub={sub!r}")

    chunks = chunk_logical(
        pages,
        target_chars=settings.chunk_size_target,
        max_chars=settings.chunk_size_max,
        overlap_sentences=settings.chunk_overlap_sentences,
    )
    print(f"\nTotal chunks: {len(chunks)}")
    sec_counts = Counter(c.section or "(none)" for c in chunks)
    print(f"Distinct section labels: {len(sec_counts)}")
    for s, c in sorted(sec_counts.items()):
        print(f"  {c:3d}  {s}")

    print("\nFirst chunk per distinct section:")
    seen: OrderedDict[str, tuple] = OrderedDict()
    for c in chunks:
        key = c.section or "(none)"
        if key not in seen:
            seen[key] = c
    for sec, c in seen.items():
        preview = c.text[:160].replace("\n", " ")
        print(f"  [{sec!r}] page={c.page_number}  hint={c.page_hint!r}")
        print(f"      {preview!r}")

    preamble_chunks = [c for c in chunks if (c.section or "").lower() == "preamble"]
    print(f"\nPreamble chunks: {len(preamble_chunks)}")
    for c in preamble_chunks:
        print(f"  page={c.page_number}  hint={c.page_hint!r}")
        print(f"  text: {c.text[:300]!r}")


if __name__ == "__main__":
    main()
