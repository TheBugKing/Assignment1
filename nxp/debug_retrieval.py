"""End-to-end retrieval test (no LLM):

1. Wipe chroma_db.
2. Create RagNxp, ingest the Constitution PDF.
3. Run a Preamble query via .retrieve().
4. Print the top chunks with their section metadata.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from rag_nxp.config import settings
from rag_nxp.rag_core import RagNxp


def main() -> None:
    chroma_dir = Path(settings.chroma_dir)
    if chroma_dir.exists():
        print(f"Wiping {chroma_dir!s}")
        shutil.rmtree(chroma_dir)

    pdf = Path("data") / "US Constitution.pdf"
    if not pdf.is_file():
        raise SystemExit(f"PDF not found at {pdf!s}")

    rag = RagNxp()
    print(f"Ingesting {pdf.name} ...")
    n = rag.ingest_pdf(pdf)
    print(f"Ingested {n} chunks. Index size: {rag.index_size()}")

    queries = [
        "What are the six goals stated in the Preamble?",
        "How long is the term of a Senator?",
        "What does Article I Section 8 say about the powers of Congress?",
    ]
    for q in queries:
        print("\n" + "=" * 78)
        print(f"QUERY: {q}")
        print("=" * 78)
        chunks = rag.retrieve(q, k=settings.retrieval_k)
        for i, c in enumerate(chunks, 1):
            sec = c.section or "(none)"
            preview = c.text[:140].replace("\n", " ")
            print(f"  [{i}] section={sec!r}  page={c.page_number}  score={c.distance:.4f}")
            print(f"      {preview}")


if __name__ == "__main__":
    main()
