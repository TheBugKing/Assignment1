"""End-to-end LLM test on the Preamble question.

Assumes the chroma_db has already been (re-)populated by debug_retrieval.py.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from rag_nxp.config import settings
from rag_nxp.rag_core import RagNxp


async def main() -> None:
    rag = RagNxp()
    if rag.index_size() == 0:
        pdf = Path("data") / "US Constitution.pdf"
        print(f"Index empty, ingesting {pdf.name} ...")
        rag.ingest_pdf(pdf)

    questions = [
        "What are the six goals stated in the Preamble?",
        "How long is the term of a Senator?",
        "How many amendments are included in this document?",
    ]
    for q in questions:
        print("\n" + "#" * 78)
        print(f"# QUESTION: {q}")
        print("#" * 78)
        chunks = rag.retrieve(q, k=settings.retrieval_k)
        sections = [c.section for c in chunks]
        print(f"sections fed to LLM: {sections}")
        answer, meta = await rag.generate_grounded_answer(q, chunks)
        print(f"\nANSWER: {answer}")
        print(f"meta: {meta}")


if __name__ == "__main__":
    asyncio.run(main())
