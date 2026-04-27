"""Peek at the chunks currently indexed in ChromaDB.

No Ollama call, no HTTP — just opens the persistent client and prints a
summary of the section metadata so we can verify ingest worked.

Run from the rag_nxp directory:

    python peek_index.py
"""
from __future__ import annotations

from collections import Counter

from rag_nxp.config import settings
import chromadb


def main() -> None:
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    coll = client.get_or_create_collection(name="rag_nxp_chunks")
    n = coll.count()
    print(f"Index size: {n} chunk(s)")
    if n == 0:
        print("Index is empty — ingest a PDF first.")
        return
    res = coll.get(include=["metadatas", "documents"])
    metas = res.get("metadatas") or []
    docs = res.get("documents") or []

    sections = [(m or {}).get("section") or "(none)" for m in metas]
    pages = [(m or {}).get("page_number") or 0 for m in metas]

    section_counts = Counter(sections)
    print(f"\nDistinct section labels: {len(section_counts)}")
    for s, c in sorted(section_counts.items()):
        print(f"  {c:3d}  {s}")

    page_counts = Counter(pages)
    print(f"\nDistinct page numbers: {len(page_counts)}")
    for p, c in sorted(page_counts.items()):
        print(f"  page {p}: {c} chunk(s)")

    print("\nFirst chunk text + metadata:")
    print(f"  section    = {sections[0]!r}")
    print(f"  page       = {pages[0]}")
    print(f"  page_hint  = {(metas[0] or {}).get('page_hint')!r}")
    preview = (docs[0] or "")[:200].replace("\n", " ")
    print(f"  text[:200] = {preview!r}")

    preamble_hits = [
        (i, (m or {}).get("page_hint"), (d or "")[:160])
        for i, (m, d) in enumerate(zip(metas, docs))
        if ((m or {}).get("section") or "").lower() == "preamble"
    ]
    print(f"\nPreamble chunks: {len(preamble_hits)}")
    for i, hint, preview in preamble_hits:
        print(f"  [{i}] {hint}")
        print(f"      {preview!r}")


if __name__ == "__main__":
    main()
