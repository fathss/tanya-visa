"""Probe the retrieval index with ad-hoc queries. Retrieval only, no LLM.

Usage:
    .venv/bin/python scripts/probe.py "Apa syarat paspor untuk anak?"
    .venv/bin/python scripts/probe.py # interactive
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.embedder import embed_query
from rag.retriever import Retriever


def probe(retriever: Retriever, query: str) -> None:
    hits = retriever.search(embed_query(query))
    print(f"\n{query}")

    if not hits:
        closest = retriever.search(embed_query(query), threshold=None)
        print(f"  -> REFUSED (below {config.SIMILARITY_THRESHOLD})")
        if closest:
            top = closest[0]
            print(f"     closest: {top.score:.4f}  {top.chunk.source} | {top.chunk.locator_label or '(dokumen)'}")
        return

    for hit in hits:
        print(f"  {hit.score:.4f}  {hit.chunk.source} | {hit.chunk.locator_label or '(dokumen)'}")


def main(argv: list[str]) -> None:
    retriever = Retriever.load()
    print(
        f"index={retriever.index.ntotal} vectors  dim={retriever.index.d}  "
        f"top_k={config.TOP_K_RETRIEVAL}  threshold={config.SIMILARITY_THRESHOLD}"
    )

    if argv:
        for query in argv:
            probe(retriever, query)
        return

    print("Enter a query, or Ctrl-D to quit.")
    while True:
        try:
            query = input("\n> ").strip()
        except EOFError:
            print()
            return
        if query:
            probe(retriever, query)


if __name__ == "__main__":
    main(sys.argv[1:])
