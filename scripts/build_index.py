"""Rebuild the FAISS index from data/raw_docs/.

Run it after changing the corpus, chunking, or embedding model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.embedder import embed_documents
from rag.loader import load_documents
from rag.retriever import Retriever


def main() -> None:
    chunks = load_documents()
    if not chunks:
        raise SystemExit(f"No supported documents found in {config.RAW_DOCS_DIR}")

    print(f"Loaded    {len(chunks)} chunks from {config.RAW_DOCS_DIR}")

    vectors = embed_documents(chunks)
    print(f"Embedded  {len(vectors)} chunks with {config.EMBEDDING_MODEL}")

    retriever = Retriever.build(chunks, vectors)
    retriever.save()
    print(
        f"Saved     {retriever.index.ntotal} vectors (dim {retriever.index.d}) "
        f"to {config.INDEX_DIR}"
    )


if __name__ == "__main__":
    main()
