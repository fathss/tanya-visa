"""Build the FAISS index from data/raw_docs/.

Two modes:

    python scripts/build_index.py                 # full rebuild: chunk + embed everything
    python scripts/build_index.py --add FILE...   # chunk + embed only FILE, append to the
                                                  # existing index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag.embedder import embed_documents
from rag.loader import Chunk, load_documents, load_file
from rag.retriever import Retriever


def rebuild() -> None:
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


def append(paths: list[Path]) -> None:
    retriever = Retriever.load()
    before = retriever.index.ntotal

    added: list[Chunk] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Not found: {path}")
        if path.suffix.lower() not in config.SUPPORTED_SUFFIXES:
            raise SystemExit(f"Unsupported file type: {path}")

        chunks = load_file(path)
        if not chunks:
            print(f"Skipped   {path.name} (no extractable text)")
            continue

        print(f"Loaded    {len(chunks)} chunks from {path.name}")
        added.extend(chunks)

    if not added:
        raise SystemExit("Nothing to append.")

    vectors = embed_documents(added)
    print(f"Embedded  {len(vectors)} chunks with {config.EMBEDDING_MODEL}")

    retriever.add(added, vectors)
    retriever.save()
    print(
        f"Saved     {retriever.index.ntotal} vectors "
        f"({retriever.index.ntotal - before} appended) to {config.INDEX_DIR}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--add",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="chunk + embed only these files and append them to the existing index",
    )
    args = parser.parse_args()

    if args.add:
        append(args.add)
    else:
        rebuild()


if __name__ == "__main__":
    main()
