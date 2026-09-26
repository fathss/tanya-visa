"""FAISS index construction, persistence, and similarity search.

Vectors are unit-normalized, so IndexFlatIP inner product equals cosine
similarity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

import config
from rag.loader import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def build_index(vectors: list[list[float]]) -> faiss.Index:
    if not vectors:
        raise ValueError("Cannot build an index from an empty vector list.")

    matrix = np.asarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


class Retriever:
    def __init__(self, index: faiss.Index, chunks: list[Chunk]):
        if index.ntotal != len(chunks):
            raise ValueError(
                f"Index holds {index.ntotal} vectors but {len(chunks)} chunks were supplied."
            )
        self.index = index
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: list[Chunk], vectors: list[list[float]]) -> "Retriever":
        return cls(build_index(vectors), list(chunks))

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Append chunks to this index.
        """
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(f"Got {len(vectors)} vectors for {len(chunks)} chunks.")

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape[1] != self.index.d:
            raise ValueError(
                f"Vector dimension {matrix.shape[1]} does not match "
                f"index dimension {self.index.d}."
            )

        self.index.add(matrix)
        self.chunks.extend(chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int = config.TOP_K_RETRIEVAL,
        threshold: float | None = config.SIMILARITY_THRESHOLD,
    ) -> list[Hit]:
        if not self.chunks:
            return []

        query = np.asarray([query_vector], dtype=np.float32)
        scores, positions = self.index.search(query, min(top_k, len(self.chunks)))

        hits = [
            Hit(chunk=self.chunks[position], score=float(score))
            for score, position in zip(scores[0], positions[0])
            if position != -1
        ]

        if threshold is not None and hits and hits[0].score < threshold:
            return []

        return hits

    @staticmethod
    def _config_signature() -> dict:
        return {
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_dimensions": config.EMBEDDING_DIMENSIONS,
            "query_template": config.QUERY_TEMPLATE,
            "document_template": config.DOCUMENT_TEMPLATE,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
        }

    def _metadata(self) -> dict:
        return {
            **self._config_signature(),
            "chunk_count": len(self.chunks),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": sorted({chunk.source for chunk in self.chunks}),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    def save(
        self,
        index_path: str | Path = config.INDEX_FILE,
        metadata_path: str | Path = config.METADATA_FILE,
    ) -> None:
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        metadata_path.write_text(
            json.dumps(self._metadata(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        index_path: str | Path = config.INDEX_FILE,
        metadata_path: str | Path = config.METADATA_FILE,
    ) -> "Retriever":
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Index not found ({index_path} / {metadata_path}). Run scripts/build_index.py first."
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        mismatched = {
            key: {"stored": metadata.get(key), "expected": expected}
            for key, expected in cls._config_signature().items()
            if metadata.get(key) != expected
        }
        if mismatched:
            raise RuntimeError(
                f"Stored index does not match the current configuration: {mismatched}. "
                "Rebuild it with scripts/build_index.py."
            )

        index = faiss.read_index(str(index_path))
        chunks = [Chunk(**stored) for stored in metadata["chunks"]]
        return cls(index, chunks)
