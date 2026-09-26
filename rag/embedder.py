"""Generate Gemini embeddings.

gemini-embedding-2 ignores task_type, so the task is carried in the text itself
(see config.query_embedding_text / config.document_embedding_text).
"""

from __future__ import annotations

from google import genai
from google.genai import types

import config
from rag.loader import Chunk


def make_client(api_key: str | None = None) -> genai.Client:
    return genai.Client(api_key=api_key or config.GEMINI_API_KEY)


def _embed(client: genai.Client, texts: list[str]) -> list[list[float]]:
    contents = [types.Content(parts=[types.Part.from_text(text=text)]) for text in texts]
    result = client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=contents,
        config=types.EmbedContentConfig(output_dimensionality=config.EMBEDDING_DIMENSIONS),
    )

    vectors = [list(embedding.values) for embedding in result.embeddings]
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embeddings but got {len(vectors)}. "
            "gemini-embedding-2 aggregates multiple inputs into one vector unless "
            "each input is wrapped in its own types.Content."
        )

    return vectors


def embed_documents(chunks: list[Chunk], client: genai.Client | None = None) -> list[list[float]]:
    client = client or make_client()
    texts = [config.document_embedding_text(chunk.title, chunk.text) for chunk in chunks]

    vectors: list[list[float]] = []
    for start in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
        vectors.extend(_embed(client, texts[start : start + config.EMBEDDING_BATCH_SIZE]))

    return vectors


def embed_query(query: str, client: genai.Client | None = None) -> list[float]:
    client = client or make_client()
    return _embed(client, [config.query_embedding_text(query)])[0]
