"""Generate Gemini embeddings.

gemini-embedding-2 ignores task_type, so the task is carried in the text itself
(see config.query_embedding_text / config.document_embedding_text).

Each chunk counts as one request against the free-tier quota (100/minute), so
rate-limit responses are retried using the delay the server asks for.
"""

from __future__ import annotations

import re
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

import config
from rag.loader import Chunk

_RETRY_ATTEMPTS = 6
_RETRY_MAX_WAIT = 90.0


def make_client(api_key: str | None = None) -> genai.Client:
    return genai.Client(api_key=api_key or config.GEMINI_API_KEY)


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if isinstance(code, int) and (code == 429 or code >= 500):
        return True
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def _retry_wait(exc: Exception, attempt: int) -> float:
    match = re.search(r"retry in ([\d.]+)s", str(exc))
    if match:
        return min(float(match.group(1)) + 2.0, _RETRY_MAX_WAIT)
    return min(5.0 * (2**attempt), _RETRY_MAX_WAIT)


def _embed(client: genai.Client, texts: list[str], label: str) -> list[list[float]]:
    contents = [types.Content(parts=[types.Part.from_text(text=text)]) for text in texts]

    result = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            result = client.models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=config.EMBEDDING_DIMENSIONS),
            )
            break
        except genai_errors.APIError as exc:
            if attempt == _RETRY_ATTEMPTS - 1 or not _is_retryable(exc):
                raise
            wait = _retry_wait(exc, attempt)
            print(
                f"{label}rate limited; waiting {wait:.0f}s "
                f"(attempt {attempt + 1}/{_RETRY_ATTEMPTS})",
                flush=True,
            )
            time.sleep(wait)

    vectors = [list(embedding.values) for embedding in result.embeddings]
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embeddings but got {len(vectors)}. "
            "gemini-embedding-2 aggregates multiple inputs into one vector unless "
            "each input is wrapped in its own types.Content."
        )

    return vectors


def _embed_all(client: genai.Client, texts: list[str]) -> list[list[float]]:
    total = len(texts)
    batch_size = config.EMBEDDING_BATCH_SIZE
    batch_count = max(1, -(-total // batch_size))
    vectors: list[list[float]] = []

    for number, start in enumerate(range(0, total, batch_size), start=1):
        stop = min(start + batch_size, total)
        label = f"  [embed {number}/{batch_count}] "
        print(f"{label}{stop - start} chunks ({start + 1}-{stop} of {total})…", flush=True)
        vectors.extend(_embed(client, texts[start:stop], label))

    return vectors


def embed_documents(chunks: list[Chunk], client: genai.Client | None = None) -> list[list[float]]:
    client = client or make_client()
    texts = [config.document_embedding_text(chunk.title, chunk.text) for chunk in chunks]
    return _embed_all(client, texts)


def embed_queries(queries: list[str], client: genai.Client | None = None) -> list[list[float]]:
    client = client or make_client()
    texts = [config.query_embedding_text(query) for query in queries]
    return _embed_all(client, texts)


def embed_query(query: str, client: genai.Client | None = None) -> list[float]:
    return embed_queries([query], client)[0]
