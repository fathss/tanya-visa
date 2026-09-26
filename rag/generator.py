"""LLM calls: follow-up condensation and grounded generation.

Condensation turns a context-dependent follow-up into a standalone query so that
retrieval has something to work with; generation answers from the retrieved
chunks plus the original message and the conversation history.
"""

from __future__ import annotations

import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

import config

from rag.embedder import _RETRY_ATTEMPTS, _is_retryable, _retry_wait, make_client
from rag.prompt_builder import build_system_instruction, format_history, window_history
from rag.retriever import Hit

_CONDENSE_INSTRUCTION = """Tugasmu menulis ulang pertanyaan terakhir pengguna menjadi satu pertanyaan mandiri (standalone) untuk keperluan pencarian dokumen.

Aturan:
- Lengkapi topik yang tidak disebut ulang memakai informasi dari riwayat percakapan. Contoh: dari "Apa syarat visa turis ke Jepang?" lalu "kalau untuk anak di bawah umur gimana?" menghasilkan "Apa syarat visa turis ke Jepang untuk anak di bawah umur?".
- Pertahankan istilah aslinya: nama negara, jenis visa, jenis dokumen.
- Jangan menjawab pertanyaan dan jangan menambahkan informasi baru.
- Balas HANYA dengan pertanyaan hasil penulisan ulang, tanpa penjelasan, tanpa tanda kutip."""


def _generate(
    client: genai.Client,
    contents: str,
    *,
    system_instruction: str | None,
    temperature: float,
) -> str:
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=config.LLM_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            return (response.text or "").strip()
        except genai_errors.APIError as exc:
            if attempt == _RETRY_ATTEMPTS - 1 or not _is_retryable(exc):
                raise
            time.sleep(_retry_wait(exc, attempt))

    return ""


def condense_query(
    history: list[dict],
    user_message: str,
    client: genai.Client | None = None,
) -> str:
    """Rewrite a follow-up into a standalone query. Skipped on the first turn."""
    if not history:
        return user_message

    client = client or make_client()
    prompt = (
        f"{_CONDENSE_INSTRUCTION}\n\n"
        f"RIWAYAT PERCAKAPAN:\n{format_history(window_history(history))}\n\n"
        f"PERTANYAAN TERAKHIR:\n{user_message}\n\n"
        "PERTANYAAN MANDIRI:"
    )

    rewritten = _generate(
        client, prompt, system_instruction=None, temperature=config.CONDENSE_TEMPERATURE
    )
    return rewritten or user_message


def generate_answer(
    hits: list[Hit],
    history: list[dict],
    user_message: str,
    client: genai.Client | None = None,
) -> str:
    """Answer the original message using the retrieved chunks and the history.

    With no hits the context block is empty and §6 rule 2 produces the honest
    refusal (DECISIONS §7).
    """
    client = client or make_client()
    return _generate(
        client,
        user_message,
        system_instruction=build_system_instruction(hits, history),
        temperature=config.LLM_TEMPERATURE,
    )
