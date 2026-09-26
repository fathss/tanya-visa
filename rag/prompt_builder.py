"""Assemble the prompt sent to the generation call.
"""

from __future__ import annotations

import tiktoken

import config
from rag.retriever import Hit

_ENCODER = tiktoken.get_encoding(config.TOKENIZER_ENCODING)

_NO_CONTEXT = "(Tidak ada konteks relevan yang ditemukan.)"
_NO_HISTORY = "(Belum ada riwayat percakapan.)"

_ROLE_LABELS = {"user": "Pengguna", "assistant": "Asisten"}

SYSTEM_PROMPT = """Kamu adalah Asisten Panduan Traveling & Visa.
Tugasmu adalah menjawab pertanyaan seputar syarat visa, prosedur paspor,
dan aturan perjalanan HANYA berdasarkan dokumen resmi yang diberikan
sebagai konteks (retrieved context).

ATURAN UTAMA:
1. Jawab HANYA berdasarkan informasi di dalam konteks yang diberikan.
   Jangan gunakan pengetahuan umum atau asumsi di luar dokumen.
2. Jika informasi yang ditanyakan TIDAK ADA di konteks, katakan dengan jujur:
   "Maaf, informasi ini tidak saya temukan di dokumen yang tersedia."
   Jangan mengarang jawaban.
3. Jawab langsung dalam prosa. Jangan menuliskan daftar sumber atau penanda
   kutipan apa pun (misalnya "[1]" atau "[Sumber: ...]").
4. Gunakan riwayat percakapan (conversation history) untuk memahami
   pertanyaan lanjutan/follow-up.
5. Jika pertanyaan ambigu atau bisa merujuk ke beberapa topik berbeda
   (misal negara tujuan tidak disebutkan), tanyakan klarifikasi singkat
   sebelum menjawab.
6. Gaya bahasa: formal tapi ramah, hindari jargon birokrasi yang membingungkan.

KONTEKS DOKUMEN (retrieved chunks):
{retrieved_context}

RIWAYAT PERCAKAPAN:
{conversation_history}"""


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def format_context(hits: list[Hit]) -> str:
    """Inject each chunk under its document header, with no numbering."""
    if not hits:
        return _NO_CONTEXT

    blocks = []
    for hit in hits:
        chunk = hit.chunk
        header = chunk.source
        if chunk.locator_label:
            header = f"{header} — {chunk.locator_label}"
        blocks.append(f"{header}\n{chunk.text}")

    return "\n\n".join(blocks)


def window_history(
    history: list[dict], budget: int = config.MEMORY_TOKEN_BUDGET
) -> list[dict]:
    """Keep the newest messages until the token budget is reached (oldest dropped)."""
    kept: list[dict] = []
    used = 0

    for message in reversed(history):
        cost = count_tokens(message.get("content", ""))
        if kept and used + cost > budget:
            break
        kept.append(message)
        used += cost

    kept.reverse()
    return kept


def format_history(history: list[dict]) -> str:
    if not history:
        return _NO_HISTORY

    lines = []
    for message in history:
        label = _ROLE_LABELS.get(message.get("role"), message.get("role", "?"))
        lines.append(f"{label}: {message.get('content', '')}")
    return "\n\n".join(lines)


def build_system_instruction(hits: list[Hit], history: list[dict]) -> str:
    """Fill §6's placeholders with the labelled context and the bounded history."""
    return (
        SYSTEM_PROMPT.replace("{retrieved_context}", format_context(hits))
        .replace("{conversation_history}", format_history(window_history(history)))
    )
