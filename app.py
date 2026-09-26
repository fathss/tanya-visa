"""Streamlit chat UI: RAG chat with conversation memory and a document sidebar.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import errors as genai_errors

import config
from rag.embedder import embed_documents, embed_query, make_client
from rag.generator import condense_query, generate_answer
from rag.loader import Chunk, load_file
from rag.retriever import Retriever

st.set_page_config(page_title="Panduan Traveling & Visa", layout="centered")

_UPLOAD_TYPES = ["pdf", "html", "htm"]


def _api_key() -> str | None:
    if config.GEMINI_API_KEY:
        return config.GEMINI_API_KEY
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _client(api_key: str) -> genai.Client:
    return make_client(api_key)


@st.cache_resource(show_spinner=False)
def _shipped_corpus() -> tuple[list[Chunk], list[list[float]]]:
    """The committed index, split into chunks and raw vectors so uploads can extend it."""
    retriever = Retriever.load()
    vectors = retriever.index.reconstruct_n(0, retriever.index.ntotal).tolist()
    return list(retriever.chunks), vectors


def _init_state() -> None:
    if "messages" in st.session_state:
        return
    chunks, vectors = _shipped_corpus()
    st.session_state.messages = []
    st.session_state.chunks = list(chunks)
    st.session_state.vectors = list(vectors)
    st.session_state.retriever = Retriever.build(st.session_state.chunks, st.session_state.vectors)


def _retriever() -> Retriever:
    if st.session_state.get("retriever") is None:
        st.session_state.retriever = Retriever.build(
            st.session_state.chunks, st.session_state.vectors
        )
    return st.session_state.retriever


def _add_uploads(uploads: list) -> None:
    client = _client(_api_key())
    added: list[Chunk] = []

    with st.spinner("Memproses dokumen…"):
        for upload in uploads:
            suffix = Path(upload.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(upload.getvalue())
                temp_path = Path(handle.name)
            try:
                added.extend(load_file(temp_path))
            finally:
                temp_path.unlink()

        if not added:
            st.warning("Tidak ada teks yang bisa diambil dari dokumen tersebut.")
            return

        vectors = embed_documents(added, client)

    st.session_state.chunks.extend(added)
    st.session_state.vectors.extend(vectors)
    st.session_state.retriever = None
    st.toast(f"{len(added)} chunk ditambahkan dari {len(uploads)} dokumen.")
    st.rerun()


def _reset_documents() -> None:
    chunks, vectors = _shipped_corpus()
    st.session_state.chunks = list(chunks)
    st.session_state.vectors = list(vectors)
    st.session_state.retriever = None
    st.rerun()


def _sidebar() -> None:
    with st.sidebar:
        st.subheader("Dokumen")
        st.caption(
            f"{len(st.session_state.chunks)} chunk aktif · "
            f"{len({chunk.source for chunk in st.session_state.chunks})} dokumen"
        )

        uploads = st.file_uploader(
            "Tambah dokumen resmi (PDF/HTML)",
            type=_UPLOAD_TYPES,
            accept_multiple_files=True,
        )
        if st.button("Proses dokumen", disabled=not uploads, use_container_width=True):
            _add_uploads(uploads)
        if st.button("Reset dokumen", use_container_width=True):
            _reset_documents()

        with st.expander("Daftar dokumen"):
            for source in sorted({chunk.source for chunk in st.session_state.chunks}):
                st.markdown(f"- {source}")

        st.divider()
        if st.button("Reset percakapan", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _answer(prompt: str, client: genai.Client) -> str:
    history = st.session_state.messages[:-1]

    with st.spinner("Mencari di dokumen…"):
        standalone = condense_query(history, prompt, client)
        hits = _retriever().search(embed_query(standalone, client))
        return generate_answer(hits, history, prompt, client)


def main() -> None:
    api_key = _api_key()
    if not api_key:
        st.error(
            "GEMINI_API_KEY belum diset. Isi `.env` (lihat `.env.example`) "
            "atau tambahkan secret dengan nama yang sama."
        )
        st.stop()

    st.title("Panduan Traveling & Visa")
    st.caption("Jawaban diambil hanya dari dokumen resmi yang tersedia.")

    _init_state()
    _sidebar()
    _render_history()

    prompt = st.chat_input("Tanya soal paspor, visa, atau aturan perjalanan…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = _client(api_key)
    with st.chat_message("assistant"):
        try:
            answer = _answer(prompt, client)
        except genai_errors.APIError as exc:
            answer = f"Maaf, terjadi kendala saat menghubungi layanan model: {exc}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


main()
