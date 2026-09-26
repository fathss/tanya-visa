# Dokumentasi

Referensi teknis untuk chatbot RAG traveling & visa. Dokumen di sini menjelaskan
implementasi; untuk setup dan cara pakai lihat [`README.md`](../README.md) di root

| Dokumen                    | Cakupan                                                          |
| -------------------------- | ---------------------------------------------------------------- |
| [`rag.md`](rag.md)         | Paket `rag/`: pemuatan, chunking, embedding, indexing, retrieval |
| [`data.md`](data.md)       | Direktori `data/`: dokumen mentah, index FAISS, eval set         |
| [`scripts.md`](scripts.md) | Utilitas CLI: build, evaluasi, probe                             |

## Alur Pipeline Sekilas

```
┌──────────────────────────────────────────────┐
│                data/raw_docs/                │
│        dokumen sumber: *.html, *.pdf         │
└───────────────────────┬──────────────────────┘
                        │
                        │  load_documents()
                        ▼
┌──────────────────────────────────────────────┐
│                rag/loader.py                 │
│  buang chrome (nav, script, style, footer)   │
│  pecah jadi Chunk (teks + metadata sitasi)   │
└───────────────────────┬──────────────────────┘
                        │
                        │  list[Chunk]
                        ▼
┌──────────────────────────────────────────────┐
│               rag/embedder.py                │
│     gemini-embedding-2 + prefix dokumen      │
└───────────────────────┬──────────────────────┘
                        │
                        │  list[vektor] 3072-d (unit-norm)
                        ▼
┌──────────────────────────────────────────────┐
│               rag/retriever.py               │
│      faiss.IndexFlatIP + metadata.json       │
└───────────────────────┬──────────────────────┘
                        │
                        │
                        ▼
┌──────────────────────────────────────────────┐
│                 data/index/                  │
│         index.faiss + metadata.json          │
└──────────────────────────────────────────────┘
                        │
═══════════ BUILD - offline, sekali ════════════
                        │
                        │  Retriever.load()  (staleness guard)
                        ▼
┌──────────────────────────────────────────────┐
│            query pengguna (teks)             │
└───────────────────────┬──────────────────────┘
                        │
                        │
                        ▼
┌──────────────────────────────────────────────┐
│               rag/embedder.py                │
│      gemini-embedding-2 + prefix query       │
└───────────────────────┬──────────────────────┘
                        │
                        │  vektor query
                        ▼
┌──────────────────────────────────────────────┐
│              Retriever.search()              │
│    top-k inner product -> gate threshold     │
└───────────────────────┬──────────────────────┘
                        │
                        │
                        ▼
┌──────────────────────────────────────────────┐
│          list[Hit]  (chunk + skor)           │
└──────────────────────────────────────────────┘
```

Indexing berjalan offline lewat `scripts/build_index.py`. Saat query hanya index
tersimpan yang dimuat dan yang di-embed hanya query masuk.

## Referensi Konfigurasi

Seluruh parameter bisa di-tuning dan berada di [`config.py`](../config.py).
Konstanta di bawah ini yang memengaruhi pipeline; mengubah salah satunya juga
mengubah signature index (lihat staleness guard di [`rag.md`](rag.md)).

| Konstanta                       | Nilai                                       | Arti                                                         |
| ------------------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| `EMBEDDING_MODEL`               | `gemini-embedding-2`                        | Model embedding (Google AI Studio)                           |
| `EMBEDDING_DIMENSIONS`          | `3072`                                      | Dimensi keluaran, unit-normalized                            |
| `EMBEDDING_BATCH_SIZE`          | `100`                                       | Chunk per panggilan `embed_content` (batas free tier)        |
| `QUERY_TEMPLATE`                | `task: question answering \| query: {text}` | Prefix untuk query                                           |
| `DOCUMENT_TEMPLATE`             | `title: {title} \| text: {text}`            | Prefix untuk chunk                                           |
| `CHUNK_SIZE` / `CHUNK_OVERLAP`  | `500` / `50`                                | Dalam **token** (tiktoken `cl100k_base`), bukan karakter     |
| `TOP_K_RETRIEVAL`               | `5`                                         | Jumlah hit yang dikembalikan per query                       |
| `SIMILARITY_THRESHOLD`          | `0.65`                                      | Di bawah skor top-1 ini query ditolak; `None` mematikan gate |
| `LLM_MODEL` / `LLM_TEMPERATURE` | `gemini-3.5-flash-lite` / `0.2`             | Generasi (Day 2, belum tersambung)                           |
| `MEMORY_TOKEN_BUDGET`           | `2000`                                      | Anggaran riwayat percakapan (Day 2)                          |

Konstanta path (`DATA_DIR`, `RAW_DOCS_DIR`, `INDEX_DIR`, `EVAL_FILE`, `INDEX_FILE`,
`METADATA_FILE`) semuanya diturunkan dari `PROJECT_ROOT`, jadi script tetap bekerja
di mana pun current working directory-nya.
