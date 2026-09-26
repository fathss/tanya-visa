# `data/` — Direktori Data

Menampung input (dokumen mentah), output (index FAISS), dan eval set.

```
data/
  raw_docs/            dokumen sumber (input)
  index/               hasil build FAISS (output)
    index.faiss
    metadata.json
  eval/
    queries.json       eval set retrieval
```

---

## `data/raw_docs/`

Korpus dokumen sumber. `rag.loader` memuat semua file `.html`/`.htm`/`.pdf` di sini
(mengabaikan file lain), terurut berdasarkan nama file.

- Total: **24 dokumen, 203 chunk** (index terakhir di-build 2026-09-26).
- Dominan berbahasa Indonesia, dengan 2 dokumen resmi Singapura (EN) dan 1 dokumen
  resmi Malaysia (MS).
- Daftar lengkap dokumen beserta bahasa, topik, URL sumber, dan tanggal undang ada
  di tabel korpus pada [`README.md`](../README.md) di root.

Untuk menambah dokumen baru, lihat `scripts/fetch_docs.py` dan
`scripts/fetch_manifest.txt`.

---

## `data/index/`

Output dari `scripts/build_index.py`.

### `index.faiss`

Index `faiss.IndexFlatIP` berisi seluruh vektor chunk (unit-normalized). Dibaca
hanya lewat `Retriever.load()`

### `metadata.json`

JSON UTF-8 berisi:

| Field                                     | Isi                                     |
| ----------------------------------------- | --------------------------------------- |
| `embedding_model`, `embedding_dimensions` | Signature konfigurasi                   |
| `query_template`, `document_template`     | Template prefix saat build              |
| `chunk_size`, `chunk_overlap`             | Parameter chunking saat build           |
| `chunk_count`                             | Jumlah chunk                            |
| `built_at`                                | Timestamp build (UTC, ISO 8601)         |
| `sources`                                 | Daftar nama dokumen unik                |
| `chunks`                                  | Array seluruh chunk (`Chunk.to_dict()`) |

Enam field pertama membentuk **staleness guard**: `Retriever.load()` menolak index
yang signature-nya tidak cocok dengan `config.py`. Jangan mengedit `metadata.json`
manual, bangun ulang lewat script.

---

## `data/eval/queries.json`

Eval set untuk mengukur kualitas retrieval, dijalankan oleh
`scripts/test_retrieval.py`.

```jsonc
{
  "version": 1,
  "description": "...",
  "queries": [
    /* ... */
  ],
}
```

Setiap entri query:

| Field              | Isi                                                             |
| ------------------ | --------------------------------------------------------------- |
| `id`               | `r00`… untuk query relevan, `o00`… untuk query di luar topik    |
| `kind`             | `"relevant"` atau `"off_topic"`                                 |
| `query`            | Teks pertanyaan                                                 |
| `expected_source`  | Substring nama dokumen yang diharapkan muncul (boleh `null`)    |
| `expected_section` | Substring `locator_value` yang diharapkan muncul (boleh `null`) |

Aturan penilaian (lihat `_is_relevant_hit` di `scripts/test_retrieval.py`): query
`relevant` lulus bila ada satu hit yang cocok `expected_source` **atau**
`expected_section`. Query `off_topic` tidak dinilai per-query — yang diukur adalah
pemisahan skornya terhadap query relevan, untuk kalibrasi `SIMILARITY_THRESHOLD`.

Saat ini: 23 query relevan (r01–r23) dan 5 query di luar topik (o01–o05).
Menambah dokumen baru sebaiknya disertai query relevan baru di sini.
