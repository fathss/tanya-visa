# `scripts/` — Utilitas CLI

Script operasional pipeline. Semuanya menambahkan root proyek ke `sys.path`,
sehingga dijalankan dari mana pun tetap menemukan paket `rag` dan `config`.
Gunakan interpreter di venv proyek (`.venv/bin/python`).

```
scripts/
  build_index.py       build ulang index FAISS dari data/raw_docs/
  test_retrieval.py    jalankan eval set dan cetak metrik relevansi
  probe.py             coba query manual (retrieval saja, tanpa LLM)
```

---

## `build_index.py`

Membangun ulang seluruh index dari `data/raw_docs/`.

```bash
.venv/bin/python scripts/build_index.py
```

Alur:

1. `load_documents()` memuat semua dokumen jadi `Chunk`.
2. `embed_documents()` meng-embed tiap chunk dengan `EMBEDDING_MODEL`.
3. `Retriever.build()` menyusun index, lalu `save()` menulis `index.faiss` dan
   `metadata.json` ke `data/index/`.

Melempar `SystemExit` bila tidak ada dokumen yang didukung. Jalankan hanya saat
korpus, chunking, atau model embedding berubah — index yang ada sudah di-commit.
Perhatikan bahwa 1 chunk = 1 request embedding, jadi rebuild korpus penuh akan
memakan waktu dan kuota (embedder sudah menangani retry `429`).

---

## `test_retrieval.py`

Menjalankan eval set (`data/eval/queries.json`) dan mencetak metrik relevansi.

```bash
.venv/bin/python scripts/test_retrieval.py
```

- Memuat index, meng-embed semua query dari eval set.
- Untuk tiap query mencetak status `PASS`/`FAIL` (query relevan), skor top-1, dan
  untuk yang gagal menampilkan 3 hit teratas.
- Ringkasan: **retrieval relevance** `n/total (%)` dengan target ≥ 80%.
- Mencetak rentang skor top-1 query relevan vs query di luar topik, serta
  **separation**: `max(off_topic) < min(relevant)` bila terpisah, beserta saran
  threshold `(lo + hi) / 2`. Kalau tidak terpisah, dilaporkan `NO separation`.

Pencarian di sini memakai `threshold=None` (gate dimatikan) supaya bisa melihat
skor mentah untuk kalibrasi. Jalankan ulang setelah setiap perubahan chunking
atau embedding.

---

## `probe.py`

Coba query secara manual — **retrieval saja, tanpa LLM**.

```bash
.venv/bin/python scripts/probe.py "Apa syarat paspor untuk anak di bawah umur?"
.venv/bin/python scripts/probe.py           # mode interaktif
```

Saat start mencetak ringkasan index (`jumlah vektor`, `dim`, `top_k`, `threshold`).

- **Satu/lebih argumen**: setiap argumen diperlakukan sebagai query.
- **Tanpa argumen**: mode interaktif; ketik query, `Ctrl-D` untuk keluar.

Untuk tiap query menampilkan hit beserta skor dan lokasi (`source | locator_label`).
Bila hasilnya kosong, query ditandai `REFUSED (below ...)` dan ditampilkan hit
terdekat beserta skornya — berguna untuk melihat seberapa dekat query di luar
topik sebelum benar-benar ditolak gate.

---
