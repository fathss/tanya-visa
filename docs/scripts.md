# `scripts/` — Utilitas CLI

Script operasional pipeline. Semuanya menambahkan root proyek ke `sys.path`,
sehingga dijalankan dari mana pun tetap menemukan paket `rag` dan `config`.
Gunakan interpreter di venv proyek (`.venv/bin/python`).

```
scripts/
  build_index.py       build ulang index FAISS dari data/raw_docs/
  fetch_docs.py        ambil dokumen sumber ke data/raw_docs/
  fetch_manifest.txt   daftar filename dan URL dokumen sumber
  test_retrieval.py    jalankan eval set dan cetak metrik relevansi
  probe.py             coba query manual (retrieval saja, tanpa LLM)
```

---

## `build_index.py`

Mengelola index di `data/index/`. Dua mode:

```bash
.venv/bin/python scripts/build_index.py                                # rebuild penuh
.venv/bin/python scripts/build_index.py --add data/raw_docs/baru.html  # tambah file
```

**Rebuild penuh** (tanpa argumen):

1. `load_documents()` memuat semua dokumen jadi `Chunk`.
2. `embed_documents()` meng-embed tiap chunk dengan `EMBEDDING_MODEL`.
3. `Retriever.build()` menyusun index dari nol, lalu `save()` menulis
   `index.faiss` dan `metadata.json` ke `data/index/`.

Melempar `SystemExit` bila tidak ada dokumen yang didukung.

**`--add FILE...`** membangun ulang hanya file yang disebut:

1. `Retriever.load()` memuat index yang ada — sekaligus menjalankan staleness
   guard, sehingga append tidak mungkin mencampur dua model embedding.
2. `load_file()` meng-chunk tiap file; file tanpa teks dilaporkan lalu dilewati.
3. `embed_documents()` meng-embed **hanya chunk baru**.
4. `Retriever.add()` menambahkan vektor ke index dan chunk ke daftar, lalu
   `save()`. Vektor lama tidak disentuh dan tidak dihitung ulang.

Ini jalur normal setelah `fetch_docs.py` mengambil dokumen baru.
Satu dokumen ≈ 20-an chunk, sedangkan rebuild penuh 226 chunk — dan karena 1
chunk = 1 request embedding terhadap kuota 100/menit, rebuild penuh **pasti**
kena `429` dan masuk backoff panjang. `embedder.py` mencetak progres per batch
dan setiap penantian `429`, jadi backoff itu terlihat, bukan tampak hang.

Tanpa de-duplikasi: `--add` pada file yang sudah ada menyimpan chunk kembar.
Identitas dokumen tidak bisa dipakai untuk mendeteksi itu — lihat catatan
`source` di [`rag.md`](rag.md).

---

## `fetch_docs.py` dan `fetch_manifest.txt`

`fetch_docs.py` mengambil dokumen sumber dari URL ke `data/raw_docs/` berdasarkan
manifest. Baris kosong dan baris yang diawali `#` diabaikan. Setiap target aktif
memakai format dua kolom:

```text
<output_filename>\t<url>
```

Sebelum mengambil URL, script memeriksa `robots.txt` per host dan melewati path
yang tidak diizinkan. Antar-request diberi jeda 1,5 detik secara default. Gunakan
`--dry-run` untuk memeriksa robots.txt tanpa mengunduh, atau `--delay` untuk
mengubah jeda:

```bash
.venv/bin/python scripts/fetch_docs.py scripts/fetch_manifest.txt --dry-run
.venv/bin/python scripts/fetch_docs.py scripts/fetch_manifest.txt
.venv/bin/python scripts/fetch_docs.py scripts/fetch_manifest.txt --delay 2
```

Setelah dokumen baru berhasil diunduh, tambahkan hanya file tersebut ke index
dengan `scripts/build_index.py --add`. Jangan menjalankan rebuild penuh hanya
untuk satu dokumen karena seluruh chunk akan di-embed ulang.

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
