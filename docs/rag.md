# `rag/` — Paket Retrieval

Berisi seluruh logika pipeline RAG: memuat dokumen, memecahnya jadi chunk,
membuat embedding, membangun/menyimpan index FAISS, mencari, menyusun prompt,
dan memanggil LLM.

## Alur Retrieval

```
┌──────────────────────────────────────────────┐
│                dokumen mentah                │
│      *.html / *.pdf dari data/raw_docs/      │
└───────────────────────┬──────────────────────┘
                        │
                        │
                        ▼
┌──────────────────────────────────────────────┐
│                rag/loader.py                 │
│    Chunk: text + source + locator + title    │
└───────────────────────┬──────────────────────┘
                        │
                        │  list[Chunk]
                        ▼
┌──────────────────────────────────────────────┐
│               rag/embedder.py                │
│    vektor 3072-d (prefix dokumen / query)    │
└───────────────────────┬──────────────────────┘
                        │
                        │
                        ▼
┌──────────────────────────────────────────────┐
│               rag/retriever.py               │
│    IndexFlatIP: build, save/load, search     │
└───────────────────────┬──────────────────────┘
                        │
                        │
                        ▼
┌──────────────────────────────────────────────┐
│              Hit: chunk + skor               │
└──────────────────────────────────────────────┘
```

---

## `rag/loader.py`

Mengubah file di `data/raw_docs/` menjadi daftar objek `Chunk`. Mendukung PDF
(`.pdf`) dan HTML (`.html`/`.htm`); suffix lain diabaikan.

### `Chunk` (dataclass, frozen)

Unit terkecil yang di-embed dan diberi metadata lokasi.

| Field           | Isi                                                           |
| --------------- | ------------------------------------------------------------- |
| `text`          | Potongan teks yang sudah dibersihkan                          |
| `source`        | Nama dokumen (judul)                                          |
| `locator_type`  | `"page"` (PDF), `"section"` (HTML), atau `"document"`         |
| `locator_value` | Nomor halaman, nama heading, atau nama dokumen                |
| `title`         | Judul gabungan untuk prefix embedding: `"{source} — {label}"` |

Properti `locator_label` menghasilkan label lokasi berbahasa Indonesia:
`"Halaman N"` untuk PDF, `"Bagian: X"` untuk section, dan `""` untuk document.
`to_dict()` dipakai saat menyimpan chunk ke `metadata.json`.

`source` adalah **judul dokumen, bukan nama file**, dan judul tidak dijamin unik:
`imigrasi_faq_visa.html` dan `imigrasi_faq_negara_e_voa.html` sama-sama berjudul
`"Pertanyaan Paling Sering Diajukan"`, sehingga `metadata.json.sources` (sebuah
himpunan) hanya menyimpan satu entri untuk keduanya. Karena itu `source` **tidak
bisa** dipakai untuk menyimpulkan "dokumen ini sudah terindeks" — itulah sebabnya
`build_index.py --add` meminta daftar file eksplisit.

### Fungsi publik

- **`load_file(path) -> list[Chunk]`** — memuat satu file.
- **`load_documents(raw_docs_dir=config.RAW_DOCS_DIR) -> list[Chunk]`** — memuat
  semua file bersuffix yang didukung di direktori, terurut berdasarkan nama.

### Chunking

`RecursiveCharacterTextSplitter` memecah per blok (halaman PDF atau section HTML)
memakai `CHUNK_SIZE`/`CHUNK_OVERLAP` dari config. **Penting:** `length_function`
di-set ke penghitung token `tiktoken` (`cl100k_base`), sehingga `CHUNK_SIZE = 500`
berarti ~500 token, bukan 500 karakter. Tanpa ini splitter default menghitung
karakter dan chunk jadi jauh lebih pendek. `tiktoken` hanyalah proksi lokal —
model Gemini tidak memakai tokenizer itu.

### Ekstraksi HTML

Aturan yang sudah menyelesaikan masalah nyata di korpus:

1. **Penamaan dokumen** (`_document_name`): pakai `<h1>` **hanya bila jumlahnya
   tepat satu**. Kalau ada beberapa `<h1>` (kemungkinan heading section), jatuh ke
   `<title>` (dipotong pada pemisah `–`, `—`, `|`, `-`). Kalau tidak ada juga,
   pakai nama file. Ini menghindari nama dokumen seperti `"F.A.Q"`.
2. **Membuang chrome halaman**: tag `script`, `style`, `nav`, `footer`, `header`,
   `noscript`, `svg`, `form`, `iframe` dibuang sebelum ekstraksi.
3. **Memilih container konten** (`_content_root`): kandidat adalah `<main>`, semua
   `<section>`, dan elemen dengan class/id mengandung `content`/`article`/`post`/
   `entry`/`main`. Kandidat dengan teks terbanyak dipilih, selama ≥ 200 karakter.
   Bila yang terpilih ternyata sebuah `<section>`, root dinaikkan satu level ke
   parent-nya. Ini perlu karena (a) halaman biaya menaruh tabel di
   `<section class="blog">` sementara widget `div class="content"` memenangkan
   heuristik naif, dan (b) halaman paspor menaruh tiap bagian sebagai `<section>`
   bersaudara. Efek sampingnya juga membuang menu navigasi.
4. **Blok teks**: heading (`h1`–`h4`) membuka section baru; `p`/`li`/`td`
   mengisi body section tersebut.

### Ekstraksi PDF

`pypdf` membaca per halaman; satu halaman = satu blok dengan `locator_type="page"`.
Judul diambil dari metadata PDF bila ada, jika tidak dari nama file.

> **Catatan:** jalur PDF belum pernah dieksekusi — seluruh korpus saat ini berupa
> HTML. Kode `pypdf` belum teruji.

---

## `rag/embedder.py`

Membungkus `google-genai` untuk menghasilkan vektor.

### Fungsi publik

- **`make_client(api_key=None) -> genai.Client`** — memakai `GEMINI_API_KEY` bila
  tidak diberikan. Timeout HTTP di-set dari `HTTP_TIMEOUT_MS`; default SDK adalah
  `None`, yang berarti httpx **tanpa timeout sama sekali** sehingga koneksi yang
  macet menggantung selamanya alih-alih masuk ke jalur retry.
- **`embed_documents(chunks, client=None) -> list[list[float]]`**
- **`embed_queries(queries, client=None) -> list[list[float]]`**
- **`embed_query(query, client=None) -> list[float]`** — kenyamanan untuk satu query.

### Retry & progres

`_embed_all()` memanggil `embed_content` satu kali per batch berisi
`EMBEDDING_BATCH_SIZE` chunk. Karena satu chunk dihitung satu request terhadap
kuota free tier, `429` justru hal biasa; `_retry_wait()` memakai petunjuk
`retry in Ns` dari server (dibatasi 90 s) dan `_RETRY_ATTEMPTS = 6`.

Penantian itu bisa berjalan menit-menitan, jadi `_embed_all()` mencetak progres
tiap batch dan setiap penantian `429`. Tanpa itu, rebuild terlihat seperti hang
padahal hanya menunggu (DECISIONS §15).

---

## `rag/retriever.py`

Membangun index, menyimpan/memuatnya, dan mencari.

### Objek

- **`Hit`** (frozen dataclass): `chunk: Chunk` + `score: float` (inner product =
  cosine, karena vektor unit-norm).
- **`Retriever`**: membungkus `faiss.Index` + daftar `Chunk`. Konstruktor
  memverifikasi `index.ntotal == len(chunks)`.

### Fungsi / method

- **`build_index(vectors) -> faiss.Index`** — `IndexFlatIP`; dimensi diambil dari
  panjang vektor. Melempar error bila list vektor kosong.
- **`Retriever.build(chunks, vectors)`** — konstruktor kelas untuk index baru.
- **`Retriever.add(chunks, vectors)`** — menambahkan vektor ke index yang sudah
  ada (`index.add`) dan chunk ke daftar internal. Hanya sah selama konfigurasi
  chunking/embedding tidak berubah — `load()` sudah menegakkannya lewat staleness
  guard. Melempar `ValueError` bila jumlah vektor ≠ jumlah chunk atau dimensinya
  tidak cocok dengan `index.d`. Tidak ada de-duplikasi.
- **`Retriever.search(query_vector, top_k=config.TOP_K_RETRIEVAL, threshold=config.SIMILARITY_THRESHOLD)`**
  — mencari `top_k` terdekat. Bila `threshold` di-set (bukan `None`) **dan** skor
  top-1 di bawah threshold, kembalikan list kosong (query ditolak). `threshold=None`
  mematikan gate. `top_k` otomatis dibatasi jumlah chunk.
- **`Retriever.save(index_path=config.INDEX_FILE, metadata_path=config.METADATA_FILE)`** —
  menulis index FAISS dan `metadata.json` (UTF-8, indent 2).
- **`Retriever.load(...)`** — memuat keduanya; melempar `FileNotFoundError` dengan
  pesan agar menjalankan `scripts/build_index.py` bila file tidak ada.

### Staleness guard

`_config_signature()` menyimpan: `embedding_model`, `embedding_dimensions`,
`query_template`, `document_template`, `chunk_size`, `chunk_overlap`. Saat `load()`,
semua nilai ini dibandingkan dengan isi `metadata.json`; bila ada yang beda,
`RuntimeError` dilempar dan index harus di-build ulang.

Ini mencegah skenario paling berbahaya: **memuat index lama setelah model
embedding atau prefix berubah** — hasilnya salah tapi tidak error. Selalu baca
ulang index lewat guard ini, jangan memuat `index.faiss` secara mentah.

`metadata.json` juga menyimpan `chunk_count`, `built_at`, `sources` (daftar unik),
dan seluruh `chunks`, sehingga `Retriever.load()` bisa merekonstruksi objek `Chunk`
tanpa membaca ulang `data/raw_docs/`.

---

## `rag/prompt_builder.py`

Menyusun system prompt yang dikirim ke LLM. Isinya adalah prompt PRD §6 dengan
**satu amendemen pada aturan 3** (DECISIONS §8): aturan yang meminta blok
`[Sumber: …]` diganti dengan larangan menulis daftar sumber atau penanda kutipan.
Konteks disuntik sebagai blok berlabel **tanpa nomor**, sehingga tidak ada nomor
yang bisa dikutip model, dan jawaban tampil sebagai prosa murni. Aturan §6 lain
dipertahankan apa adanya.

| Fungsi / konstanta                                  | Isi                                                                                            |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `SYSTEM_PROMPT`                                     | Teks §6 dengan placeholder `{retrieved_context}` dan `{conversation_history}`                   |
| `format_context(hits)`                              | Blok `nama_dokumen — Bagian: X` + teks chunk (tanpa nomor); `"(Tidak ada konteks relevan yang ditemukan.)"` bila kosong |
| `window_history(history, budget=MEMORY_TOKEN_BUDGET)` | Menyimpan pesan terbaru sampai anggaran token habis, membuang yang tertua lebih dulu          |
| `format_history(history)`                           | Menulis riwayat sebagai `Pengguna:` / `Asisten:`                                               |
| `build_system_instruction(hits, history)`           | Menggabungkan konteks + riwayat menjadi system instruction                                     |

Placeholder diganti dengan `str.replace`, bukan `str.format`, supaya kurung
kurawal lain di prompt tidak perlu di-escape.

Token dihitung dengan `tiktoken` (`cl100k_base`), proksi lokal yang sama seperti
`loader.py`; model Gemini tidak memakai tokenizer tersebut.

## `rag/generator.py`

Dua panggilan `gemini-3.5-flash-lite`:

- **`condense_query(history, user_message, client=None)`** — menulis ulang
  pertanyaan lanjutan menjadi pertanyaan mandiri memakai riwayat, dengan
  `CONDENSE_TEMPERATURE = 0.0`. **Dilewati bila riwayat kosong** (giliran
  pertama) dan mengembalikan pesan asli apa adanya. Hasilnya hanya dipakai untuk
  retrieval; LLM tetap menjawab **pesan asli** (DECISIONS §6).
- **`generate_answer(hits, history, user_message, client=None)`** — menyusun
  system instruction lewat `build_system_instruction()`, lalu meminta jawaban
  dengan `LLM_TEMPERATURE = 0.2`. Bila `hits` kosong, konteks yang disuntik juga
  kosong sehingga aturan §6 nomor 2 menghasilkan penolakan jujur (DECISIONS §7).

Keduanya melewati `_generate()` internal yang me-retry error `429`/`5xx` memakai
kebijakan yang sama dengan `embedder.py` (menghormati petunjuk `retry in Ns`).
Automatic function calling dimatikan karena tidak ada tool yang dipakai.

## Alur Generasi

Diagram alur runtime lengkap (UI → condensation → retrieval → generation) ada di
[`app.md`](app.md).
