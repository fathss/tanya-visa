# Chatbot Panduan Traveling & Visa Berbasis RAG

Chatbot yang menjawab pertanyaan seputar paspor dan visa **hanya berdasarkan dokumen resmi** yang disediakan, lengkap dengan kutipan sumber. Dibangun dengan Streamlit + Google Gemini.

Setiap jawaban wajib menyertakan sumber, dan chatbot harus jujur menyatakan tidak tahu bila informasi tidak ada di dokumen.

## Status

| Tahap                                                           | Status                      |
| --------------------------------------------------------------- | --------------------------- |
| [1] pipeline RAG (load → chunk → embedding → FAISS → retrieval) | **Selesai & terverifikasi** |
| [2] prompt + LLM + UI Streamlit + memory                        | Belum dikerjakan            |

UI chat belum tersedia. Yang bisa dijalankan sekarang adalah pengujian retrieval (lihat bagian [Menjalankan](#menjalankan)).

## Prasyarat

- Python 3.11+
- Gemini API key dari [Google AI Studio](https://aistudio.google.com/apikey)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Lalu isi `GEMINI_API_KEY` pada `.env`.

## Menjalankan

### Uji retrieval

```bash
.venv/bin/python scripts/test_retrieval.py
```

Menjalankan eval set (`data/eval/queries.json`) dan mencetak tingkat relevansi, skor top-1, serta pemisahan skor untuk kalibrasi threshold.

### Coba pertanyaan sendiri

```bash
.venv/bin/python scripts/probe.py "Apa syarat paspor untuk anak di bawah umur?"
.venv/bin/python scripts/probe.py # mode interaktif
```

Menampilkan chunk yang terambil beserta skor dan lokasinya. Query yang di bawah threshold akan ditolak, disertai skor terdekatnya.

### Build ulang index

```bash
.venv/bin/python scripts/build_index.py
```

Jalankan hanya jika korpus, chunking, atau model embedding berubah.

## Struktur Folder

```
app.py                 # Entry point Streamlit
config.py              # Semua parameter yang bisa di-tuning
rag/
  loader.py            # Muat PDF/HTML, chunking + metadata lokasi
  embedder.py          # Generate embeddings
  retriever.py         # Build & query FAISS index
  prompt_builder.py    # Gabungkan system prompt + context + history
data/
  eval/queries.json    # Eval set untuk uji relevansi
  index/               # FAISS index hasil build (di-commit)
  raw_docs/            # Dokumen sumber
scripts/
  build_index.py       # Build index dari data/raw_docs/
  probe.py             # Coba query manual
  test_retrieval.py    # Jalankan eval set
```

## Dokumen Sumber

Semua dokumen bersifat publik, resmi (`.go.id`), dan berbahasa Indonesia. Tanggal unduh dicatat agar jelas per-kapan informasi tersebut berlaku.

| File                                         | Judul                                              | Sumber                                                                                                  | Tanggal unduh |
| -------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------- |
| `ditjen_imigrasi_paspor_baru.html`           | WNI — Paspor Baru                                  | https://www.imigrasi.go.id/wni/paspor-baru                                                              | 2026-09-26    |
| `imigrasi_malang_prosedur_paspor_2026.html`  | Syarat dan Prosedur Lengkap Pengurusan Paspor 2026 | https://malang.imigrasi.go.id/info-publik/syarat-dan-prosedur-lengkap-pengurusan-paspor-2026            | 2026-09-26    |
| `imigrasi_visa_b1_saat_kedatangan.html`      | WNA — Visa Saat Kedatangan (B1)                    | https://www.imigrasi.go.id/wna/daftar-visa-indonesia/B1                                                 | 2026-09-26    |
| `imigrasi_voa_vs_visa_kunjungan_wisata.html` | Beda Visa Kunjungan Wisata dan Visa on Arrival     | https://www.imigrasi.go.id/berita/jangan-salah-pilih-ini-beda-visa-kunjungan-wisata-dan-visa-on-arrival | 2026-09-26    |

### Catatan Freshness

Aturan visa dan imigrasi dapat berubah sewaktu-waktu. Bila dokumen di atas dinilai sudah lama, unduh ulang dari sumber resmi, jalankan `scripts/build_index.py`, lalu perbarui tanggal unduh pada tabel ini.
