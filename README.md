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

Semua dokumen bersifat publik dan berasal dari situs resmi pemerintah/kedutaan. Tanggal unduh dicatat agar jelas per-kapan informasi tersebut berlaku.

| File                                               | Bahasa | Topik                                              | Sumber                                                                                                                    | Tanggal unduh |
| -------------------------------------------------- | ------ | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `ditjen_imigrasi_paspor_baru.html`                 | ID     | Paspor baru                                        | https://www.imigrasi.go.id/wni/paspor-baru                                                                                | 2026-09-26    |
| `imigrasi_malang_prosedur_paspor_2026.html`        | ID     | Syarat dan Prosedur Lengkap Pengurusan Paspor 2026 | https://malang.imigrasi.go.id/info-publik/syarat-dan-prosedur-lengkap-pengurusan-paspor-2026                              | 2026-09-26    |
| `imigrasi_daftar_visa_indonesia.html`              | ID     | Daftar Visa Indonesia                              | https://www.imigrasi.go.id/wna/daftar-visa-indonesia                                                                      | 2026-09-26    |
| `imigrasi_voa_vs_visa_kunjungan_wisata.html`       | ID     | Beda Visa Kunjungan Wisata dan Visa on Arrival     | https://www.imigrasi.go.id/berita/jangan-salah-pilih-ini-beda-visa-kunjungan-wisata-dan-visa-on-arrival                   | 2026-09-26    |
| `imigrasi_visa_a1_bebas_visa_wisata.html`          | ID     | A1 Bebas Visa Wisata                               | https://www.imigrasi.go.id/wna/daftar-visa-indonesia/A1                                                                   | 2026-09-26    |
| `imigrasi_visa_b1_saat_kedatangan.html`            | ID     | B1 Visa Kunjungan Wisata (Visa Saat Kedatangan)    | https://www.imigrasi.go.id/wna/daftar-visa-indonesia/B1                                                                   | 2026-09-26    |
| `imigrasi_visa_c1_kunjungan_wisata.html`           | ID     | C1 Visa Kunjungan Wisata                           | https://www.imigrasi.go.id/wna/daftar-visa-indonesia/C1                                                                   | 2026-09-26    |
| `imigrasi_visa_c2_kunjungan_bisnis.html`           | ID     | C2 Visa Kunjungan Bisnis                           | https://www.imigrasi.go.id/wna/daftar-visa-indonesia/C2                                                                   | 2026-09-26    |
| `imigrasi_visa_d1_kunjungan_wisata.html`           | ID     | D1 Visa Kunjungan Wisata                           | https://www.imigrasi.go.id/wna/daftar-visa-indonesia/D1                                                                   | 2026-09-26    |
| `imigrasi_visa_f1_kunjungan_wisata.html`           | ID     | F1 Visa Kunjungan Wisata                           | https://www.imigrasi.go.id/wna/daftar-visa-indonesia/F1                                                                   | 2026-09-26    |
| `imigrasi_visa_e31a_keluarga_suami_istri_wni.html` | ID     | E31A Visa Keluarga Suami/Istri WNI                 | https://www.imigrasi.go.id/wna/daftar-visa-indonesia/E31A                                                                 | 2026-09-26    |
| `imigrasi_visa_e33g_pekerja_jarak_jauh.html`       | ID     | E33G Visa Rumah Kedua Pekerja Jarak Jauh           | https://www.imigrasi.go.id/wna/daftar-visa-indonesia/E33G                                                                 | 2026-09-26    |
| `imigrasi_paspor_masyarakat_umum.html`             | ID     | Permohonan Paspor Baru — Masyarakat Umum           | https://www.imigrasi.go.id/layanan-wni/paspor-republik-indonesia/permohonan-baru/masyarakat-umum                            | 2026-09-26    |
| `imigrasi_biaya_keimigrasian.html`                 | ID     | Biaya Keimigrasian (PNBP)                          | https://www.imigrasi.go.id/biaya_imigrasi/index                                                                           | 2026-09-26    |
| `imigrasi_batam_tata_cara_m_paspor.html`           | ID     | Tata Cara Penggunaan M-Paspor                      | https://batam.imigrasi.go.id/page//info-publik/tata-cara-penggunaan-m-paspor                                             | 2026-09-26    |
| `imigrasi_evisa_info_evoa.html`                    | EN     | Informasi e-Visa dan e-VOA                         | https://evisa.imigrasi.go.id/front/info/evoa                                                                              | 2026-09-26    |
| `imigrasi_faq_visa.html`                           | ID     | FAQ Visa                                           | https://www.imigrasi.go.id/faq/visa                                                                                       | 2026-09-26    |
| `imigrasi_daftar_negara_voa_bvk_calling_visa.html` | ID     | Daftar Negara Subjek VoA, BVK & Calling Visa       | https://www.imigrasi.go.id/wna/daftar-negara-voa-bvk-calling-visa                                                         | 2026-09-26    |
| `imigrasi_faq_negara_e_voa.html`                   | ID     | FAQ: Negara yang Dapat Mengajukan e-VOA            | https://www.imigrasi.go.id/faq/visa/negara-mana-saja-yang-terdaftar-dalam-daftar-electronic-visa-on-arrival-e-voa        | 2026-09-26    |
| `imigrasi_uu_keimigrasian_bab_4_5.html`            | ID     | UU Keimigrasian — Bab IV (Dokumen Perjalanan) & V (Visa/Izin Tinggal) | https://depok.imigrasi.go.id/uu-keimigrasian/ (dipangkas)                                             | 2026-09-26    |
| `sg_ica_visa_requirements.html`                    | EN     | Singapura — Check if You Need an Entry Visa        | https://www.ica.gov.sg/enter-transit-depart/entering-singapore/visa_requirements                                          | 2026-09-26    |
| `sg_ica_visa_free_transit_facility.html`           | EN     | Singapura — Visa Free Transit Facility             | https://www.ica.gov.sg/enter-transit-depart/entering-singapore/visa-free-transit-facility                                 | 2026-09-26    |
| `my_pas_lawatan_sosial_jangka_panjang.html`        | MS     | Malaysia — Pas Lawatan Sosial Jangka Panjang       | https://www.imi.gov.my/index.php/perkhidmatan-utama/pas/pas-lawatan/pas-lawatan-sosial/pas-lawatan-sosial-jangka-panjang/ | 2026-09-26    |

Sembilan belas dokumen berbahasa Indonesia, ditambah tiga dokumen berbahasa Inggris (dua dari Singapura, satu portal e-Visa Indonesia) dan satu berbahasa Melayu (Malaysia), karena tidak selalu ada sumber resmi berbahasa Indonesia yang dapat diambil otomatis.

### Catatan Freshness

Aturan visa dan imigrasi dapat berubah sewaktu-waktu. Bila dokumen di atas dinilai sudah lama, unduh ulang dari sumber resmi, jalankan `scripts/build_index.py`, lalu perbarui tanggal unduh pada tabel ini.
