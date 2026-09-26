# `app.py` — Aplikasi Streamlit

Entry point UI: chat RAG, memory percakapan, dan sidebar dokumen.
Dijalankan dengan `streamlit run app.py`.

```
app.py                 # UI + loop chat (tidak ada logika retrieval di sini)
rag/
  prompt_builder.py    # system prompt §6 + konteks berlabel + riwayat
  generator.py         # condensation + LLM call
```

Seluruh logika berat tetap di paket `rag/`; `app.py` hanya merangkai UI,
session state, dan pemanggilan fungsi.

---

## Alur Runtime

```
┌──────────────────────────────────────────────────────┐
│ st.chat_input — pesan pengguna                       │
└──────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│ rag/generator.py · condense_query()                  │
│ riwayat + pesan terakhir → pertanyaan mandiri        │
│ (dilewati bila riwayat kosong)                       │
└──────────────────────────────────────────────────────┘
                           │ pertanyaan mandiri
                           ▼
┌──────────────────────────────────────────────────────┐
│ rag/embedder.py · embed_query()                      │
│ gemini-embedding-2 + prefix query                    │
└──────────────────────────────────────────────────────┘
                           │ vektor query
                           ▼
┌──────────────────────────────────────────────────────┐
│ rag/retriever.py · Retriever.search()                │
│ top-k IndexFlatIP + gate SIMILARITY_THRESHOLD        │
└──────────────────────────────────────────────────────┘
                           │ list[Hit] (bisa kosong)
                           ▼
┌──────────────────────────────────────────────────────┐
│ rag/prompt_builder.py · build_system_instruction()   │
│ §6 + konteks berlabel + riwayat beranggaran token    │
└──────────────────────────────────────────────────────┘
                           │ system instruction
                           ▼
┌──────────────────────────────────────────────────────┐
│ rag/generator.py · generate_answer()                 │
│ gemini-3.5-flash-lite · temperature 0.2              │
└──────────────────────────────────────────────────────┘
                           │ jawaban prosa (tanpa sitasi)
                           ▼
┌──────────────────────────────────────────────────────┐
│ st.markdown(jawaban) — prosa tanpa daftar sumber     │
└──────────────────────────────────────────────────────┘
```

Perhatikan bahwa yang di-embed dan dicari adalah **pertanyaan mandiri** hasil
condensation, sedangkan yang dijawab LLM adalah **pesan asli** pengguna
(DECISIONS §6). Bila `Retriever.search()` mengembalikan list kosong karena gate
threshold, konteks yang disuntikkan kosong dan aturan §6 nomor 2 menghasilkan
penolakan jujur.

---

## Session State

| Key                     | Isi                                                                |
| ----------------------- | ------------------------------------------------------------------ |
| `messages`              | Daftar `{"role", "content"}` — riwayat percakapan                  |
| `chunks`                | Seluruh `Chunk` aktif (index bawaan + hasil upload)                |
| `vectors`               | Vektor untuk tiap chunk di `chunks`                                |
| `retriever`             | `Retriever` aktif; di-set `None` untuk memaksa build ulang         |

Riwayat disimpan di `st.session_state`, jadi tetap ada selama sesi (refresh
menghapusnya, sesuai non-goal PRD §2). Jendela riwayat yang dikirim ke model
dibatasi `MEMORY_TOKEN_BUDGET` oleh `window_history()`.

## Cache

`@st.cache_resource` dipakai untuk dua hal (DECISIONS §13), karena Streamlit
menjalankan ulang seluruh skrip setiap interaksi:

- `_client()` — `genai.Client`, agar tidak dibuat ulang tiap pesan.
- `_shipped_corpus()` — index bawaan (`data/index/`) dipecah jadi `chunks` +
  `vectors` mentah (`index.reconstruct_n`), sehingga upload bisa menambah vektor
  tanpa meng-embed ulang 203 chunk bawaan.

## Sidebar

- **Info dokumen**: jumlah chunk aktif dan jumlah dokumen unik, plus expander
  berisi daftar nama dokumen.
- **Tambah dokumen** (`st.file_uploader`, PDF/HTML): isi file ditulis ke berkas
  sementara, dipecah `load_file()`, di-embed, lalu **ditambahkan** ke index sesi
  (bukan menggantikan korpus bawaan). Index dibangun ulang dari daftar chunk +
  vektor.
- **Reset dokumen**: mengembalikan korpus ke index bawaan.
- **Reset percakapan**: mengosongkan `messages`.

Nama dokumen hasil upload diambil dari `<h1>`/`<title>` seperti loader biasa,
bukan dari nama file, sehingga nama berkas sementara tidak bocor ke judul
dokumen.

## Sumber Jawaban

Jawaban tampil sebagai prosa murni — tidak ada daftar sumber, expander, maupun
penanda kutipan. Ini penyimpangan sadar dari FR5/FR8 dengan alasan di DECISIONS
§8. Yang menjaga perilakunya:

- `format_context()` menyuntik tiap chunk dengan header dokumen (nama dokumen +
  bagian/halaman) **tanpa nomor**, sehingga tidak ada nomor yang bisa dikutip
  model.
- Aturan 3 `SYSTEM_PROMPT` melarang model menulis daftar sumber atau penanda
  kutipan.

Konsekuensinya tidak ada sitasi yang bisa dikarang. Metadata lokasi chunk tetap
ada dan tetap dipakai sebagai header konteks; yang dihapus hanya penyajiannya di
UI.

## API Key

`config.GEMINI_API_KEY` dibaca dari `.env` lewat `python-dotenv`. Untuk deploy,
`_api_key()` juga membaca `st.secrets["GEMINI_API_KEY"]`, sehingga kunci bisa
diisi lewat dashboard Streamlit Cloud tanpa `.env`. Bila keduanya kosong,
aplikasi berhenti dengan `st.error`.

## Konfigurasi yang Dipakai

| Konstanta                       | Peran di UI                                  |
| ------------------------------- | -------------------------------------------- |
| `LLM_MODEL` / `LLM_TEMPERATURE` | Generation                                   |
| `CONDENSE_TEMPERATURE`          | Penulisan ulang pertanyaan lanjutan          |
| `TOP_K_RETRIEVAL`               | Jumlah hit                                   |
| `SIMILARITY_THRESHOLD`          | Gate penolakan (aturan §6 nomor 2)           |
| `MEMORY_TOKEN_BUDGET`           | Batas riwayat yang dikirim ke model          |

## Belum Diuji

- **Upload PDF.** Korpusnya masih sepenuhnya HTML; jalur `pypdf` di `loader.py`
  belum pernah dieksekusi (lihat [`data.md`](data.md)).
- **Dokumen duplikat.** Mengunggah dokumen yang sudah ada akan menambah chunk
  kembar; tidak ada de-duplikasi berdasarkan nama dokumen.
