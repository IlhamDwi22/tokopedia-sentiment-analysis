# Tokopedia Play Store Sentiment Analysis — Full-Stack AI Web App

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Hugging%20Face%20Spaces-2496ED?logo=docker&logoColor=white)

Aplikasi web **Full-Stack Monolith** berbasis **FastAPI + HTML5/CSS3/Vanilla JavaScript** untuk analisis sentimen 3 kelas (`Positif`, `Netral`, `Negatif`) pada ulasan aplikasi Google Play Store berbahasa Indonesia.

Proyek ini merupakan hasil refactoring dari eksperimen Jupyter Notebook (scraping + pelatihan 3 skema model ML/DL) menjadi sistem inferensi siap *deploy* ke **Hugging Face Spaces (Docker SDK)** maupun **Render**.

---

## Fitur Utama

1. **Global Model Switcher** — Beralih secara langsung di header antara 3 arsitektur model terlatih:
   - `bilstm` — **Deep Learning (CNN + Bi-LSTM + Word Embedding)** *(default)*
   - `svm` — **Machine Learning (SVM LinearSVC + TF-IDF)** dengan konversi *decision function* ke *confidence score* via Softmax
   - `logreg` — **Machine Learning (Logistic Regression + Bag of Words)**
2. **Live Single-Review Analyzer** — Analisis teks ulasan tunggal lengkap dengan *preset sample pills*, badge sentimen berkontras tinggi, *confidence progress bar*, dan *collapsible NLP preprocessing accordion*.
3. **Batch CSV Upload** — Unggah file `.csv` (drag-and-drop, batas hingga 2.000 baris), tampilkan kartu metrik, grafik Doughnut interaktif (**Chart.js**), tabel pratinjau 10 baris, serta unduh ulang hasil CSV lengkap dengan kolom `clean_text`, `predicted_sentiment`, dan `confidence`.
4. **Live On-Demand Play Store Scraper** — Tarik 25 / 50 / 100 ulasan terbaru langsung dari Google Play Store Indonesia (`com.tokopedia.tkpd`, `com.gojek.app`, `id.dana`, atau App ID lainnya), lengkap dengan grafik distribusi, **Top 10 Keywords** (Positif vs Negatif), dan feed ulasan live.

---

## Struktur Direktori

```text
tokopedia-sentiment-analysis/
├── Dockerfile                  # Container image siap Hugging Face Spaces (Port 7860)
├── .dockerignore
├── .gitignore
├── README.md
├── requirements.txt
├── main.py                     # FastAPI entrypoint, lifespan loader & static mount
├── notebooks/
│   ├── 01_data_scraping.ipynb
│   └── 02_model_training_and_evaluation.ipynb
├── artifacts/                  # 8 artefak model & NLP terlatih
│   ├── model_bilstm.keras
│   ├── model_svm.joblib
│   ├── model_logreg.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── bow_vectorizer.joblib
│   ├── tokenizer.joblib
│   ├── label_encoder.joblib
│   └── unified_lexicon.joblib
├── src/
│   ├── __init__.py
│   ├── config.py               # Konstanta path, MAX_LEN=50, kamus emoji & slang
│   ├── preprocessing.py        # Pipeline NLP (emoji, regex, slang, stopwords)
│   ├── inference.py            # Multi-model SentimentEngine & Neutral Fallback
│   └── scraper.py              # Wrapper google-play-scraper & top keywords
└── static/
    ├── index.html              # Single Page Application (3 tab)
    ├── css/style.css           # UI modern minimalis responsif
    └── js/app.js               # Fetch API, DOM state, & Chart.js rendering
```

---

## Pipeline NLP & Pelabelan

Setiap ulasan melewati urutan preprocessing yang konsisten antara pelatihan dan produksi (`src/preprocessing.py`):

1. **Konversi Emoji Sentimen** (`👍, ❤️, 🔥` → `bagus`; `👎, 😡, 😢` → `buruk`) *sebelum* menghapus karakter non-ASCII.
2. **Pembersihan Teks** — *case folding*, penghapusan URL, mention/hashtag, angka, tanda baca, dan karakter non-ASCII.
3. **Normalisasi Slang & Code-Mixing** — memetakan bahasa gaul & istilah e-commerce (`gk/gak/ga` → `tidak`, `lemot/lelet` → `lambat`, `eror/bug/fc` → `error`, `ongkir` → `ongkos kirim`, dll.).
4. **Stopword Removal Selektif** — menghapus stopwords NLTK Bahasa Indonesia & Inggris dasar sambil mempertahankan kata negasi/emosi (`tidak`, `bukan`, `jangan`, `kurang`, `belum`, `lama`, `hilang`, `bagus`, `buruk`).
5. **Neutral Fallback** — jika `clean_text` kosong atau tidak mengandung kata dari `unified_lexicon`, sistem mengembalikan label `Netral` (`90.0%`) agar kalimat faktual tidak salah terklasifikasi.

---

## Evaluasi 3 Skema Model

Evaluasi dilakukan pada **12.272 ulasan bersih** (pelabelan 3 kelas berbasis *High-Confidence InSet Lexicon*):

| Skema | Algoritma | Ekstraksi Fitur | Split Data | Akurasi Training | Akurasi Testing |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Skema 1 *(Default)* | **CNN + Bi-LSTM** | Word Embedding | 80 / 20 | 95.85% | **90.31%** |
| Skema 2 | **SVM (LinearSVC)** | TF-IDF | 80 / 20 | 98.25% | 86.60% |
| Skema 3 | **Logistic Regression** | Bag of Words | 70 / 30 | 99.66% | 89.71% |

---

## Dokumentasi REST API

| Method | Endpoint | Deskripsi |
| :--- | :--- | :--- |
| `GET` | `/` | Menyajikan antarmuka web (`static/index.html`) |
| `GET` | `/api/health` | Status server dan daftar model yang dimuat di memori |
| `POST` | `/api/predict` | Prediksi sentimen ulasan tunggal (`text`, `model_type`) |
| `POST` | `/api/predict-batch` | Prediksi massal file `.csv` (`file`, `text_column`, `model_type`) |
| `POST` | `/api/scrape-analyze` | Live scraping Play Store + prediksi + top keywords (`app_id`, `count`, `model_type`) |

Dokumentasi interaktif Swagger UI tersedia di `/docs` saat server berjalan.

---

## Menjalankan Secara Lokal

### 1. Instal Dependensi
```bash
cd tokopedia-sentiment-analysis
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Jalankan Server FastAPI
```bash
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```
Buka browser ke `http://localhost:7860`.

---

## Deploy ke Hugging Face Spaces (Docker SDK) / Render

### Hugging Face Spaces
1. Buat Space baru di [Hugging Face Spaces](https://huggingface.co/spaces) dengan SDK **Docker**.
2. Push isi direktori `tokopedia-sentiment-analysis/` (termasuk folder `artifacts/` — gunakan Git LFS jika ukuran model melebihi batas file normal Git).
3. Space akan otomatis membangun `Dockerfile` dan menjalankan aplikasi pada port `7860`.

### Docker Lokal
```bash
docker build -t tokopedia-sentiment-app .
docker run -p 7860:7860 tokopedia-sentiment-app
```
