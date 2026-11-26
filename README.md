# Tokopedia Play Store Sentiment Analysis

Aplikasi web full-stack monolith berbasis **FastAPI** dan **HTML5, CSS3, Vanilla JavaScript** (tema Neo-Brutalism) untuk klasifikasi sentimen 3 kelas (`Positif`, `Netral`, `Negatif`) pada ulasan aplikasi Google Play Store berbahasa Indonesia.

## Fitur Utama

1. **Global Model Switcher**: Beralih langsung di header antara 3 model terlatih:
   - `bilstm`: Deep Learning (**CNN + Bi-LSTM** dengan Word Embedding, diekspor ke **ONNX Runtime** agar hemat memori di produksi, default)
   - `svm`: Machine Learning (**SVM LinearSVC + TF-IDF**) dengan konversi `decision_function` ke confidence score via Softmax
   - `logreg`: Machine Learning (**Logistic Regression + Bag of Words**)
2. **Ulasan Tunggal (Live Single-Review)**: Pengujian teks ulasan secara langsung lengkap dengan contoh cepat (Positif, Negatif, Netral), stempel sentimen, confidence score, dan panel collapsible untuk melihat hasil `clean_text` dari pipeline NLP.
3. **Batch CSV Upload**: Unggah file `.csv` (drag-and-drop, batas maksimal 2.000 baris), tampilkan ringkasan proporsi sentimen, grafik Doughnut interaktif (**Chart.js**), tabel pratinjau 10 baris pertama, dan tombol unduh CSV hasil lengkap dengan kolom `clean_text`, `predicted_sentiment`, serta `confidence`.
4. **Live Play Store Scraper**: Tarik 25, 50, atau 100 ulasan terbaru langsung dari Google Play Store Indonesia (`com.tokopedia.tkpd`, `com.gojek.app`, `id.dana`, atau App ID lainnya), lengkap dengan grafik distribusi sentimen, **Top 10 Keywords** (Positif vs Negatif), dan feed ulasan terbaru.

## Struktur Direktori

```text
tokopedia-sentiment-analysis/
├── .gitignore
├── .vercelignore
├── README.md
├── main.py
├── requirements.txt
├── vercel.json
├── notebooks/
│   ├── 01_data_scraping.ipynb
│   └── 02_model_training_and_evaluation.ipynb
├── artifacts/
│   ├── bow_vectorizer.joblib
│   ├── label_encoder.joblib
│   ├── model_bilstm.onnx
│   ├── model_logreg.joblib
│   ├── model_svm.joblib
│   ├── stopwords_id.json
│   ├── tfidf_vectorizer.joblib
│   ├── tokenizer_config.json
│   └── unified_lexicon.joblib
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── inference.py
│   ├── preprocessing.py
│   └── scraper.py
└── static/
    ├── index.html
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

## Pipeline NLP & Evaluasi Model

Setiap ulasan diproses secara konsisten melalui `src/preprocessing.py`:

1. **Konversi Emoji Sentimen**: Memetakan emoji ekspresif ke token teks (`bagus` / `buruk`) sebelum penghapusan karakter non-ASCII.
2. **Pembersihan Teks**: Case folding serta penghapusan URL, mention/hashtag, angka, tanda baca, dan karakter non-ASCII.
3. **Normalisasi Slang & Code-Mixing**: Memetakan singkatan dan istilah e-commerce Indonesia (misalnya `gk`/`gak` menjadi `tidak`, `lemot` menjadi `lambat`, `eror`/`fc` menjadi `error`, `ongkir` menjadi `ongkos kirim`).
4. **Stopword Removal Selektif**: Menghapus stopwords NLTK Bahasa Indonesia dan Inggris dasar sambil mempertahankan kata negasi/emosi (`tidak`, `bukan`, `jangan`, `kurang`, `belum`, `lama`, `hilang`, `bagus`, `buruk`).
5. **Neutral Fallback**: Jika `clean_text` kosong atau tidak mengandung satu pun kata dari `unified_lexicon`, sistem mengembalikan label `Netral` (`90.0%`) agar kalimat faktual tidak salah terklasifikasi.

### Ringkasan Evaluasi (12.272 Ulasan Bersih)

| Skema | Algoritma | Ekstraksi Fitur | Split Data | Akurasi Training | Akurasi Testing |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Skema 1 (Default) | **CNN + Bi-LSTM (ONNX)** | Word Embedding | 80 / 20 | 95.85% | **90.31%** |
| Skema 2 | **SVM (LinearSVC)** | TF-IDF | 80 / 20 | 98.25% | 86.60% |
| Skema 3 | **Logistic Regression** | Bag of Words | 70 / 30 | 99.66% | 89.71% |

## Endpoint REST API

| Method | Endpoint | Deskripsi |
| :--- | :--- | :--- |
| `GET` | `/` | Menyajikan antarmuka web (`static/index.html`) |
| `GET` | `/api/health` | Status server dan daftar model yang dimuat di memori |
| `POST` | `/api/predict` | Prediksi sentimen ulasan tunggal (`text`, `model_type`) |
| `POST` | `/api/predict-batch` | Prediksi massal file `.csv` (`file`, `text_column`, `model_type`) |
| `POST` | `/api/scrape-analyze` | Live scraping Play Store, prediksi, dan top keywords (`app_id`, `count`, `model_type`) |

Dokumentasi Swagger UI otomatis tersedia di `/docs` saat server berjalan.

## Prasyarat & Instalasi Lokal

- Python 3.12 atau lebih baru
- Koneksi internet (hanya saat menjalankan fitur Live Play Store Scraper)

```bash
cd tokopedia-sentiment-analysis
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Cara Menjalankan di Lokal

```bash
uvicorn main:app --host 127.0.0.1 --port 7860 --reload
```

Setelah server aktif, buka `http://127.0.0.1:7860` di browser (atau `http://127.0.0.1:7860/docs` untuk mencoba endpoint API secara langsung).
